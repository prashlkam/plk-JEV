# HARDWARE.md — the GTX 1650 re-plan

Everything in `PLAN.md` that depended on hardware, re-derived from measurements on this
machine. Where this file and `PLAN.md` disagree, **this file wins** — `PLAN.md` was
written against an unknown "local consumer GPU" and its VRAM table started at 8GB.

Measured 2026-10-08. Regenerate with `make spikes`.

---

## The machine

| | |
|---|---|
| GPU | GeForce GTX 1650, **4096 MiB**, compute capability **7.5** (Turing TU117) |
| usable VRAM | **3.32 GiB** with the desktop running, **2.37 GiB** if Ollama holds a model |
| driver / CUDA | 615.71.09 / UMD 13.4, system nvcc 13.4 |
| tensor cores | **none** (TU117 ships without them) |
| native bf16 | **no** — `is_bf16_supported(including_emulation=False)` is `False` |
| FlashAttention | **unavailable** (FA2 needs sm_80+); SDPA `EFFICIENT_ATTENTION` works |
| CPU / RAM | 16 cores / 15 GiB + 15 GiB swap |
| disk | `/home` **7.2 GiB free** ← binding; `/` 63 GiB free |
| Ollama | 0.32.15 (logprobs landed in 0.12.11, so well supported) |
| Python | system 3.14.7 (too new for the ML stack) → **uv-managed 3.12.13** |

---

## Revision 1 — train in fp32, not fp16 or bf16

**This is the single most consequential finding and it contradicts every standard
fine-tuning recipe.** Measured matmul throughput, 2048³, reproducible across sizes:

| dtype | TFLOP/s | vs fp32 |
|---|---|---|
| **fp32** | **2.64** | 1.00× |
| bf16 (emulated) | 1.53 | 0.58× |
| fp16 | 0.31 | **0.12×** |

And on a real SmolLM2-135M train step (seq 192, batch 8, 8-bit Adam, sliced LM head):

| config | tok/s | peak VRAM |
|---|---|---|
| **fp32** | **1818** | 1.49 GiB |
| bf16 | 1385 | 0.88 GiB |
| fp16 AMP + GradScaler | **346** | 1.39 GiB |

TU117 has no tensor cores and no fast FP16x2 path, so fp16 falls back to a pathological
kernel: **fp16 AMP makes training 5.3× slower**. bf16 is emulated — cheaper on memory
but 24% slower than fp32.

- **Default: fp32 compute.** `--dtype fp32`, no `autocast`, no `GradScaler`.
- **bf16 is the memory escape hatch only** — reach for it when a config OOMs in fp32
  (it roughly halves activation + weight memory for a 24% throughput cut), never for speed.
- Never enable fp16 AMP on this card.
- TF32 does not exist pre-Ampere; `allow_tf32` is a no-op here (measured 2.63 vs 2.64).

`PLAN.md` §5 said "bf16, flash-attention if supported". Both are wrong for this GPU.

## Revision 2 — slice the LM head to the loss positions

Decision SFT computes loss on **exactly one token**, so materializing
`[batch, seq, vocab]` logits is pure waste. Computing the LM head only at the decision
position:

| | tok/s | peak VRAM |
|---|---|---|
| full logits | 1522 | 1.72 GiB |
| **sliced to last position** | **1818** | **1.49 GiB** |

+19% throughput and −13% memory, free. Implemented in `train/sft.py`
(`--sliced-lm-head`, on by default). The win scales with vocab, so it is larger for
Qwen3-0.6B (vocab 151936) than for SmolLM2-135M (49152).

## Revision 3 — gradient checkpointing is now a memory *option*, not a default

| config | tok/s | peak VRAM |
|---|---|---|
| no checkpointing, b8 | **2194** | 2.82 GiB |
| checkpointing, b8 | 1818 | 1.49 GiB |
| checkpointing, b16 | 1767 | 1.49 GiB |
| checkpointing, b32 | 1555 | 1.62 GiB |

Checkpointing costs 17% throughput and buys 1.33 GiB. Note activation memory is nearly
**batch-independent** once checkpointing is on (1.49 GiB at b8 and b16 alike), so large
micro-batches are almost free in that mode.

- **Default: checkpointing on, micro-batch 16.** 1767 tok/s at 1.49 GiB leaves room for
  Ollama to stay resident and survives fragmentation.
- `--no-grad-ckpt --batch 8` is the 24%-faster config for a dedicated run, but 2.82 GiB
  against 3.32 GiB available is a ~0.5 GiB margin — it will OOM if anything else touches
  the GPU.

## Revision 4 — Ollama and training cannot share this GPU

A resident `qwen3:0.6b` costs **964 MiB**, cutting the training budget from 3.32 to
2.37 GiB. Before any training run:

```bash
ollama stop qwen3:0.6b && export OLLAMA_KEEP_ALIVE=0
```

`make train-*` does this automatically. The eval harness needs Ollama *up*, so eval and
training are sequential, never concurrent. Reclaim a further ~300 MiB by stopping the
desktop compositor if a config is genuinely borderline.

## Revision 5 — model selection, rescoped

`PLAN.md` §3 picked Qwen3-0.6B-Base with a "full fine-tune at 0.6B over QLoRA at 1.7B"
preference. At 3.32 GiB a 0.6B full fine-tune needs ~2.4 GiB for fp32 weights alone
before gradients or optimizer state, so it is out.

| role | model | mode | fits | throughput |
|---|---|---|---|---|
| **primary** | **SmolLM2-135M-Base** | **full FT, fp32** | 1.49 GiB | 1767 tok/s |
| quality variant | Qwen3-0.6B-Base | QLoRA NF4 | ~1.6 GiB est. | ~3-5× slower |
| baseline | DeBERTa-v3-small / distilbert | full FT | easily | minutes |
| baseline | TF-IDF + logistic regression | CPU | n/a | seconds |

SmolLM2-135M is the right primary: Apache 2.0, Llama architecture (so
`convert_hf_to_gguf.py` handles it with no custom converter), 135M params, and a 49k
vocab that keeps the LM head cheap. For six closed-set decision tasks, 135M is not
obviously underpowered — and the DeBERTa baseline exists precisely to test that claim.

The 0.6B QLoRA variant stays in the plan as a deliberate multi-day run, not the default.
Note it needs `bnb_4bit_compute_dtype=float32` here, not the usual float16.

## Revision 6 — compute budget, recomputed

### First, a correction to the headline rate

The 1767 tok/s above is a **fixed-length** benchmark (every sequence exactly 192
tokens). The real SFT run measures **866–954 tok/s of real tokens**, and the gap is
not a regression — it is padding, counted honestly:

- real prompts are ~45–192 tokens, p50 around 80;
- a naive batch pads every member to the batch maximum, so ~55% of the compute goes
  to pad tokens;
- the throughput counter reports only **non-pad** tokens (`attention_mask.sum()`),
  so padded work is correctly excluded from the numerator.

`train/sft.py --group-by-length` (default on) fixes the cause rather than the
accounting: it shuffles, chunks into mega-batches, sorts each by length, then
reshuffles the batch order, so batches are length-homogeneous while the composition
stays random. It also hoists tokenization out of the training loop.

**Budget below uses the conservative 950 tok/s measured on real data**, not the
1767 fixed-length figure. Treat 1767 as the ceiling for a packed corpus — which the
CPT phase *is*, since `data/prepare_cpt.py` emits fixed-length packed blocks, so CPT
really does run near 1767.

| run | tokens | at 950 tok/s (SFT, real) | at 1767 tok/s (CPT, packed) |
|---|---|---|---|
| decision SFT, 100k × 128 tok | 12.8 M | **3.7 h** | — |
| decision SFT, 200k × 160 tok | 32 M | **9.4 h** | — |
| CPT, 25 M tokens | 25 M | — | **3.9 h** |
| CPT, 50 M tokens | 50 M | — | **7.9 h** |
| CPT, 100 M tokens | 100 M | — | **15.7 h** |

Original estimates, for reference, at the fixed-length rate:

| run | tokens | wall clock |
|---|---|---|
| decision SFT, 100k × 128 tok | 12.8 M | **2.0 h** |
| decision SFT, 200k × 160 tok | 32 M | **5.0 h** |
| CPT, 25 M tokens | 25 M | **3.9 h** |
| CPT, 50 M tokens | 50 M | **7.9 h** |
| CPT, 100 M tokens | 100 M | **15.7 h** |

So `PLAN.md` §4a's "0.5–2B tokens" for CPT is 8–30 **days** and is cut. **CPT budget is
50M tokens** (one overnight run) with 100M as a stretch. That is a real reduction in
ambition: expect a modest distribution shift, not a domain-expert base. The §5 gate
("domain perplexity ≥15% better") is kept, but if 50M tokens does not clear it, the
correct response is to **skip CPT entirely** and spend the time on SFT data instead —
for closed-set classification, CPT is the most cuttable phase in the plan.

Phases 4–6 were budgeted to loop 2–4 times. At these rates a full loop is ~1 day, so the
loop is affordable.

## Revision 7 — everything heavy lives on `/`, not `/home`

`/home` has 7.2 GiB free; the venv alone is **7.0 GiB**. Installing into the project
would have filled the partition.

```
/var/tmp/jev-workspace/     (63 GiB free, on /)
├── .venv/        7.0 GiB   Python 3.12 + torch 2.14.1+cu126
├── hf-cache/               HF_HOME — model + dataset downloads
├── data/                   generated corpora, tokenized caches
├── out/                    checkpoints, GGUF files
└── pip-cache/
```

The project symlinks it as `./workspace`. `env.sh` exports `HF_HOME` and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (fragmentation mitigation, which
matters a lot at 3.3 GiB).

**Caveat:** `/usr/lib/tmpfiles.d/tmp.conf` carries `q /var/tmp 1777 root root 30d`, so
systemd will prune `/var/tmp` entries after 30 days of inactivity. Install the shipped
exclusion to make the workspace durable:

```bash
sudo cp packaging/tmpfiles-jev.conf /etc/tmpfiles.d/jev-workspace.conf
```

Or set `JEV_WORKSPACE` to somewhere permanent with ≥15 GiB and re-run `make setup`.

## Revision 8 — toolchain hazards avoided

- **Python 3.14.7 is the system interpreter** and the ML stack has no wheels for it.
  Everything runs under uv-managed **3.12.13**.
- **gcc 16.2.1 is far newer than nvcc 13.4 accepts.** This would break a CUDA build of
  llama.cpp — so `serve/convert.sh` builds llama.cpp **CPU-only**
  (`-DGGML_CUDA=OFF`). We only need `convert_hf_to_gguf.py` (pure Python) and
  `llama-quantize` (CPU); GPU inference is Ollama's job. No CUDA compile, no gcc pin.
- torch wheels are **cu126**, chosen over cu128/cu130 because `sm_75` is confirmed
  present in its arch list (`sm_50 … sm_90`). Verify after any upgrade with
  `make check-gpu` — a wheel that drops Turing would silently fall back to CPU.

---

## The config that falls out of all this

`train/configs/sft_smollm135.yaml`:

```yaml
model: HuggingFaceTB/SmolLM2-135M
dtype: fp32                # Revision 1 — NOT bf16, never fp16
attn_implementation: sdpa  # EFFICIENT_ATTENTION; no FlashAttention on sm_75
sliced_lm_head: true       # Revision 2
gradient_checkpointing: true
micro_batch_size: 16       # Revision 3 — activations are batch-independent here
gradient_accumulation: 4   # effective batch 64
max_seq_len: 192           # decisions are short; truncate hard
optim: adamw_8bit          # bitsandbytes; fp32 AdamW would cost +0.8 GiB
lr: 1.0e-5
epochs: 3
early_stop_metric: dev_ece # not accuracy — see spikes/RESULTS.md
```

Expected: ~950 tok/s of real tokens (1767 is the fixed-length ceiling; the
difference is padding, see revision 6), ~1.65 GiB peak, ~3.7 h for 100k examples.
