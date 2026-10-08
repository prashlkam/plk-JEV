# JEV — Build Plan

> **STATUS: this plan has been implemented, and parts of it are superseded by
> measurement.** It is kept as written so the reasoning and the corrections stay
> legible side by side. Before following any section here, check:
>
> - **[HARDWARE.md](HARDWARE.md)** — 8 measured revisions for the actual GPU (a GTX
>   1650, 4GB). It replaces §3 (model selection), §5 (CPT scope and dtype) and the
>   VRAM table, which assumed ≥8GB. Headline: **fp32 is the fastest dtype on this
>   card** — fp16 AMP is 5.3× slower — and the CPT budget drops from 0.5–2B tokens
>   to 50M.
> - **[spikes/RESULTS.md](spikes/RESULTS.md)** — Phase 0 findings. §6–§7's output
>   encoding is **reversed**: Ollama reports logprobs *before* the grammar mask, and
>   forced JSON scaffolding corrupts the decision token, so decisions are emitted as
>   **bare integer enums, one field per request** rather than as a compact JSON
>   object. 1 decode step instead of 27.
> - **[data/README.md](data/README.md)** — §4's data phase, plus a negative result:
>   on the synthetic stand-in corpus, TF-IDF + logistic regression reaches 1.000
>   accuracy on five of six tasks, so **risk #3 in §12 has already fired**.
> - **[README.md](README.md)** — what is built, what is verified, what is not done.

A from-scratch pipeline for a **System One model**: a small model that makes fast,
schema-constrained decisions with calibrated confidence, served through Ollama.

Reference concept: [TypeSafe AI — Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
This plan reproduces Jev's *three observable properties* on open infrastructure. It does
not attempt to clone their unpublished architecture or RLCD training recipe.

---

## 0. The central constraint (read this first)

Jev's headline mechanism is **parallel sampling**: all outputs produced in a single
forward pass, via classifier heads over a fixed schema rather than autoregressive token
generation.

**Ollama cannot serve that.** Ollama exposes three model shapes — `generate`/`chat`
(causal LM), and `embed`. It has **no classification or rerank endpoint**: for
cross-encoder models it exposes only the embedding layer, never the classification head.
This has been an open request since March 2024
([#3368](https://github.com/ollama/ollama/issues/3368),
[#5399](https://github.com/ollama/ollama/issues/5399),
[#13183](https://github.com/ollama/ollama/issues/13183)) with no landed support.

So "JEV, compatible with Ollama" forces a choice:

| | Path A — faithful architecture | **Path B — Ollama-native (recommended)** |
|---|---|---|
| Model shape | Encoder trunk + per-field classifier heads | Causal decoder LM |
| Output | 1 forward pass, all fields | 1 decode step *per field* |
| Type safety | By construction (fixed head arity) | By construction (grammar-constrained decoding) |
| Confidence | Head softmax | Token logprobs |
| Ollama support | **Requires patching llama.cpp + Ollama's Go runner** | **Works on stock Ollama today** |
| Effort | +4–8 weeks of C++/Go runtime work | Core plan below |

**Path B is the plan.** It matches your stated choices (continued pretrain + fine-tune on
an open base; local consumer GPU) and reaches all three properties. The latency story is
weaker than Jev's claim — *k* decode steps instead of one — but §7 gets that to the
~10–40ms range for small *k*, which is the same order as their 70–500ms figure.

Path A is scoped in §10 as optional future work. Don't start there.

### What Ollama *does* give us

Two primitives make Path B work, and they map 1:1 onto Jev's guarantees:

1. **Structured outputs** — the `format` parameter accepts a full JSON Schema and
   constrains decoding to it ([docs](https://docs.ollama.com/capabilities/structured-outputs)).
   Output outside the schema is not merely unlikely, it is unreachable. This is Jev's
   "hallucinations are mathematically impossible," by the same mechanism.
2. **Logprobs** — `"logprobs": true` and `"top_logprobs": N`, added in
   [v0.12.11](https://github.com/ollama/ollama/releases/tag/v0.12.11). Gives per-token
   probability for the decision tokens, which is the raw material for calibration.

---

## 1. Phase 0 — Feasibility spikes (2 days, do before anything else)

Three assumptions carry the whole design. Verify empirically, not by reading docs.

**Spike 0.1 — Do `format` and `logprobs` compose?**
The load-bearing unknown. Grammar-constrained sampling and logprob reporting are
implemented in different layers; if Ollama renormalizes logprobs over the grammar mask,
great (that is exactly what we want). If it reports *pre-mask* logprobs, confidence must
be recovered by renormalizing over the legal token set client-side from `top_logprobs`.
If `format` suppresses `logprobs` entirely, Path B's calibration story needs rework.

```bash
curl http://localhost:11434/api/chat -d '{"model":"qwen3:0.6b","messages":[{"role":"user","content":"Is this spam? body: free crypto now"}],"format":{"type":"object","properties":{"spam":{"type":"boolean"}},"required":["spam"]},"logprobs":true,"top_logprobs":5,"stream":false}'
```

Record: are logprobs present, and do the top-k candidates sum to ~1.0 within the mask?
**This result determines §6's design. Do not proceed past it blind.**

**Spike 0.2 — Latency floor.**
Measure p50/p99 for a 0.6B model at Q4_K_M on your GPU, with the prompt prefix cached,
for 1, 4, and 16 output tokens. Note: grammar-forced tokens with exactly one legal
continuation **still cost a forward pass** in llama.cpp — there is no token-skipping
fast path. So JSON scaffolding (`{"`, `field`, `":`, …) is not free. This measurement is
what motivates the compact output encoding in §7.

**Spike 0.3 — Single-token label feasibility.**
For your candidate base model's tokenizer, check which label surface forms are exactly
one token. `" yes"`, `" no"`, `" A"`, `" 1"` usually are; `"URGENT"`, `"needs_review"`
usually are not. Dump the mapping now — §5's label vocabulary depends on it.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
for s in [" yes"," no"," A"," B"," 0"," 1","URGENT","needs_review"]:
    print(repr(s), tok.encode(s, add_special_tokens=False))
```

**Gate:** all three green → proceed. 0.1 red → revisit §6 before writing training code.

---

## 2. Phase 1 — Decide what JEV decides (3–5 days)

The hardest part of this project is not training. It is defining the task family
narrowly enough that a sub-1B model beats a frontier LLM on it.

A "System One task" here means: **bounded output space, decidable from the input alone,
no multi-step reasoning, no world knowledge lookup.** Pick 6–10 task types, e.g.

- **Route** — which of *N* handlers/queues/agents handles this input
- **Gate** — boolean preconditions (`is_safe`, `needs_human`, `is_duplicate`)
- **Triage** — ordinal severity (`P0..P3`)
- **Classify** — flat or shallow-hierarchical labels (intent, topic, sentiment)
- **Extract-enum** — field whose value is drawn from a closed set (currency, country, status)
- **Score** — bucketed ordinal (`1..5` confidence/quality/relevance)

Deliverables:
- `spec/tasks.md` — one page per task type: input shape, output schema, decision
  boundary, 10 hand-labeled examples including 3 genuine edge cases.
- `spec/schemas/*.json` — the JSON Schemas, which are simultaneously the training
  target, the eval contract, and the runtime `format` payload. One source of truth.
- **Explicit non-goals.** Write down what JEV must refuse to be: no free text, no
  summarization, no generation, no chain-of-thought. The value proposition *is* the
  restriction.

> Scope warning: if the task list grows open-ended, you are building a general
> instruction model and the whole speed/calibration thesis collapses. Cap it at 10.

---

## 3. Phase 2 — Base model selection (1 day)

Primary: **Qwen3-0.6B-Base** — Apache 2.0, strong for its size, small enough that
*k* decode steps stay fast, well-supported by `convert_hf_to_gguf.py`.

- Quality fallback: **Qwen3-1.7B-Base** if 0.6B underfits after Phase 4.
- Also evaluate **Gemma 4 ~2B** (Apache 2.0) if VRAM allows — newer; verify its GGUF
  conversion path works before committing.
- Use the **`-Base`** checkpoint, not `-Instruct`. You are doing continued pretraining;
  instruct-tuning artifacts (chat scaffolding, verbosity priors, refusal behavior) are
  pure interference for a classifier.

VRAM sizing (full-finetune vs QLoRA, bf16, 2048 ctx):

| VRAM | Continued pretrain | Decision SFT |
|---|---|---|
| 8 GB | 0.6B QLoRA, ctx 1024, grad-ckpt | 0.6B QLoRA |
| 12 GB | 0.6B QLoRA, ctx 2048 | 0.6B full FT |
| 16 GB | 0.6B full FT | 0.6B full FT / 1.7B QLoRA |
| 24 GB | 1.7B QLoRA or 0.6B full FT @ 4096 | 1.7B full FT |

Prefer **full fine-tune at 0.6B over QLoRA at 1.7B** for this workload. You are
reshaping the output distribution, not adding knowledge; LoRA's low-rank constraint
fights that, and calibration quality suffers measurably.

---

## 4. Phase 3 — Data (2–3 weeks; the real bottleneck)

Three distinct datasets. Build them in this order.

### 4a. Continued-pretrain corpus (`data/cpt/`) — target 0.5–2B tokens
Domain text in the distribution JEV will be judged on: support tickets, logs, commit
messages, API payloads, emails, product events — whatever your §2 tasks consume. Goal is
representation shift toward your input distribution, not new capability.
- Dedupe (MinHash/LSH), strip PII, hold out a 5M-token eval slice.
- Pack to fixed-length sequences; document the mix ratios in `data/cpt/MIXTURE.md`.
- **Include a replay slice** — 5–15% general web/instruction text from the base model's
  original distribution, to limit catastrophic forgetting of basic language competence.

### 4b. Decision SFT set (`data/sft/`) — target 50k–200k examples
`(input, schema_id, label, [rationale])` tuples. Sources, in descending preference:
1. **Existing production labels.** Historical routing/triage/moderation decisions already
   made by humans or rules. Free, in-distribution, correctly biased. Mine these first.
2. **Frontier-model distillation.** Label with Claude/GPT under the exact §2 schemas.
   Fast and cheap. **Check the provider's terms** before using outputs to train a model
   you will distribute.
3. **Human labeling** for the hard slice — ambiguous, adversarial, and near-boundary
   cases. Budget most of your human hours here, not on easy examples.

Critical requirements:
- **Label every example with 3+ independent annotators on a ≥2k subset.** Inter-annotator
  agreement is your accuracy ceiling. If humans agree only 80% of the time, a model
  scoring 85% is overfitting the labeler, not the task — and you will never know without
  this measurement.
- **Preserve label disagreement as soft targets.** 2-of-3 annotators → target 0.67, not
  1.0. This is the single highest-leverage thing you can do for calibration, and it is
  destroyed forever if you collapse to majority vote during collection.
- Stratify so rare classes are learnable; record true priors separately for §6.

### 4c. Calibration + eval splits (`data/eval/`)
Carve **three** disjoint splits, stratified by task and class:
- `dev` — training signal, early stopping
- `cal` — **temperature fitting only, never trained on** (5–10k examples)
- `test` — touched once per milestone, in distribution
- `ood` — deliberately out-of-distribution: unseen phrasings, adversarial inputs, other
  domains. Measures whether confidence correctly *drops*. A model that is confidently
  wrong off-distribution is worse than useless in an automated pipeline.

---

## 5. Phase 4 — Continued pretraining (3–7 days compute)

Standard causal LM objective on 4a. Nothing clever here; the goal is a modest
distribution shift without wrecking the base.

- LR 1e-5 to 5e-5, cosine, warmup 2%. **Much lower than pretraining-from-scratch LRs** —
  high LR here is the usual cause of catastrophic forgetting.
- 1–2 epochs. Watch held-out perplexity on both your domain slice *and* the replay
  slice; if replay perplexity climbs sharply, lower LR or raise replay ratio.
- bf16, grad checkpointing, `packing=True`, flash-attention if supported.
- Checkpoint every ~500 steps and keep them. You will want to restart Phase 5 from an
  earlier checkpoint at least once.
- Tooling: HF `transformers` + `trl`, or `axolotl` for config-driven runs. Log to a local
  `wandb`/`tensorboard`.

**Gate:** domain perplexity improves ≥15% vs base, replay perplexity degrades <5%.

---

## 6. Phase 5 — Decision SFT with single-token labels (1 week)

This phase is where JEV becomes JEV. The core design decision:

### Single-token label vocabulary
Map every schema enum value to a **one-token** surface form (from Spike 0.3). Then:
- one decision = **one decode step**
- the logprob of that single token **is** the decision's probability — no multi-token
  probability aggregation, no length normalization, no ambiguity
- `top_logprobs` gives you the full distribution over alternatives for free

Keep the mapping in `spec/label_vocab.json`, versioned alongside the schemas. The client
SDK (§9) translates symbol → human-readable enum value, so the compact encoding never
leaks into the API surface.

### Training format
Train on a fixed, minimal prompt template with the schema identified by ID, not inlined —
inlining full JSON Schema text per request wastes prompt tokens and defeats prefix
caching:

```
<task:route_v1>
{input text}
<decide>
```
→ target: a single label token (plus the field separator for multi-field schemas).

- **Loss on the label token(s) only.** Mask the prompt. This is essential — unmasked
  prompt loss dilutes gradient onto text the model never needs to produce.
- **Train on soft targets** from 4b where available (KL against the annotator
  distribution rather than cross-entropy against a one-hot). Preserves the uncertainty
  you paid annotators to capture.
- **Label smoothing ε≈0.05** on hard-labeled examples. Cheap partial substitute for soft
  targets; prevents the overconfidence that SFT reliably induces.
- Interleave all task types in every batch so the model learns to condition on the task
  tag rather than drifting toward whichever task it saw last.
- Add an explicit **abstain** token per schema (`<unk_label>`) where the task permits it.
  "I don't know" is a legitimate System One output and far more useful downstream than a
  confident coin flip.
- LR 1e-5 to 2e-5, 2–3 epochs, early stop on `dev` ECE (not accuracy — see §7).

### On RLCD
Jev's "Reinforcement Learning for Calibrated Decisions" is unpublished; there is nothing
to reimplement. The open equivalent that gets most of the benefit:
**soft targets + label smoothing during SFT, then post-hoc temperature scaling (§7).**
If you later want to close the gap, the natural next step is a proper scoring-rule
objective — train directly against Brier or log-loss on held-out calibration data with
a learned per-field temperature in the loop. Treat that as a Phase 6+ experiment, not
part of v1.

---

## 7. Phase 6 — Calibration and evaluation (1 week)

### Calibration
Fit **per-field temperature scaling** on the `cal` split: a single scalar *T* per schema
field, minimizing NLL. One parameter, cannot hurt accuracy (monotone transform preserves
argmax), typically cuts ECE by 3–5×. Best ratio of benefit to effort in the entire plan.

- Store temperatures in `spec/calibration.json`, applied client-side in the SDK.
- If logprobs come back pre-grammar-mask (Spike 0.1), renormalize over the legal token
  set *before* applying *T*.
- Consider vector scaling or isotonic regression only if per-class ECE stays bad after
  temperature scaling — and refit on `cal`, never `test`.

### Eval harness (`eval/`)
Report every milestone:

| Metric | Why |
|---|---|
| Accuracy / macro-F1 | Baseline quality |
| **ECE** (15-bin) + reliability diagram | Is stated confidence truthful |
| **Brier score** | Joint accuracy+calibration, single number |
| **Selective accuracy @ coverage** 50/80/95% | The metric that actually matters: accuracy when you auto-act above a confidence threshold and defer the rest |
| Latency p50/p99, tokens/decision | The speed claim |
| Schema violation rate | Must be exactly 0.00% — a regression test on the grammar, not the model |
| OOD confidence delta | Does confidence drop off-distribution |

Baselines to beat, on identical inputs and schemas:
- A frontier LLM via API with the same `format` constraint — your **quality ceiling**
- `qwen3:0.6b` stock + `format` — proves continued pretrain + SFT earned anything
- Logistic regression on TF-IDF, and a fine-tuned DeBERTa-v3-small classifier — **run
  these early.** If a 100M-param encoder matches your model on accuracy *and* ECE, that
  is genuinely important information about whether this project needs an LLM at all.

### Latency optimization
From Spike 0.2: forced tokens still cost forward passes, so minimize *emitted* tokens.
- Prefer a **compact output encoding** over verbose JSON: have the model emit
  `A,2,yes` (≈5 tokens) and let the SDK reconstruct the typed object, rather than
  `{"route":"billing","severity":2,"needs_human":true}` (≈20 tokens). 4× fewer forward
  passes for identical type safety — the grammar constrains either encoding equally.
- Keep the system prompt byte-identical across calls so Ollama's prefix cache hits.
  Set `"keep_alive"` generously to avoid reload stalls.
- Quantize to Q5_K_M or Q4_K_M; **re-measure ECE after quantizing**, since quantization
  perturbs the logit distribution and can silently decalibrate the model. Refit *T*
  post-quantization if needed.

---

## 8. Phase 7 — Ollama packaging (2–3 days)

```bash
python llama.cpp/convert_hf_to_gguf.py ./out/jev-0.6b-sft --outfile jev-0.6b-f16.gguf --outtype f16
./llama.cpp/build/bin/llama-quantize jev-0.6b-f16.gguf jev-0.6b-q5_k_m.gguf Q5_K_M
ollama create jev -f Modelfile
```

`Modelfile`:

```
FROM ./jev-0.6b-q5_k_m.gguf
TEMPLATE """<task:{{ .System }}>
{{ .Prompt }}
<decide>
"""
PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER top_k 0
PARAMETER num_predict 16
PARAMETER stop "<end>"
```

Greedy decoding (`temperature 0`) is deliberate: for a decision model the argmax *is*
the answer, and the uncertainty is reported via logprobs rather than expressed through
sampling. Sampling here would add variance with no upside.

Ship `jev-0.6b-q5_k_m.gguf`, `Modelfile`, the schemas, the label vocab, and
`calibration.json` together as one versioned release. The weights are not usable
correctly without the last three — a consumer that applies the wrong temperatures gets
plausible-looking, miscalibrated confidence.

---

## 9. Phase 8 — Client SDK (1 week)

The honest interface. Python first, mirroring Jev's ergonomics:

```python
class Decision(BaseModel):
    route: Literal["billing", "technical", "sales"]
    severity: int
    needs_human: bool

result = jev.decide(Decision, "card declined on renewal, third time")
# result.value      -> Decision(route="billing", severity=2, needs_human=True)
# result.confidence -> {"route": 0.94, "severity": 0.71, "needs_human": 0.88}
# result.latency_ms -> 31
```

Responsibilities:
- Pydantic model → JSON Schema → compact grammar + `format` payload
- Token symbols → typed enum values
- Apply per-field temperatures, return calibrated probabilities
- `decide_or_defer(..., threshold=0.9)` → `None` when below threshold. **The primary
  intended usage pattern** — a calibrated model's value is knowing when not to trust it.
- Batch helper; connection reuse; `keep_alive` management

Also provide a thin OpenAI-compatible shim so it drops into existing stacks.

---

## 10. Optional Phase 9 — True parallel architecture (4–8 weeks, don't start here)

Only if §7 latency proves insufficient in production. The faithful design:

1. Take the Phase 3 checkpoint, drop the LM head, attach *N* per-field classification
   heads over a pooled representation (standard `AutoModelForSequenceClassification`
   pattern, multi-head). Train with per-head cross-entropy. **This part is easy** — a
   few days, and gives you genuine single-forward-pass inference in PyTorch.
2. The hard part is serving it. llama.cpp *does* support classifier heads for rerankers
   (`pooling = rank`, `cls.output.weight`) — that is the hook to build on — but it
   supports exactly one scalar output head, not *N* multi-class heads. You would need:
   a new arch in `gguf-py/constants.py` + `tensor_mapping.py`, a converter in
   `convert_hf_to_gguf.py`, a graph builder in `src/llama-model.cpp`, and a multi-head
   output path.
3. Then surface it through Ollama, which has **no rerank/classification endpoint at
   all**. This means patching Ollama's Go runner and API — and maintaining that fork.

Realistic assessment: step 1 is a week, steps 2–3 are the real cost and commit you to
tracking two upstream projects indefinitely. **Keeping the PyTorch multi-head model as a
separate FastAPI service, and the Ollama GGUF as the portable distribution, is almost
certainly the better trade** — same weights, two serving paths, no fork to maintain.
Revisit only with measured evidence that *k* decode steps is the actual bottleneck.

---

## 11. Repo layout

```
plk-JEV/
├── PLAN.md
├── spec/
│   ├── tasks.md                 # §2 task definitions
│   ├── schemas/*.json           # single source of truth
│   ├── label_vocab.json         # enum -> single-token symbol
│   └── calibration.json         # per-field temperatures
├── data/
│   ├── cpt/        MIXTURE.md, prepare.py
│   ├── sft/        build.py, annotate/, agreement.py
│   └── eval/       dev/ cal/ test/ ood/
├── train/
│   ├── cpt.yaml  sft.yaml       # axolotl/trl configs
│   └── calibrate.py             # temperature fitting
├── eval/
│   ├── harness.py               # all §7 metrics
│   ├── baselines/               # frontier API, stock base, tfidf, deberta
│   └── reports/
├── serve/
│   ├── Modelfile  convert.sh
│   └── jev_sdk/                 # §9
└── spikes/                      # Phase 0 scripts + recorded results
```

---

## 12. Timeline and risks

| Phase | Effort |
|---|---|
| 0 — Spikes | 2 days |
| 1 — Task spec | 3–5 days |
| 2 — Base selection | 1 day |
| 3 — Data | **2–3 weeks** ← bottleneck |
| 4 — Continued pretrain | 3–7 days compute |
| 5 — Decision SFT | 1 week |
| 6 — Calibration + eval | 1 week |
| 7 — Ollama packaging | 2–3 days |
| 8 — SDK | 1 week |

**~7–9 weeks to a working, calibrated, Ollama-served v1.** Phases 4–6 will loop 2–4
times; budget for it.

### Risks, highest first

1. **Data quality, not modeling, decides this project.** 50k well-labeled examples with
   measured annotator agreement beats 500k distilled ones. Over-investing in training
   infrastructure while under-investing in labels is the standard way this fails.
2. **Spike 0.1 comes back red** (`format` suppresses logprobs). Mitigation: drop `format`
   and constrain via a `stop`-token + single-token label vocabulary, validating
   client-side; or run llama.cpp's `llama-server` directly, which exposes grammar and
   logprobs together, and treat Ollama as the distribution channel only.
3. **A 100M encoder matches you.** Entirely possible for closed-set classification. Run
   the DeBERTa baseline in week 1, not week 8 — it reframes the project cheaply if true.
4. **Calibration degrades under quantization.** Measure ECE at every quant level; refit
   temperatures post-quantization.
5. **Scope creep in §2.** Each added "just one more" task type dilutes capacity and
   pushes you toward a general instruct model with none of the speed advantage.
6. **The latency claim is weaker than Jev's.** *k* decode steps, not one. Be precise
   about this in your own docs rather than inheriting their benchmark framing — their
   published numbers also carry an acknowledged team bias.

---

## Sources

- [Introducing System One Models and Jev — TypeSafe AI](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Ollama — Structured outputs](https://docs.ollama.com/capabilities/structured-outputs) · [blog](https://ollama.com/blog/structured-outputs)
- [Ollama v0.12.11 release notes (logprobs)](https://github.com/ollama/ollama/releases/tag/v0.12.11)
- [Ollama API docs](https://github.com/ollama/ollama/blob/main/docs/api.md) · [Modelfile reference](https://docs.ollama.com/modelfile)
- Ollama rerank/classification gaps: [#3368](https://github.com/ollama/ollama/issues/3368) · [#5399](https://github.com/ollama/ollama/issues/5399) · [#13183](https://github.com/ollama/ollama/issues/13183)
- [llama.cpp — HOWTO add a model architecture](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md) · [adding new architectures (discussion #16770)](https://github.com/ggml-org/llama.cpp/discussions/16770)
- [llama.cpp reranker / `pooling=rank` / `cls.output.weight` notes](https://gist.github.com/VooDisss/42bce4eb5c76d3c325633886c5e348ee)
