#!/usr/bin/env python3
"""Phase 5 — decision SFT. One decode step, calibrated, on a 4GB GPU.

The loss formulation is the heart of this file, so it is worth stating plainly.

For each (input, task) pair the model sees a prompt and must produce ONE token: a
digit. We build a target distribution `q` over the vocabulary that is supported
*only on the legal digit tokens* for that task, and minimize cross-entropy against
the model's full-vocabulary softmax:

    loss = - sum_v q_v * log p_v        (v ranges over the whole vocab)

One formulation, three jobs:

  * **Hard labels** put 1-eps on the gold digit and spread eps over the *other legal
    digits* — not over all 49k tokens, which is what `CrossEntropyLoss(
    label_smoothing=...)` would wrongly do.
  * **Soft targets** from annotator disagreement (data/agreement.py) drop straight
    in as `q`, preserving the uncertainty that was paid for. PLAN.md 6.
  * **Full-vocab denominator** is what drives `legal_mass` toward 1 — the model
    learns to natively concentrate on legal digits, so the grammar mask becomes
    non-binding and Ollama's pre-mask logprobs become trustworthy. That is the
    direct fix for spike 0.1's AMBER finding, and it is only achieved because the
    softmax is over the whole vocabulary rather than the legal subset.

`legal_mass` is therefore measurable during training, not just at serve time, and is
reported every eval alongside dev ECE.

Hardware choices (all measured — see HARDWARE.md):
  * **fp32 by default.** fp16 AMP is 5.3x SLOWER on this GPU; bf16 is emulated.
  * **Sliced LM head.** Loss touches one position, so logits are computed for one
    position: +19% throughput, -13% memory.
  * **Right padding with per-row gather**, not left padding — avoids RoPE
    position_id subtleties entirely.
  * Gradient checkpointing on, micro-batch 16, 8-bit Adam.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev import metrics as M  # noqa: E402
from jev.spec import ABSTAIN_CODE, load_specs, load_template, render_prompt  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


# --- data ----------------------------------------------------------------

class PairDataset(Dataset):
    """(input, task) pairs, pre-tokenized once.

    Each source example can yield up to one pair per task. Using all six multiplies
    the token budget by six for highly redundant prompts, so `tasks_per_example`
    samples a subset per input by default; see --tasks-per-example.
    """

    def __init__(self, records, tokenizer, specs, max_seq_len, tasks_per_example,
                 soft_targets=None, seed=0, task_filter=None):
        self.tok = tokenizer
        self.specs = specs
        self.max_seq_len = max_seq_len
        self.rows = []
        self._cache: dict[int, list[int]] = {}
        rng = random.Random(seed)
        task_ids = [t for t in specs if (not task_filter or t in task_filter)]
        self.digit_ids = {t: specs[t].digit_token_ids(tokenizer) for t in task_ids}

        for r in records:
            avail = [t for t in task_ids if t in r["labels"]]
            if tasks_per_example and tasks_per_example < len(avail):
                chosen = rng.sample(avail, tasks_per_example)
            else:
                chosen = avail
            soft = (soft_targets or {}).get(r["id"], {})
            for t in chosen:
                self.rows.append((r["input"], t, r["labels"][t], soft.get(t)))

    def __len__(self):
        return len(self.rows)

    def lengths(self) -> list[int]:
        """Token length of every row, tokenizing once and caching.

        Needed by the length-grouped sampler, and it moves tokenization out of the
        training loop as a side effect. Real support messages average ~80 tokens
        against a 192 cap, so naive batching pads ~55% of every batch into waste —
        measured 954 tok/s of real tokens against the 1767 tok/s fixed-length
        benchmark in HARDWARE.md.
        """
        if len(self._cache) != len(self.rows):
            self._cache = {}
            for i, (text, task_id, _, _) in enumerate(self.rows):
                self._cache[i] = self._truncate(text, task_id)
        return [len(self._cache[i]) for i in range(len(self.rows))]

    def _truncate(self, text, task_id):
        """Tokenize the rendered prompt, truncating the INPUT if needed.

        The task tag and the `answer: ` terminator must survive truncation — they
        are what tells the model which decision to make and where position 0 is.
        So we never truncate the rendered prompt; we shrink the input and re-render.

        Truncation keeps the head AND tail of the input: support messages carry
        their framing in the opener and their urgency markers in the closer, and
        dropping either systematically biases urgency_v1 and gate_human_v1.
        """
        prompt = render_prompt(task_id, text)
        ids = self.tok.encode(prompt, add_special_tokens=False)
        if len(ids) <= self.max_seq_len:
            return ids
        overhead = len(self.tok.encode(render_prompt(task_id, ""),
                                       add_special_tokens=False))
        budget = max(8, self.max_seq_len - overhead)
        body = self.tok.encode(text, add_special_tokens=False)
        head = budget // 2
        tail = budget - head
        kept = self.tok.decode(body[:head]) + " … " + self.tok.decode(body[-tail:])
        ids = self.tok.encode(render_prompt(task_id, kept), add_special_tokens=False)
        return ids[-self.max_seq_len:]

    def __getitem__(self, i):
        text, task_id, label, soft = self.rows[i]
        spec = self.specs[task_id]
        ids = self._cache.get(i) or self._truncate(text, task_id)
        dmap = self.digit_ids[task_id]
        gold_code = spec.value_to_code(label) if label is not None else ABSTAIN_CODE
        return {
            "input_ids": ids,
            "task_id": task_id,
            "gold_code": gold_code,
            "gold_token": dmap[gold_code],
            "legal_tokens": [dmap[c] for c in spec.legal_codes],
            "legal_codes": list(spec.legal_codes),
            "soft": soft,
        }


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffle, chunk into mega-batches, sort each by length, shuffle the batches.

    Keeps batches length-homogeneous so padding is near-zero, while retaining
    enough randomness that gradient estimates stay unbiased — within a mega-batch
    the composition is random, and the batch order is reshuffled every epoch.

    Worth ~1.8x on this corpus: lengths range ~45-192 tokens, and the alternative
    pads every batch to its longest member.
    """

    def __init__(self, lengths, batch_size, mega=50, seed=0):
        self.lengths = lengths
        self.batch_size = batch_size
        self.mega = mega * batch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = random.Random(self.seed + self.epoch)
        self.epoch += 1
        idx = list(range(len(self.lengths)))
        g.shuffle(idx)
        batches = []
        for s in range(0, len(idx), self.mega):
            chunk = sorted(idx[s:s + self.mega], key=lambda i: self.lengths[i])
            batches.extend(chunk[b:b + self.batch_size]
                           for b in range(0, len(chunk), self.batch_size))
        g.shuffle(batches)
        return iter([i for b in batches for i in b])


def collate(batch, pad_id, specs, label_smoothing):
    """Right-pad, record true last index, and build the target distribution."""
    n = len(batch)
    L = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((n, L), pad_id, dtype=torch.long)
    attn = torch.zeros((n, L), dtype=torch.long)
    last_idx = torch.zeros(n, dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        input_ids[i, :k] = torch.tensor(b["input_ids"], dtype=torch.long)
        attn[i, :k] = 1
        last_idx[i] = k - 1

    # Sparse target distribution over the vocab, supported on legal digits only.
    tgt_tok, tgt_p, tgt_row = [], [], []
    for i, b in enumerate(batch):
        spec = specs[b["task_id"]]
        legal_codes = b["legal_codes"]
        legal_tokens = b["legal_tokens"]
        if b["soft"]:
            # annotator / teacher vote distribution
            q = {}
            for val, p in b["soft"].items():
                code = (ABSTAIN_CODE if val in (None, "__abstain__")
                        else spec.value_to_code(val))
                if code in legal_codes:
                    q[code] = q.get(code, 0.0) + float(p)
            Z = sum(q.values())
            if Z <= 0:
                q = {b["gold_code"]: 1.0}
                Z = 1.0
            q = {k: v / Z for k, v in q.items()}
        else:
            # hard label, smoothed across the OTHER LEGAL codes only
            others = [c for c in legal_codes if c != b["gold_code"]]
            eps = label_smoothing if others else 0.0
            q = {b["gold_code"]: 1.0 - eps}
            for c in others:
                q[c] = eps / len(others)
        code_to_tok = dict(zip(legal_codes, legal_tokens))
        for c, p in q.items():
            tgt_row.append(i)
            tgt_tok.append(code_to_tok[c])
            tgt_p.append(p)

    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "last_idx": last_idx,
        "tgt_row": torch.tensor(tgt_row, dtype=torch.long),
        "tgt_tok": torch.tensor(tgt_tok, dtype=torch.long),
        "tgt_p": torch.tensor(tgt_p, dtype=torch.float32),
        "gold_code": torch.tensor([b["gold_code"] for b in batch], dtype=torch.long),
        "task_ids": [b["task_id"] for b in batch],
        "legal_tokens": [b["legal_tokens"] for b in batch],
        "legal_codes": [b["legal_codes"] for b in batch],
    }


# --- model ---------------------------------------------------------------

def last_token_logits(model, input_ids, attention_mask, last_idx):
    """Hidden states for the sequence; LM head for the final real position only.

    HARDWARE.md revision 2. Materializing [B, L, vocab] logits when loss touches
    one position wastes ~0.25 GiB and 19% of throughput at these shapes.
    """
    base = getattr(model, "model", None) or getattr(model, "transformer")
    out = base(input_ids=input_ids, attention_mask=attention_mask)
    h = out.last_hidden_state                                  # [B, L, H]
    h = h[torch.arange(h.size(0), device=h.device), last_idx]  # [B, H]
    head = model.get_output_embeddings()
    return head(h)                                             # [B, vocab]


def loss_fn(logits, batch):
    """-sum_v q_v log p_v with q sparse over legal digits. See module docstring."""
    logp = F.log_softmax(logits.float(), dim=-1)
    sel = logp[batch["tgt_row"], batch["tgt_tok"]]
    per = torch.zeros(logits.size(0), device=logits.device, dtype=sel.dtype)
    per.index_add_(0, batch["tgt_row"], -sel * batch["tgt_p"])
    return per.mean()


@torch.no_grad()
def evaluate(model, loader, specs, device, autocast_dtype=None):
    """Dev metrics, including the legal_mass gate and per-task ECE."""
    model.eval()
    per_task: dict[str, dict] = {}
    tot_loss, n_batch = 0.0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        li = batch["last_idx"].to(device, non_blocking=True)
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        ctx = (torch.autocast("cuda", dtype=autocast_dtype)
               if autocast_dtype else torch.autocast("cuda", enabled=False))
        with ctx:
            logits = last_token_logits(model, ids, attn, li)
        tot_loss += float(loss_fn(logits, b))
        n_batch += 1

        p_full = F.softmax(logits.float(), dim=-1)
        for i, task_id in enumerate(batch["task_ids"]):
            spec = specs[task_id]
            toks = batch["legal_tokens"][i]
            codes = batch["legal_codes"][i]
            pl = p_full[i, toks]
            legal_mass = float(pl.sum())
            # Restrict to the real label classes, exactly as the SDK does at serve
            # time, so the ECE measured here is the ECE that will be reported.
            keep = [j for j, c in enumerate(codes) if c != ABSTAIN_CODE]
            pr = pl[keep]
            pr = pr / pr.sum().clamp_min(1e-12)
            d = per_task.setdefault(task_id, {"probs": [], "labels": [], "mass": []})
            gold = int(batch["gold_code"][i])
            if gold == ABSTAIN_CODE:
                continue  # abstain is a decision, not a class; excluded from ECE
            d["probs"].append(pr.cpu().numpy())
            d["labels"].append(gold)
            d["mass"].append(legal_mass)

    out = {"loss": tot_loss / max(1, n_batch), "tasks": {}}
    eces, accs, masses = [], [], []
    for task_id, d in per_task.items():
        if not d["labels"]:
            continue
        probs = np.stack(d["probs"])
        labels = np.array(d["labels"])
        rep = M.report(probs, labels, ordinal=specs[task_id].ordinal,
                       masses=d["mass"])
        out["tasks"][task_id] = rep
        eces.append(rep["ece"])
        accs.append(rep["accuracy"])
        masses.extend(d["mass"])
    out["macro_ece"] = float(np.mean(eces)) if eces else float("nan")
    out["macro_accuracy"] = float(np.mean(accs)) if accs else float("nan")
    out["legal_mass"] = M.legal_mass_stats(masses)
    model.train()
    return out


# --- main ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="yaml/json config; CLI flags override")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--train", default="workspace/data/eval/train.jsonl")
    ap.add_argument("--dev", default="workspace/data/eval/dev.jsonl")
    ap.add_argument("--soft-targets", default="workspace/data/sft/soft_targets.jsonl")
    ap.add_argument("--out", default="workspace/out/jev-sft")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES),
                    help="fp32 is FASTEST on sm_75; fp16 is 5.3x slower (HARDWARE.md)")
    ap.add_argument("--max-seq-len", type=int, default=192)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--tasks-per-example", type=int, default=3,
                    help="0 = all six; fewer trades coverage for a smaller token budget")
    ap.add_argument("--max-pairs", type=int, default=0, help="0 = no cap")
    ap.add_argument("--dev-max-pairs", type=int, default=3000,
                    help="subsample dev; a full 6-task dev pass is ~36k pairs and "
                         "would dominate wall clock at --eval-every 500")
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--sliced-lm-head", action="store_true", default=True)
    ap.add_argument("--optim", default="adamw_8bit",
                    choices=["adamw_8bit", "adamw", "adafactor"])
    ap.add_argument("--qlora", action="store_true",
                    help="4-bit NF4 base + LoRA; required for Qwen3-0.6B at 4GB")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--early-stop-patience", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--group-by-length", action="store_true", default=True,
                    help="length-homogeneous batches; ~1.8x on this corpus because "
                         "lengths span 45-192 and naive batching pads to the max")
    ap.add_argument("--no-group-by-length", dest="group_by_length",
                    action="store_false")
    args = ap.parse_args()

    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        # Readable config keys that differ from the argparse dest.
        ALIASES = {"gradient_checkpointing": "grad_ckpt",
                   "gradient_accumulation": "grad_accum",
                   "micro_batch_size": "micro_batch",
                   "attn_implementation": None}  # informational only
        explicit = {a.lstrip("-").replace("-", "_") for a in sys.argv[1:]
                    if a.startswith("--")}
        unknown = []
        for k, v in cfg.items():
            k = ALIASES.get(k.replace("-", "_"), k.replace("-", "_"))
            if k is None:
                continue
            if not hasattr(args, k):
                unknown.append(k)
                continue
            if k not in explicit:
                setattr(args, k, v)
        if unknown:
            # Silently ignoring a typo'd config key is how a run quietly trains
            # with the wrong settings for two hours.
            sys.exit(f"unknown keys in {args.config}: {sorted(unknown)}")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.dtype == "fp16":
        print("WARNING: fp16 measured 5.3x SLOWER than fp32 on this GPU "
              "(TU117 has no fast FP16 path). See HARDWARE.md revision 1.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    specs = load_specs()

    # Fail fast if the single-token invariant is broken for this tokenizer.
    for s in specs.values():
        s.digit_token_ids(tok)

    model_kwargs = {"dtype": DTYPES[args.dtype], "attn_implementation": "sdpa"}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # fp32 compute, not the usual fp16: see HARDWARE.md revision 1
            bnb_4bit_compute_dtype=torch.float32,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if not args.qlora:
        model = model.to(device)
    if args.qlora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.grad_ckpt)
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    soft = {}
    sp = Path(args.soft_targets)
    if sp.exists():
        for line in sp.open():
            d = json.loads(line)
            soft[d["id"]] = d["soft"]
        print(f"loaded {len(soft):,} soft targets from {sp}")
    else:
        print(f"no soft targets at {sp}; using label smoothing "
              f"eps={args.label_smoothing} only")

    tr_rec = [json.loads(l) for l in Path(args.train).open()]
    dv_rec = [json.loads(l) for l in Path(args.dev).open()]
    train_ds = PairDataset(tr_rec, tok, specs, args.max_seq_len,
                           args.tasks_per_example, soft, args.seed)
    dev_ds = PairDataset(dv_rec, tok, specs, args.max_seq_len, 0, None, args.seed)
    if args.max_pairs:
        random.Random(args.seed).shuffle(train_ds.rows)
        train_ds.rows = train_ds.rows[: args.max_pairs]
    if args.dev_max_pairs and len(dev_ds.rows) > args.dev_max_pairs:
        # Stratify the subsample by task so every task keeps enough examples for
        # its per-task ECE to mean anything.
        by_task: dict[str, list] = {}
        for row in dev_ds.rows:
            by_task.setdefault(row[1], []).append(row)
        per = max(1, args.dev_max_pairs // len(by_task))
        rng = random.Random(args.seed)
        kept = []
        for rows in by_task.values():
            rng.shuffle(rows)
            kept.extend(rows[:per])
        dev_ds.rows = kept

    coll = lambda b: collate(b, tok.pad_token_id, specs, args.label_smoothing)
    dev_coll = lambda b: collate(b, tok.pad_token_id, specs, 0.0)
    sampler = None
    if args.group_by_length:
        print("pre-tokenizing for length-grouped batching...", flush=True)
        t_pre = time.perf_counter()
        lens = train_ds.lengths()
        sampler = LengthGroupedSampler(lens, args.micro_batch, seed=args.seed)
        print(f"  {len(lens):,} rows in {time.perf_counter()-t_pre:.0f}s; "
              f"lengths p50 {int(np.percentile(lens,50))} "
              f"p95 {int(np.percentile(lens,95))} max {max(lens)}; "
              f"padding waste {1 - np.mean(lens)/max(lens):.0%} -> "
              f"near zero with grouping")
    train_dl = DataLoader(train_ds, batch_size=args.micro_batch,
                          shuffle=(sampler is None), sampler=sampler,
                          collate_fn=coll, num_workers=args.workers,
                          pin_memory=True, drop_last=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.micro_batch, shuffle=False,
                        collate_fn=dev_coll, num_workers=2, pin_memory=True)

    steps_per_epoch = max(1, len(train_dl) // args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"\n{len(train_ds):,} train pairs ({len(tr_rec):,} inputs x "
          f"{args.tasks_per_example or len(specs)} tasks), {len(dev_ds):,} dev pairs")
    print(f"micro-batch {args.micro_batch} x accum {args.grad_accum} = "
          f"effective {args.micro_batch * args.grad_accum}")
    print(f"{total_steps:,} optimizer steps ({args.epochs} epochs)")
    # Estimate from the step count actually being run, not the epoch count:
    # --max-steps truncates training and the two diverge badly.
    est_tok = (total_steps * args.micro_batch * args.grad_accum
               * args.max_seq_len * 0.55)
    print(f"~{est_tok/1e6:.0f}M tokens -> ~{est_tok/1767/3600:.1f} h at 1767 tok/s\n")

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=args.lr,
                                      weight_decay=args.weight_decay)
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}); falling back to fp32 AdamW")
            opt = torch.optim.AdamW(params, lr=args.lr,
                                    weight_decay=args.weight_decay)
    elif args.optim == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=args.lr, relative_step=False,
                        scale_parameter=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    warmup = max(1, int(total_steps * args.warmup_frac))
    def lr_at(s):
        if s < warmup:
            return s / warmup
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    try:
        from torch.utils.tensorboard import SummaryWriter
        tb = SummaryWriter(str(outdir / "tb"))
    except Exception:
        tb = None

    best_ece, best_step, bad_evals = float("inf"), -1, 0
    history = []
    step, micro, t0, seen_tok = 0, 0, time.perf_counter(), 0
    done = False

    while not done:
        for batch in train_dl:
            ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            li = batch["last_idx"].to(device, non_blocking=True)
            b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                 for k, v in batch.items()}
            logits = last_token_logits(model, ids, attn, li)
            loss = loss_fn(logits, b) / args.grad_accum
            loss.backward()
            seen_tok += int(attn.sum())
            micro += 1

            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 20 == 0:
                el = time.perf_counter() - t0
                print(f"step {step:>6}/{total_steps}  loss {float(loss.detach())*args.grad_accum:.4f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  "
                      f"{seen_tok/el:.0f} tok/s  "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                      flush=True)
                if tb:
                    tb.add_scalar("train/loss", float(loss.detach()) * args.grad_accum, step)
                    tb.add_scalar("train/tok_per_s", seen_tok / el, step)

            if step % args.eval_every == 0 or step == total_steps:
                ev = evaluate(model, dev_dl, specs, device)
                lm = ev["legal_mass"]
                print(f"\n  EVAL step {step}: loss {ev['loss']:.4f}  "
                      f"macro_acc {ev['macro_accuracy']:.4f}  "
                      f"macro_ECE {ev['macro_ece']:.4f}  "
                      f"legal_mass {lm.get('mean', float('nan')):.4f}")
                for t, r in sorted(ev["tasks"].items()):
                    print(f"    {t:<16} acc {r['accuracy']:.3f}  ece {r['ece']:.3f}  "
                          f"brier {r['brier']:.3f}" +
                          (f"  mae {r['ordinal_mae']:.3f}" if "ordinal_mae" in r else ""))
                if lm.get("mean", 1.0) < 0.95:
                    print(f"    ! legal_mass {lm['mean']:.3f} < 0.95 — the grammar "
                          "mask is deciding, not the model (spike 0.1)")
                history.append({"step": step, **ev})
                (outdir / "history.json").write_text(json.dumps(history, indent=2))
                if tb:
                    tb.add_scalar("dev/macro_ece", ev["macro_ece"], step)
                    tb.add_scalar("dev/macro_accuracy", ev["macro_accuracy"], step)
                    tb.add_scalar("dev/legal_mass", lm.get("mean", 0), step)

                # Early stop on ECE, not accuracy. PLAN.md 6: accuracy improves
                # easily here while calibration stays bad, so accuracy would stop
                # in the wrong place.
                if ev["macro_ece"] < best_ece - 1e-5:
                    best_ece, best_step, bad_evals = ev["macro_ece"], step, 0
                    model.save_pretrained(outdir / "best")
                    tok.save_pretrained(outdir / "best")
                    (outdir / "best" / "jev_meta.json").write_text(json.dumps({
                        "step": step, "macro_ece": ev["macro_ece"],
                        "macro_accuracy": ev["macro_accuracy"],
                        "legal_mass_mean": lm.get("mean"),
                        "base_model": args.model, "dtype": args.dtype,
                        "max_seq_len": args.max_seq_len,
                        "template": load_template()["template"],
                    }, indent=2))
                    print(f"    new best macro_ECE {best_ece:.4f} -> {outdir/'best'}")
                else:
                    bad_evals += 1
                    print(f"    no ECE improvement ({bad_evals}/"
                          f"{args.early_stop_patience}); best {best_ece:.4f} "
                          f"@ step {best_step}")
                    if bad_evals >= args.early_stop_patience:
                        print("  early stop")
                        done = True
                print()

            if step % args.save_every == 0:
                model.save_pretrained(outdir / "last")
                tok.save_pretrained(outdir / "last")
            if step >= total_steps:
                done = True
            if done:
                break

    model.save_pretrained(outdir / "last")
    tok.save_pretrained(outdir / "last")
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min, {seen_tok:,} tokens ({seen_tok/el:.0f} tok/s)")
    print(f"best macro_ECE {best_ece:.4f} at step {best_step} -> {outdir/'best'}")
    print(f"\nnext: train/calibrate.py --model {outdir/'best'}")
    if tb:
        tb.close()


if __name__ == "__main__":
    main()
