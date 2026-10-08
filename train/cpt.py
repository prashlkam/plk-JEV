#!/usr/bin/env python3
"""Phase 4 — continued pretraining. Plain causal LM on the packed domain corpus.

Nothing clever here by design: the goal is a modest distribution shift toward the
input domain without wrecking the base model.

READ THIS BEFORE RUNNING IT
---------------------------
HARDWARE.md revision 6 cut this phase from PLAN.md's 0.5-2B tokens to **50M**,
because 2B tokens is ~30 days on a GTX 1650. And the honest assessment stands: for
six closed-set classification tasks, CPT is **the most cuttable phase in the plan**.
If the gate below does not clear, skip it and spend the time on SFT data instead.

Gate (PLAN.md 5): domain perplexity improves >=15% vs the base model AND replay
perplexity degrades <5%. `--eval-only` reports both against any checkpoint, so the
gate is a measurement rather than a hope.

The low LR (1e-5..5e-5, far below from-scratch pretraining rates) is the main
defense against catastrophic forgetting; the replay slice from
data/prepare_cpt.py is the other.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


class BlockDataset(Dataset):
    """Pre-packed fixed-length token blocks from data/prepare_cpt.py."""

    def __init__(self, path, limit=0):
        self.rows = []
        for line in Path(path).open():
            self.rows.append(json.loads(line)["input_ids"])
            if limit and len(self.rows) >= limit:
                break
        if not self.rows:
            raise ValueError(f"{path} is empty — run data/prepare_cpt.py first")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = torch.tensor(self.rows[i], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def collate(batch):
    return {"input_ids": torch.stack([b["input_ids"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch])}


@torch.no_grad()
def perplexity(model, loader, device, max_batches=0):
    model.eval()
    tot, n = 0.0, 0
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        out = model(input_ids=b["input_ids"].to(device),
                    labels=b["labels"].to(device))
        tot += float(out.loss)
        n += 1
    model.train()
    return math.exp(tot / max(1, n)) if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--data", default="workspace/data/cpt/train.jsonl")
    ap.add_argument("--domain-eval", default="workspace/data/cpt/domain_eval.jsonl")
    ap.add_argument("--replay-eval", default="workspace/data/cpt/replay_eval.jsonl")
    ap.add_argument("--out", default="workspace/out/jev-cpt")
    ap.add_argument("--dtype", default="fp32", choices=list(DTYPES),
                    help="fp32 is fastest on sm_75; see HARDWARE.md revision 1")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5,
                    help="1e-5..5e-5. Higher is the usual cause of forgetting.")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    ap.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    ap.add_argument("--optim", default="adamw_8bit", choices=["adamw_8bit", "adamw"])
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-blocks", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true",
                    help="report the section-5 gate for --model and exit")
    ap.add_argument("--seed", type=int, default=20261008)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dtype == "fp16":
        print("WARNING: fp16 is 5.3x slower than fp32 here (HARDWARE.md rev 1).")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=DTYPES[args.dtype], attn_implementation="sdpa").to(device)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    evals = {}
    for name, path in [("domain", args.domain_eval), ("replay", args.replay_eval)]:
        if Path(path).exists():
            evals[name] = DataLoader(BlockDataset(path), batch_size=args.micro_batch,
                                     collate_fn=collate)
        else:
            print(f"  no {name} eval at {path}: the section-5 gate cannot be "
                  f"evaluated for {name}")

    def gate_report(tag):
        out = {}
        for name, dl in evals.items():
            out[name] = perplexity(model, dl, device, args.eval_batches)
        print(f"  [{tag}] " + "  ".join(f"{k} ppl {v:.3f}" for k, v in out.items()))
        return out

    if args.eval_only:
        base = gate_report("eval-only")
        print(json.dumps(base, indent=2))
        return

    ds = BlockDataset(args.data, args.limit_blocks)
    dl = DataLoader(ds, batch_size=args.micro_batch, shuffle=True,
                    collate_fn=collate, num_workers=2, pin_memory=True,
                    drop_last=True)
    seq_len = len(ds.rows[0])
    total_tok = len(ds) * seq_len
    steps_per_epoch = max(1, len(dl) // args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    print(f"{len(ds):,} blocks x {seq_len} = {total_tok:,} tokens/epoch")
    print(f"{total_steps:,} steps, effective batch "
          f"{args.micro_batch * args.grad_accum * seq_len:,} tokens")
    print(f"est. {total_tok * args.epochs / 1767 / 3600:.1f} h at 1767 tok/s\n")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    print("baseline perplexity before any update (this IS the gate denominator):")
    baseline = gate_report("base")
    (outdir / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")

    params = list(model.parameters())
    if args.optim == "adamw_8bit":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(params, lr=args.lr,
                                      weight_decay=args.weight_decay)
        except Exception as e:
            print(f"bitsandbytes unavailable ({e}); fp32 AdamW")
            opt = torch.optim.AdamW(params, lr=args.lr,
                                    weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    warmup = max(1, int(total_steps * args.warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (
        s / warmup if s < warmup else
        0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) /
                                          max(1, total_steps - warmup))))))

    history, step, micro, seen = [], 0, 0, 0
    t0 = time.perf_counter()
    done = False
    while not done:
        for b in dl:
            out = model(input_ids=b["input_ids"].to(device, non_blocking=True),
                        labels=b["labels"].to(device, non_blocking=True))
            (out.loss / args.grad_accum).backward()
            seen += b["input_ids"].numel()
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
                print(f"step {step:>6}/{total_steps}  loss {float(out.loss):.4f}  "
                      f"ppl {math.exp(min(20, float(out.loss))):.2f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {seen/el:.0f} tok/s  "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                      flush=True)
            if step % args.eval_every == 0 or step == total_steps:
                cur = gate_report(f"step {step}")
                rec = {"step": step, **cur}
                for k, v in cur.items():
                    if k in baseline and baseline[k] and baseline[k] == baseline[k]:
                        rec[f"{k}_delta_pct"] = (v - baseline[k]) / baseline[k] * 100
                dom = rec.get("domain_delta_pct")
                rep = rec.get("replay_delta_pct")
                if dom is not None:
                    ok_d = dom <= -15.0
                    msg = f"    domain ppl {dom:+.1f}% (gate: <= -15%) {'PASS' if ok_d else 'not yet'}"
                    if rep is not None:
                        ok_r = rep < 5.0
                        msg += f" | replay {rep:+.1f}% (gate: < +5%) {'PASS' if ok_r else 'FAIL'}"
                        if not ok_r:
                            msg += "\n    ! forgetting: lower --lr or raise the replay fraction"
                    print(msg)
                history.append(rec)
                (outdir / "history.json").write_text(json.dumps(history, indent=2))
            if step % args.save_every == 0:
                model.save_pretrained(outdir / "last")
                tok.save_pretrained(outdir / "last")
            if step >= total_steps:
                done = True
                break

    model.save_pretrained(outdir / "last")
    tok.save_pretrained(outdir / "last")
    (outdir / "args.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    el = time.perf_counter() - t0
    print(f"\ndone in {el/3600:.2f} h, {seen:,} tokens ({seen/el:.0f} tok/s)")
    final = history[-1] if history else {}
    dom = final.get("domain_delta_pct")
    if dom is not None and dom > -15.0:
        print(f"\nGATE NOT MET: domain ppl moved {dom:+.1f}%, wanted <= -15%.")
        print("Per HARDWARE.md revision 6 the right response is to SKIP CPT and run\n"
              "SFT from the base model, spending the time on data instead.")
    print(f"\nnext: train/sft.py --model {outdir/'last'}")


if __name__ == "__main__":
    main()
