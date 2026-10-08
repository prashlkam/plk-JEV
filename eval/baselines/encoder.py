#!/usr/bin/env python3
"""Encoder baseline — AND the Phase 9 / Path A prototype, in one file.

PLAN.md 7 wants a fine-tuned small encoder as a baseline, because if a ~100M
encoder matches JEV on accuracy *and* ECE then the project does not need a decoder
LLM. PLAN.md 10 step 1 separately describes the "faithful Jev architecture": a
shared trunk with per-field classification heads, giving genuine single-forward-pass
inference for all fields at once.

Those are the same model. So this trains it once and reports both roles:

  * as a **baseline**, per-task accuracy / ECE / Brier directly comparable to
    `eval/harness.py` output;
  * as the **Path A prototype**, single-forward-pass latency for all six decisions
    at once, which is the number that decides whether Path A is worth the 4-8 weeks
    of llama.cpp and Ollama runtime work that step 2-3 would cost.

PLAN.md 10's conclusion is that the right home for this model is a small FastAPI
service rather than a patched Ollama fork — same weights, two serving paths, no
upstream fork to maintain. This file produces those weights.

Heads are trained with the same soft-target / label-smoothing loss as
`train/sft.py`, so the calibration comparison is apples to apples rather than
accidentally favouring whichever model got the better loss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from jev import metrics as M  # noqa: E402
from jev.spec import ABSTAIN_CODE, load_specs  # noqa: E402


class MultiHead(nn.Module):
    """Shared encoder trunk + one linear head per task.

    This is the architecture Jev's blog post describes: all fields produced in a
    single forward pass, each head's softmax giving a calibrated-by-construction
    distribution over a fixed output space. Type safety comes from head arity
    rather than from a grammar — there is literally no way to emit an illegal value.
    """

    def __init__(self, encoder_name, task_sizes, dropout=0.1):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        cfg = AutoConfig.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = getattr(cfg, "hidden_size", None) or cfg.dim
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict(
            {t: nn.Linear(h, k) for t, k in task_sizes.items()})
        self.task_sizes = task_sizes

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        # Mean pooling over real tokens; more robust than [CLS] for small encoders.
        m = attention_mask.unsqueeze(-1).float()
        pooled = (hs * m).sum(1) / m.sum(1).clamp_min(1e-9)
        pooled = self.drop(pooled)
        return {t: head(pooled) for t, head in self.heads.items()}


class EncDataset(Dataset):
    def __init__(self, records, tokenizer, specs, max_len, soft=None):
        self.rec = records
        self.tok = tokenizer
        self.specs = specs
        self.max_len = max_len
        self.soft = soft or {}

    def __len__(self):
        return len(self.rec)

    def __getitem__(self, i):
        r = self.rec[i]
        enc = self.tok(r["input"], truncation=True, max_length=self.max_len)
        labels, softs = {}, {}
        for t, spec in self.specs.items():
            if t not in r["labels"]:
                labels[t] = -100
                continue
            c = spec.value_to_code(r["labels"][t])
            labels[t] = -100 if c == ABSTAIN_CODE else c
            s = self.soft.get(r["id"], {}).get(t)
            if s:
                v = np.zeros(spec.n_classes, dtype=np.float32)
                for val, p in s.items():
                    if val in spec.values:
                        v[spec.values.index(val)] = p
                if v.sum() > 0:
                    softs[t] = v / v.sum()
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels, "soft": softs}


def make_collate(pad_id, specs):
    def collate(batch):
        L = max(len(b["input_ids"]) for b in batch)
        n = len(batch)
        ids = torch.full((n, L), pad_id, dtype=torch.long)
        am = torch.zeros((n, L), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            am[i, :k] = torch.tensor(b["attention_mask"])
        labels = {t: torch.tensor([b["labels"][t] for b in batch], dtype=torch.long)
                  for t in specs}
        soft = {}
        for t, spec in specs.items():
            if any(t in b["soft"] for b in batch):
                arr = np.zeros((n, spec.n_classes), dtype=np.float32)
                mask = np.zeros(n, dtype=bool)
                for i, b in enumerate(batch):
                    if t in b["soft"]:
                        arr[i] = b["soft"][t]
                        mask[i] = True
                soft[t] = (torch.from_numpy(arr), torch.from_numpy(mask))
        return {"input_ids": ids, "attention_mask": am, "labels": labels,
                "soft": soft}
    return collate


def head_loss(logits, labels, soft, eps):
    """Same objective as train/sft.py: soft targets where available, otherwise a
    hard label smoothed across the other legal classes."""
    valid = labels >= 0
    if not valid.any():
        return logits.sum() * 0.0
    logp = F.log_softmax(logits[valid].float(), dim=-1)
    y = labels[valid]
    K = logits.size(-1)
    q = torch.full_like(logp, eps / max(1, K - 1))
    q.scatter_(1, y.unsqueeze(1), 1.0 - eps)
    if soft is not None:
        arr, mask = soft
        m = mask.to(logits.device)[valid]
        if m.any():
            q[m] = arr.to(logits.device)[valid][m]
    return -(q * logp).sum(1).mean()


@torch.no_grad()
def collect(model, loader, specs, device, temps=None):
    model.eval()
    per = {t: {"probs": [], "labels": []} for t in specs}
    for b in loader:
        out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        for t, spec in specs.items():
            lg = out[t].float().cpu()
            if temps:
                lg = lg / temps.get(t, 1.0)
            p = torch.softmax(lg, dim=-1).numpy()
            y = b["labels"][t].numpy()
            for i in range(len(y)):
                if y[i] >= 0:
                    per[t]["probs"].append(p[i])
                    per[t]["labels"].append(int(y[i]))
    model.train()
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="distilbert-base-uncased",
                    help="distilbert-base-uncased (66M) fits comfortably in 4GB; "
                         "microsoft/deberta-v3-small (142M) is the stronger option")
    ap.add_argument("--train", default="workspace/data/eval/train.jsonl")
    ap.add_argument("--dev", default="workspace/data/eval/dev.jsonl")
    ap.add_argument("--cal", default="workspace/data/eval/cal.jsonl")
    ap.add_argument("--test", default="workspace/data/eval/test.jsonl")
    ap.add_argument("--ood", default="workspace/data/sft/seed_ood.jsonl")
    ap.add_argument("--soft-targets", default="workspace/data/sft/soft_targets.jsonl")
    ap.add_argument("--train-limit", type=int, default=30000)
    ap.add_argument("--eval-limit", type=int, default=3000)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--out", default="workspace/out/baseline-encoder")
    ap.add_argument("--report", default="eval/reports/baseline_encoder.json")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = load_specs()
    tok = AutoTokenizer.from_pretrained(args.encoder)

    def load(p, lim):
        rows = [json.loads(l) for l in Path(p).open()]
        return rows[:lim] if lim else rows

    tr = load(args.train, args.train_limit)
    dv = load(args.dev, args.eval_limit)
    cal = load(args.cal, args.eval_limit)
    te = load(args.test, args.eval_limit)
    ood = load(args.ood, 400) if Path(args.ood).exists() else []

    soft = {}
    if Path(args.soft_targets).exists():
        for line in Path(args.soft_targets).open():
            d = json.loads(line)
            soft[d["id"]] = d["soft"]

    sizes = {t: s.n_classes for t, s in specs.items()}
    model = MultiHead(args.encoder, sizes).to(device)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{args.encoder}: {nparam:.0f}M params, {len(sizes)} heads "
          f"({sum(sizes.values())} output logits total)")

    coll = make_collate(tok.pad_token_id, specs)
    dl = DataLoader(EncDataset(tr, tok, specs, args.max_len, soft),
                    batch_size=args.batch, shuffle=True, collate_fn=coll,
                    num_workers=4, pin_memory=True, drop_last=True)
    mk = lambda rs: DataLoader(EncDataset(rs, tok, specs, args.max_len),
                               batch_size=args.batch, collate_fn=coll)
    dl_dev, dl_cal, dl_te = mk(dv), mk(cal), mk(te)

    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    except Exception:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total = args.max_steps or int(len(dl) / args.grad_accum * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=max(1, total), pct_start=0.1)
    print(f"{len(tr):,} train, {total:,} steps\n")

    step, micro, t0 = 0, 0, time.perf_counter()
    done = False
    while not done:
        for b in dl:
            out = model(b["input_ids"].to(device, non_blocking=True),
                        b["attention_mask"].to(device, non_blocking=True))
            loss = sum(head_loss(out[t], b["labels"][t].to(device),
                                 b["soft"].get(t), args.label_smoothing)
                       for t in specs) / len(specs)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{total}  loss {float(loss):.4f}  "
                      f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB",
                      flush=True)
            if step >= total:
                done = True
                break

    train_s = time.perf_counter() - t0
    print(f"\ntrained in {train_s/60:.1f} min")

    # --- temperature scaling on cal, same method as train/calibrate.py -----
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "train"))
    from calibrate import fit_temperature

    cal_per = collect(model, dl_cal, specs, device)
    temps = {}
    for t, d in cal_per.items():
        if len(d["labels"]) > 50:
            lg = np.log(np.clip(np.stack(d["probs"]), 1e-12, None))
            temps[t], _ = fit_temperature(lg, np.array(d["labels"]))
        else:
            temps[t] = 1.0

    te_per = collect(model, dl_te, specs, device, temps)
    report = {"label": f"encoder:{args.encoder}", "model": args.encoder,
              "backend": "pytorch-multihead", "params_m": nparam,
              "train_seconds": train_s, "temperatures": temps,
              "n_inputs": len(te), "tasks": {}, "synthetic": True,
              "role": "PLAN.md 7 baseline AND PLAN.md 10 step-1 Path A prototype"}

    print(f"\n{'task':<16} {'n':>6} {'acc':>7} {'mF1':>7} {'ECE':>7} {'Brier':>7} {'T':>6}")
    print("-" * 62)
    accs, eces, briers = [], [], []
    for t, d in te_per.items():
        if not d["labels"]:
            continue
        probs, labels = np.stack(d["probs"]), np.array(d["labels"])
        rep = M.report(probs, labels, ordinal=specs[t].ordinal)
        rep["temperature"] = temps[t]
        report["tasks"][t] = rep
        accs.append(rep["accuracy"])
        eces.append(rep["ece"])
        briers.append(rep["brier"])
        print(f"{t:<16} {rep['n']:>6} {rep['accuracy']:>7.3f} "
              f"{rep['macro_f1']:>7.3f} {rep['ece']:>7.3f} {rep['brier']:>7.3f} "
              f"{temps[t]:>6.2f}")
    report["macro_accuracy"] = float(np.mean(accs)) if accs else float("nan")
    report["macro_ece"] = float(np.mean(eces)) if eces else float("nan")
    report["macro_brier"] = float(np.mean(briers)) if briers else float("nan")
    print("-" * 62)
    print(f"{'MACRO':<16} {'':>6} {report['macro_accuracy']:>7.3f} {'':>7} "
          f"{report['macro_ece']:>7.3f} {report['macro_brier']:>7.3f}")

    if ood:
        ood_per = collect(model, mk(ood), specs, device, temps)
        report["ood"] = {}
        for t, d in ood_per.items():
            if d["labels"]:
                p, y = np.stack(d["probs"]), np.array(d["labels"])
                report["ood"][t] = {"accuracy": M.accuracy(p, y),
                                    "mean_confidence": float(p.max(1).mean()),
                                    "ece": M.ece(p, y)}

    # --- the Path A number: single forward pass, all six decisions ---------
    model.eval()
    b = next(iter(dl_te))
    ids = b["input_ids"][:1].to(device)
    am = b["attention_mask"][:1].to(device)
    with torch.no_grad():
        for _ in range(10):
            model(ids, am)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(100):
            model(ids, am)
        torch.cuda.synchronize()
        single = (time.perf_counter() - t1) / 100 * 1000
    report["path_a_single_forward_ms_all_6_fields"] = single
    print(f"\nPath A prototype: {single:.2f} ms for ALL {len(specs)} decisions in one "
          f"forward pass (batch 1, fp32, GTX 1650)")
    print("Compare against eval/harness.py's p50 for JEV, which pays one decode step "
          "PER FIELD.\nPLAN.md 10: if Path B latency is adequate, serving this as a "
          "small FastAPI service beats\nmaintaining a patched llama.cpp + Ollama fork.")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "encoder": args.encoder,
                "task_sizes": sizes, "temperatures": temps},
               Path(args.out) / "model.pt")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {args.report} and {args.out}/model.pt")


if __name__ == "__main__":
    main()
