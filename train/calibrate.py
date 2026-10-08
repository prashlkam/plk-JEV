#!/usr/bin/env python3
"""Phase 6 — fit per-field temperature scaling on the `cal` split.

One scalar T per task, fitted by minimizing NLL. It cannot change accuracy (a
monotone transform preserves the argmax) and typically cuts ECE by 3-5x. Best
benefit-to-effort ratio in the entire plan.

TWO BACKENDS, AND THE ORDER MATTERS
-----------------------------------
  --backend hf      score the PyTorch checkpoint directly. Fast, exact, needs no
                    Ollama. Use during training iteration.
  --backend ollama  score through the deployed GGUF. SLOWER but authoritative.

PLAN.md 7 is explicit that quantization perturbs the logit distribution and can
silently decalibrate a model. So temperatures fitted on the fp32 checkpoint are
*preliminary*; the ones that ship must be refitted against the quantized GGUF the
SDK will actually call. `make calibrate-final` enforces that order, and the output
records which backend produced it so a mismatch is auditable rather than invisible.

Also fits, for comparison:
  * vector scaling (per-class scale+bias) — only worth it if per-class ECE stays
    bad after temperature scaling (PLAN.md 7)
  * the identity (T=1) baseline, so the improvement is always quantified
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev import metrics as M  # noqa: E402
from jev.spec import ABSTAIN_CODE, load_specs  # noqa: E402


# --- scoring backends ----------------------------------------------------

def score_hf(model_path, records, specs, device, max_seq_len, batch_size, tasks):
    """Collect per-task logits over the LEGAL label codes from a HF checkpoint."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sft import PairDataset, collate, last_token_logits

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, attn_implementation="sdpa").to(device)
    model.eval()
    model.config.use_cache = False

    ds = PairDataset(records, tok, specs, max_seq_len, 0, None, 0, task_filter=tasks)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=lambda b: collate(b, tok.pad_token_id, specs, 0.0))

    out: dict[str, dict] = {}
    with torch.no_grad():
        for batch in dl:
            ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            li = batch["last_idx"].to(device)
            logits = last_token_logits(model, ids, attn, li).float().cpu()
            logp = torch.log_softmax(logits, dim=-1)
            for i, task_id in enumerate(batch["task_ids"]):
                gold = int(batch["gold_code"][i])
                if gold == ABSTAIN_CODE:
                    continue
                codes = batch["legal_codes"][i]
                toks = batch["legal_tokens"][i]
                lp = logp[i, toks].numpy()
                keep = [j for j, c in enumerate(codes) if c != ABSTAIN_CODE]
                d = out.setdefault(task_id, {"logits": [], "labels": [], "mass": []})
                d["logits"].append(lp[keep])
                d["labels"].append(gold)
                d["mass"].append(float(np.exp(lp).sum()))
    return {k: {"logits": np.stack(v["logits"]), "labels": np.array(v["labels"]),
                "mass": v["mass"]} for k, v in out.items()}


def score_ollama(model, records, specs, host, tasks, top_logprobs):
    """Collect the same quantities through the deployed GGUF.

    Uses jev.ollama so the renormalization, pre-mask handling and warning logic are
    identical to what the SDK does in production — calibrating against a different
    code path than you serve with is how temperatures end up wrong.
    """
    from jev.ollama import JevOllama

    out: dict[str, dict] = {}
    n_warn = 0
    with JevOllama(model=model, host=host, calibration={"temperatures": {}},
                   top_logprobs=top_logprobs) as cli:
        t0 = time.perf_counter()
        for n, r in enumerate(records):
            for task_id in tasks:
                if task_id not in r["labels"]:
                    continue
                spec = specs[task_id]
                gold = spec.value_to_code(r["labels"][task_id])
                if gold == ABSTAIN_CODE:
                    continue
                d = cli.decide(task_id, r["input"])
                n_warn += bool(d.warnings)
                # raw_confidence is post-renormalization, pre-temperature; recover
                # per-class log-probs in the same ordering the HF path uses.
                vals = list(spec.values)
                p = np.array([d.distribution.get(v, 1e-12) for v in vals])
                p = p / p.sum()
                e = out.setdefault(task_id,
                                   {"logits": [], "labels": [], "mass": []})
                e["logits"].append(np.log(p))
                e["labels"].append(gold)
                e["mass"].append(d.legal_mass)
            if (n + 1) % 50 == 0:
                el = time.perf_counter() - t0
                print(f"  {n+1}/{len(records)} inputs  "
                      f"{(n+1)*len(tasks)/el:.1f} req/s", flush=True)
    if n_warn:
        print(f"  {n_warn} decisions carried warnings (see jev/ollama.py)")
    return {k: {"logits": np.stack(v["logits"]), "labels": np.array(v["labels"]),
                "mass": v["mass"]} for k, v in out.items()}


# --- fitting -------------------------------------------------------------

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits, labels, lo=0.05, hi=20.0, iters=200):
    """Golden-section search on NLL over T. Convex in log T, so this is reliable
    and needs no gradients or torch."""
    def nll(T):
        return float(-np.log(
            softmax(logits / T)[np.arange(len(labels)), labels] + 1e-12).mean())

    gr = (np.sqrt(5) - 1) / 2
    a, b = np.log(lo), np.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if nll(np.exp(c)) < nll(np.exp(d)):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
        if abs(b - a) < 1e-6:
            break
    T = float(np.exp((a + b) / 2))
    return T, nll(T)


def fit_vector_scaling(logits, labels, steps=400, lr=0.05):
    """Per-class scale + bias, fitted by plain gradient descent on NLL.

    More capacity than a single T, and more ways to overfit — only adopted when it
    beats temperature scaling on a held-out slice of `cal`, never on `cal` itself.
    """
    K = logits.shape[1]
    w, b = np.ones(K), np.zeros(K)
    Y = np.zeros_like(logits)
    Y[np.arange(len(labels)), labels] = 1.0
    for _ in range(steps):
        p = softmax(logits * w + b)
        g = (p - Y) / len(labels)
        w -= lr * (g * logits).sum(0)
        b -= lr * g.sum(0)
    nll = float(-np.log(softmax(logits * w + b)[
        np.arange(len(labels)), labels] + 1e-12).mean())
    return w.tolist(), b.tolist(), nll


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="workspace/out/jev-sft/best")
    ap.add_argument("--backend", default="hf", choices=["hf", "ollama"])
    ap.add_argument("--cal", default="workspace/data/eval/cal.jsonl")
    ap.add_argument("--out", default="spec/calibration.json")
    ap.add_argument("--report", default="eval/reports/calibration.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-seq-len", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--vector-scaling", action="store_true")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of cal held back to compare fitting methods")
    args = ap.parse_args()

    specs = load_specs()
    tasks = list(specs)
    records = [json.loads(l) for l in Path(args.cal).open()]
    if args.limit:
        records = records[: args.limit]
    print(f"{len(records):,} cal examples from {args.cal}  backend={args.backend}")
    if args.backend == "hf":
        print("NOTE: these temperatures are PRELIMINARY. PLAN.md 7 — quantization\n"
              "      perturbs logits, so refit with --backend ollama against the\n"
              "      deployed GGUF before shipping.")

    if args.backend == "hf":
        import torch
        scored = score_hf(args.model, records, specs,
                          "cuda" if torch.cuda.is_available() else "cpu",
                          args.max_seq_len, args.batch_size, tasks)
    else:
        scored = score_ollama(args.model, records, specs, args.host, tasks,
                              args.top_logprobs)

    rng = np.random.default_rng(0)
    temps, report = {}, {"backend": args.backend, "model": args.model,
                         "n_cal": len(records), "tasks": {}}

    print(f"\n{'task':<16} {'n':>6} {'T':>6} {'ECE@1':>8} {'ECE@T':>8} "
          f"{'NLL@1':>8} {'NLL@T':>8} {'Brier@1':>8} {'Brier@T':>8} {'mass':>6}")
    print("-" * 96)
    for task_id in tasks:
        if task_id not in scored:
            continue
        s = scored[task_id]
        logits, labels = s["logits"], s["labels"]
        n = len(labels)
        idx = rng.permutation(n)
        n_ho = int(n * args.holdout)
        ho, fit = idx[:n_ho], idx[n_ho:]

        T, _ = fit_temperature(logits[fit], labels[fit])
        p1, pT = softmax(logits), softmax(logits / T)
        ece1, eceT = M.ece(p1, labels), M.ece(pT, labels)
        rec = {
            "n": int(n), "temperature": T,
            "before": {"ece": ece1, "nll": M.nll(p1, labels),
                       "brier": M.brier(p1, labels),
                       "ece_adaptive": M.ece_adaptive(p1, labels),
                       "accuracy": M.accuracy(p1, labels)},
            "after": {"ece": eceT, "nll": M.nll(pT, labels),
                      "brier": M.brier(pT, labels),
                      "ece_adaptive": M.ece_adaptive(pT, labels),
                      "accuracy": M.accuracy(pT, labels)},
            "legal_mass": M.legal_mass_stats(s["mass"]),
            "reliability_after": M.reliability(pT, labels),
        }
        if args.vector_scaling and n_ho > 50:
            w, b, _ = fit_vector_scaling(logits[fit], labels[fit])
            pv = softmax(logits * np.array(w) + np.array(b))
            rec["vector_scaling"] = {
                "w": w, "b": b,
                "holdout_nll_temperature": M.nll(softmax(logits[ho] / T), labels[ho]),
                "holdout_nll_vector": M.nll(pv[ho], labels[ho]),
                "holdout_ece_temperature": M.ece(softmax(logits[ho] / T), labels[ho]),
                "holdout_ece_vector": M.ece(pv[ho], labels[ho]),
            }
            better = (rec["vector_scaling"]["holdout_nll_vector"]
                      < rec["vector_scaling"]["holdout_nll_temperature"])
            rec["vector_scaling"]["adopt"] = bool(better)

        temps[task_id] = T
        report["tasks"][task_id] = rec
        lm = rec["legal_mass"].get("mean", float("nan"))
        print(f"{task_id:<16} {n:>6,} {T:>6.3f} {ece1:>8.4f} {eceT:>8.4f} "
              f"{rec['before']['nll']:>8.4f} {rec['after']['nll']:>8.4f} "
              f"{rec['before']['brier']:>8.4f} {rec['after']['brier']:>8.4f} "
              f"{lm:>6.3f}")

    if report["tasks"]:
        e1 = float(np.mean([r["before"]["ece"] for r in report["tasks"].values()]))
        eT = float(np.mean([r["after"]["ece"] for r in report["tasks"].values()]))
        report["macro_ece_before"], report["macro_ece_after"] = e1, eT
        print(f"\nmacro ECE {e1:.4f} -> {eT:.4f}"
              + (f"  ({e1/eT:.1f}x better)" if eT > 1e-9 else ""))
        bad = {t: r["legal_mass"].get("mean") for t, r in report["tasks"].items()
               if (r["legal_mass"].get("mean") or 1) < 0.95}
        if bad:
            print(f"\n! legal_mass below 0.95 for {bad}")
            print("  The grammar mask is deciding rather than the model, so these "
                  "temperatures\n  are fitted on partly artificial probabilities. "
                  "Train longer before trusting them.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "version": 1, "fitted": True, "backend": args.backend,
        "model": str(args.model), "n_cal": len(records),
        "method": "per-field temperature scaling (NLL, golden-section)",
        "temperatures": temps,
        "authoritative": args.backend == "ollama",
        "note": ("Fitted against the deployed GGUF." if args.backend == "ollama"
                 else "PRELIMINARY: fitted on the fp32 checkpoint. Refit with "
                      "--backend ollama after quantization before shipping."),
    }, indent=2) + "\n")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {args.out} and {args.report}")


if __name__ == "__main__":
    main()
