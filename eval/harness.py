#!/usr/bin/env python3
"""Phase 6 — the eval harness. Every metric in PLAN.md 7, plus the spike gates.

Reports, per task and macro:

  accuracy / macro-F1          baseline quality
  ECE (15-bin) + adaptive ECE  is the stated confidence truthful
  MCE                          worst-bin calibration
  Brier                        joint accuracy + calibration, one number
  selective accuracy @ 50/80/95/100% coverage, and AURC
  latency p50/p99, tokens per decision
  schema violation rate        MUST be exactly 0.00%
  legal_mass                   the spike 0.1 gate: is the MODEL deciding, or the mask
  decision_token_index         MUST be 0: the spike 0.3 template gate
  OOD confidence delta         does confidence correctly DROP off-distribution

The last three are not in PLAN.md; Phase 0 added them. A model can post a beautiful
ECE while being untrustworthy if legal_mass is low, because then the grammar picked
the answer and the confidence is an artifact of renormalizing noise.

HONESTY STAMP
-------------
Any report scored against the synthetic seed corpus is stamped `synthetic: true`
with a caveat embedded in the JSON. Labels there are a deterministic function of
latent facts, so a model can approach 100% by learning the generator. These numbers
measure that the pipeline works, not that the model does.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev import metrics as M  # noqa: E402
from jev.spec import ABSTAIN_CODE, load_specs  # noqa: E402

CAVEAT = (
    "SYNTHETIC CORPUS. Labels are a deterministic function of latent facts in "
    "data/gen_seed.py, so a model can approach 100% by learning the generator. "
    "These numbers measure pipeline correctness, NOT model quality. Replace with "
    "real labeled data before reporting any of this as performance."
)


def score_ollama(model, records, specs, host, tasks, top_logprobs, keep_alive):
    """Score through the deployed GGUF — the production code path."""
    from jev.ollama import JevOllama

    per: dict[str, dict] = {}
    violations, bad_index, latencies, tokens = 0, 0, [], []
    # Prefix reuse is not directly observable (see jev.ollama.Decision), so compare
    # prompt-eval time on the FIRST task of an input against the follow-ups.
    prompt_ms_first, prompt_ms_followup = [], []
    t0 = time.perf_counter()
    with JevOllama(model=model, host=host, top_logprobs=top_logprobs,
                   keep_alive=keep_alive) as cli:
        h = cli.health()
        if not h["model_present"]:
            sys.exit(f"model {model!r} not in Ollama. Have: {h['models']}\n"
                     f"Run `make serve` to build and register it.")
        for n, r in enumerate(records):
            for i, task_id in enumerate(tasks):
                if task_id not in r["labels"]:
                    continue
                spec = specs[task_id]
                gold = spec.value_to_code(r["labels"][task_id])
                try:
                    d = cli.decide(task_id, r["input"])
                except ValueError:
                    # jev.ollama raises when the response is not a legal code,
                    # which the grammar should make impossible.
                    violations += 1
                    continue
                latencies.append(d.latency_ms)
                tokens.append(d.eval_count)
                if d.decision_token_index != 0:
                    bad_index += 1
                (prompt_ms_followup if i > 0 else prompt_ms_first).append(
                    d.prompt_eval_ms)
                e = per.setdefault(task_id, {"probs": [], "labels": [],
                                             "mass": [], "abstain": 0, "lat": []})
                e["lat"].append(d.latency_ms)
                if d.abstained:
                    e["abstain"] += 1
                    continue
                if gold == ABSTAIN_CODE:
                    continue
                p = np.array([d.distribution.get(v, 0.0) for v in spec.values])
                s = p.sum()
                p = p / s if s > 0 else np.full(len(spec.values), 1 / len(spec.values))
                e["probs"].append(p)
                e["labels"].append(gold)
                e["mass"].append(d.legal_mass)
            if (n + 1) % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  {n+1}/{len(records)} inputs  "
                      f"{len(latencies)/el:.1f} req/s", flush=True)
    return per, {
        "schema_violations": violations,
        "bad_decision_index": bad_index,
        "n_requests": len(latencies),
        "latency_ms_p50": float(np.percentile(latencies, 50)) if latencies else None,
        "latency_ms_p90": float(np.percentile(latencies, 90)) if latencies else None,
        "latency_ms_p99": float(np.percentile(latencies, 99)) if latencies else None,
        "latency_ms_mean": float(np.mean(latencies)) if latencies else None,
        "tokens_per_decision_mean": float(np.mean(tokens)) if tokens else None,
        "prompt_eval_ms_first_task": float(np.mean(prompt_ms_first))
            if prompt_ms_first else None,
        "prompt_eval_ms_followup_tasks": float(np.mean(prompt_ms_followup))
            if prompt_ms_followup else None,
    }


def score_hf(model_path, records, specs, tasks, max_seq_len, batch_size):
    """Score the PyTorch checkpoint. No latency or schema numbers — those are
    properties of the deployment, not the weights."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
    from sft import PairDataset, collate, last_token_logits

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    # Repo-relative, not cwd-relative: `make eval-hf` and a direct invocation from
    # another directory must pick up the same temperatures.
    cal_path = Path(__file__).resolve().parent.parent / "spec" / "calibration.json"
    cal = json.loads(cal_path.read_text()) if cal_path.exists() \
        else {"temperatures": {}}
    if not cal.get("fitted"):
        print("  NOTE: no fitted spec/calibration.json; scoring with T=1 "
              "(UNCALIBRATED). Run train/calibrate.py.")

    per: dict[str, dict] = {}
    with torch.no_grad():
        for batch in dl:
            logits = last_token_logits(model, batch["input_ids"].to(device),
                                       batch["attention_mask"].to(device),
                                       batch["last_idx"].to(device)).float().cpu()
            p_full = torch.softmax(logits, dim=-1)
            for i, task_id in enumerate(batch["task_ids"]):
                gold = int(batch["gold_code"][i])
                if gold == ABSTAIN_CODE:
                    continue
                codes, toks = batch["legal_codes"][i], batch["legal_tokens"][i]
                pl = p_full[i, toks].numpy()
                mass = float(pl.sum())
                keep = [j for j, c in enumerate(codes) if c != ABSTAIN_CODE]
                T = float(cal.get("temperatures", {}).get(task_id, 1.0))
                lg = np.log(np.clip(pl[keep], 1e-12, None)) / T
                pr = np.exp(lg - lg.max())
                pr = pr / pr.sum()
                e = per.setdefault(task_id, {"probs": [], "labels": [],
                                             "mass": [], "abstain": 0, "lat": []})
                e["probs"].append(pr)
                e["labels"].append(gold)
                e["mass"].append(mass)
    return per, {"note": "hf backend: no latency/schema metrics (deployment properties)"}


def summarize(per, specs, ceilings=None):
    out, eces, accs, briers, masses = {}, [], [], [], []
    for task_id, e in sorted(per.items()):
        if not e["labels"]:
            continue
        probs, labels = np.stack(e["probs"]), np.array(e["labels"])
        rep = M.report(probs, labels, ordinal=specs[task_id].ordinal,
                       masses=e["mass"])
        rep["abstain_rate"] = e["abstain"] / max(1, e["abstain"] + len(labels))
        if e["lat"]:
            rep["latency_ms_p50"] = float(np.percentile(e["lat"], 50))
        if ceilings and task_id in ceilings:
            c = ceilings[task_id]
            rep["annotator_ceiling"] = c
            rep["headroom_to_ceiling"] = c - rep["accuracy"]
            if rep["accuracy"] > c + 0.01:
                rep["WARNING"] = (
                    f"accuracy {rep['accuracy']:.3f} exceeds the annotator ceiling "
                    f"{c:.3f}: the model is fitting the labeler, not the task")
        out[task_id] = rep
        eces.append(rep["ece"])
        accs.append(rep["accuracy"])
        briers.append(rep["brier"])
        masses.extend(e["mass"])
    macro = {
        "macro_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "macro_ece": float(np.mean(eces)) if eces else float("nan"),
        "macro_brier": float(np.mean(briers)) if briers else float("nan"),
        "legal_mass": M.legal_mass_stats(masses),
    }
    return out, macro


def plot_reliability(per, specs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  (skipping reliability plot: {e})")
        return
    tasks = [t for t in sorted(per) if per[t]["labels"]]
    if not tasks:
        return
    cols = min(3, len(tasks))
    rows = (len(tasks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.4 * rows), squeeze=False)
    for ax, task_id in zip([a for r in axes for a in r], tasks):
        e = per[task_id]
        probs, labels = np.stack(e["probs"]), np.array(e["labels"])
        rel = M.reliability(probs, labels)
        xs = [b["mean_conf"] for b in rel["bins"] if b["n"] > 0]
        ys = [b["accuracy"] for b in rel["bins"] if b["n"] > 0]
        ns = [b["n"] for b in rel["bins"] if b["n"] > 0]
        ax.plot([0, 1], [0, 1], "--", lw=1, color="#888", label="perfect")
        ax.scatter(xs, ys, s=[max(8, min(140, n / 2)) for n in ns], alpha=0.75)
        ax.plot(xs, ys, lw=1, alpha=0.6)
        ax.set_title(f"{task_id}\nECE {M.ece(probs, labels):.3f}", fontsize=9)
        ax.set_xlabel("confidence", fontsize=8)
        ax.set_ylabel("accuracy", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(labelsize=7)
    for a in [a for r in axes for a in r][len(tasks):]:
        a.axis("off")
    fig.suptitle("Reliability — point size is bin count", fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"  reliability diagram -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev")
    ap.add_argument("--backend", default="ollama", choices=["ollama", "hf"])
    ap.add_argument("--split", default="workspace/data/eval/test.jsonl")
    ap.add_argument("--ood", default="workspace/data/sft/seed_ood.jsonl")
    ap.add_argument("--no-ood", action="store_true")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--ood-limit", type=int, default=200)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--keep-alive", default="30m")
    ap.add_argument("--max-seq-len", type=int, default=192)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--agreement", default="data/sft/agreement_report.json")
    ap.add_argument("--out", default="eval/reports/report.json")
    ap.add_argument("--plot", default="eval/reports/reliability.png")
    ap.add_argument("--label", default=None, help="name for this run in the report")
    args = ap.parse_args()

    specs = load_specs()
    tasks = args.tasks or list(specs)
    records = [json.loads(l) for l in Path(args.split).open()]
    if args.limit:
        records = records[: args.limit]

    ceilings = None
    if Path(args.agreement).exists():
        ag = json.loads(Path(args.agreement).read_text())
        ceilings = {t: v["accuracy_ceiling"] for t, v in ag.get("tasks", {}).items()}

    synthetic = all(str(r.get("source", "")).startswith("synthetic")
                    for r in records)
    print(f"model={args.model} backend={args.backend}  {len(records):,} inputs x "
          f"{len(tasks)} tasks")
    if synthetic:
        print(f"\n*** {CAVEAT} ***\n")

    score = (lambda rs: score_ollama(args.model, rs, specs, args.host, tasks,
                                     args.top_logprobs, args.keep_alive)
             ) if args.backend == "ollama" else (
        lambda rs: score_hf(args.model, rs, specs, tasks, args.max_seq_len,
                            args.batch_size))

    per, runtime = score(records)
    tasks_rep, macro = summarize(per, specs, ceilings)

    report = {
        "label": args.label or f"{args.model}:{args.backend}",
        "model": args.model, "backend": args.backend, "split": args.split,
        "n_inputs": len(records), "tasks": tasks_rep, **macro, "runtime": runtime,
        "synthetic": synthetic,
    }
    if synthetic:
        report["CAVEAT"] = CAVEAT

    # --- OOD: does confidence correctly drop? ---------------------------
    if not args.no_ood and Path(args.ood).exists():
        ood_rec = [json.loads(l) for l in Path(args.ood).open()][: args.ood_limit]
        print(f"\nOOD split: {len(ood_rec)} inputs from {args.ood}")
        ood_per, ood_runtime = score(ood_rec)
        ood_tasks, ood_macro = summarize(ood_per, specs)
        deltas = {}
        for t in ood_tasks:
            if t in tasks_rep:
                deltas[t] = {
                    "id_confidence": tasks_rep[t]["mean_confidence"],
                    "ood_confidence": ood_tasks[t]["mean_confidence"],
                    "confidence_drop": tasks_rep[t]["mean_confidence"]
                                       - ood_tasks[t]["mean_confidence"],
                    "id_accuracy": tasks_rep[t]["accuracy"],
                    "ood_accuracy": ood_tasks[t]["accuracy"],
                    "accuracy_drop": tasks_rep[t]["accuracy"]
                                     - ood_tasks[t]["accuracy"],
                }
        report["ood"] = {"tasks": ood_tasks, **ood_macro, "runtime": ood_runtime,
                         "deltas": deltas}

    # --- gates ----------------------------------------------------------
    gates = {}
    v = runtime.get("schema_violations")
    if v is not None:
        gates["schema_violation_rate_is_zero"] = (v == 0)
    bi = runtime.get("bad_decision_index")
    if bi is not None:
        gates["decision_token_index_always_0"] = (bi == 0)
    lm = macro["legal_mass"].get("mean")
    if lm is not None:
        gates["legal_mass_ge_0.95"] = (lm >= 0.95)
    if "ood" in report and report["ood"]["deltas"]:
        drops = [d["confidence_drop"] for d in report["ood"]["deltas"].values()]
        acc_drops = [d["accuracy_drop"] for d in report["ood"]["deltas"].values()]
        # Confidence must fall at least as fast as accuracy, else the model is
        # confidently wrong off-distribution — worse than useless in a pipeline.
        gates["ood_confidence_tracks_accuracy"] = bool(
            np.mean(drops) >= np.mean(acc_drops) * 0.5)
    report["gates"] = gates

    # --- print ----------------------------------------------------------
    print(f"\n{'task':<16} {'n':>5} {'acc':>6} {'mF1':>6} {'ECE':>6} {'aECE':>6} "
          f"{'Brier':>6} {'AURC':>6} {'mass':>6} {'p50ms':>7} {'ceil':>6}")
    print("-" * 94)
    for t, r in sorted(tasks_rep.items()):
        print(f"{t:<16} {r['n']:>5} {r['accuracy']:>6.3f} {r['macro_f1']:>6.3f} "
              f"{r['ece']:>6.3f} {r['ece_adaptive']:>6.3f} {r['brier']:>6.3f} "
              f"{r['aurc']:>6.3f} "
              f"{r['legal_mass'].get('mean', float('nan')):>6.3f} "
              f"{r.get('latency_ms_p50', float('nan')):>7.1f} "
              f"{r.get('annotator_ceiling', float('nan')):>6.3f}")
        if "WARNING" in r:
            print(f"  ! {r['WARNING']}")
    print("-" * 94)
    print(f"{'MACRO':<16} {'':>5} {macro['macro_accuracy']:>6.3f} {'':>6} "
          f"{macro['macro_ece']:>6.3f} {'':>6} {macro['macro_brier']:>6.3f}")

    print("\nselective accuracy (auto-act above threshold, defer the rest):")
    for t, r in sorted(tasks_rep.items()):
        s = r["selective"]
        print(f"  {t:<16} " + "  ".join(
            f"@{float(k):.0%}={s[k]['accuracy']:.3f}" for k in
            sorted(s, key=float)))

    if runtime.get("latency_ms_p50"):
        print(f"\nlatency p50 {runtime['latency_ms_p50']:.1f} ms  "
              f"p90 {runtime['latency_ms_p90']:.1f}  "
              f"p99 {runtime['latency_ms_p99']:.1f}  "
              f"{runtime['tokens_per_decision_mean']:.2f} tokens/decision")
        pf, pl = (runtime.get("prompt_eval_ms_first_task"),
                  runtime.get("prompt_eval_ms_followup_tasks"))
        if pf and pl:
            print(f"prompt-eval {pf:.1f} ms on the first task of an input vs "
                  f"{pl:.1f} ms on follow-ups ({1 - pl/pf:.0%} lower) — "
                  f"the only observable proxy for prefix reuse")

    if "ood" in report:
        print("\nOOD (confidence MUST drop at least as fast as accuracy):")
        for t, d in sorted(report["ood"]["deltas"].items()):
            flag = "" if d["confidence_drop"] >= d["accuracy_drop"] * 0.5 else "  <-- !"
            print(f"  {t:<16} conf {d['id_confidence']:.3f}->{d['ood_confidence']:.3f} "
                  f"({d['confidence_drop']:+.3f})   "
                  f"acc {d['id_accuracy']:.3f}->{d['ood_accuracy']:.3f} "
                  f"({d['accuracy_drop']:+.3f}){flag}")

    print("\ngates:")
    for k, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    if args.plot:
        plot_reliability(per, specs, args.plot)
    if synthetic:
        print(f"\n*** {CAVEAT} ***")
    if not all(gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
