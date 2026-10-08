#!/usr/bin/env python3
"""Put every run in eval/reports/ side by side, and answer the only question that
matters: is JEV earning its parameters?

PLAN.md 7 lists the baselines to beat. This renders them as one table and then
states a verdict, because the failure mode of a project like this is a report full
of respectable-looking numbers that nobody compares to a linear model.

The verdict rules, in order of severity:

  1. If a baseline matches JEV on accuracy AND on ECE, say so plainly. A decoder
     LLM is then the wrong tool, and shipping the baseline is the honest call.
  2. If JEV's accuracy exceeds the annotator ceiling, it is fitting the labeler.
  3. If JEV's legal_mass < 0.95, its confidences are mask artifacts and every
     calibration number in its column is meaningless regardless of how good it looks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TOL_ACC = 0.01   # within 1pp of accuracy counts as "matched"
TOL_ECE = 0.01   # within 1pp of ECE counts as "matched"


def load_reports(d: Path) -> list[dict]:
    out = []
    for p in sorted(d.glob("*.json")):
        if p.name == "calibration.json":
            continue
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if "tasks" not in r:
            continue
        r["_file"] = p.name
        r.setdefault("label", p.stem)
        out.append(r)
    return out


def cell(r, task, key):
    t = r.get("tasks", {}).get(task)
    if not t or key not in t:
        return None
    v = t[key]
    return v.get("mean") if isinstance(v, dict) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/reports")
    ap.add_argument("--jev", default=None,
                    help="label substring identifying the JEV run (default: any "
                         "report whose label is not a baseline_*)")
    ap.add_argument("--out", default="eval/reports/COMPARISON.md")
    args = ap.parse_args()

    reports = load_reports(Path(args.dir))
    if not reports:
        raise SystemExit(f"no reports in {args.dir}; run `make eval` and `make baselines`")

    jev = next((r for r in reports if args.jev and args.jev in r["label"]), None)
    if jev is None:
        jev = next((r for r in reports
                    if not r["_file"].startswith("baseline")), None)
    baselines = [r for r in reports if r is not jev]

    tasks = sorted({t for r in reports for t in r.get("tasks", {})})
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    synth = any(r.get("synthetic") for r in reports)
    emit("# Model comparison\n")
    if synth:
        emit("> **SYNTHETIC CORPUS.** Labels are a deterministic function of latent\n"
             "> facts in `data/gen_seed.py`. These numbers measure pipeline\n"
             "> correctness, not model quality. See `data/README.md`.\n")

    emit("## Accuracy\n")
    hdr = f"| {'run':<28} | " + " | ".join(f"{t:<14}" for t in tasks) + " | macro |"
    emit(hdr)
    emit("|" + "-" * 30 + "|" + "|".join(["-" * 16] * len(tasks)) + "|-------|")
    for r in ([jev] if jev else []) + baselines:
        row = f"| {r['label'][:28]:<28} | " + " | ".join(
            (f"{cell(r,t,'accuracy'):<14.3f}" if cell(r, t, "accuracy") is not None
             else f"{'-':<14}") for t in tasks)
        m = r.get("macro_accuracy")
        emit(row + f" | {m:.3f} |" if m is not None else row + " |   -   |")
    if po := next((r.get("prior_only") for r in reports if r.get("prior_only")), None):
        emit(f"| {'(prior only — the floor)':<28} | " + " | ".join(
            (f"{po[t]['accuracy']:<14.3f}" if t in po else f"{'-':<14}")
            for t in tasks) + " |   -   |")

    emit("\n## ECE (lower is better)\n")
    emit(hdr)
    emit("|" + "-" * 30 + "|" + "|".join(["-" * 16] * len(tasks)) + "|-------|")
    for r in ([jev] if jev else []) + baselines:
        row = f"| {r['label'][:28]:<28} | " + " | ".join(
            (f"{cell(r,t,'ece'):<14.3f}" if cell(r, t, "ece") is not None
             else f"{'-':<14}") for t in tasks)
        m = r.get("macro_ece")
        emit(row + f" | {m:.3f} |" if m is not None else row + " |   -   |")

    emit("\n## Latency and cost\n")
    emit(f"| {'run':<28} | {'p50 ms':>8} | {'tok/decision':>13} | {'params':>8} |")
    emit("|" + "-" * 30 + "|----------|---------------|----------|")
    for r in ([jev] if jev else []) + baselines:
        rt = r.get("runtime", {}) or {}
        p50 = rt.get("latency_ms_p50")
        tpd = rt.get("tokens_per_decision_mean")
        pm = r.get("params_m")
        sfp = r.get("path_a_single_forward_ms_all_6_fields")
        note = f"{p50:.1f}" if p50 else (f"{sfp:.2f}*" if sfp else "-")
        emit(f"| {r['label'][:28]:<28} | {note:>8} | "
             f"{(f'{tpd:.2f}' if tpd else '-'):>13} | "
             f"{(f'{pm:.0f}M' if pm else '-'):>8} |")
    emit("\n`*` single forward pass for ALL fields at once (the Path A prototype), "
         "not per field.")

    # --- verdict --------------------------------------------------------
    emit("\n## Verdict\n")
    verdicts = []
    if jev is None:
        verdicts.append("No JEV run found — only baselines. Run `make eval`.")
    else:
        lm = (jev.get("legal_mass") or {}).get("mean")
        if lm is not None and lm < 0.95:
            verdicts.append(
                f"**JEV's legal_mass is {lm:.3f} (< 0.95).** The grammar mask is "
                "choosing the answer, not the model, so every calibration number in "
                "JEV's column is an artifact of renormalizing noise. Fix this before "
                "reading anything else: train longer, or check the Modelfile and "
                "prompt template. See spikes/RESULTS.md 0.1.")

        for b in baselines:
            ja, ba = jev.get("macro_accuracy"), b.get("macro_accuracy")
            je, be = jev.get("macro_ece"), b.get("macro_ece")
            if None in (ja, ba, je, be):
                continue
            if ba >= ja - TOL_ACC and be <= je + TOL_ECE:
                verdicts.append(
                    f"**`{b['label']}` matches or beats JEV on BOTH accuracy "
                    f"({ba:.3f} vs {ja:.3f}) and ECE ({be:.3f} vs {je:.3f}).** "
                    "PLAN.md risk #3: a decoder LLM is the wrong tool for these "
                    "tasks. Shipping the baseline is the honest call.")
            elif ba >= ja - TOL_ACC:
                verdicts.append(
                    f"`{b['label']}` matches JEV on accuracy ({ba:.3f} vs {ja:.3f}) "
                    f"but is worse calibrated ({be:.3f} vs {je:.3f} ECE). JEV's "
                    "value here is confidence quality, not accuracy — so lead with "
                    "selective accuracy, not accuracy.")

        over = [(t, r["accuracy"], r["annotator_ceiling"])
                for t, r in jev.get("tasks", {}).items()
                if "annotator_ceiling" in r
                and r["accuracy"] > r["annotator_ceiling"] + 0.01]
        for t, a, c in over:
            verdicts.append(
                f"**{t}: accuracy {a:.3f} exceeds the annotator ceiling {c:.3f}.** "
                "The model is fitting the labeler, not the task.")

    if not verdicts:
        verdicts.append("JEV beats every baseline on accuracy and ECE, legal_mass "
                        "is healthy, and no task exceeds its annotator ceiling.")
    for v in verdicts:
        emit(f"- {v}")

    if synth:
        emit("\nAll of the above is measured on synthetic data and says nothing "
             "about real-world quality.")

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
