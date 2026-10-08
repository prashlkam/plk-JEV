#!/usr/bin/env python3
"""Baselines that cost nothing: majority-class prior, and TF-IDF + logistic regression.

PLAN.md 7 says run these in week 1, not week 8. Two reasons they matter more than
they look:

1. **Logistic regression on TF-IDF is well calibrated almost for free.** It is
   fitted by maximizing likelihood on the exact objective ECE measures, so it
   frequently beats a fine-tuned LLM on ECE while losing on accuracy. If it beats
   JEV on *both*, the project does not need an LLM and should be reframed.

2. **The prior-only baseline sets the floor.** On an imbalanced task like
   `gate_human_v1`, predicting the majority class with its base rate is perfectly
   calibrated by construction and can score high accuracy. Any model that does not
   clearly beat it has learned nothing, and reporting raw accuracy without this
   comparison hides that completely.

Both run on CPU in seconds, so there is no excuse for not having the numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from jev import metrics as M  # noqa: E402
from jev.spec import ABSTAIN_CODE, load_specs  # noqa: E402


def load(path, limit=0):
    rows = [json.loads(l) for l in Path(path).open()]
    return rows[:limit] if limit else rows


def xy(records, task_id, spec):
    X, y = [], []
    for r in records:
        if task_id not in r["labels"]:
            continue
        c = spec.value_to_code(r["labels"][task_id])
        if c == ABSTAIN_CODE:
            continue
        X.append(r["input"])
        y.append(c)
    return X, np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="workspace/data/eval/train.jsonl")
    ap.add_argument("--test", default="workspace/data/eval/test.jsonl")
    ap.add_argument("--cal", default="workspace/data/eval/cal.jsonl")
    ap.add_argument("--ood", default="workspace/data/sft/seed_ood.jsonl")
    ap.add_argument("--train-limit", type=int, default=40000)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--out", default="eval/reports/baseline_tfidf.json")
    ap.add_argument("--calibrate", action="store_true", default=True,
                    help="fit a temperature on `cal` too, for a fair comparison")
    args = ap.parse_args()

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    specs = load_specs()
    tr = load(args.train, args.train_limit)
    te = load(args.test, args.test_limit)
    cal = load(args.cal, 4000)
    ood = load(args.ood, 400) if Path(args.ood).exists() else []
    print(f"train {len(tr):,}  test {len(te):,}  cal {len(cal):,}  ood {len(ood):,}")

    report = {"label": "tfidf+logreg", "model": "tfidf+logreg",
              "backend": "sklearn", "split": args.test,
              "n_inputs": len(te), "tasks": {}, "prior_only": {}}

    print(f"\n{'task':<16} {'prior_acc':>10} {'prior_ece':>10} {'acc':>7} "
          f"{'mF1':>7} {'ECE':>7} {'ECE@T':>7} {'Brier':>7} {'T':>6} {'fit_s':>7}")
    print("-" * 96)
    accs, eces, briers = [], [], []
    for task_id, spec in specs.items():
        Xtr, ytr = xy(tr, task_id, spec)
        Xte, yte = xy(te, task_id, spec)
        if not len(ytr) or not len(yte):
            continue

        # --- prior-only floor ------------------------------------------
        counts = np.bincount(ytr, minlength=spec.n_classes).astype(float)
        prior = counts / counts.sum()
        p_prior = np.tile(prior, (len(yte), 1))
        report["prior_only"][task_id] = {
            "accuracy": M.accuracy(p_prior, yte), "ece": M.ece(p_prior, yte),
            "brier": M.brier(p_prior, yte), "macro_f1": M.macro_f1(p_prior, yte),
            "distribution": prior.tolist(),
        }

        # --- tfidf + logreg --------------------------------------------
        t0 = time.perf_counter()
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2),
                            min_df=2, max_features=200_000,
                            strip_accents="unicode", lowercase=True),
            LogisticRegression(max_iter=2000, C=4.0, class_weight=None,
                               n_jobs=-1),
        )
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

        pte = pipe.predict_proba(Xte)
        # Align columns to the full class axis (a class may be absent from train).
        full = np.zeros((len(yte), spec.n_classes))
        for j, c in enumerate(pipe.classes_):
            full[:, c] = pte[:, j]
        full = full / full.sum(1, keepdims=True).clip(1e-12)

        T = 1.0
        if args.calibrate:
            Xc, yc = xy(cal, task_id, spec)
            if len(yc) > 50:
                pc = pipe.predict_proba(Xc)
                fc = np.zeros((len(yc), spec.n_classes))
                for j, c in enumerate(pipe.classes_):
                    fc[:, c] = pc[:, j]
                lg = np.log(fc.clip(1e-12))
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent
                                       / "train"))
                from calibrate import fit_temperature
                T, _ = fit_temperature(lg, yc)
        lg = np.log(full.clip(1e-12)) / T
        pT = np.exp(lg - lg.max(1, keepdims=True))
        pT = pT / pT.sum(1, keepdims=True)

        rep = M.report(pT, yte, ordinal=spec.ordinal)
        rep["temperature"] = T
        rep["ece_uncalibrated"] = M.ece(full, yte)
        rep["fit_seconds"] = fit_s
        report["tasks"][task_id] = rep
        accs.append(rep["accuracy"])
        eces.append(rep["ece"])
        briers.append(rep["brier"])

        pr = report["prior_only"][task_id]
        print(f"{task_id:<16} {pr['accuracy']:>10.3f} {pr['ece']:>10.3f} "
              f"{rep['accuracy']:>7.3f} {rep['macro_f1']:>7.3f} "
              f"{rep['ece_uncalibrated']:>7.3f} {rep['ece']:>7.3f} "
              f"{rep['brier']:>7.3f} {T:>6.2f} {fit_s:>7.1f}")

        if ood:
            Xo, yo = xy(ood, task_id, spec)
            if len(yo):
                po = pipe.predict_proba(Xo)
                fo = np.zeros((len(yo), spec.n_classes))
                for j, c in enumerate(pipe.classes_):
                    fo[:, c] = po[:, j]
                fo = fo / fo.sum(1, keepdims=True).clip(1e-12)
                rep["ood"] = {"accuracy": M.accuracy(fo, yo),
                              "mean_confidence": float(fo.max(1).mean()),
                              "ece": M.ece(fo, yo)}

    report["macro_accuracy"] = float(np.mean(accs)) if accs else float("nan")
    report["macro_ece"] = float(np.mean(eces)) if eces else float("nan")
    report["macro_brier"] = float(np.mean(briers)) if briers else float("nan")
    report["macro_prior_accuracy"] = float(np.mean(
        [v["accuracy"] for v in report["prior_only"].values()]))

    print("-" * 96)
    print(f"{'MACRO':<16} {report['macro_prior_accuracy']:>10.3f} {'':>10} "
          f"{report['macro_accuracy']:>7.3f} {'':>7} {'':>7} "
          f"{report['macro_ece']:>7.3f} {report['macro_brier']:>7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out}")
    print("\nRead this as the bar. If JEV does not clearly beat the TF-IDF ECE as "
          "well as its\naccuracy, temperature scaling is carrying the result and a "
          "600M-parameter decoder\nis the wrong tool for these six tasks.")


if __name__ == "__main__":
    main()
