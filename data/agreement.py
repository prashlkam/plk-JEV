#!/usr/bin/env python3
"""Inter-annotator agreement and soft-target construction.

PLAN.md 4b calls this the single highest-leverage thing in the project, for two
reasons that are worth keeping distinct:

1. **Agreement is the accuracy ceiling.** If three annotators agree only 80% of the
   time on `route_v1`, a model scoring 85% is fitting the labeler, not the task.
   Without this number you cannot tell the difference, ever.

2. **Disagreement is training signal.** 2-of-3 for `billing` means the right target
   is 0.67, not 1.0. Collapsing to a majority vote throws away the only
   cheap source of genuine aleatoric uncertainty you will ever have — and it is
   irreversible once collected that way.

Computes Krippendorff's alpha (nominal, and ordinal-weighted for the ordinal tasks),
observed agreement, Fleiss' kappa, and emits per-example soft targets consumed by
`train/sft.py --soft-targets`.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.spec import load_specs  # noqa: E402


def observed_agreement(votes: list[list[str]]) -> float:
    """Mean fraction of agreeing annotator PAIRS, averaged over examples."""
    tot, n = 0.0, 0
    for v in votes:
        if len(v) < 2:
            continue
        pairs = list(itertools.combinations(v, 2))
        tot += sum(a == b for a, b in pairs) / len(pairs)
        n += 1
    return tot / n if n else float("nan")


def fleiss_kappa(votes: list[list[str]], categories: list[str]) -> float:
    """Fleiss' kappa. Requires a constant number of raters per item."""
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    rows = [v for v in votes if len(v) == len(votes[0])]
    if not rows or len(rows[0]) < 2:
        return float("nan")
    n_raters = len(rows[0])
    N = len(rows)
    M = np.zeros((N, k))
    for i, v in enumerate(rows):
        for a in v:
            M[i, idx[a]] += 1
    p_j = M.sum(0) / (N * n_raters)
    P_i = ((M**2).sum(1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar, P_e = P_i.mean(), (p_j**2).sum()
    return float((P_bar - P_e) / (1 - P_e)) if abs(1 - P_e) > 1e-12 else float("nan")


def krippendorff_alpha(votes: list[list[str]], categories: list[str],
                       ordinal: bool = False) -> float:
    """Krippendorff's alpha via the coincidence-matrix formulation.

    Handles a variable number of raters per item, which Fleiss cannot, and
    supports ordinal weighting — essential for triage_v1 and urgency_v1 where
    P0-vs-P3 disagreement is far worse than P0-vs-P1.
    """
    k = len(categories)
    idx = {c: i for i, c in enumerate(categories)}
    # Coincidence matrix: each item contributes its pairs, weighted 1/(m_u - 1).
    O = np.zeros((k, k))
    for v in votes:
        m = len(v)
        if m < 2:
            continue
        for a, b in itertools.permutations(v, 2):
            O[idx[a], idx[b]] += 1.0 / (m - 1)
    n = O.sum()
    if n < 2:
        return float("nan")
    nc = O.sum(1)

    if ordinal:
        # Krippendorff's ordinal delta: squared distance in cumulative-count space.
        def delta(a, b):
            if a == b:
                return 0.0
            lo, hi = min(a, b), max(a, b)
            s = nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2.0
            return float(s**2)
    else:
        def delta(a, b):
            return 0.0 if a == b else 1.0

    D = np.array([[delta(a, b) for b in range(k)] for a in range(k)])
    Do = (O * D).sum() / n
    E = np.outer(nc, nc).astype(float)
    np.fill_diagonal(E, nc * (nc - 1))
    De = (E * D).sum() / (n * (n - 1))
    return float(1 - Do / De) if De > 1e-12 else float("nan")


def soft_target(votes: list[str], categories: list[str],
                laplace: float = 0.0) -> dict[str, float]:
    """Annotator vote distribution as a probability vector.

    `laplace > 0` adds pseudo-counts; left at 0 by default because smoothing here
    and label-smoothing in the SFT loss would compound into double-counted
    uncertainty.
    """
    c = collections.Counter(votes)
    tot = len(votes) + laplace * len(categories)
    return {cat: (c.get(cat, 0) + laplace) / tot
            for cat in categories if c.get(cat, 0) or laplace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="workspace/data/sft/seed.jsonl")
    ap.add_argument("--report", default="data/sft/agreement_report.json")
    ap.add_argument("--soft-out", default="workspace/data/sft/soft_targets.jsonl")
    ap.add_argument("--laplace", type=float, default=0.0)
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.inp).open()]
    annotated = [r for r in records if "annotations" in r]
    print(f"{len(records):,} records, {len(annotated):,} with annotator votes")
    if not annotated:
        sys.exit("no annotated records; nothing to measure")
    if len(annotated) < 2000:
        print(f"  WARNING: PLAN.md 4b wants >=2000 annotated examples, have "
              f"{len(annotated):,}")

    specs = load_specs()
    report = {
        "n_records": len(records),
        "n_annotated": len(annotated),
        "source": sorted({r.get("source", "?") for r in annotated}),
        "synthetic": all(r.get("source", "").startswith("synthetic")
                         for r in annotated),
        "tasks": {},
    }

    print(f"\n{'task':<16} {'n':>6} {'raters':>7} {'obs.agr':>8} {'Fleiss':>8} "
          f"{'alpha':>8} {'unanim':>8} {'ceiling':>8}")
    print("-" * 80)
    for task_id, spec in specs.items():
        votes = [r["annotations"][task_id]["votes"] for r in annotated
                 if task_id in r["annotations"]]
        if not votes:
            continue
        cats = list(spec.values)
        oa = observed_agreement(votes)
        fk = fleiss_kappa(votes, cats)
        al = krippendorff_alpha(votes, cats, ordinal=spec.ordinal)
        unan = float(np.mean([len(set(v)) == 1 for v in votes]))
        # The practical ceiling: how often a single random annotator matches the
        # modal label. A model cannot meaningfully exceed this on this label set.
        ceiling = float(np.mean([
            collections.Counter(v).most_common(1)[0][1] / len(v) for v in votes]))
        report["tasks"][task_id] = {
            "n": len(votes), "raters_per_item": sorted({len(v) for v in votes}),
            "observed_agreement": oa, "fleiss_kappa": fk,
            "krippendorff_alpha": al, "alpha_weighting":
                "ordinal" if spec.ordinal else "nominal",
            "fraction_unanimous": unan, "accuracy_ceiling": ceiling,
            "class_distribution": dict(collections.Counter(
                v for vv in votes for v in vv)),
        }
        print(f"{task_id:<16} {len(votes):>6,} "
              f"{str(sorted({len(v) for v in votes})):>7} {oa:>8.3f} {fk:>8.3f} "
              f"{al:>8.3f} {unan:>8.1%} {ceiling:>8.1%}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")

    outp = Path(args.soft_out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    n_soft = 0
    with outp.open("w") as fh:
        for r in annotated:
            soft = {t: soft_target(a["votes"], list(specs[t].values), args.laplace)
                    for t, a in r["annotations"].items() if t in specs}
            fh.write(json.dumps({"id": r["id"], "soft": soft}) + "\n")
            n_soft += 1

    print(f"\nwrote {args.report}")
    print(f"wrote {n_soft:,} soft targets -> {outp}")
    print("\nRead accuracy_ceiling as the number model accuracy must be compared "
          "against.\nA model above it is fitting the annotators, not the task.")
    if report["synthetic"]:
        print("\nNOTE: these annotators are SIMULATED (data/gen_seed.py). The "
              "agreement numbers\n      validate the pipeline; they are not "
              "human agreement and must not be reported as such.")


if __name__ == "__main__":
    main()
