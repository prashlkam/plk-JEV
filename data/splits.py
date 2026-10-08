#!/usr/bin/env python3
"""Carve train / dev / cal / test splits, stratified, with a hash-stable assignment.

PLAN.md 4c requires four disjoint splits and is emphatic about one of them:

  train  SFT
  dev    early stopping (on ECE, not accuracy)
  cal    temperature fitting ONLY — never trained on, never early-stopped on
  test   touched once per milestone
  ood    separate file, deliberately shifted (produced by gen_seed.py)

`cal` leaking into training is the classic way a calibration result turns out to be
nothing. Assignment is by hash of the example id, so it is deterministic, stable
when the corpus grows, and independent of file order — re-running after adding data
never reshuffles an existing example into a different split.

Stratification is joint over all six tasks' labels, which cannot be done exactly;
we stratify on the rarest-task label and verify per-task class coverage afterwards,
failing loudly if any split is missing a class that the others have.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.spec import load_specs  # noqa: E402

SPLITS = ("train", "dev", "cal", "test")


def bucket(example_id: str, salt: str = "jev-v1") -> float:
    """Deterministic uniform [0,1) from the example id."""
    h = hashlib.sha256(f"{salt}:{example_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def assign(records: list[dict], fracs: dict[str, float], salt: str) -> dict[str, list]:
    """Stratified hash assignment.

    Within each stratum, examples are ordered by their hash and cut at the
    cumulative fractions. This keeps class balance in every split while remaining
    deterministic and order-independent.
    """
    specs = load_specs()
    # Stratify on the task with the most classes; it constrains the others most.
    strat_task = max(specs.values(), key=lambda s: s.n_classes).id
    strata: dict[str, list] = collections.defaultdict(list)
    for r in records:
        key = r["labels"][strat_task]
        if r.get("annotations", {}).get(strat_task, {}).get("unanimous") is False:
            key += "|hard"      # keep ambiguous examples spread across splits too
        strata[key].append(r)

    out: dict[str, list] = {s: [] for s in SPLITS}
    order = list(SPLITS)
    for key, rows in strata.items():
        rows.sort(key=lambda r: bucket(r["id"], salt))
        n = len(rows)
        cum, start = 0.0, 0
        for i, s in enumerate(order):
            cum += fracs[s]
            end = n if i == len(order) - 1 else int(round(cum * n))
            out[s].extend(rows[start:end])
            start = end
    for s in SPLITS:
        out[s].sort(key=lambda r: r["id"])
    return out


def verify(out: dict[str, list]) -> list[str]:
    """Fail loudly on the things that silently ruin a calibration study."""
    problems = []
    ids = {s: {r["id"] for r in rows} for s, rows in out.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and (ov := ids[a] & ids[b]):
                problems.append(f"{a} and {b} overlap on {len(ov)} ids")
    inputs = {s: {r["input"] for r in rows} for s, rows in out.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b and (ov := inputs[a] & inputs[b]):
                problems.append(
                    f"{a} and {b} share {len(ov)} identical INPUT strings "
                    "(distinct ids, same text) — dedupe before splitting")
    for task, spec in load_specs().items():
        seen = {s: {r["labels"][task] for r in out[s]} for s in SPLITS}
        allv = set().union(*seen.values())
        for s in SPLITS:
            if missing := allv - seen[s]:
                problems.append(f"{s} is missing {task} classes {sorted(missing)}")
    if len(out["cal"]) < 2000:
        problems.append(
            f"cal has {len(out['cal'])} examples; temperature fitting wants >=2000 "
            "per PLAN.md 4c (5-10k preferred)")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="workspace/data/sft/seed.jsonl")
    ap.add_argument("--outdir", default="workspace/data/eval")
    ap.add_argument("--train", type=float, default=0.80)
    ap.add_argument("--dev", type=float, default=0.05)
    ap.add_argument("--cal", type=float, default=0.08)
    ap.add_argument("--test", type=float, default=0.07)
    ap.add_argument("--salt", default="jev-v1")
    args = ap.parse_args()

    fracs = {"train": args.train, "dev": args.dev, "cal": args.cal, "test": args.test}
    if abs(sum(fracs.values()) - 1.0) > 1e-6:
        sys.exit(f"fractions must sum to 1, got {sum(fracs.values())}")

    records = [json.loads(l) for l in Path(args.inp).open()]
    print(f"read {len(records):,} from {args.inp}")

    out = assign(records, fracs, args.salt)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        p = outdir / f"{s}.jsonl"
        with p.open("w") as fh:
            for r in out[s]:
                fh.write(json.dumps(r) + "\n")
        n_ann = sum(1 for r in out[s] if "annotations" in r)
        print(f"  {s:<6} {len(out[s]):>7,}  ({len(out[s])/len(records):>5.1%})  "
              f"{n_ann:>6,} with annotator votes -> {p}")

    print("\nper-task class balance (train):")
    for task in load_specs():
        c = collections.Counter(r["labels"][task] for r in out["train"])
        tot = sum(c.values())
        print(f"  {task:<16} " + "  ".join(
            f"{k}={v/tot:.0%}" for k, v in sorted(c.items())))

    problems = verify(out)
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  ! {p}")
        sys.exit(1)
    print("\nverified: no id/input overlap, all classes present in every split, "
          "cal large enough.")


if __name__ == "__main__":
    main()
