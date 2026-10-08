#!/usr/bin/env python3
"""Merge per-annotator files into the shape data/agreement.py consumes.

One file per annotator is the point: `tools/annotate.py` never lets an annotator
see another's vote, and this is the only place the votes meet. It still does not
collapse them — the output carries the full vote list per (example, task), and
`data/agreement.py` turns that into soft targets.

Emits the same record shape as the synthetic generator so everything downstream
(`splits.py`, `agreement.py`, `train/sft.py --soft-targets`) works unchanged on
real labels.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.spec import load_specs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="workspace/data/sft/annot_*.jsonl")
    ap.add_argument("--out", default="workspace/data/sft/annotated.jsonl")
    ap.add_argument("--min-annotators", type=int, default=2,
                    help="drop (example, task) pairs with fewer votes than this; "
                         "PLAN.md 4b wants 3+ for the agreement subset")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    if not files:
        sys.exit(f"no files matched {args.glob!r}; run tools/annotate.py first")
    specs = load_specs()

    # id -> {"input":..., task -> {annotator: value}}
    merged: dict[str, dict] = {}
    per_annotator = collections.Counter()
    for f in files:
        for line in Path(f).open():
            d = json.loads(line)
            who = d["annotator"]
            e = merged.setdefault(d["id"], {"input": d["input"], "tasks": {}})
            for t, v in d.get("labels", {}).items():
                if t not in specs:
                    continue
                # Last write wins per annotator (a re-run corrects a mistake);
                # different annotators never overwrite each other.
                e["tasks"].setdefault(t, {})[who] = v
                per_annotator[who] += 1

    print(f"{len(files)} annotator file(s): " +
          ", ".join(f"{k}={v}" for k, v in sorted(per_annotator.items())))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rec, n_pairs, dropped = 0, 0, 0
    counts = collections.Counter()
    with out.open("w") as fh:
        for ex_id, e in sorted(merged.items()):
            annotations, labels = {}, {}
            for t, votes_by_who in e["tasks"].items():
                votes = list(votes_by_who.values())
                if len(votes) < args.min_annotators:
                    dropped += 1
                    continue
                c = collections.Counter(votes)
                annotations[t] = {
                    # No `gold` key: with real annotators there is no privileged
                    # answer, only the vote distribution. agreement.py treats a
                    # missing gold as "use the modal vote for the ceiling".
                    "votes": votes,
                    "annotators": list(votes_by_who),
                    "soft": {k: round(v / len(votes), 6)
                             for k, v in sorted(c.items())},
                    "majority": max(c, key=lambda k: (c[k], k)),
                    "unanimous": len(c) == 1,
                }
                labels[t] = annotations[t]["majority"]
                counts[len(votes)] += 1
                n_pairs += 1
            if not annotations:
                continue
            fh.write(json.dumps({
                "id": ex_id, "input": e["input"], "labels": labels,
                "annotations": annotations, "source": "human",
            }) + "\n")
            n_rec += 1

    print(f"wrote {n_rec:,} examples / {n_pairs:,} (example, task) pairs -> {out}")
    if dropped:
        print(f"dropped {dropped:,} pairs with < {args.min_annotators} votes")
    print("raters per pair: " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    if counts and max(counts) < 3:
        print(f"\nWARNING: no pair has 3+ annotators. PLAN.md 4b wants 3+ on a "
              f">=2000-example\nsubset — with 2 raters you can measure agreement but "
              f"soft targets are coarse (0/0.5/1).")
    print(f"\nNext: python data/agreement.py --inp {out}")
    print("`labels` holds the majority for convenience, but train/sft.py should "
          "consume the\nsoft targets — that is the whole point of collecting "
          "multiple votes.")


if __name__ == "__main__":
    main()
