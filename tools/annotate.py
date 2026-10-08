#!/usr/bin/env python3
"""Blind terminal annotation tool for the hard slice.

PLAN.md 4b: human effort belongs on the ambiguous, adversarial and near-boundary
cases, not on easy examples. This tool exists to make that practical, and to
enforce the three properties that make the resulting labels worth anything:

  1. **Blind.** The annotator never sees the model's prediction, the generator's
     latent label, or another annotator's vote. Showing any of those produces
     agreement numbers that are measuring anchoring, not the task.
  2. **Disagreement preserved.** Each annotator writes their own file. Nothing is
     merged here; `data/agreement.py` combines them into soft targets. There is no
     code path in this tool that collapses votes to a majority.
  3. **Ordered by need.** `--mode hard` prioritizes examples the model is least
     certain about (active learning), which is where a human hour buys the most.

Usage:

    # annotate the 500 examples JEV is least sure about, as annotator "pk"
    python tools/annotate.py --annotator pk --mode hard --limit 500

    # second annotator, same examples, in the same order, blind to the first
    python tools/annotate.py --annotator sam --mode hard --limit 500

    # then build soft targets from all annotator files
    python data/agreement.py --inp workspace/data/sft/annotated.jsonl

Resumable: re-running skips anything that annotator already labeled.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.spec import get_spec, load_specs  # noqa: E402


def order_examples(records, mode, model, host, tasks, seed):
    """Decide what a human should look at first."""
    if mode == "random":
        rng = random.Random(seed)
        rows = list(records)
        rng.shuffle(rows)
        return rows
    if mode == "sequential":
        return list(records)

    # mode == "hard": lowest model confidence first, i.e. active learning.
    from jev.ollama import JevOllama
    print(f"scoring {len(records)} examples with {model} to rank by uncertainty...")
    scored = []
    with JevOllama(model=model, host=host) as cli:
        for i, r in enumerate(records):
            worst = 1.0
            for t in tasks:
                try:
                    d = cli.decide(t, r["input"])
                    worst = min(worst, d.confidence)
                except Exception:
                    worst = 0.0
            scored.append((worst, r))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(records)}", flush=True)
    scored.sort(key=lambda x: x[0])
    print(f"  ranked; least-confident example sits at {scored[0][0]:.3f}\n")
    return [r for _, r in scored]


def prompt_one(spec, text, idx, total, done_tasks):
    print("\n" + "=" * 74)
    print(f"[{idx}/{total}]  task: {spec.id}   ({spec.kind})")
    print("=" * 74)
    print(text)
    print("-" * 74)
    print(f"Q: {spec.question}")
    for c in spec.codes:
        v = spec.values[c]
        b = spec.boundaries.get(v)
        print(f"   {c}) {v}" + (f"  — {b}" if isinstance(b, str) else ""))
    if spec.abstain_enabled:
        print(f"   9) CANNOT DECIDE from this message alone")
    else:
        print("   (no abstain: this is a safety gate, choose the safer option)")
    print("   s) skip   q) save and quit")
    legal = {str(c) for c in spec.legal_codes} | {"s", "q"}
    while True:
        a = input("> ").strip().lower()
        if a in legal:
            return a
        print(f"   pick one of {sorted(legal)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotator", required=True,
                   help="your id; one output file per annotator, never merged here")
    ap.add_argument("--inp", default="workspace/data/eval/test.jsonl")
    ap.add_argument("--out", default=None,
                    help="default workspace/data/sft/annot_<annotator>.jsonl")
    ap.add_argument("--mode", default="hard",
                    choices=["hard", "random", "sequential"],
                    help="hard = lowest model confidence first (active learning)")
    ap.add_argument("--model", default="jev")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--pool", type=int, default=1500,
                    help="how many candidates to score before ranking (mode=hard)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    specs = load_specs()
    tasks = args.tasks or list(specs)
    out = Path(args.out or
               f"workspace/data/sft/annot_{args.annotator}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str]] = set()
    if out.exists():
        for line in out.open():
            d = json.loads(line)
            for t in d.get("labels", {}):
                done.add((d["id"], t))
        print(f"resuming: {len(done)} (example, task) pairs already labeled by "
              f"{args.annotator}")

    records = [json.loads(l) for l in Path(args.inp).open()][: args.pool]
    records = order_examples(records, args.mode, args.model, args.host, tasks,
                             args.seed)

    todo = []
    for r in records:
        for t in tasks:
            if (r["id"], t) not in done:
                todo.append((r, t))
        if len(todo) >= args.limit:
            break
    todo = todo[: args.limit]

    print(f"{len(todo)} judgements queued. Order is fixed by --mode/--seed so a "
          f"second annotator sees the same examples.\n"
          f"You will NOT see the model's answer or anyone else's — that is "
          f"deliberate (PLAN.md 4b).")
    input("press enter to start...")

    t0 = time.perf_counter()
    buf: dict[str, dict] = {}
    n = 0
    quit_early = False
    for i, (r, task_id) in enumerate(todo, 1):
        spec = get_spec(task_id)
        a = prompt_one(spec, r["input"], i, len(todo), None)
        if a == "q":
            quit_early = True
            break
        if a == "s":
            continue
        code = int(a)
        rec = buf.setdefault(r["id"], {"id": r["id"], "input": r["input"],
                                       "annotator": args.annotator, "labels": {}})
        rec["labels"][task_id] = spec.code_to_value(code)
        rec.setdefault("codes", {})[task_id] = code
        n += 1

    # Append-only: one annotator's file is never rewritten by another's run.
    with out.open("a") as fh:
        for rec in buf.values():
            rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
            fh.write(json.dumps(rec) + "\n")

    el = time.perf_counter() - t0
    print(f"\n{'quit early — ' if quit_early else ''}{n} judgements in "
          f"{el/60:.1f} min ({el/max(1,n):.1f} s each) -> {out}")
    print("\nNext: have at least 2 more annotators run the same command with their "
          "own --annotator id,\nthen merge and measure:")
    print(f"  python tools/merge_annotations.py --glob "
          f"'workspace/data/sft/annot_*.jsonl'")
    print(f"  python data/agreement.py --inp workspace/data/sft/annotated.jsonl")


if __name__ == "__main__":
    main()
