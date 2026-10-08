#!/usr/bin/env python3
"""Label unlabeled inputs with a stronger teacher model (PLAN.md 4b, source 2).

Teacher labels are the realistic bulk source once real *inputs* exist but labels do
not. Two things this does that a naive labeling script does not:

1. **Asks under the exact production schema.** The teacher is constrained by the
   same integer-enum `format` as JEV itself, so a teacher label is always a legal
   code and never needs parsing or repair.

2. **Collects the teacher's distribution, not just its argmax.** With
   `--samples N > 1` at non-zero temperature, repeated draws give a vote
   distribution that feeds `train/sft.py --soft-targets` exactly like human
   annotators do. Teacher uncertainty is a weaker signal than human disagreement,
   but it is far better than a one-hot label, and it is nearly free.

DATA EGRESS
-----------
`--teacher` is resolved against the local Ollama instance. Ollama cloud models
(names ending `-cloud`) send every input to a third party. This script refuses to
use one unless `--allow-cloud` is passed, and prints exactly what will leave the
machine first. Check your corpus for PII before you pass that flag.

LICENSING
---------
PLAN.md 4b: check the teacher's terms before training a model you will distribute
on its outputs. This is a real constraint, not a formality, and it varies by
provider. `--teacher` is recorded in the output manifest so provenance is auditable.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev.ollama import as_code  # noqa: E402
from jev.spec import get_spec, load_specs, render_prompt  # noqa: E402

TEACHER_PROMPT = """You are labeling support messages for a classification dataset.

Task: {question}

Options:
{options}

Answer with the single digit of the best option. No explanation.

Message:
{input}

Digit:"""


def build_prompt(task_id: str, text: str) -> str:
    spec = get_spec(task_id)
    lines = []
    for c in spec.codes:
        v = spec.values[c]
        b = spec.boundaries.get(v)
        lines.append(f"  {c} = {v}" + (f" ({b})" if isinstance(b, str) else ""))
    if spec.abstain_enabled:
        lines.append(f"  9 = cannot be decided from this message alone")
    return TEACHER_PROMPT.format(question=spec.question, input=text,
                                 options="\n".join(lines))


def ask(client: httpx.Client, teacher: str, task_id: str, text: str,
        samples: int, temperature: float) -> dict:
    spec = get_spec(task_id)
    votes: list[int] = []
    logprob_dist: dict[int, float] | None = None
    for i in range(samples):
        body = {
            "model": teacher,
            "prompt": build_prompt(task_id, text),
            "stream": False,
            "format": spec.format_payload(),
            "options": {
                "temperature": 0.0 if samples == 1 else temperature,
                "num_predict": 4, "seed": i,
            },
            "keep_alive": "30m",
        }
        if i == 0:
            body["logprobs"] = True
            body["top_logprobs"] = 20
        r = client.post("/api/generate", json=body)
        r.raise_for_status()
        d = r.json()
        code = as_code(d.get("response") or "")
        if code is None or code not in spec.legal_codes:
            continue
        votes.append(code)
        if i == 0 and (lp := d.get("logprobs")):
            idx = next((j for j, t in enumerate(lp)
                        if as_code(t["token"]) is not None), None)
            if idx is not None:
                cand = {}
                for c in lp[idx].get("top_logprobs") or []:
                    k = as_code(c["token"])
                    if k is not None and k in spec.legal_codes:
                        cand[k] = max(cand.get(k, -math.inf), c["logprob"])
                if cand:
                    mx = max(cand.values())
                    ex = {k: math.exp(v - mx) for k, v in cand.items()}
                    Z = sum(ex.values())
                    logprob_dist = {k: v / Z for k, v in ex.items()}
    if not votes:
        return {}
    counts = collections.Counter(votes)
    return {
        "votes": [spec.values[c] if c != 9 else None for c in votes],
        "codes": votes,
        "majority_code": counts.most_common(1)[0][0],
        "soft": {(spec.values[c] if c != 9 else "__abstain__"): n / len(votes)
                 for c, n in sorted(counts.items())},
        "teacher_logprob_dist": (
            {(spec.values[c] if c != 9 else "__abstain__"): round(p, 6)
             for c, p in sorted(logprob_dist.items())} if logprob_dist else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True,
                    help="jsonl with an `input` field (labels ignored if present)")
    ap.add_argument("--out", default="workspace/data/sft/distilled.jsonl")
    ap.add_argument("--teacher", default="gemma4:latest")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--samples", type=int, default=1,
                    help=">1 draws a teacher vote distribution for soft targets")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--allow-cloud", action="store_true",
                    help="required to use a *-cloud teacher; sends data off-machine")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.teacher.endswith("-cloud") or ":cloud" in args.teacher:
        if not args.allow_cloud:
            sys.exit(
                f"\nREFUSING: {args.teacher!r} is an Ollama cloud model. Every input "
                f"in\n  {args.inp}\nwould be sent to a third party for labeling.\n\n"
                "If that is intended, re-run with --allow-cloud. Check the corpus for "
                "PII first,\nand check the provider's terms on training from outputs "
                "(PLAN.md 4b).\n")
        print(f"\n*** {args.teacher} is a CLOUD model: inputs from {args.inp} "
              f"will be sent off-machine. ***\n")

    tasks = args.tasks or list(load_specs())
    rows = [json.loads(l) for l in Path(args.inp).open()]
    if args.limit:
        rows = rows[: args.limit]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if args.resume and outp.exists():
        done = {json.loads(l)["id"] for l in outp.open()}
        print(f"resuming: {len(done):,} already labeled")

    todo = [r for r in rows if r.get("id") not in done]
    total = len(todo) * len(tasks) * args.samples
    print(f"{len(todo):,} inputs x {len(tasks)} tasks x {args.samples} samples "
          f"= {total:,} teacher calls")

    t0 = time.perf_counter()
    n_calls = 0
    with httpx.Client(base_url=args.host, timeout=600.0) as client, \
            outp.open("a" if args.resume else "w") as fh:
        for i, r in enumerate(todo):
            labels: dict = {}
            for t in tasks:
                res = ask(client, args.teacher, t, r["input"],
                          args.samples, args.temperature)
                n_calls += args.samples
                if res:
                    labels[t] = res
            if not labels:
                continue
            fh.write(json.dumps({
                "id": r.get("id"),
                "input": r["input"],
                "labels": {t: (v["soft"] and max(v["soft"], key=v["soft"].get))
                           for t, v in labels.items()},
                "teacher": labels,
                "source": f"distilled:{args.teacher}",
            }) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                el = time.perf_counter() - t0
                rate = n_calls / el
                print(f"  {i+1:,}/{len(todo):,} inputs  {n_calls:,} calls  "
                      f"{rate:.1f} call/s  eta {(total-n_calls)/max(rate,1e-9)/60:.0f} min",
                      flush=True)

    (outp.parent / "distill_manifest.json").write_text(json.dumps({
        "teacher": args.teacher, "host": args.host, "tasks": tasks,
        "samples": args.samples, "temperature": args.temperature,
        "n_inputs": len(todo), "n_calls": n_calls,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "cloud": args.allow_cloud,
        "note": "Check the teacher's terms before distributing a model trained on these labels.",
    }, indent=2) + "\n")
    print(f"\nwrote {outp}")
    print("Teacher labels are source 2 of 3 in PLAN.md 4b. Reserve human effort for "
          "the hard slice\nwhere the teacher's own vote distribution is split.")


if __name__ == "__main__":
    main()
