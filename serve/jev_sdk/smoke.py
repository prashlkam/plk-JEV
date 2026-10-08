#!/usr/bin/env python3
"""End-to-end smoke check against a deployed JEV model.

Exercises the whole serving path the way a caller would: a Pydantic model mapped
to schemas, k prefix-cached requests, renormalized probabilities, temperature
applied, and the deferral path. Prints the diagnostics that matter rather than
just the happy answer.
"""
import argparse
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pydantic import BaseModel  # noqa: E402

from serve.jev_sdk import Jev  # noqa: E402


class Triage(BaseModel):
    route_v1: Optional[Literal["billing", "technical", "account", "sales", "abuse"]]
    triage_v1: Optional[Literal["P0", "P1", "P2", "P3"]]
    gate_human_v1: bool
    urgency_v1: Optional[Literal["1", "2", "3", "4", "5"]]


SAMPLES = [
    "I was charged $49 twice this month. Please advise.",
    "THE WHOLE DASHBOARD IS RETURNING 503. Production is down. Fix this or we're leaving.",
    "just wanted to say the new editor is excellent, thank you",
    "can you send a quote for 40 seats with SSO",
    "zzzz qqq asdf",  # nonsense: confidence SHOULD be low
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--threshold", type=float, default=0.9)
    args = ap.parse_args()

    with Jev(model=args.model, host=args.host) as jev:
        h = jev.health()
        if not h["model_present"]:
            sys.exit(f"model {args.model!r} not registered. Have: {h['models']}\n"
                     f"Run: bash serve/convert.sh")
        print(f"model {args.model} OK at {args.host}\n")

        for text in SAMPLES:
            r = jev.decide_or_defer(Triage, text, threshold=args.threshold)
            print("=" * 74)
            print(f"in   : {text[:68]}")
            print(f"out  : {r.value}")
            print(f"conf : " + "  ".join(f"{k}={v:.3f}" for k, v in r.confidence.items()))
            print(f"meta : {r.latency_ms:.0f} ms total"
                  f"   trustworthy={r.trustworthy}"
                  f"   deferred={r.deferred or '-'}")
            for name, d in r.raw.items():
                print(f"       {name:<16} code={d.code} mass={d.legal_mass:.3f} "
                      f"cov={d.legal_coverage:.2f} T={d.temperature:.2f} "
                      f"tok_idx={d.decision_token_index} "
                      f"{d.latency_ms:.0f}ms prompt_eval={d.prompt_eval_ms:.0f}ms")
            for w in dict.fromkeys(r.warnings):
                print(f"  !  {w}")
            print()


if __name__ == "__main__":
    main()
