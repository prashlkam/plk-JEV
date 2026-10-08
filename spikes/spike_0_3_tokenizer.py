#!/usr/bin/env python3
"""Spike 0.3 — single-token feasibility for label symbols, task tags and separators.

Spike 0.1 settled on bare-integer label encoding, so the labels themselves are
single digits. What still needs checking per candidate base model:

  * are digits 0-9 single tokens, and are they single tokens *in position 0*
    (no leading-space variant problem)?
  * is the task tag a small, fixed number of tokens? It sits in the cached
    prefix so its length costs nothing at decode time, but it must tokenize
    IDENTICALLY every call or the prefix cache misses.
  * does the prompt template end on a token boundary that does not merge with
    the first generated digit? (BPE merge across the boundary would corrupt
    position 0 — the exact thing spike 0.1 told us to protect.)
"""

import json
import sys

from transformers import AutoTokenizer

MODELS = sys.argv[1:] or [
    "HuggingFaceTB/SmolLM2-135M",
    "Qwen/Qwen3-0.6B-Base",
]

DIGITS = [str(i) for i in range(10)]
TAGS = ["<task:route_v1>", "task:route_v1", "[route_v1]", "route_v1"]
TERMINATORS = ["\nanswer:", "\nanswer: ", "\n>", "\ndecide:", "\n"]

out = {}
for name in MODELS:
    try:
        tok = AutoTokenizer.from_pretrained(name)
    except Exception as e:
        print(f"\n{name}: SKIP ({type(e).__name__}: {str(e)[:60]})")
        continue

    enc = lambda s: tok.encode(s, add_special_tokens=False)
    print(f"\n{'=' * 72}\n{name}   vocab={tok.vocab_size}\n{'=' * 72}")

    print("  digits as single tokens:")
    bad = [d for d in DIGITS if len(enc(d)) != 1]
    print(f"    bare   0-9 single-token: {'ALL' if not bad else f'FAILED {bad}'}")
    badsp = [d for d in DIGITS if len(enc(' ' + d)) != 1]
    print(f"    ' '+0-9 single-token: {'ALL' if not badsp else f'not single: {badsp}'}")

    print("  task tag token cost (lives in the cached prefix):")
    for t in TAGS:
        print(f"    {t!r:<22} {len(enc(t)):>2} tokens  {enc(t)}")

    print("  boundary safety — does the terminator merge with the first digit?")
    rows = []
    for term in TERMINATORS:
        base = enc(term)
        merged = []
        for d in DIGITS:
            joint = enc(term + d)
            # safe iff appending the digit only appends tokens, leaving the
            # terminator's own token ids untouched
            safe = joint[:len(base)] == base and len(joint) == len(base) + 1
            if not safe:
                merged.append(d)
        ok = not merged
        rows.append((term, ok, merged))
        print(f"    {term!r:<14} {'SAFE' if ok else f'MERGES on {merged}'}")

    safe_terms = [r[0] for r in rows if r[1]]
    print(f"  --> recommended terminator: {safe_terms[0]!r}" if safe_terms
          else "  --> WARNING: no safe terminator found")

    out[name] = {
        "vocab_size": tok.vocab_size,
        "digits_single_token": not bad,
        "space_digits_single_token": not badsp,
        "tag_costs": {t: len(enc(t)) for t in TAGS},
        "safe_terminators": safe_terms,
        "recommended_terminator": safe_terms[0] if safe_terms else None,
    }

json.dump(out, open("spikes/RESULTS_tokenizer.json", "w"), indent=2)
print("\nwrote spikes/RESULTS_tokenizer.json")
