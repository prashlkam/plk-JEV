#!/usr/bin/env python3
"""Spike 0.1 — do Ollama's `format` (grammar) and `logprobs` compose?

The load-bearing question for PLAN.md section 6. Three possible outcomes:

  GREEN  logprobs are renormalized over the grammar mask -> use them directly.
  AMBER  logprobs are reported PRE-mask -> must renormalize over the legal
         token set client-side, and that is only possible when every legal
         label appears in `top_logprobs`.
  RED    `format` suppresses logprobs -> Path B calibration needs rework.

Run with system python; uses stdlib only so it works before the venv exists.
"""

import json
import statistics
import sys
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3:0.6b"


def chat(messages, fmt=None, top_logprobs=5, num_predict=16, raw_prompt=None):
    body = {
        "model": MODEL,
        "logprobs": True,
        "top_logprobs": top_logprobs,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    if raw_prompt is not None:
        body["prompt"] = raw_prompt
        body["raw"] = True
        url = f"{OLLAMA}/api/generate"
    else:
        body["messages"] = messages
        url = f"{OLLAMA}/api/chat"
    if fmt is not None:
        body["format"] = fmt
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def content_of(d):
    return d.get("message", {}).get("content", d.get("response", ""))


def report(label, d):
    lp = d.get("logprobs")
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(f"  output   : {content_of(d)!r}")
    print(f"  logprobs : {'PRESENT' if lp else 'ABSENT'}  ({len(lp or [])} tokens)")
    print(f"  eval_count: {d.get('eval_count')}  "
          f"eval_ms: {round((d.get('eval_duration') or 0) / 1e6, 1)}")
    if not lp:
        return None
    print(f"  {'tok':<14} {'logprob':>9} {'p':>8}  in_top_k  top_k_sum")
    for t in lp:
        tops = t.get("top_logprobs") or []
        in_top = any(c["token"] == t["token"] for c in tops)
        ssum = sum(pow(2.718281828459045, c["logprob"]) for c in tops)
        print(f"  {t['token']!r:<14} {t['logprob']:>9.3f} "
              f"{pow(2.718281828459045, t['logprob']):>8.2e}  "
              f"{str(in_top):<8}  {ssum:>7.3f}")
    return lp


def verdict(lp):
    """PRE-mask iff the emitted token is sometimes absent from its own top_logprobs,
    or its logprob is wildly lower than the top candidate's."""
    if not lp:
        return "RED"
    missing = [t for t in lp if not any(
        c["token"] == t["token"] for c in (t.get("top_logprobs") or []))]
    return "AMBER (pre-mask)" if missing else "GREEN (post-mask or non-binding)"


print(f"model = {MODEL}")

# --- A: verbose JSON schema, the PLAN.md section 1 probe -------------------
d = chat(
    [{"role": "user", "content": "Is this spam? body: free crypto now"}],
    fmt={"type": "object", "properties": {"spam": {"type": "boolean"}},
         "required": ["spam"]},
)
lp_a = report("A. format=JSON object {spam: boolean}", d)
print(f"  --> VERDICT: {verdict(lp_a)}")

# --- B: same question, NO format. Where does the model actually want to go? -
d = chat([{"role": "user", "content": "Is this spam? body: free crypto now"}],
         num_predict=4)
lp_b = report("B. no format (unconstrained) — the model's natural distribution", d)
print("  --> if A's forced tokens differ from B's natural ones, the mask is BINDING")

# --- C: bare enum, compact encoding. Does the mask become non-binding when ---
#        the prompt is shaped so the label IS the natural continuation?
d = chat(
    [{"role": "user", "content":
      "Classify as spam. Answer with exactly one word, yes or no.\n"
      "body: free crypto now\nAnswer:"}],
    fmt={"type": "string", "enum": ["yes", "no"]},
    top_logprobs=20,
)
lp_c = report("C. format=enum[yes,no] + label-shaped prompt, top_logprobs=20", d)
print(f"  --> VERDICT: {verdict(lp_c)}")

# --- D: legal-set coverage. Can we even renormalize? ----------------------
#     For the decision token, are BOTH legal labels present in top_logprobs?
print(f"\n{'=' * 72}\nD. legal-set coverage in top_logprobs (can we renormalize?)\n{'=' * 72}")
LEGAL = {"yes", "no", "Yes", "No", " yes", " no"}
cov, rows = [], []
for body in ["free crypto now", "meeting moved to 3pm", "CLAIM YOUR PRIZE!!!",
             "quarterly report attached", "hot singles in your area"]:
    d = chat(
        [{"role": "user", "content":
          "Classify as spam. Answer with exactly one word, yes or no.\n"
          f"body: {body}\nAnswer:"}],
        fmt={"type": "string", "enum": ["yes", "no"]},
        top_logprobs=20,
    )
    lp = d.get("logprobs") or []
    if not lp:
        continue
    tok0 = lp[0]
    tops = tok0.get("top_logprobs") or []
    found = {c["token"].strip().lower() for c in tops} & {"yes", "no"}
    full = len(found) == 2
    cov.append(full)
    # renormalized probability over the legal set
    legal = {c["token"].strip().lower(): c["logprob"]
             for c in tops if c["token"].strip().lower() in {"yes", "no"}}
    if legal:
        mx = max(legal.values())
        ex = {k: pow(2.718281828459045, v - mx) for k, v in legal.items()}
        Z = sum(ex.values())
        renorm = {k: round(v / Z, 4) for k, v in ex.items()}
    else:
        renorm = {}
    leaked = 1.0 - sum(pow(2.718281828459045, v) for v in legal.values())
    rows.append((body[:28], content_of(d).strip(), sorted(found), renorm, leaked))
    print(f"  {body[:28]:<30} out={content_of(d).strip():<5} "
          f"legal_in_topk={sorted(found)} renorm={renorm} leaked_mass={leaked:.3f}")

print(f"\n  full legal-set coverage: {sum(cov)}/{len(cov)} requests")
if rows:
    print(f"  mean leaked mass (prob the mask had to discard): "
          f"{statistics.mean(r[4] for r in rows):.3f}")

# --- E: token cost, verbose JSON vs compact enum ---------------------------
print(f"\n{'=' * 72}\nE. decode-step cost: verbose JSON vs compact enum\n{'=' * 72}")
for name, fmt in [
    ("verbose JSON 3 fields", {
        "type": "object",
        "properties": {"route": {"type": "string", "enum": ["billing", "technical", "sales"]},
                       "severity": {"type": "integer"},
                       "needs_human": {"type": "boolean"}},
        "required": ["route", "severity", "needs_human"]}),
    ("compact enum 1 field", {"type": "string", "enum": ["b", "t", "s"]}),
]:
    d = chat([{"role": "user", "content":
               "card declined on renewal, third time. route it."}],
             fmt=fmt, num_predict=48)
    print(f"  {name:<24} eval_count={d.get('eval_count'):<4} "
          f"out={content_of(d)!r}")

print("\nDone. Record conclusions in spikes/RESULTS.md")
