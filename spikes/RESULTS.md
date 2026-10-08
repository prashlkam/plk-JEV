# Phase 0 — Spike results

Measured 2026-10-08 on the target machine. Ollama 0.32.15, `qwen3:0.6b`, GTX 1650 4GB.
Reproduce with `python3 spikes/spike_0_1_format_logprobs.py` and
`python3 spikes/spike_0_2_latency.py`, `python3 spikes/spike_0_3_tokenizer.py`.

---

## Spike 0.1 — do `format` and `logprobs` compose?

**Verdict: AMBER — they compose, but logprobs are reported PRE-grammar-mask.**
This was branch 2 of the three predicted outcomes in PLAN.md §1. Path B survives, but
the output encoding had to change. That change is the most important result in Phase 0.

### Evidence that logprobs are pre-mask

With `format = {"type":"object","properties":{"spam":{"type":"boolean"}}}`:

| emitted token | logprob | p | in its own top_logprobs? |
|---|---|---|---|
| `{"` | −14.534 | 4.9e−07 | **no** |
| `spam` | −4.871 | 7.7e−03 | **no** |
| `":` | −0.051 | 0.950 | yes |
| ` true` | −0.342 | 0.710 | yes |
| `}` | −1.283 | 0.277 | yes |

The grammar forced `{"` at p≈5e−7. Unconstrained (same prompt, no `format`) the model
emits `Yes, the text` with p(`Yes`)=0.449. So the mask is strongly **binding** on
scaffolding tokens, and the reported logprob is the model's own pre-mask belief.

### The real problem this causes

Forced low-probability scaffolding **corrupts the decision token downstream**. With
`format = {"type":"string","enum":["yes","no"]}` the model emits `"no"` for all five
test inputs — including `CLAIM YOUR PRIZE!!!` and `hot singles in your area` — because
the forced opening `"` (p=7.8e−09) puts the model off-distribution before it ever
reaches the decision. Its *unconstrained* preference on those same inputs was `yes`.

Grammar-constrained decoding is not free. It guarantees the output *parses*; it does not
guarantee the output is what the model would have chosen.

### The fix: bare-integer enum encoding

Make the decision token **position 0**, so nothing is forced before it. JSON string
enums require quotes; JSON *integer* enums do not:

```json
"format": {"type": "integer", "enum": [0, 1]}
```

Measured, same five inputs, `top_logprobs=20`:

| input | out | decode steps | p(token 0) | legal mass | renormalized |
|---|---|---|---|---|---|
| free crypto now | 1 | 1 | 0.708 | 0.995 | {1: 0.711, 0: 0.289} |
| meeting moved to 3pm | 0 | 1 | 0.573 | 0.995 | {0: 0.576, 1: 0.424} |
| CLAIM YOUR PRIZE!!! | 1 | 1 | 0.595 | 0.987 | {1: 0.602, 0: 0.398} |
| quarterly report attached | 1 | 1 | 0.570 | 0.995 | {1: 0.573, 0: 0.427} |
| hot singles in your area | 1 | 1 | 0.570 | 0.993 | {1: 0.574, 0: 0.426} |

Three things fall out at once:

1. **One decode step per decision.** No scaffolding.
2. **Logprobs are trustworthy** — p(token 0) = 0.57–0.71, not 1e−9. Nothing was forced.
3. **Legal mass 0.987–0.995.** The model already puts ~99% of its *unmasked* mass on the
   legal set, so the mask is effectively non-binding and renormalization is a near
   no-op (0.708 → 0.711). Type safety is retained for free.

### Encoding comparison (measured)

| encoding | decode steps | min p of an emitted token | logprobs usable |
|---|---|---|---|
| **integer enum `1`** | **1** | **0.708** | **yes** |
| string enum `"1"` | 3 | 7.8e−09 | no (position 0 forced) |
| integer array `[3, 0, 0]` | 9 | 5.8e−05 | no (`[` forced, interiors 0.1–0.28) |
| JSON object, 3 fields | 27 | — | no |

**Multi-field decisions must therefore be k prefix-cached single-field requests, not one
array/object call.** The array call is 9× the decode cost *and* returns corrupted
probabilities. This reverses the "compact JSON" suggestion in PLAN.md §7, which assumed
scaffolding was merely expensive rather than also distribution-corrupting.

### Consequences for the rest of the build

- `spec/label_vocab.json` maps each enum value to a **small integer**, not a word.
- The SDK issues one request per field against a byte-identical cached prefix, and
  renormalizes over the legal integer set before applying temperature.
- `legal_mass` becomes a **gate metric** in the eval harness (§7). If a trained model
  ever drops below ~0.95 legal mass, its confidences are not trustworthy regardless of
  how good its ECE looks, because the mask is doing the work instead of the model.
- The "stock base + format" baseline must be reported with a caveat: its probabilities
  are mask-corrupted on any encoding with scaffolding, so compare only on integer enums.

### Two API limits found while wiring the SDK (added after the serving path ran)

**`top_logprobs` is capped at 20.** Above that Ollama returns HTTP 400
(`"top_logprobs must be between 0 and 20"`). `blocker_v1` and `intent_v1` have 7
legal codes, so 20 candidates is adequate but not purchasable — there is no headroom
to buy. Whether 20 sufficed is therefore a *measurement*: `Decision.legal_coverage`
reports it per request and the harness aggregates it. Observed coverage on the
deployed model: **1.00 on every request**.

**There is no cache-hit signal.** `prompt_eval_count` reports the full prompt length
on every call — measured 55 tokens on four *identical* consecutive requests, and 19
for a shorter input — so it cannot be used to detect prefix reuse. An earlier
`prompt_cached` boolean derived from it was unfalsifiable and has been removed. The
only observable is `prompt_eval_duration`, which falls ~40% (38 ms → 22 ms) on the
follow-up tasks of one input. The input-first template ordering is retained because
it is free and is what makes reuse possible at all, not because a cache hit was
confirmed.

### Secondary finding — stock 0.6B is nearly uninformative here

The five renormalized confidences span 0.573–0.711, and two of five are wrong. A model
whose confidence barely moves between `quarterly report attached` and `free crypto now`
has almost no resolution. This is the baseline Phase 5 has to beat, and it sets a low
bar — but it also means **accuracy alone will look easy to improve while calibration
stays bad**. Hence early-stopping on ECE, not accuracy.

---

## Spike 0.2 — throughput floor (`spike_0_2_trainstep.py`, raw JSON alongside)

Scope widened from the planned inference-latency probe to a **real train step**, because
the dtype question turned out to dominate everything. Full numbers and all eight
revisions they force are in [HARDWARE.md](../HARDWARE.md). Headline:

**fp16 AMP is 5.3× slower than fp32 on this GPU** (346 vs 1818 tok/s). TU117 has no
tensor cores and no fast FP16 path. bf16 is emulated: 1385 tok/s, but roughly half the
memory. Every standard fine-tuning recipe says "use bf16 or fp16"; here that advice costs
a 1.7–5.3× slowdown.

| config (SmolLM2-135M, seq 192, AdamW8bit, sliced LM head) | tok/s | peak VRAM |
|---|---|---|
| fp32, no ckpt, b8 | 2194 | 2.82 GiB |
| **fp32, ckpt, b16** ← default | **1767** | **1.49 GiB** |
| fp32, ckpt, b8 | 1818 | 1.49 GiB |
| fp32, ckpt, b8, **full** logits | 1522 | 1.72 GiB |
| bf16, ckpt, b8 | 1385 | 0.88 GiB |
| fp16 AMP, ckpt, b8 | 346 | 1.39 GiB |

Also measured: a resident Ollama model costs **964 MiB**, so training and serving cannot
share the card. And from spike 0.1's encoding table, decode cost per decision is
**1 step** for integer enums vs 27 for a 3-field JSON object.

At 1767 tok/s: 100k-example SFT ≈ **2.0 h**, 50M-token CPT ≈ **7.9 h**. PLAN.md §4a's
0.5–2B-token CPT would be 8–30 days and is cut to 50M.

---

## Spike 0.3 — single-token feasibility (`spike_0_3_tokenizer.py`)

**Green, with one trap found.**

| | SmolLM2-135M (49152) | Qwen3-0.6B-Base (151643) |
|---|---|---|
| bare `0`–`9` single-token | **all 10** | **all 10** |
| `" 0"`–`" 9"` single-token | **none** | **none** |
| `route_v1` tag cost | 4 tokens | 3 tokens |
| `<task:route_v1>` tag cost | 8 tokens | 7 tokens |
| terminator merges with digit | none of 5 tested | none of 5 tested |

The trap: **space-prefixed digits are never single tokens** in either tokenizer (both
split digits individually). After a prefix ending `answer:` the model's natural
continuation is ` 1`, which is two tokens — so the grammar would be forced to pick the
bare `1`, reintroducing exactly the position-0 corruption spike 0.1 told us to avoid.

**Fix:** the prompt prefix owns the trailing space — terminate with `"\nanswer: "`, so
the first generated token is an unambiguous bare digit. Verified non-merging. The eval
harness asserts `decision_token_index == 0` on every request so a template regression
cannot silently pass.

Task tags are cheap and, more importantly, tokenize identically every call — they live in
the shared prefix, so their length costs nothing at decode time, but any prefix reuse
depends on byte-identical rendering. `jev/spec.py` renders them from a single template.

---

## Gate decision

| Spike | Result | Action |
|---|---|---|
| 0.1 | **AMBER** | Proceed with integer-enum encoding + client-side renormalization over the legal set; add `legal_mass` as a gate metric |
| 0.2 | **GREEN** | fp32 + sliced LM head + checkpointing @ b16; CPT rescoped 2B → 50M tokens |
| 0.3 | **GREEN** | Bare-digit labels; prefix owns the trailing space; assert decision index 0 |

**Proceeding to Phase 1.** PLAN.md §6–§7 are superseded by the encoding decision and
§3–§5 by [HARDWARE.md](../HARDWARE.md).

**Proceeding to Phase 1.** PLAN.md §6 and §7 are superseded by the encoding decision
above; `HARDWARE.md` records that plus the GTX 1650 revisions.
