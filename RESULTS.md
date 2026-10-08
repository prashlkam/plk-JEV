# RESULTS — first full run

Measured 2026-10-08 on the target machine (GTX 1650 4GB). Every number here is
reproducible with `make all`. Raw reports in `eval/reports/`.

> **SYNTHETIC CORPUS.** Labels are a deterministic function of latent facts in
> `data/gen_seed.py`. Accuracy figures below measure that the pipeline works, **not
> that the model works**. See [data/README.md](data/README.md).

The model: SmolLM2-135M, full fine-tune, fp32, **stopped at step 400 of 1200**
(deliberately — the pipeline mattered more than the last 800 steps on a corpus a
linear model already solves). 107 MB as Q5_K_M.

---

## 1. The design works

Every mechanism the project is built on was verified end to end.

| claim | measured | status |
|---|---|---|
| one decode step per decision | **2.00 tokens/decision** (1 digit + EOS) | ✅ |
| type-safe by construction | **0 schema violations** in 1800 requests | ✅ |
| decision always at token 0 | **0 violations** (spike 0.3 gate) | ✅ |
| the model decides, not the mask | **`legal_mass` 0.997–1.000** | ✅ |
| legal set visible in top-20 | **`legal_coverage` 1.00 on every request** | ✅ |
| calibration improves ECE | **0.066 → 0.018, 3.6×** (fp32, n=3000) | ✅ |
| latency | **p50 22.9 ms**, p90 42.7, p99 64.7 per field | ✅ |

### `legal_mass` — the spike 0.1 problem is solved by training

Spike 0.1's AMBER finding was that Ollama's logprobs are pre-grammar-mask. The SFT
loss was designed to make that irrelevant by driving the model to natively
concentrate on legal digits. It worked:

| | legal_mass |
|---|---|
| untrained base | 0.82 |
| step 200 | 0.9961 |
| step 400 | 0.9988 |
| deployed Q5_K_M GGUF | **0.997–1.000** |

At 0.999 the grammar mask is non-binding, so the pre-mask logprob *is* the model's
own belief and renormalization is a near no-op. The workaround became unnecessary
because the training objective removed the cause.

### Quantization really does decalibrate — PLAN.md §7 confirmed

Temperatures refitted against the quantized GGUF differ materially from those fitted
on the fp32 checkpoint:

| task | T (fp32) | T (Q5_K_M) | |
|---|---|---|---|
| route_v1 | 0.731 | **1.076** | flipped under- → over-confident |
| intent_v1 | 0.930 | **1.149** | flipped |
| triage_v1 | 0.752 | 0.860 | +14% |
| blocker_v1 | 0.693 | 0.746 | |
| urgency_v1 | 0.588 | 0.643 | |
| gate_human_v1 | 0.462 | 0.522 | |

Shipping the fp32 temperatures would have made `route_v1` and `intent_v1`
confidences wrong **in the wrong direction**. `make calibrate-final` exists for
exactly this, and it is not optional.

**And this bit me during the run.** `serve/convert.sh` necessarily copies
`spec/calibration.json` into the shipping bundle *before* `calibrate-final` has
produced the authoritative version, so the first bundle went out with the fp32
temperatures while `spec/` held the correct ones — silently, with no error. That is
precisely the "plausible-looking, miscalibrated confidence" failure PLAN.md §8
describes, caught only because the bundle was inspected by hand.

The fix is a check rather than a reminder: `serve/verify_bundle.py` refuses to call
a bundle shippable unless its calibration is `authoritative: true`, matches `spec/`
exactly, covers every task, and its template matches the one the model was *trained*
on. `make calibrate-final` now refreshes the bundle and runs it; `make verify-bundle`
runs it standalone. On the stale bundle it reported all three faults with the exact
per-task deltas.

Side note: every temperature is < 1 on the fp32 model, i.e. the model is
*under*-confident. That is the expected consequence of label smoothing (ε=0.05)
plus soft targets, and temperature scaling sharpens it back.

---

## 2. Two findings that matter more than the metrics

### 2a. The OOD gate FAILS — confidence does not fall when accuracy does

This is the most actionable result in the run.

| task | accuracy ID → OOD | confidence ID → OOD |
|---|---|---|
| urgency_v1 | 0.918 → 0.479 (**−0.439**) | 0.914 → 0.812 (−0.102) |
| gate_human_v1 | 0.887 → 0.613 (−0.273) | 0.874 → 0.854 (−0.020) |
| triage_v1 | 0.830 → 0.567 (−0.263) | 0.832 → 0.744 (−0.087) |
| blocker_v1 | 0.820 → 0.580 (−0.240) | 0.877 → 0.831 (−0.046) |
| route_v1 | 0.837 → 0.607 (−0.230) | 0.809 → 0.762 (−0.047) |
| intent_v1 | 0.467 → 0.387 (−0.080) | 0.478 → **0.493 (+0.015)** |

`urgency_v1` loses 44 points of accuracy while shedding 10 points of confidence.
A pipeline auto-acting above 0.8 would keep acting on nearly all off-distribution
inputs while being wrong half the time — **worse than useless**, which is precisely
why PLAN.md §4c demanded this split.

In-distribution the confidences are good (AURC 0.017–0.059, selective accuracy at
50% coverage reaches 0.973–0.987). The failure is specifically a *distribution-shift*
failure: temperature scaling fits one scalar on in-distribution data and has no
mechanism to detect that an input is unlike anything it was fitted on.

Fixes worth trying, in order of cost:
1. **Train on shifted data.** Fold OOD-style augmentations (the `to_ood` transforms)
   into training so the shift is in-distribution. Cheapest, and likely sufficient
   for the surface-form shifts measured here.
2. **An abstain head with OOD supervision.** The `9` code exists in every schema but
   is never trained — there are no abstain labels in the corpus. Give it some.
3. **Density/energy-based OOD scoring** at serve time, gating on input novelty
   rather than output confidence. `legal_mass` is *not* that signal — it stayed at
   0.997+ on OOD inputs, confirming it measures mask-bindingness, not familiarity.

### 2b. A 66M encoder beats JEV on accuracy, calibration AND latency

| run | macro acc | macro ECE | latency | params |
|---|---|---|---|---|
| **jev** (SmolLM2-135M, Ollama) | 0.793 | 0.060 | 22.9 ms **per field** | 135M |
| encoder (distilbert, 6 heads) | **0.939** | **0.005** | **4.71 ms for all 6 fields** | 66M |
| tfidf + logreg | **0.939** | 0.009 | CPU, ms | — |
| prior only (floor) | 0.408 | ~0.004 | — | — |

The encoder trained in **3.8 minutes** and is ~29× faster per complete decision
(4.71 ms for six fields vs ~137 ms for six sequential calls).

`eval/compare.py` renders the verdict without hedging: *"a decoder LLM is the wrong
tool for these tasks; shipping the baseline is the honest call."*

**Two caveats, both important.** The accuracy and ECE columns are inflated by the
synthetic corpus being a lookup table — the encoder's 1.000s are the same artifact
as TF-IDF's, and this comparison would need real data to mean anything. But the
**latency comparison is architectural and holds regardless of data**: one forward
pass over six heads will always beat six sequential decode calls.

So the honest read of the Ollama requirement: it costs roughly **29× latency** versus
a model you serve yourself. What it buys is distribution — `ollama pull` on any
machine, no Python, no serving code, no fork. That is a real benefit, just not a
performance one, and it should be chosen deliberately rather than by default.

`eval/baselines/encoder.py` *is* PLAN.md §10 step 1, so those weights already exist
at `workspace/out/baseline-encoder/model.pt` if the FastAPI route is preferred.

---

## 3. Per-task detail (test split, n=300)

| task | acc | mF1 | ECE | Brier | AURC | legal_mass | p50 | ceiling |
|---|---|---|---|---|---|---|---|---|
| urgency_v1 | 0.918 | 0.914 | 0.051 | 0.125 | 0.017 | 0.998 | 22.8 ms | 0.957 |
| gate_human_v1 | 0.887 | 0.886 | 0.042 | 0.167 | 0.034 | 0.999 | 22.9 ms | 0.962 |
| route_v1 | 0.837 | 0.743 | 0.052 | 0.234 | 0.039 | 0.999 | 16.9 ms | 0.949 |
| triage_v1 | 0.830 | 0.785 | 0.051 | 0.238 | 0.045 | 0.999 | 22.8 ms | 0.958 |
| blocker_v1 | 0.820 | 0.679 | 0.097 | 0.289 | 0.059 | 1.000 | 43.8 ms | 0.956 |
| intent_v1 | 0.467 | 0.430 | 0.065 | 0.661 | 0.404 | 0.997 | 22.9 ms | 0.953 |
| **macro** | **0.793** | | **0.060** | 0.286 | | 0.999 | | |

No task exceeds its annotator ceiling, so nothing here is fitting the labeler.

**`intent_v1` is the outlier worth staring at.** It is the only task whose label
depends on a three-way interaction (frame × tone × topic) rather than on topic
alone — and it is simultaneously the only task TF-IDF cannot solve (0.641) and the
one JEV is worst at (0.467). JEV loses to a linear model on the single genuinely
compositional task, which is the opposite of what the architecture is supposed to
buy. On real data, `intent_v1`-shaped tasks are where this project lives or dies.

### Selective accuracy — the metric for production use

| task | @50% coverage | @80% | @95% | @100% |
|---|---|---|---|---|
| route_v1 | **0.987** | 0.921 | 0.863 | 0.837 |
| urgency_v1 | 0.986 | 0.970 | 0.939 | 0.918 |
| gate_human_v1 | 0.980 | 0.946 | 0.902 | 0.887 |
| triage_v1 | 0.980 | 0.908 | 0.842 | 0.830 |
| blocker_v1 | 0.973 | 0.867 | 0.835 | 0.820 |
| intent_v1 | 0.567 | 0.521 | 0.481 | 0.467 |

Acting only on the most-confident half takes `route_v1` from 0.837 to 0.987. That is
`decide_or_defer` working as designed — and the reason calibration, not accuracy, is
the thing to optimize. `intent_v1` barely improves with coverage, consistent with its
AURC of 0.404: its confidences do not rank correctness.

---

## 4. Live SDK output

```
in   : I was charged $49 twice this month. Please advise.
out  : route_v1='billing' triage_v1=None gate_human_v1=True urgency_v1='2'
conf : route_v1=0.949  triage_v1=0.562  gate_human_v1=0.809  urgency_v1=0.994
meta : 139 ms total   trustworthy=True   deferred=['triage_v1','gate_human_v1']
       route_v1   code=0 mass=1.000 cov=1.00 T=1.08 tok_idx=0 55ms
       triage_v1  code=3 mass=0.999 cov=1.00 T=0.86 tok_idx=0 28ms
  !  gate_human_v1: confidence 0.809 < 0.9 and abstain is disabled; failed closed
     to 'true' rather than deferring
```

Note the deferral earning its keep: on `THE WHOLE DASHBOARD IS RETURNING 503`,
`triage_v1` returned P3 — wrong, it is a P0 — at confidence 0.573, and was deferred
rather than acted on. The safety gate meanwhile fired at 0.994.

---

## 5. What to do next

1. **Get real labels.** Nothing above distinguishes "the pipeline works" from "the
   model works", and it cannot. `make annotate` / `make merge-annotations` are built
   and tested; `data/distill.py` does teacher labeling under the production schema.
2. **Re-run `make baselines` on that data, before training.** If TF-IDF still ties
   JEV, ship TF-IDF.
3. **Fix the OOD failure** (§2a) — it is the one outright gate failure and the most
   likely source of production harm.
4. **Finish the training run.** This model saw 400 of 1200 steps; `make train-sft`
   runs the full thing in ~3.7 h.
5. **Decide the Ollama question deliberately.** §2b quantifies the cost at ~29×
   latency. If portability is not actually required, the multi-head encoder weights
   already exist.
