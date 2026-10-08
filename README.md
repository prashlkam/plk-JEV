# JEV

A **System One model**: it makes one bounded, schema-constrained decision per call,
in one decode step, with a calibrated probability attached — and it runs on Ollama.

Inspired by [TypeSafe AI's Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev).
This is an independent open implementation of the three *observable* properties
(type safety, calibrated confidence, low latency), not a clone of their unpublished
architecture or RLCD recipe.

```python
class Triage(BaseModel):
    route_v1:      Optional[Literal["billing","technical","account","sales","abuse"]]
    triage_v1:     Optional[Literal["P0","P1","P2","P3"]]
    gate_human_v1: bool

r = Jev().decide_or_defer(Triage, "charged twice after the outage", threshold=0.9)
r.value       # Triage(route_v1='billing', triage_v1='P1', gate_human_v1=True)
r.confidence  # {'route_v1': 0.94, 'triage_v1': 0.71, 'gate_human_v1': 0.88}
r.deferred    # ['triage_v1']   <- below threshold; hand this one to a human
```

---

## Start here

| document | what it is |
|---|---|
| **[RESULTS.md](RESULTS.md)** | **what the first full run actually measured, including one failed gate** |
| [PLAN.md](PLAN.md) | the original 9-phase plan |
| [HARDWARE.md](HARDWARE.md) | **8 measured revisions to that plan** for a GTX 1650. Where they disagree, this wins |
| [spikes/RESULTS.md](spikes/RESULTS.md) | Phase 0 findings, including the one that changed the output encoding |
| [data/README.md](data/README.md) | the data pipeline, and an important **negative result** |
| [spec/tasks.md](spec/tasks.md) | the six tasks, their decision boundaries, and the non-goals |

```bash
make setup     # workspace venv + torch cu126 (verifies sm_75)
make data      # 120k-example corpus, splits, agreement        ~20 s
make baselines # prior + TF-IDF + multi-head encoder           ~10 m
make train-sft # Phase 5 decision SFT                        ~3.7 h
make calibrate && make serve && make calibrate-final
make eval && make smoke
```

`make help` lists every phase. `source env.sh` for an interactive shell.

---

## What Phase 0 changed

Three measurements reshaped the implementation. All reproducible via `make spikes`.

### 1. Ollama's logprobs are reported *before* the grammar mask

`format` and `logprobs` do compose — but with a JSON object schema, the forced `{"`
token comes back at **p = 4.9e-07**, and the model's actual preference (`Yes`,
p = 0.449) is what it would have emitted unconstrained.

Worse, forced scaffolding *corrupts the decision downstream*: with
`format: {"type":"string","enum":["yes","no"]}` the model answered `"no"` to all
five test inputs — including `CLAIM YOUR PRIZE!!!` — because the forced opening
quote (p = 7.8e-09) pushed it off-distribution before it reached the decision.

**Grammar-constrained decoding guarantees the output parses. It does not guarantee
the output is what the model would have chosen.**

The fix is to make the decision **token 0**, which JSON integer enums allow and
string enums do not:

```json
"format": {"type": "integer", "enum": [0, 1, 2, 3, 9]}
```

| encoding | decode steps | min p of an emitted token | logprobs usable |
|---|---|---|---|
| **integer enum → `1`** | **1** | **0.708** | **yes** |
| string enum → `"1"` | 3 | 7.8e-09 | no |
| integer array → `[3, 0, 0]` | 9 | 5.8e-05 | no |
| JSON object, 3 fields | 27 | — | no |

So multi-field decisions are **k single-field requests**, not one object call — 9×
cheaper *and* the only version whose probabilities mean anything. This reverses
PLAN.md §7, which assumed scaffolding was merely expensive rather than also
distribution-corrupting.

The template puts the input before the task tag so those k requests share the long
prefix; prompt-eval time drops ~40% on the follow-ups (38 ms → 22 ms). Ollama
exposes no cache-hit flag, though — `prompt_eval_count` reports the full prompt
length even on an identical repeated request — so that timing is the only
observable, and the SDK reports it rather than claiming a cache hit.

The residual risk gets its own gate metric: **`legal_mass`**, the pre-mask
probability on the legal set. At 0.99 the mask is non-binding and confidence is the
model's own. Below 0.95 the grammar is doing the deciding and the confidence is an
artifact of renormalizing noise — so the harness **fails the run** regardless of how
good ECE looks. The SFT loss is designed to drive this up (see below).

### 2. On this GPU, fp32 is the fastest dtype

A GTX 1650 (TU117) has no tensor cores and no fast FP16 path. Measured on a real
train step:

| dtype | tok/s | vs fp32 |
|---|---|---|
| **fp32** | **1818** | 1.00× |
| bf16 (emulated) | 1385 | 0.76× |
| fp16 AMP | 346 | **0.19×** |

Every standard recipe says "use bf16 or fp16". Here that costs **1.3–5.3×**.
bf16 is kept purely as a memory escape hatch (0.88 GiB vs 1.49).

### 3. Space-prefixed digits are never single tokens

` 1` is two tokens in both candidate tokenizers, so a prefix ending `answer:` would
leave the model wanting ` 1` and the grammar would force a bare `1` against its
preference — reintroducing problem 1. The prompt prefix therefore owns the trailing
space (`"answer: "`), and the harness asserts `decision_token_index == 0` on every
request so a template regression fails loudly.

---

## How it works

**Prompt** (rendered only by `jev.spec.render_prompt`, so the text is
byte-identical across the k requests of one input):

```
{input}
#{task_id}
answer: 
```

Input first, task tag last, so k requests for one input share the long prefix.

**The SFT loss** is one formulation doing three jobs. For each pair we build a
target distribution `q` supported *only on the legal digit tokens*, and minimize
cross-entropy against the model's **full-vocabulary** softmax:

```
loss = − Σ_v  q_v · log p_v
```

- **Hard labels** put `1−ε` on the gold digit and spread `ε` over the *other legal
  digits* — not over all 49k tokens, which is what `CrossEntropyLoss(label_smoothing=…)`
  would wrongly do.
- **Soft targets** from annotator disagreement drop in as `q` directly, preserving
  the uncertainty that was paid for (2-of-3 → 0.67, never a majority vote).
- The **full-vocab denominator** is what drives `legal_mass` → 1. The model learns to
  natively concentrate on legal digits, so the mask becomes non-binding and finding 1
  stops mattering. A softmax restricted to the legal subset would not do this.

**Calibration**: one temperature per task, fitted on a `cal` split that is never
trained on and never early-stopped on. Cannot change accuracy (monotone transform),
typically cuts ECE 3–5×.

**Early stopping is on dev ECE, not accuracy** — on these tasks accuracy improves
easily while calibration stays bad, so accuracy stops in the wrong place.

---

## Honest status

### Built and verified
Phases 0–8 are implemented and **a trained model is deployed and measured** — full
numbers in [RESULTS.md](RESULTS.md). 84 unit tests pass. Highlights:

| claim | measured |
|---|---|
| one decode step per decision | **2.00 tokens/decision** |
| type-safe by construction | **0 schema violations** in 1800 requests |
| the model decides, not the mask | **`legal_mass` 0.997–1.000** |
| calibration | **ECE 0.066 → 0.018 (3.6×)** |
| latency | **p50 22.9 ms** per field |

The design's central claim is confirmed: `legal_mass` rises from **0.82 on the
untrained base to 0.999 after 400 steps**, so the grammar mask becomes non-binding
and finding 1 is neutralized *by the training objective* rather than worked around.
Quantization also measurably decalibrated the model — `route_v1`'s temperature
flipped from 0.73 to 1.08 — confirming PLAN.md §7 and making `make calibrate-final`
mandatory rather than optional.

### One gate fails, and it is the important one
**Confidence does not fall when accuracy does off-distribution.** `urgency_v1` loses
44 points of accuracy on the OOD split while shedding only 10 points of confidence.
A pipeline auto-acting above 0.8 would keep acting while being wrong half the time.
`legal_mass` stays at 0.997+ on those same inputs, which confirms it measures
mask-bindingness and **not** input familiarity. See [RESULTS.md §2a](RESULTS.md) for
three candidate fixes.

### The limitation that matters
**The synthetic corpus cannot show whether JEV is worth building.** TF-IDF +
logistic regression scores **1.000 accuracy and 0.000 ECE on five of six tasks** in
under 7 seconds per task. That is PLAN.md risk #3 firing completely, and it is not
fixable by making the generator harder — a finite template grammar *is* a lookup
table. See [data/README.md](data/README.md) for the attempt and why it failed.

A 66M distilbert with six classification heads, trained in 3.8 minutes, then beat
JEV on **accuracy (0.939 vs 0.793), calibration (ECE 0.005 vs 0.060) and latency
(4.71 ms for all six fields vs ~137 ms)**. The accuracy columns are inflated by the
lookup-table corpus, but the ~29× latency gap is architectural and holds on any data:
one forward pass over six heads beats six sequential decode calls.

So the Ollama requirement costs roughly **29× latency** versus a model you serve
yourself. What it buys is distribution — `ollama pull`, no Python, no fork. A real
benefit, just not a performance one.

The one instructive exception: `intent_v1` is the only label depending on a
**three-way interaction** rather than on topic alone — and it is simultaneously the
only task TF-IDF cannot solve (0.641) and the one JEV is *worst* at (0.467). On real
data, `intent_v1`-shaped tasks are where this project lives or dies.

So: **real labeled data is the next step, and it is not optional.** Every report
scored against the synthetic corpus is stamped `synthetic: true` with the caveat
embedded in the JSON. Don't strip it, and don't quote those numbers as performance.

The tooling for that step is built and tested:

```bash
make annotate ANNOTATOR=you      # blind, lowest-model-confidence-first labeling
make annotate ANNOTATOR=someone  # same examples, same order, blind to your votes
make merge-annotations           # combine -> soft targets + Krippendorff alpha
```

`tools/annotate.py` never shows the annotator the model's prediction or another
annotator's vote, and has no code path that collapses votes to a majority —
disagreement survives all the way into the SFT loss as `q`.

### Not done
- **Phase 4 (CPT) is implemented but unrun.** HARDWARE.md revision 6 cut it from
  2B tokens to 50M (~8 h) and argues it is the first phase to drop for closed-set
  classification. Running it on synthetic data would prove nothing.
- **The Qwen3-0.6B QLoRA variant** has a config but has not been trained — it is a
  multi-day run on this card.
- **Phase 9 (true single-forward-pass Path A)** is deliberately not attempted.
  Ollama has no classification endpoint, so it needs a patched llama.cpp *and* a
  patched Ollama Go runner, maintained as a fork. But
  `eval/baselines/encoder.py` builds the architecture (shared trunk + 6 heads) and
  reports its single-forward-pass latency, so the trade-off is measurable before
  committing 4–8 weeks. PLAN.md §10's conclusion stands: serve those weights from a
  small FastAPI service instead of forking two upstream projects.

---

## Layout

```
spec/        task schemas (source of truth), prompt template, generated label vocab,
             fitted calibration.json
jev/         shared library: spec loading + prompt rendering, metrics, Ollama client
data/        generator, teacher distillation, agreement/soft targets, splits, CPT prep
tools/       annotate.py (blind, active-learning labeling), merge_annotations.py
train/       cpt.py, sft.py, calibrate.py, configs/
eval/        harness.py (all metrics + gates), baselines/{tfidf,encoder}.py, reports/
serve/       convert.sh (GGUF + Ollama), Modelfile.tmpl, jev_sdk/
spikes/      Phase 0 scripts and RESULTS.md
tests/       57 tests pinning the invariants the design rests on
workspace -> /var/tmp/jev-workspace   (venv, HF cache, data, checkpoints)
```

`/home` has 7.2 GiB free and the venv alone is 7.0 GiB, so everything heavy lives on
`/`. `/var/tmp` is swept after 30 days — install the exclusion:

```bash
sudo cp packaging/tmpfiles-jev.conf /etc/tmpfiles.d/jev-workspace.conf
```

## Shipping

The weights are **not usable alone**. A consumer with the GGUF and the wrong
temperatures gets plausible-looking, miscalibrated confidence — no error, just quietly
wrong numbers. `serve/convert.sh` bundles the GGUF with `schemas/`,
`label_vocab.json`, `template.json`, `calibration.json` and the `Modelfile` as one
versioned artifact.

That failure mode is real: the first run shipped a bundle carrying the *preliminary*
fp32 temperatures, because `convert.sh` necessarily copies them before
`calibrate-final` exists. So it is now checked, not remembered:

```bash
make verify-bundle
```

refuses to call a bundle shippable unless its calibration is `authoritative: true`,
matches `spec/` exactly, covers every task, and its template matches the one the
model was **trained** on. `make calibrate-final` refreshes the bundle and runs the
check automatically.
