# data/ — and an important negative result

## The headline finding

**The synthetic seed corpus cannot serve as a quality benchmark, by construction.**

Measured (`eval/reports/baseline_tfidf.json`): TF-IDF + logistic regression, fitted
on CPU in under 7 seconds per task, scores

| task | prior-only acc | TF-IDF acc | TF-IDF ECE (after T) |
|---|---|---|---|
| route_v1 | 0.316 | **1.000** | 0.000 |
| triage_v1 | 0.468 | **1.000** | 0.000 |
| gate_human_v1 | 0.513 | **1.000** | 0.000 |
| blocker_v1 | 0.489 | **1.000** | 0.000 |
| urgency_v1 | 0.305 | **0.995** | 0.002 |
| intent_v1 | 0.357 | 0.641 | 0.051 |

This is `PLAN.md` risk #3 firing immediately and completely. A 135M-parameter decoder
cannot beat 1.000, so **no result on this corpus can demonstrate that JEV is worth
building.**

### Why, and why it is not fixable here

Each topic has a handful of fixed body templates, and topic determines the label. So
*body template → label* is a lookup table with ~50 entries, and bag-of-ngrams learns
it exactly.

I tried to break it. `--hard` (now the default) adds a cross-topic distractor clause
to 55% of messages, demoted by a discourse marker, so both topics' vocabulary is
present and only sentence structure says which is the live ask:

> *"Not about the data this time — my last 5 days of records are gone — anyway.
> Morning, my role was changed and now half the pages 404. The entire team is
> blocked."* → `route_v1: account`, `blocker_v1: permissions`

Accuracy stayed at 1.000. The reason is structural: the primary body n-grams still
correlate with the label 100% of the time, while distractor n-grams correlate only
~1/14 of the time, so logistic regression simply down-weights the distractors. **Any
finite template grammar is a lookup table, and no amount of injected surface noise
changes that.** Making the generator "harder" produces fake difficulty, not a real
benchmark.

So the generator is kept honest about what it is, and `--hard` is retained only
because the text is more realistic, not because it is harder.

### The one instructive exception

`intent_v1` is the only task TF-IDF cannot solve (0.641), and it is the only label
that depends on a **three-way interaction** — frame × tone × topic — rather than on
topic alone. That is a genuine signal about where a language model earns its keep:
not on closed-set lookup, but on composition. If the real task distribution looks
like `route_v1`, ship logistic regression. If it looks like `intent_v1`, JEV has a
reason to exist.

### What this means for the project

1. **Real labeled data is not optional.** It is the only thing that can answer
   whether this project should exist. `PLAN.md` 4b's source ranking stands: mine
   production labels first, then distill, then pay humans for the hard slice.
2. Every report scored against this corpus is stamped `synthetic: true` with the
   caveat embedded in the JSON, and `eval/harness.py` prints it before and after
   the numbers. Do not strip it.
3. Run `make baselines` **before** any training run on real data. If TF-IDF matches
   JEV on accuracy *and* ECE there too, the honest move is to ship the linear model.

---

## Files

| file | what it does |
|---|---|
| `gen_seed.py` | compositional synthetic generator: latent facts → message + 6 labels, simulated annotators, OOD shift |
| `distill.py` | label real inputs with a teacher model under the exact production schema; collects the teacher's vote distribution for soft targets |
| `agreement.py` | Krippendorff α (ordinal-weighted where relevant), Fleiss κ, observed agreement, accuracy ceiling, soft targets |
| `splits.py` | hash-stable stratified train/dev/cal/test, with overlap and class-coverage verification |
| `prepare_cpt.py` | dedupe (exact + MinHash LSH), replay mix, pack, held-out domain/replay eval slices |

Real-label tooling lives in `tools/` and is the intended next step:

| file | what it does |
|---|---|
| `tools/annotate.py` | blind terminal labeling, ordered **lowest model confidence first** (active learning). Never shows the model's answer or another annotator's vote. One file per annotator. |
| `tools/merge_annotations.py` | combines annotator files into the record shape the rest of the pipeline already consumes. Has no majority-collapse path. |

```bash
make annotate ANNOTATOR=you      # then repeat with 2+ more annotators
make merge-annotations           # -> soft targets + Krippendorff alpha + ceilings
```

## The splits

`PLAN.md` 4c, with one addition — `cal` is verified disjoint from everything, by id
*and* by input string:

| split | frac | n | purpose |
|---|---|---|---|
| train | 80% | 96.0k | SFT |
| dev | 5% | 6.0k | early stopping on **ECE**, not accuracy |
| cal | 8% | 9.6k | **temperature fitting only** — never trained on, never early-stopped on |
| test | 7% | 8.4k | touched once per milestone |
| ood | — | 4.0k | separate file; deliberately shifted, measures whether confidence *drops* |

Assignment is `sha256(salt:id)`, so it is deterministic, order-independent, and
stable as the corpus grows — adding data never reshuffles an existing example into a
different split. `splits.py` exits non-zero if any split overlaps another or is
missing a class the others have.

## Soft targets

The thing `PLAN.md` 4b is most emphatic about. 6000 examples carry 3 independent
simulated annotators, and disagreement is **preserved**:

```json
{"route_v1": {"votes": ["technical","account","technical"],
              "soft": {"account": 0.333, "technical": 0.667}}}
```

Disagreement is concentrated on the latent combinations in `AMBIGUOUS` (login caused
by an outage, cancellation framed as a complaint, firm-but-not-urgent tone), so the
soft targets carry structure instead of uniform label noise. `train/sft.py` consumes
them as the target distribution `q` directly.

Measured agreement on the seed corpus — α 0.845–0.882, accuracy ceiling 94.9–96.2%.
**These annotators are simulated.** The numbers validate the agreement code against
a corpus whose true label distribution is known exactly; they are not human
agreement and must never be reported as such.

## Regenerating

```bash
make data
```

Equivalent to:

```bash
python data/gen_seed.py --n 120000     # ~9 s
python data/splits.py                  # stratified, verified
python data/agreement.py               # α, ceilings, soft targets
```

CPT data (optional — see `HARDWARE.md` revision 6 on why this phase is the first
thing to cut):

```bash
python data/prepare_cpt.py --target-tokens 50000000
```
