# JEV task specification (v1)

Six System One tasks over a single domain: **inbound software-support messages**.

One domain, six decisions. The domain is fixed deliberately — it keeps the CPT corpus
coherent, lets all six tasks share one prompt prefix per input (see
[Prompt template](#prompt-template)), and keeps the task count at the low end of
`PLAN.md` §2's 6–10 cap.

A task qualifies as System One here only if it is: **bounded output**, **decidable from
the input alone**, **no multi-step reasoning**, **no external lookup**.

---

## Non-goals — what JEV must refuse to be

The value proposition *is* the restriction. JEV does not:

- emit free text, prose, summaries, explanations, or rationales
- generate chain-of-thought (the output is one token; there is nowhere to put it)
- answer open questions, hold a conversation, or follow arbitrary instructions
- produce any value outside a declared enum — enforced by grammar, not by prompting
- make decisions requiring information absent from the input (account history, SLA
  tables, prior tickets). Those are System Two; JEV returns **abstain**.

If a request needs any of the above, it is out of scope by construction, and the correct
integration is to route it to a general LLM. JEV is the fast gate in front of that.

---

## Output encoding

Every task emits **exactly one bare digit**. From
[spikes/RESULTS.md](../spikes/RESULTS.md) §0.1, this is not a cosmetic choice:

- `format: {"type":"integer","enum":[...]}` emits a bare digit with **no JSON
  scaffolding**, so the decision is generated token **and the logprob is uncorrupted**.
- A string enum (`"1"`) forces an opening quote at p≈7.8e−09 and derails the decision.
- A JSON object costs 27 decode steps instead of 1.

`9` is reserved for **abstain** across all schemas, so digits `0`–`5` are always real
labels and `9` never collides.

---

## Prompt template

```
{input}
#{task_id}
answer: 
```

Three deliberate details:

1. **The input comes first, the task tag last**, so the k requests of a multi-field
   decision share the long input prefix rather than only the ~4-token tag.
   Measured effect: prompt-eval time drops from ~38 ms on the first task of an
   input to ~22 ms on the follow-ups. Note that Ollama exposes **no cache-hit
   flag** — `prompt_eval_count` reports the full prompt length even for an
   identical repeated request — so prompt-eval *time* is the only observable, and
   this ordering is the cheap choice rather than a confirmed cache hit.
2. **The terminator carries the trailing space** (`"answer: "`, not `"answer:"`).
   Spike 0.3: space-prefixed digits are never single tokens in either candidate
   tokenizer, so a prefix ending `answer:` would leave the model wanting ` 1` (two
   tokens) and the grammar would force the bare `1` — reintroducing the position-0
   corruption. With the space in the prefix, token 0 is an unambiguous digit.
3. **Rendered from one function** (`jev.spec.render_prompt`). Byte-identical rendering is
   what any prefix reuse depends on; a stray format change silently costs latency.

The eval harness asserts `decision_token_index == 0` on every request, so a template
regression fails loudly rather than quietly degrading confidence quality.

---

## The six tasks

### 1. `route_v1` — Route

Which team owns this message.

| code | value | boundary |
|---|---|---|
| 0 | `billing` | charges, invoices, refunds, payment methods, pricing disputes |
| 1 | `technical` | the product is broken or erroring for the user |
| 2 | `account` | login, access, permissions, profile, data export/deletion |
| 3 | `sales` | pre-purchase questions, quotes, upgrades, procurement |
| 4 | `abuse` | spam, ToS violations, harassment reported *about* another user |
| 9 | *abstain* | genuinely ambiguous, or needs account history to decide |

Edge cases that must be labeled consistently:
- "I was charged after I couldn't log in" → `billing` (the *ask* is the charge)
- "can't log in since the outage" → `technical` (cause is an outage, not access rights)
- "cancel my plan, it's too expensive" → `billing` (cancellation intent lives in `intent_v1`)

### 2. `triage_v1` — Triage (ordinal)

Severity. Ordinal, so calibration is reported with ordinal-aware distance too.

| code | value | boundary |
|---|---|---|
| 0 | `P0` | total outage, data loss, security exposure, payment pipeline down |
| 1 | `P1` | major feature broken, no workaround, blocks the user's core job |
| 2 | `P2` | degraded or broken with a workaround |
| 3 | `P3` | cosmetic, question, feature request, no functional impact |
| 9 | *abstain* | severity depends on blast radius not stated in the message |

### 3. `gate_human_v1` — Gate (boolean)

Does this need a human before any automated reply.

| code | value | boundary |
|---|---|---|
| 0 | `false` | an automated answer is safe and probably sufficient |
| 1 | `true` | legal threat, churn risk, distress, security report, repeat contact, or a P0 |

No abstain: this is a safety gate, and "unsure" must fail closed. The SDK enforces
`abstain_disabled` on this schema — low confidence routes to `true`, not to `None`.

### 4. `intent_v1` — Classify

What the sender is trying to do.

| code | value |
|---|---|
| 0 | `question` |
| 1 | `bug_report` |
| 2 | `feature_request` |
| 3 | `complaint` |
| 4 | `praise` |
| 5 | `cancel` |
| 9 | *abstain* |

### 5. `blocker_v1` — Extract-enum

The closed-set field describing what is actually blocking the user *right now*. This is
extraction, not classification: the answer is usually stated in the message.

| code | value |
|---|---|
| 0 | `none` |
| 1 | `credentials` |
| 2 | `payment` |
| 3 | `outage` |
| 4 | `data_loss` |
| 5 | `permissions` |
| 9 | *abstain* |

### 6. `urgency_v1` — Score (ordinal 1–5)

How urgently the sender needs a reply, as *they* express it — deliberately distinct from
`triage_v1`, which is objective severity. A calm P0 and a furious P3 both exist, and
keeping the two apart is the point.

| code | value | meaning |
|---|---|---|
| 0 | `1` | no time pressure expressed |
| 1 | `2` | mild |
| 2 | `3` | clear deadline or repeated follow-up |
| 3 | `4` | blocked now, explicit frustration |
| 4 | `5` | emergency framing, threats to leave, all-caps insistence |
| 9 | *abstain* | |

---

## Labeling protocol

Per `PLAN.md` §4b, the measurement that sets the accuracy ceiling:

- A **≥2000-example subset** is labeled by **3+ independent annotators** per task.
- **Disagreement is preserved as soft targets.** 2-of-3 for `billing` becomes
  `{billing: 0.67, technical: 0.33}`, never a collapsed majority vote. Collapsing is
  irreversible and destroys the calibration signal.
- Inter-annotator agreement (Krippendorff's α, ordinal-weighted for `triage_v1` and
  `urgency_v1`) is reported in `data/sft/agreement_report.json` and is the **ceiling**
  against which model accuracy is read. A model above the human ceiling is fitting the
  labeler, not the task.
- `tools/annotate.py` serves the blind labeling UI; annotators never see each other's
  answers or the model's.

See `data/README.md` for how the synthetic seed corpus stands in for this until real
labels exist, and why its agreement numbers must not be reported as if they were human.
