# Model comparison

> **SYNTHETIC CORPUS.** Labels are a deterministic function of latent
> facts in `data/gen_seed.py`. These numbers measure pipeline
> correctness, not model quality. See `data/README.md`.

## Accuracy

| run                          | blocker_v1     | gate_human_v1  | intent_v1      | route_v1       | triage_v1      | urgency_v1     | macro |
|------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|-------|
| jev:ollama                   | 0.820          | 0.887          | 0.467          | 0.837          | 0.830          | 0.918          | 0.793 |
| encoder:distilbert-base-unca | 1.000          | 1.000          | 0.640          | 1.000          | 1.000          | 0.995          | 0.939 |
| tfidf+logreg                 | 1.000          | 1.000          | 0.641          | 1.000          | 1.000          | 0.995          | 0.939 |
| (prior only — the floor)     | 0.489          | 0.513          | 0.357          | 0.316          | 0.468          | 0.305          |   -   |

## ECE (lower is better)

| run                          | blocker_v1     | gate_human_v1  | intent_v1      | route_v1       | triage_v1      | urgency_v1     | macro |
|------------------------------|----------------|----------------|----------------|----------------|----------------|----------------|-------|
| jev:ollama                   | 0.097          | 0.042          | 0.065          | 0.052          | 0.051          | 0.051          | 0.060 |
| encoder:distilbert-base-unca | 0.000          | 0.000          | 0.026          | 0.000          | 0.000          | 0.003          | 0.005 |
| tfidf+logreg                 | 0.000          | 0.000          | 0.051          | 0.000          | 0.000          | 0.002          | 0.009 |

## Latency and cost

| run                          |   p50 ms |  tok/decision |   params |
|------------------------------|----------|---------------|----------|
| jev:ollama                   |     22.9 |          2.00 |        - |
| encoder:distilbert-base-unca |    4.71* |             - |      66M |
| tfidf+logreg                 |        - |             - |        - |

`*` single forward pass for ALL fields at once (the Path A prototype), not per field.

## Verdict

- **`encoder:distilbert-base-uncased` matches or beats JEV on BOTH accuracy (0.939 vs 0.793) and ECE (0.005 vs 0.060).** PLAN.md risk #3: a decoder LLM is the wrong tool for these tasks. Shipping the baseline is the honest call.
- **`tfidf+logreg` matches or beats JEV on BOTH accuracy (0.939 vs 0.793) and ECE (0.009 vs 0.060).** PLAN.md risk #3: a decoder LLM is the wrong tool for these tasks. Shipping the baseline is the honest call.

All of the above is measured on synthetic data and says nothing about real-world quality.
