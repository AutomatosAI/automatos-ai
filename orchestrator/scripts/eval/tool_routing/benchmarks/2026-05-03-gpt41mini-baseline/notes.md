# Benchmark — 2026-05-03-gpt41mini-baseline

Single-model baseline: gpt-4.1-mini, both modes, all 47 queries.

Headline: filtered uses 78% fewer prompt tokens but loses 8.5pp accuracy on this model.
- Full:     93.6% / 7319 in-tok / $0.0032/correct / 1.9s p95
- Filtered: 85.1% / 1595 in-tok / $0.0008/correct / 2.0s p95

In-set rate is 87.2% so the filter drops the correct action entirely in 6 of 47 queries. Mostly analytics/memory/reports/external — categories with multiple plausible actions.

Caveat: hypothesis is small-model gap closes with filtering, which needs a frontier comparison. This run alone doesn't test the hypothesis.

Snapshotted 4 files from results/.
