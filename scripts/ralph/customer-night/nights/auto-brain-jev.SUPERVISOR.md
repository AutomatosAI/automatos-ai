# Night 9 — Auto's brain with the decision seam on (supervisor notes, never part of the persona's prompt)

`auto-brain-jev.md` is byte-identical to `auto-brain.md` and a test keeps it so
(`orchestrator/tests/test_prd248_s4_tool_rerank.py::test_night_briefs_are_the_same_evening_word_for_word`).
The customer's evening never changes; the only variable is what the supervising session sets
before launch. Everything below is the supervisor's job. The persona must never touch
Settings → System (the brief's "no database" rule covers it; say it explicitly in the launch).

## The sub-nights, one variable each

| sub-night | `classifier_mode` | `tool_rerank_mode` | `provider` | what it proves |
|---|---|---|---|---|
| 8 (baseline) | off | off | — | the yardstick; nothing from PRD-248 runs |
| 9a | shadow | shadow | openrouter | the seam is inert; agreement and rerank comparisons, latency and cost of a decision |
| 9b | live | off | openrouter | the classifier hook alone, against 8 |
| 9c | off | live | openrouter | the rerank hook alone, against 8 |
| 9d (optional) | shadow | shadow | llm | is it Jev, or would any typed judge on the cheap tier do the same |

If the rotation cannot afford 9b and 9c as separate nights, run both live in one night and
write the confound into the findings ledger. Never change a dial mid-night: a night is one setting.

## Before launch

1. The test checkout carries `feat/prd-248-decision-seam` (merged with the dials off). On the
   backend's restart the startup seed creates the nine `decision_engine` rows; confirm in
   Settings → System, or `select key, value from system_settings where category='decision_engine'`.
2. Set the sub-night's dials there. Leave `model` empty (pinned defaults: `typesafe/jev-1.13`
   via OpenRouter, `jev-1.13.0` direct), `timeout_seconds` 2.5, `min_confidence` 0.7,
   `rerank_candidates` 30, `rerank_min_probability` 0.5, `rerank_min_keep` 5.
3. The OpenRouter route uses the platform's existing OpenRouter key; nothing else to set. The
   direct route needs `TYPESAFE_API_KEY` in the compose override environment, never the shared
   env file. `provider=llm` needs no key.
4. Within the first three chat turns of a shadow or live night, one row with
   `request_type='decision'` must appear in `llm_usage` and one line in
   `/app/logs/decision_shadow.jsonl` (shadow only). If neither appears, the seam is idle
   (no key, or the dial did not take): stop and fix before the persona runs on.
5. Launch as any other night: `./scripts/ralph/overnight-customer.sh` with the same persona,
   model and stop time as night 8.

## The first assertion of 9a: the seam changed nothing

From the ledger row for 9a against the row for 8, with `request_type='decision'` rows
excluded from every chat number: asks, first-time rate (the persona's tally), chat turns,
input tokens per turn, cost per turn, tool calls per turn, wrong-tool rate, the judge's grades.
All within the night-to-night noise of two baseline nights. If any of them moved, the shadow
is not inert and 9b/9c do not run until it is.

## The morning after each sub-night

```sh
# the seam's own numbers for the window, as JSON for campaign.sqlite (STARTED/FINISHED from night.status)
docker exec automatos_backend python -m scripts.eval.decision_shadow.score --since <STARTED> --until <FINISHED> --purpose all --json
# the same, readable
docker exec automatos_backend python -m scripts.eval.decision_shadow.score --since <STARTED> --until <FINISHED>
# the two offline gates, every sub-night, same as night 8
cd orchestrator && DECISION_PROVIDER=openrouter python -m scripts.eval.tool_routing.run_eval --mode jev_rerank --models openai/gpt-4.1-mini && python -m scripts.eval.tool_routing.score
docker exec automatos_backend python -m evals.operating_graph_uplift --ranker jev --from-telemetry --json
```

Read against night 8, in this order: the persona's first-time rate and its five worst misses;
first-try selection rate and wrong-tool rate; input tokens and cost per chat turn; latency
per turn; the `complexity_assessor` call count (9b should take it toward zero); the decision
lane's own count, latency p50/p95 and cost; the judge's grades. From the shadow log (9a):
agreement per field, per tier and per engine-confidence band, rerank overlap and the
nothing-fits rate. Label every finding trace-backed or inferred, as every night.

## What goes in the findings ledger

One row per sub-night: the setting, the persona's tally, the deltas against 8, and the
seam's own numbers. A hook earns a "flip to live for the programme" recommendation only when
its live night beats night 8 on the first-time rate or the wrong-tool rate without losing on
the judge's grades or the cost per turn. Anything else is an honest sub-threshold result.
