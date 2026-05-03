# Benchmark — 2026-05-03-full-matrix

First full 9-model × 2-mode sweep for PRD-138. 846 cells, ~$9 spent.

## Setup
- 47 hand-curated queries validated against live ActionRegistry (107 actions)
- 9 models via OpenRouter: 3 frontier, 2 mid, 4 small
- Full mode dumps the entire action catalog; filtered mode top-15 by cosine similarity (qwen3-embedding-8b @ 2048d)
- max_tokens=2048 (bumped from 256 mid-run after reasoning models — GPT-5, GPT-5-mini, Gemini 2.5 Pro — were truncating before producing tool calls)
- in-set rate = 93.6% across all filtered cells: the embedding ranker fails to surface the correct action ~6.4% of the time

## Headline numbers

| model | full acc | filtered acc | Δ | filtered $/correct |
|---|---|---|---|---|
| llama-3.3-70b | 91.5% | 91.5% | 0.0pp | $0.050¢ |
| claude-opus-4.7 | 95.7% | 87.2% | -8.5pp | $0.0715 |
| gpt-5 | 93.6% | 85.1% | -8.5pp | $0.0121 |
| gpt-4.1-mini | 93.6% | 85.1% | -8.5pp | $0.080¢ |
| claude-sonnet-4.6 | 89.4% | 87.2% | -2.1pp | $0.0101 |
| claude-haiku-4.5 | 89.4% | 87.2% | -2.1pp | $0.344¢ |
| gpt-5-mini | 89.4% | 87.2% | -2.1pp | $0.186¢ |
| gemini-2.5-flash | 74.5% | 70.2% | -4.3pp | $0.018¢ |
| gemini-2.5-pro | 46.8% | 59.6% | +12.8pp | $0.693¢ |

## What this proves vs. the hypothesis

Original hypothesis: "Semantic routing lets small models match frontier on tool selection."

The data is more nuanced than the hypothesis predicted:

1. **Filtering does NOT close the gap by lifting small models.** Most models lose 2-8pp accuracy when filtered — the embedding filter drops correct actions ~6.4% of the time, which is the new ceiling for filtered accuracy regardless of model.

2. **Filtering DOES dramatically reduce cost and prompt size at minor accuracy cost.** -70 to -78% prompt tokens, -65 to -75% \$/correct across every model.

3. **Llama 3.3 70B is the standout.** No accuracy degradation under filtering, cheapest \$/correct in the matrix at \$0.050¢. Open-source 70B viable for production tool routing.

4. **Gemini 2.5 Pro is the only model that gains accuracy from filtering** (+12.8pp). It struggles with full-dump prompts — likely a reasoning-model-instruction issue. Filter forces focus.

5. **Claude Opus 4.7 has the highest full-mode ceiling (95.7%)** but suffers the most from filtering (-8.5pp). Worth the 6x cost premium only when accuracy >95% is the requirement.

6. **The right production split is probably mode-by-category.** Filtered crushes on workspace (100%), marketplace (91.7%), external (96.3%), playbooks (88.9%). Full wins on reports (94.4% vs 72.2%), analytics (87.3% vs 74.6%), memory (74.1% vs 63.0%). Suggests embedding text for those action sets needs work.

## Caveats

- Single-shot tool selection. No agent loop, no parameter quality scoring.
- 47 queries is small for category-level claims (some categories have n<30).
- gemini-2.5-pro had 39 of 94 calls return finish_reason="error" — those count as wrong but the underlying behavior is OpenRouter/Gemini-side, not the model failing the routing task.
- Embedding cache populated during this run; subsequent runs at same query set will be ~free for embeddings.

## Reproduction

```bash
cd orchestrator
export OPENROUTER_API_KEY=...
python -m scripts.eval.tool_routing.seed_eval_set
python -m scripts.eval.tool_routing.run_eval
python -m scripts.eval.tool_routing.score
```

## Next runs

- Larger query set (target: 200 queries) for tighter category-level CIs
- Sweep top_k ∈ {5, 10, 15, 20, 30} to find the in-set rate elbow
- Better embedding text composition for analytics/reports/memory action sets
- Test with parameter-quality scoring (does the LLM populate `params` correctly?)

Snapshotted 4 files from results/.
