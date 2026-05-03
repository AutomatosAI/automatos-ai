# PRD-138 — Tool-routing eval results

Total cells: 94 across 2 (model, mode) pairs.

## Per (model, mode)

| model | tier | mode | n | accuracy | in-set | errors | prompt tok | comp tok | $/call | $/correct | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `openai/gpt-4.1-mini` | small | full | 47 | 93.6% | 93.6% | 0.0% | 7319 | 24 | $0.297¢ | $0.317¢ | 1149 | 1912 |
| `openai/gpt-4.1-mini` | small | filtered | 47 | 85.1% | 87.2% | 0.0% | 1595 | 25 | $0.068¢ | $0.080¢ | 1177 | 1977 |

## Δ filtered − full (per model)

Positive Δ accuracy = filtered helps. Negative Δ tokens = filtered cheaper.

| model | tier | Δ accuracy | Δ prompt tok | Δ $/correct | Δ p50 ms |
| --- | --- | --- | --- | --- | --- |
| `openai/gpt-4.1-mini` | small | -8.5pp | -5724 | -0.0024 | +28 |

## Per category × mode

Where does semantic routing actually win? Look for paraphrase + ambiguous + cross.

| category | filtered (acc) | full (acc) |
| --- | --- | --- |
| agents | 8/8 (100.0%) | 8/8 (100.0%) |
| analytics | 5/7 (71.4%) | 7/7 (100.0%) |
| cross | 2/3 (66.7%) | 2/3 (66.7%) |
| documents | 2/3 (66.7%) | 2/3 (66.7%) |
| external | 2/3 (66.7%) | 2/3 (66.7%) |
| marketplace | 4/4 (100.0%) | 4/4 (100.0%) |
| memory | 2/3 (66.7%) | 3/3 (100.0%) |
| missions | 3/3 (100.0%) | 3/3 (100.0%) |
| playbooks | 4/4 (100.0%) | 4/4 (100.0%) |
| reports | 1/2 (50.0%) | 2/2 (100.0%) |
| workspace | 3/3 (100.0%) | 3/3 (100.0%) |
| workspace_code | 4/4 (100.0%) | 4/4 (100.0%) |
