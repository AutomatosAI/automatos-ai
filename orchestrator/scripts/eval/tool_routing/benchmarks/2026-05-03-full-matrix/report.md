# PRD-138 — Tool-routing eval results

Total cells: 846 across 18 (model, mode) pairs.

## Per (model, mode)

| model | tier | mode | n | accuracy | in-set | errors | prompt tok | comp tok | $/call | $/correct | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `anthropic/claude-opus-4.7` | frontier | full | 47 | 95.7% | 100.0% | 2.1% | 12765 | 78 | $0.1932 | $0.2017 | 1392 | 3397 |
| `anthropic/claude-opus-4.7` | frontier | filtered | 47 | 87.2% | 93.6% | 0.0% | 3740 | 83 | $0.0623 | $0.0715 | 1131 | 2417 |
| `google/gemini-2.5-pro` | frontier | full | 47 | 46.8% | 100.0% | 0.0% | 7856 | 252 | $0.0123 | $0.0264 | 3740 | 10757 |
| `google/gemini-2.5-pro` | frontier | filtered | 47 | 59.6% | 93.6% | 0.0% | 1542 | 220 | $0.413¢ | $0.693¢ | 3562 | 13555 |
| `openai/gpt-5` | frontier | full | 47 | 93.6% | 100.0% | 0.0% | 7324 | 865 | $0.0178 | $0.0190 | 10509 | 19834 |
| `openai/gpt-5` | frontier | filtered | 47 | 85.1% | 93.6% | 0.0% | 1600 | 828 | $0.0103 | $0.0121 | 11557 | 31310 |
| `anthropic/claude-sonnet-4.6` | mid | full | 47 | 89.4% | 100.0% | 2.1% | 9099 | 65 | $0.0277 | $0.0310 | 2490 | 4958 |
| `anthropic/claude-sonnet-4.6` | mid | filtered | 47 | 87.2% | 93.6% | 2.1% | 2679 | 66 | $0.884¢ | $0.0101 | 1726 | 3409 |
| `openai/gpt-5-mini` | mid | full | 47 | 89.4% | 100.0% | 0.0% | 7324 | 566 | $0.296¢ | $0.332¢ | 6622 | 14300 |
| `openai/gpt-5-mini` | mid | filtered | 47 | 87.2% | 93.6% | 0.0% | 1600 | 609 | $0.162¢ | $0.186¢ | 7568 | 19214 |
| `anthropic/claude-haiku-4.5` | small | full | 47 | 89.4% | 100.0% | 2.1% | 9098 | 61 | $0.920¢ | $0.0103 | 1495 | 2520 |
| `anthropic/claude-haiku-4.5` | small | filtered | 47 | 87.2% | 93.6% | 0.0% | 2683 | 63 | $0.300¢ | $0.344¢ | 1146 | 1689 |
| `google/gemini-2.5-flash` | small | full | 47 | 74.5% | 100.0% | 0.0% | 7856 | 13 | $0.059¢ | $0.080¢ | 680 | 1304 |
| `google/gemini-2.5-flash` | small | filtered | 47 | 70.2% | 93.6% | 0.0% | 1614 | 12 | $0.012¢ | $0.018¢ | 530 | 722 |
| `meta-llama/llama-3.3-70b-instruct` | small | full | 47 | 91.5% | 100.0% | 0.0% | 7961 | 32 | $0.160¢ | $0.175¢ | 1153 | 2766 |
| `meta-llama/llama-3.3-70b-instruct` | small | filtered | 47 | 91.5% | 93.6% | 0.0% | 2251 | 33 | $0.046¢ | $0.050¢ | 1080 | 1711 |
| `openai/gpt-4.1-mini` | small | full | 47 | 93.6% | 100.0% | 0.0% | 7319 | 24 | $0.297¢ | $0.317¢ | 1149 | 1912 |
| `openai/gpt-4.1-mini` | small | filtered | 47 | 85.1% | 93.6% | 0.0% | 1595 | 25 | $0.068¢ | $0.080¢ | 1177 | 1977 |

## Δ filtered − full (per model)

Positive Δ accuracy = filtered helps. Negative Δ tokens = filtered cheaper.

| model | tier | Δ accuracy | Δ prompt tok | Δ $/correct | Δ p50 ms |
| --- | --- | --- | --- | --- | --- |
| `anthropic/claude-opus-4.7` | frontier | -8.5pp | -9025 | -0.1303 | -261 |
| `google/gemini-2.5-pro` | frontier | +12.8pp | -6314 | -0.0194 | -178 |
| `openai/gpt-5` | frontier | -8.5pp | -5724 | -0.0069 | +1048 |
| `anthropic/claude-sonnet-4.6` | mid | -2.1pp | -6420 | -0.0208 | -764 |
| `openai/gpt-5-mini` | mid | -2.1pp | -5724 | -0.0015 | +946 |
| `anthropic/claude-haiku-4.5` | small | -2.1pp | -6415 | -0.0069 | -349 |
| `google/gemini-2.5-flash` | small | -4.3pp | -6242 | -0.0006 | -150 |
| `meta-llama/llama-3.3-70b-instruct` | small | +0.0pp | -5710 | -0.0012 | -73 |
| `openai/gpt-4.1-mini` | small | -8.5pp | -5724 | -0.0024 | +28 |

## Per category × mode

Where does semantic routing actually win? Look for paraphrase + ambiguous + cross.

| category | filtered (acc) | full (acc) |
| --- | --- | --- |
| agents | 58/72 (80.6%) | 60/72 (83.3%) |
| analytics | 47/63 (74.6%) | 55/63 (87.3%) |
| cross | 17/27 (63.0%) | 17/27 (63.0%) |
| documents | 19/27 (70.4%) | 21/27 (77.8%) |
| external | 26/27 (96.3%) | 24/27 (88.9%) |
| marketplace | 33/36 (91.7%) | 29/36 (80.6%) |
| memory | 17/27 (63.0%) | 20/27 (74.1%) |
| missions | 23/27 (85.2%) | 22/27 (81.5%) |
| playbooks | 32/36 (88.9%) | 32/36 (88.9%) |
| reports | 13/18 (72.2%) | 17/18 (94.4%) |
| workspace | 27/27 (100.0%) | 26/27 (96.3%) |
| workspace_code | 36/36 (100.0%) | 36/36 (100.0%) |
