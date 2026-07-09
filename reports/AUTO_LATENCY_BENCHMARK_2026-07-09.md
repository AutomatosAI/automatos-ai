# Auto Chat Latency — Baseline Benchmark (2026-07-09)

**What this is:** the measured latency profile of Auto (prod, Railway `automatos-ai-api`) immediately after the 2026-07-09 latency chain landed (#513 chat identity, #515 S3 RAG restore, #516 parallel RAG, #517/#518 classifier decouple, PR #519 bounded/async tool narrowing). **Use these numbers as the comparison baseline for every next-phase latency claim.**

**How it was measured:** Gerard ran a ~20-probe system-test battery through Auto chat (roster, activity, usage, board, cost, errors, queue, alerts, HARNESS, autonomy, channels, apps, datasources, playbooks, missions, documents, deliverables, success/efficiency/SLA) between 20:15–20:45 UTC on code `89d90e5d0` (`perf/tool-router-latency`). Logs pulled via `railway logs`, aggregated with `scripts/analyze_auto_latency.py`. Re-run recipe at the bottom.

---

## 1. Headline: today's before → after

| Stage | Before (measured this morning) | After (this benchmark) |
|---|---|---|
| Message classification (AutoBrain Tier-3) | ~95s (built the full planner pack) | **1.3s p50 / 3.8s max** (gemini-flash, cheap roster) |
| Tool-load per call (narrowing embed) | **37–67s**, unbounded, froze the event loop | **≤2.9s hard-bounded** (2.5s embed cap + registry work) |
| Narrowing query embed | 37,406–66,965ms per live call | **776ms p50 completed · 7ms on cache hit · 2,503ms hard cap on timeout** |
| Thread-bridge engagements on chat paths | every tool-load (`THREADED 37–67s`) | **zero lines** |
| Chat 500s (Clerk-string user_id) | 100% of logged-in messages | fixed (#513) |
| RAG document plane | dark (F005 guard) → 0 docs | live, S3 returns 39 docs (#515) |

## 2. The baseline numbers (hold next phases against these)

### Tool routing & narrowing (PR #519 mechanics)
| Metric | Value |
|---|---|
| `query_embed` completed | n=4 · min 7ms · p50 776ms · max 1,742ms (2 Redis cache hits) |
| `query_embed` timed out (upstream degraded) | n=4 · all bounded at ~2,503ms → full-enum fallback |
| `ensure_indexed` (124 actions) | 0ms warm · 195ms cold (Redis-backed) |
| Dispatcher rank total | p50 1.9s · max 2.7s (timeout-dominated) |
| Tool-load wall (`Loaded 57 tools`) | min 151ms · p50 2.66s · max 2.89s |
| Narrowed enum when ranking succeeds | 15 actions (vs 129 full) |
| Thread bridge on chat paths | **0** (was every load) |

### Tool execution (the tools themselves are fast)
| Tool | Duration |
|---|---|
| `platform_execute` (24 calls) | p50 ~0ms · p95 1.0s · 2 handled failures¹ |
| `platform_list_agents` / `get_activity_feed` / `get_autonomy_level` | ≤100ms |
| `search_knowledge` (1 call) | 8.0s (S3 full-document assembly: 3 docs, 36k+47k chars) |

¹ Both failures are correct behavior, not breakage: `platform_check_budget` called without its required `run_id`, and `platform_field_stability` outside a mission ("only works during missions"). Errors surfaced to the model as data; the loop continued.

### LLM calls (now the dominant cost)
| Service · model | n | latency p50 | latency max | input tokens p50 / max |
|---|---|---|---|---|
| orchestrator · gpt-5.5 | 8 | 6.0s | **30.5s** | 7,404 / **36,281** |
| graph_extraction · gemini-2.5-flash | 2 | 25.2s | 25.2s | 797 (background, 5.4k out) |
| memory_integration · gemini-2.5-flash | 4 | 5.7s | 8.6s | 1,673 |
| orchestrator · gpt-4o-mini / gpt-4.1-mini | 4 | 4.1–6.2s | 6.2s | 19,417–24,144 |
| complexity_assessor · gemini-2.5-flash | 4 | 1.3s | 3.8s | 1,212 |
| rag · gemini-2.5-flash | 4 | 0.6s | 1.3s | 58 |

### End-to-end request spans (approximate wall, from log activity)
17s · 42s · 74s · 95s for the four heavyweight test messages. Anatomy of the 95s one (a multi-part "run all these tests" ask):
- classify 4.1s → **3× narrowing attempts ≈ 8s** (all timed out at 2.5s; triple tool-load) → GPT-5.5 turn 7.9s at 30k input tokens → fast tool storm (≤1s each) → repeat LLM turns (8 GPT-5.5 calls total, up to 30.5s each).
- **≈80% of remaining wall time is the LLM loop itself** (model latency × iterations × 20–36k-token contexts), not the platform.

## 3. Next-phase targets, ranked by measured payoff

1. **Embedding provider routing** — 4/8 live embeds still hit the 2.5s cap (OpenRouter routes `qwen3-embedding-8b` by price across Nebius/DeepInfra/SiliconFlow; the slow host wins ties). Passing `provider: {"sort": "latency"}` on embedding requests should turn 2.5s penalties into ~0.3–0.8s successes, restore narrowing (15-action enum instead of 129 → better tool choice, fewer tokens), and cure the identical latency on the RAG answer path (`modules/rag/service.py` query embeds) and document ingestion (2 chunks took 47s today). Target after fix: `query_embed p50 < 800ms, timeout rate < 5%`.
2. **Single tool-load per request** — the 95s request loaded tools 3× (service `_get_tools` + PlatformActionsSection + agent_factory): ~8s of duplicated narrowing/registry/composio work when the upstream is degraded. Target: 1 load per request, ~2.6s → ~2.6s once but not thrice.
3. **LLM-loop economics** — the new dominant term. 30k+-token contexts on GPT-5.5 at 6–30s per turn × up to 8 turns. Levers: context budget on the orchestrator prompt, iteration caps for read-only probes, model tiering per complexity. (This is also the cost driver: agent 337 ≈ $3.15 of today's $3.75.)
4. **`search_knowledge` full-content assembly** — 8s for 3 full documents from S3; fine for one call, worth watching if it becomes chatty.
5. **`graph_extraction` at 25s** — background (post-response) but long; check it never blocks a user-facing path.

## 4. How to re-run this benchmark

```bash
cd automatos-ai
railway logs --since 60m -n 5000 > /tmp/auto_bench.log   # capture a test window
python3 scripts/analyze_auto_latency.py /tmp/auto_bench.log
```

Run the same ~20-probe battery through Auto chat first (roster → SLA list above), then compare each section against §2. The analyzer prints n=0 for any log family that has been pruned, so it stays usable if the `[perf]` lines are trimmed at merge — but keep at least the `[perf] rank_actions` line: it is one INFO line per narrowing and carries `cache_hit`/`TIMED OUT`, which is the health signal for target #1.
