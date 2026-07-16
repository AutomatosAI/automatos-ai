# HANDOFF (2026-07-09, updated by Fable) — tool-router stall root-caused; PR #519 on branch awaiting test→merge

## 1. Goal
Make Auto (chat) fast. Chain today: #513 chat-500, #515 S3 RAG, #516 parallel RAG, #517/#518 classifier 95s→1.25s (all MERGED/live). Last blocker was tool/delegate messages at ~40–67s.

## 2. ROOT CAUSE (proven, supersedes the bridge theory)
**OpenRouter `qwen/qwen3-embedding-8b` serving latency — NOT the thread bridge.** Railway log evidence:
- A MAIN-LOOP `rank_actions` (PlatformActionsSection path, `[req=5d7137baff3c …]` context intact = no bridge) clocked `query_embed=56976ms`.
- HTTP level: `POST /api/v1/embeddings` logs `200 OK` (headers) in ~700ms, then the completion log lands 37–67s later — body/compute time upstream.
- RAG pays it too: doc-ingestion batch of 2 chunks = 47s; RAG answer path awaits the same manager (`modules/rag/service.py:990`).
- One delegate request ran rank_actions ×3 (service `_get_tools` bridge + PlatformActionsSection main-loop + agent_factory bridge) ≈ 140s.
- Diagnostic key: EMPTY `[req= ws= agent=]` in a log line = ran on a bridge thread (contextvars lost); populated = main loop.

## 3. Standing
- **PR #519** (`perf/tool-router-latency`, commit `89d90e5d0`, pushed): Redis query-embed cache (same CacheService/model_key as action texts) + `SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S` (default 2.5s; timeout → [] → full-enum fallback + background cache-warm) + async-native `get_tools_for_agent_async`/`_rank_actions_for_dispatcher_async`; ALL hot callers converted (service ×3 sites, ToolsSection ×3 strategies, agent_factory). Sync entries remain as thin wrappers for module-load/scripts. Tests extended in `test_tool_router_semantic.py` + `test_action_semantic_index.py`.
- Railway `automatos-ai-api` deploys this branch — Gerard tests, then merge.
- **#512 PARKED draft — do not merge** (premise falsified by #515).

## 4. Dead ends — do NOT retry
- Per-loop AsyncOpenAI client (`_client_for_loop`) as THE fix — embeds are slow on any loop. Kept only as compat-path defense; its `AsyncClient.aclose` RuntimeError noise disappears once hot paths stop bridging.
- "Run the embed on the main loop and it'll be ~200ms" — falsified; main-loop embed was 57s. The bound+cache is what guarantees latency.

## 5. Open / needs Gerard's call
1. **Provider-level fix** (also fixes RAG query embeds): OpenRouter routes qwen3-embedding-8b across Nebius/DeepInfra/SiliconFlow by PRICE; pass `provider: {"sort": "latency"}` (or pin) on embedding requests in `openrouter_embedding.py`. Same weights → no re-embed (avoid SiliconFlow fp8 for vector consistency).
2. Triple tool-load/narrowing per delegate message — dedup per request.
3. Strip `[perf]` logs before/at merge of #519.
4. Wave-1: 186 done via #515 → 188 unblocked; 187/189/190/191 not started.

## 6. Read to reconstitute
PR #519 + this file · `tool_router.py` (`get_tools_for_agent_async`, `_narrow_dispatcher_actions_*`, `_get_tools_for_agent_core`) · `action_semantic_index.py` (`_embed_query_bounded`) · `config.py` `SEMANTIC_TOOL_ROUTING_EMBED_TIMEOUT_S` · memory `auto-latency-fix-chain-2026-07-09`.

## 7. Next action
Gerard: test on the branch (tool msg + delegate msg), `railway logs -f perf` → expect bounded/cached `query_embed`, `(TIMED OUT)`+fallback while upstream degraded, no `_run_coroutine_blocking THREADED` on chat paths. Then decide §5.1 provider fix, strip `[perf]`, merge #519.
