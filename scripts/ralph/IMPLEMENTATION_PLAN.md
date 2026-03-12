# PRD-79: Unified Memory & Context Architecture — Implementation Plan

> **Scope**: Backend (orchestrator) | **Risk**: High (core memory system rewrite) | **Branch**: `ralph/79-unified-memory-context`

## Summary

Replace 12 scattered Mem0Client instances with a single UnifiedMemoryService. Add 5-layer memory stack (L0 Focus, L1 Working/Redis, L2 Short-term/Postgres, L3 Long-term/Mem0, L4 Org Knowledge). Build Context Router for intelligent pre-LLM context assembly. Add NL2SQL for live data queries.

## Reference

- **PRD**: `docs/PRDS/79-UNIFIED-MEMORY-CONTEXT-ARCHITECTURE.md`
- **Mem0Client**: `orchestrator/modules/memory/integrations/mem0_client.py`
- **SmartMemoryManager**: `orchestrator/consumers/chatbot/smart_memory.py` (854 lines)
- **Config**: `orchestrator/config.py` (all constants here)
- **Redis**: `orchestrator/core/redis/client.py`
- **Platform tool pattern**: platform_actions.py + platform_executor.py + auto.py

## Tasks

### Phase 1: Foundation

- [x] **US-001: Create UnifiedMemoryService singleton** — `orchestrator/modules/memory/unified_memory_service.py` with shared Mem0Client, Redis client, MemoryNamespace helper for standardized user_id formats. Public methods stubbed.
- [x] **US-002: Implement L3 long-term methods** — store_long_term(), search_long_term(), get_all_memories(), delete_memory() delegating to shared Mem0Client with MemoryNamespace user_ids.
- [x] **US-003: Migrate SmartMemoryManager** — Replace lazy Mem0Client init with UnifiedMemoryService. Preserve 2-min LRU cache and _track_memory_access().
- [x] **US-004: Migrate platform_executor.py** — Replace 5 inline Mem0Client() calls (~lines 533, 1050, 1272, 3368, 3417) with UnifiedMemoryService.
- [x] **US-005: Migrate RecipeMemoryService** — Replace self._mem0 with UnifiedMemoryService. Add recipe namespace to MemoryNamespace if needed.
- [x] **US-006: Migrate widget_memory.py + memory_stats.py** — Replace lazy Mem0Client inits. Fix widget's raw workspace_id scoping via MemoryNamespace.
- [x] **US-007: Migrate workflows.py + workflow_recipes.py** — Replace last 2 inline Mem0Client() calls. After this: zero direct instantiation outside UnifiedMemoryService.
- [x] **US-008: Delete MemoryInjector + mem0_system.py** — Grep all callers first. Migrate any remaining. Delete only after zero live callers confirmed.
- [x] **US-009: Implement L1 Redis session store** — SessionMemory dataclass, get/update/end session methods in UnifiedMemoryService. Redis key: mem:session:{ws}:{conv}. 24hr TTL.
- [x] **US-010: Wire L1 into SmartChatOrchestrator** — Session hydration on request start, session update after each exchange. Graceful degradation if Redis down.
- [x] **US-011: Add Redis caching for L3 Mem0 results** — 5min TTL cache in Redis. Cache key: mem:cache:{ws}:{agent}:{query_hash}. Invalidate on write. Add MEMORY_CACHE_TTL_SECONDS to config.py.

### Phase 2: Context Router

- [x] **US-016: Build Context Router signal detection** — Regex-based temporal, personal, knowledge, live-data detection. <10ms budget. Returns ContextSignals dataclass.
- [x] **US-017: Implement retrieve_context() assembly** — Multi-layer fetch with budget allocation (session=500, long_term=800, temporal=600, daily=400, awareness=200 tokens). Add CONTEXT_BUDGET_TOKENS to config.py.
- [x] **US-018: Build knowledge awareness injection** — Dynamic capability map per workspace (connected DBs, docs, tools). Cache in Redis 10min TTL.
- [ ] **US-019: Replace hardcoded memory retrieval with Context Router** — Wire into SmartChatOrchestrator. Fallback to existing SmartMemoryManager on failure.

### Phase 3: Layered Storage

- [ ] **US-012: Create memory_short_term Postgres table** — Alembic migration with workspace_id, agent_id, content, content_type, importance, decay_score, access_count, metadata JSONB. Composite indexes.
- [ ] **US-013: Implement L2 CRUD** — store_short_term(), search_short_term(), get_short_term_by_time(), touch_short_term() in UnifiedMemoryService.
- [ ] **US-014: Wire L2 for chat exchanges** — store_exchange() stores to L2 + delegates to L3. Fire-and-forget via asyncio.create_task().
- [ ] **US-015: Wire L2 for recipes + heartbeats** — store_short_term() calls alongside existing Mem0 stores in RecipeMemoryService and heartbeat_service.
- [ ] **US-020: Implement Ebbinghaus decay job** — retention = exp(-0.1 * hours) * (1 + 0.5*importance + 0.1*min(access_count, 10)). Archive items below 0.3. Batch per workspace.
- [ ] **US-021: Implement L2→L3 promotion** — Daily job. Criteria: importance > 0.7 AND access_count > 3. Mem0 add with infer=True for fact extraction.
- [ ] **US-022: Wire session consolidation L1→L2** — Extract decisions/action items from expired sessions. Store as L2 entries. Delete L1 key. Use SCAN not KEYS.
- [ ] **US-023: Register background jobs** — Hourly: consolidation + decay. Daily: promotion. Follow existing scheduler pattern.

### Phase 4: NL2SQL

- [ ] **US-024: Build NL2SQL service** — Schema caching (Redis 10min), LLM SQL generation, SELECT-only validation, 5s timeout, 1000-row limit, query audit logging.
- [ ] **US-025: Register query_data tool** — 3-file pattern: platform_actions.py, platform_executor.py, auto.py.

### Phase 5: Observability & Cleanup

- [ ] **US-026: Memory layer health endpoint** — GET /api/v1/memory/layers with per-layer stats and health score.
- [ ] **US-027: Clean up skeleton classes** — Delete service.py and storage/manager.py if zero live callers confirmed via grep.
- [ ] **US-028: Integration tests** — L1 session lifecycle, L2 CRUD + decay, L3 cache hit, Context Router signal routing, MemoryNamespace correctness.

---

## Discovered Issues

1. **US-002 L3 methods were blocking the event loop** — `store_long_term()`, `search_long_term()`, `get_all_memories()`, `delete_memory()` called synchronous `Mem0Client` methods directly in async functions. Fixed in US-003 by wrapping all calls in `asyncio.run_in_executor()`.
2. **Namespace migration breaks existing daily logs** — Old format `ws_{id}_daily` → new format `mem:{id}:daily`. Existing daily log entries in Mem0 under old keys will not be found. This is by design per PRD-79 but should be communicated to users.
3. **pgvector module not installed locally** — `modules/memory/storage/knowledge_system.py` imports `pgvector.sqlalchemy` which is not in local dev dependencies. Pre-existing issue, unrelated to US-003.

## Notes

- **US-003:** Added `store_two_tier()`, `store_daily_log()`, `get_all_daily_logs()` to UnifiedMemoryService to support SmartMemoryManager's two-tier storage and daily log patterns. SmartMemoryManager no longer has any direct Mem0Client usage. The 2-min LRU cache, tier classification, memory formatting, and _track_memory_access remain unchanged.
- **US-004:** Replaced all 5 inline `Mem0Client()` calls in `platform_executor.py` with `get_unified_memory_service()`. Added `is_mem0_configured` property to `UnifiedMemoryService` to preserve the "not configured" guard that platform handlers relied on. Handlers: `_get_memory_stats`, `_store_memory`, `_search_memory`, `_browse_memories`, `_delete_memory`. Zero direct `Mem0Client` references remain in `platform_executor.py`.
- **US-005:** Replaced `self.mem0 = mem0_client or Mem0Client()` with `get_unified_memory_service()`. Made `store_execution_memory()` and `retrieve_relevant_memories()` async. All user_ids now built via `MemoryNamespace.recipe()` and `MemoryNamespace.recipe_agent()` — zero raw string concatenation. Added `store_long_term_messages()` and `search_long_term_scoped()` to UnifiedMemoryService for consumers needing custom message formats or pre-built namespace user_ids. Updated `recipe_executor.py` callers to `await`. Also improved agent memory retrieval to use `asyncio.gather()` for concurrent searches. Widened `MemoryNamespace.recipe()` and `recipe_agent()` type hints to `Union[int, str]` since template_id can be a string.
  - **Discovered:** `workflow_recipes.py:680` imports `RecipeMemoryService` but never uses it — the import is dead code. The same block (lines 681-694) creates a raw `Mem0Client()` directly for recipe memory cleanup on delete — this is US-007 scope.
- **US-006:** Replaced lazy `Mem0Client` init in both `widget_memory.py` and `memory_stats.py` with `UnifiedMemoryService` singleton via `_get_memory_service()` helper. Widget endpoints now use `service.get_all_memories()`, `service.search_long_term()`, `service.store_long_term()`, `service.delete_memory()` instead of direct `Mem0Client` calls. Memory stats refactored to use `_fetch_all_scoped_memories()` helper which queries all scopes (workspace, per-agent, daily) in parallel via `asyncio.gather()`. The old `_all_mem0_user_ids()` helper (which built old-format `ws_{id}` strings) is replaced by `MemoryNamespace`-based scoping through the service. `_mem0_user_id()` also deleted. Fallback store preserved in widget. Removed unused `timedelta` import from memory_stats.
- **US-007:** Replaced all Mem0Client usage in `workflows.py` (2 locations: retrieval at ~line 2073, storage at ~line 2985) and `workflow_recipes.py` (1 location: recipe delete cleanup at ~line 681). In workflows.py: replaced `mem0_client` variable with `memory_service_available` boolean flag, search uses `search_long_term_scoped()`, storage uses `store_long_term_messages()`. In workflow_recipes.py: recipe delete cleanup uses `get_all_memories_scoped()` + `delete_memory()` with `MemoryNamespace.recipe()` scoping. Added `MemoryNamespace.workflow()` for workflow-scoped memory keys. Added `get_all_memories_scoped()` to UnifiedMemoryService for consumers needing custom user_ids on get_all. Removed dead `RecipeMemoryService` import from workflow_recipes.py. After this: only `mem0_system.py` (deprecated, US-008 scope) retains a direct `Mem0Client()` instantiation outside UnifiedMemoryService.
- **US-009:** Added `SessionMemory` dataclass (summary, decisions, action_items, exchange_count, last_updated, ended) with `to_json()`/`from_json()` for Redis serialisation. Implemented three L1 methods in UnifiedMemoryService: `get_session()` reads from Redis and returns `SessionMemory` or None; `update_session()` creates or updates session with rolling summary (naive truncation to 500 chars — Phase 2 adds LLM summarisation), increments exchange_count, resets 24hr TTL; `end_session()` marks session ended and reduces TTL to 1hr consolidation window. All three methods use `asyncio.run_in_executor()` for the sync Redis client, and all swallow Redis failures with `exc_info=True` logging (Redis must never break chat). Added three config constants: `MEMORY_SESSION_TTL_SECONDS` (86400), `MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS` (3600), `MEMORY_CACHE_TTL_SECONDS` (300) — all env-overridable. Return type of `get_session()` changed from `Optional[Dict]` to `Optional[SessionMemory]` (typed dataclass).
- **US-010:** Wired L1 session store into `SmartChatOrchestrator` in `smart_orchestrator.py`. Two integration points: (1) `prepare_request()` — calls `get_session()` at step 3b, if session exists injects `## Session Context` heading with exchange count and rolling summary (capped at 2000 chars ≈ 500 tokens) into system prompt between personality and daily logs. (2) `store_exchange()` — fires `update_session()` via `asyncio.create_task()` (fire-and-forget, never blocks TTFT). Both paths have full try/except with `exc_info=True` logging — Redis failures silently degrade. `_unified_memory` instance stored on orchestrator `__init__` via `get_unified_memory_service()`. No changes to `integration.py` — `chat_id` already flows through from `SmartChatIntegration.prepare()` and `.store()`.
- **US-008:** Deleted 3 files: `operations/injection.py` (MemoryInjector + get_memory_injector + get_memory_system), `storage/mem0_system.py` (Mem0MemorySystem adapter), `debug_mem0_persistence.py` (debug script). Removed dead `get_memory_injector()` import and `self.memory_injector` assignment from `consumers/chatbot/service.py` (was imported but never used after assignment). Updated `operations/__init__.py` to remove MemoryInjector/get_memory_injector/get_memory_system exports while preserving MemoryPromptInjector (which has live callers in execution_manager.py). Verified zero remaining references via grep. Zero direct Mem0Client() instantiation remains outside UnifiedMemoryService.
- **US-011:** Added Redis caching layer to `search_long_term()` in `unified_memory_service.py`. Three new private methods: `_get_cached_search()` reads from Redis, `_set_cached_search()` writes with `MEMORY_CACHE_TTL_SECONDS` (300s) TTL, `_invalidate_search_cache()` uses SCAN (not KEYS) to delete all `mem:cache:{workspace_id}:*` keys. Cache key format: `mem:cache:{workspace_id}:{agent_id|global}:{sha256(query)[:16]}`. On `store_long_term()` success, cache invalidation fires as fire-and-forget via `asyncio.ensure_future()`. `MEMORY_CACHE_TTL_SECONDS` already existed in config.py from US-009. MemoryNamespace already had `cache_key()` and `cache_pattern()` methods. Cache read/write failures are non-fatal — logged at DEBUG level, never break the search path.
- **US-016:** Created `modules/memory/context_router.py` with `ContextRouter` class and `ContextSignals` frozen dataclass. Five signal categories detected via pre-compiled regex patterns: temporal (relative time refs like "last week", "3 days ago", named weekdays), personal fact ("my name", "I prefer", "remember when"), session continuation ("as we just discussed", "earlier in this conversation"), knowledge query ("find the doc", "what's our policy"), live data ("current MRR", "how many users", metrics keywords). `_compute_temporal_window()` converts relative references to absolute `(start, end)` datetime tuples. All patterns compiled at module load — `analyze_query()` is pure regex, no I/O, <10ms. ContextSignals is frozen (immutable). Tested all 5 signal types + empty input + frozen enforcement.
- **US-017:** Added `ContextBundle` frozen dataclass (session_summary, long_term_memories, temporal_results, daily_logs, knowledge_awareness, total_tokens_estimate, signals) and `retrieve_context()` async method to `ContextRouter`. Signal-driven fetch strategy: session continuation/default → L1 Redis session; personal fact/default → L3 Mem0 (cached); temporal → L2 short-term with time filter; knowledge/live_data → static awareness text only (no pre-fetch). All layer fetches run concurrently via `asyncio.gather()` with `_safe_fetch()` wrapper that swallows per-layer failures. Budget allocation via 6 new config constants: `CONTEXT_BUDGET_TOKENS` (4000), `CONTEXT_BUDGET_SESSION` (500), `CONTEXT_BUDGET_LONG_TERM` (800), `CONTEXT_BUDGET_TEMPORAL` (600), `CONTEXT_BUDGET_DAILY` (400), `CONTEXT_BUDGET_AWARENESS` (200) — all env-overridable. Helper methods: `_estimate_tokens()` (len//4), `_truncate_to_budget()`, `_memories_to_text()`, `_window_days()`, `_build_default_awareness()`. `UnifiedMemoryService.retrieve_context()` updated to delegate to `ContextRouter.retrieve_context()` with full error handling. `_build_default_awareness()` is a minimal static placeholder — US-018 replaces it with dynamic per-workspace capability map.
- **US-018:** Replaced static `_build_default_awareness()` with dynamic `build_knowledge_awareness()` in `ContextRouter`. Queries workspace for: (1) active `DatabaseKnowledgeSource` entries (name + dialect), (2) processed `Document` count, (3) active `ComposioConnection` app names joined through `ComposioEntity`. Results formatted into `## What You Can Look Up` text block (<200 tokens) showing actual connected databases by name, document count, and connected tools. Cached in Redis via `mem:awareness:{workspace_id}` key with `MEMORY_AWARENESS_CACHE_TTL_SECONDS` (600s, env-overridable). Cache read/write uses `_get_cached_awareness()` / `_set_cached_awareness()` helpers — both non-fatal on failure. DB query runs in `run_in_executor()` via synchronous `_query_workspace_capabilities()` using `get_db_session()` context manager. Each sub-query (databases, docs, tools) has independent try/except so partial capability data still produces useful output. `retrieve_context()` now calls `build_knowledge_awareness()` instead of static fallback, with graceful degradation to `_build_fallback_awareness()` on failure. Renamed `_build_default_awareness` → `_build_fallback_awareness` for clarity.
