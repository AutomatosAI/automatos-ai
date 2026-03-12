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
- [ ] **US-007: Migrate workflows.py + workflow_recipes.py** — Replace last 2 inline Mem0Client() calls. After this: zero direct instantiation outside UnifiedMemoryService.
- [ ] **US-008: Delete MemoryInjector + mem0_system.py** — Grep all callers first. Migrate any remaining. Delete only after zero live callers confirmed.
- [ ] **US-009: Implement L1 Redis session store** — SessionMemory dataclass, get/update/end session methods in UnifiedMemoryService. Redis key: mem:session:{ws}:{conv}. 24hr TTL.
- [ ] **US-010: Wire L1 into SmartChatOrchestrator** — Session hydration on request start, session update after each exchange. Graceful degradation if Redis down.
- [ ] **US-011: Add Redis caching for L3 Mem0 results** — 5min TTL cache in Redis. Cache key: mem:cache:{ws}:{agent}:{query_hash}. Invalidate on write. Add MEMORY_CACHE_TTL_SECONDS to config.py.

### Phase 2: Context Router

- [ ] **US-016: Build Context Router signal detection** — Regex-based temporal, personal, knowledge, live-data detection. <10ms budget. Returns ContextSignals dataclass.
- [ ] **US-017: Implement retrieve_context() assembly** — Multi-layer fetch with budget allocation (session=500, long_term=800, temporal=600, daily=400, awareness=200 tokens). Add CONTEXT_BUDGET_TOKENS to config.py.
- [ ] **US-018: Build knowledge awareness injection** — Dynamic capability map per workspace (connected DBs, docs, tools). Cache in Redis 10min TTL.
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
