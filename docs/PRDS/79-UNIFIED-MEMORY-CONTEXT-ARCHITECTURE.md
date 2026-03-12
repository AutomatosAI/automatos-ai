# PRD-78: Unified Memory & Context Architecture

**Status:** DRAFT
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-12
**Priority:** P0 — Foundation for agent intelligence
**Supersedes:** PRD-05 (Memory & Knowledge), PRD-39 (Mem0 Migration)
**Extends:** PRD-77 (Memory Dashboard), PRD-69 (Agent Intelligence)
**Touches:** PRD-08 (RAG v2), PRD-03 (Context Engineering), PRD-21 (Database Knowledge)

---

## 1. Problem Statement

### 1.1 The Core Issue

Memory in Automatos is fragmented across **6 consumers**, **5 Mem0Client instances**, **5 user_id formats**, and **2 competing architectural approaches** (PRD-05 hierarchical vs PRD-39 Mem0 flat). The result:

- Agents lose all conversation context when a session ends
- No graduated importance — a passing "I like dark mode" gets the same treatment as a critical business decision
- Agents don't know what they don't know — no awareness of "I can look this up in documents" vs "I should remember this"
- Every request hits Mem0 cold — no local cache hierarchy
- At 10K users, one Mem0 instance doing embedding searches per request will choke

### 1.2 Current Fragmentation (Evidence)

| Consumer | Mem0Client | user_id Format | Stores? | Retrieves? |
|----------|-----------|---------------|---------|------------|
| Chatbot | Own lazy instance | `ws_{id}`, `ws_{id}_agent_{id}`, `ws_{id}_daily` | Yes | Yes |
| Recipes | Own instance in `__init__` | `ws_{id}_recipe_{rid}`, `ws_{id}_recipe_{rid}_agent_{aid}` | Yes | Yes |
| Widget | Own lazy instance | `ws_{id}` only | Yes | Yes |
| Platform tools | New per-call | `ws_{id}`, `ws_{id}_agent_{id}` | Yes | Yes |
| Memory stats | Own lazy getter | All tiers dynamically | No | Yes |
| Heartbeat | Via SmartMemoryManager | Inherits chatbot tiers | Yes (daily) | Yes (daily) |

**Circuit breaker state is NOT shared** — one consumer can have breaker open while another keeps hammering a dead Mem0.

### 1.3 What Gets Lost Today

- **Conversation continuity**: Chat history dies when session ends. Only extracted facts survive.
- **Recipe learnings**: Stored in recipe-scoped Mem0 keys — invisible to chatbot agents.
- **Cross-agent knowledge**: Agent A learns something useful for Agent B — no transfer mechanism.
- **Temporal context**: "What did we discuss last week?" — no retrieval path exists.
- **Operational context**: Heartbeat daily logs stored but rarely injected into chat context.

---

## 2. Vision: The Human Brain Analogy

When you hire someone, they:
1. **Focus** on what's in front of them (context window)
2. **Scribble notes** during the meeting (working memory)
3. **Remember this week's discussions** without effort (short-term)
4. **Know your preferences** after months of working together (long-term)
5. **Look things up** in Confluence/CRM when they need specifics (organizational knowledge)

They don't memorize the entire CRM. They know it EXISTS and when to look. That's the system we need.

---

## 3. Architecture: 5-Layer Memory Stack

```
                    ┌──────────────────────────────────┐
                    │     CONTEXT ROUTER (NEW)          │
                    │   "Where should I look for this?" │
                    └──────┬───────────────────────────┘
                           │
         ┌─────────────────┼─────────────────────┐
         ↓                 ↓                     ↓
  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
  │  L0-L3       │  │  L4          │  │  L4               │
  │  Memory      │  │  RAG         │  │  NL2SQL / APIs    │
  │  (personal)  │  │  (docs)      │  │  (live data)      │
  └──────────────┘  └──────────────┘  └──────────────────┘
```

### Layer 0: FOCUS (Context Window)

| Property | Value |
|----------|-------|
| **Storage** | In-memory, request-scoped |
| **What** | Current conversation, tool results, system prompt |
| **Capacity** | Model context window (128K tokens) |
| **TTL** | Request lifetime |
| **Analogy** | Your desk right now |
| **New work** | None — this already works |

### Layer 1: WORKING MEMORY (Session Cache)

| Property | Value |
|----------|-------|
| **Storage** | Redis (per-session key) |
| **What** | Conversation summaries, session decisions, temp notes, tool results |
| **Capacity** | ~50 items per session |
| **TTL** | 24 hours after last activity (configurable) |
| **Analogy** | Your notepad from today's meeting |
| **New work** | **NEW** — doesn't exist today |

**Key design:**
- Key format: `mem:session:{workspace_id}:{conversation_id}`
- Stores conversation summary (rolling, updated every 5 messages)
- Stores key decisions and action items extracted per exchange
- On session resume: hydrate L0 from L1 summary instead of replaying full history
- On session end + 1hr: consolidation job promotes important items to L2

**Why Redis:**
- Sub-millisecond reads (vs 200-500ms Mem0)
- Natural TTL support
- Already deployed on Railway
- Handles 100K+ concurrent sessions trivially

### Layer 2: SHORT-TERM MEMORY (Recent Context)

| Property | Value |
|----------|-------|
| **Storage** | Postgres (structured rows) + Mem0 (semantic search) |
| **What** | Last 7-30 days of interactions, recent preferences, project context |
| **Capacity** | ~1,000 items per user |
| **TTL** | 7-30 days (Ebbinghaus decay curve) |
| **Analogy** | What you discussed this week with a colleague |
| **New work** | Wire existing `HierarchicalMemoryManager` decay logic |

**Key design:**
- Postgres table: `memory_short_term` (workspace_id, agent_id, content, importance, decay_score, access_count, created_at, last_accessed_at)
- Mem0 stores the semantic version (for vector search)
- Consolidation job: hourly, promotes high-importance items to L3
- Importance scoring: base score + access_frequency_boost + recency_boost + content_richness_boost
- Decay: `retention = exp(-0.1 * hours_elapsed)` boosted by importance and access count
- Items below 0.3 retention score are archived or deleted

**What goes here:**
- All conversation exchanges (stored with `infer=False` — raw, not extracted)
- Recipe execution summaries
- Heartbeat daily logs
- Tool call results that had useful outcomes

### Layer 3: LONG-TERM MEMORY (Learned Knowledge)

| Property | Value |
|----------|-------|
| **Storage** | Mem0 with `infer=True` (fact extraction + deduplication) |
| **What** | Stable facts, preferences, patterns, relationships |
| **Capacity** | ~10K items per user |
| **TTL** | Permanent (refreshed on contradiction) |
| **Analogy** | What you know about a colleague after months |
| **New work** | Promotion pipeline from L2, consolidation |

**Key design:**
- Mem0 handles deduplication automatically ("user likes coffee" stored twice → merged)
- LLM fact extraction pulls out key facts, not verbatim storage
- Custom categories via Mem0: `personal`, `workflow`, `preference`, `decision`, `learning`
- Contradiction handling: latest truth wins (Mem0 native)
- Consolidation: daily job merges related memories, flags stale ones (PRD-77 Phase 3)

**What goes here:**
- User facts (name, role, company, timezone)
- Stable preferences (communication style, tool preferences)
- Learned patterns (how user likes reports formatted, which Slack channels matter)
- Business decisions (pricing strategy, target markets)
- Agent instincts (PRD-69 promoted patterns)

### Layer 4: ORGANIZATIONAL KNOWLEDGE (Look-up)

| Property | Value |
|----------|-------|
| **Storage** | S3 Vectors (RAG), Postgres (NL2SQL), External APIs |
| **What** | Company docs, databases, CRM, Jira, Confluence |
| **Capacity** | Unlimited |
| **TTL** | Permanent (updated by sync) |
| **Analogy** | Confluence, the CRM, the company wiki |
| **New work** | NL2SQL integration, Context Router awareness |

**Key design:**
- NOT pre-fetched — agent decides to search via tools
- BUT: Context Router gives agent awareness that these sources exist
- Three sub-channels:
  - **RAG** (documents): `search_knowledge` tool → S3 Vectors → chunks
  - **NL2SQL** (live data): `query_data` tool → SQL generation → Postgres → results
  - **APIs** (external): Composio tools → Jira, Slack, GitHub, etc.

---

## 4. The Context Router (Core Innovation)

The Context Router is not a tool — it's a **pre-LLM context assembly layer** that decides what context to inject BEFORE the agent sees the prompt.

### 4.1 How It Works

```
User message arrives
  ↓
Context Router analyzes the query:
  ├── Temporal signal? ("last week", "yesterday", "earlier")
  │   → Fetch from L2 (short-term) with time filter
  ├── Personal fact? ("my name", "I prefer", "remember when")
  │   → Fetch from L3 (long-term) via Mem0 search
  ├── Session continuation? (same conversation_id)
  │   → Hydrate from L1 (Redis session cache)
  ├── Knowledge query? ("what's our policy", "find the doc")
  │   → DON'T fetch — but inject awareness: "You have access to
  │     search_knowledge and query_data tools for company docs and data"
  ├── Live data? ("current MRR", "how many users", "latest deploy")
  │   → DON'T fetch — but inject awareness: "You can query live
  │     business data using the query_data tool"
  └── Default
      → L3 memories (top 5) + L1 session summary (if exists)
  ↓
Assembled context → System prompt → LLM
```

### 4.2 Context Budget Allocation

Total budget: configurable per model, default 4,000 tokens for context injection.

| Source | Budget | Priority | Pre-fetched? |
|--------|--------|----------|--------------|
| L1 Session summary | 500 tokens | Highest | Yes (Redis, <5ms) |
| L3 Long-term memories | 800 tokens (top 5) | High | Yes (Mem0, cached in Redis) |
| L2 Temporal results | 600 tokens | Medium | Only if temporal signal detected |
| Daily activity logs | 400 tokens | Low | Only if relevant |
| Knowledge awareness | 200 tokens | Always | Static text injection |
| Reserved for tools | Remainder | N/A | Tool results fill this |

### 4.3 Knowledge Awareness Injection

Instead of pre-fetching all knowledge, inject a **dynamic capability map** into the system prompt. For example, if a user has connected their Postgres metrics database:

```
## What You Can Look Up
You have access to organizational knowledge. Don't guess — look things up:
- **Company documents**: Use `search_knowledge` to search uploaded docs, policies, guides
- **External Databases**: You have a Postgres database connected called 'Production App Data'. Use `query_data(database_id=1)` to ask questions about users, payments, and activity.
- **External systems**: Use your connected tools (Jira, Slack, GitHub, etc.)
- **Past conversations**: Your memories include recent interactions — check before asking the user to repeat themselves
```

This is ~100 tokens but gives the agent the "I know Confluence exists" awareness without pre-fetching.

---

## 5. Unified Memory Service (Consolidation)

### 5.1 Single Entry Point

Replace the 5 scattered Mem0Client instances with ONE service:

```python
# orchestrator/modules/memory/unified_memory_service.py

class UnifiedMemoryService:
    """Single entry point for all memory operations across all consumers."""

    _instance: Optional['UnifiedMemoryService'] = None

    @classmethod
    def get_instance(cls) -> 'UnifiedMemoryService':
        """Singleton — shared circuit breaker, connection pool, caching."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._mem0 = Mem0Client()          # L3: long-term (shared instance)
        self._redis = get_redis_client()    # L1: session cache
        # CRITICAL: DB sessions must be request-scoped or acquired from an async pool per-call
        # DO NOT hold a single DB session in this Singleton to prevent cross-tenant data leaks.
        self._db_pool = get_db_pool()       # L2: short-term structured


    # --- Layer 1: Session (Redis) ---
    async def get_session(self, workspace_id: str, conversation_id: str) -> SessionMemory
    async def update_session(self, workspace_id: str, conversation_id: str, exchange: Exchange)
    async def end_session(self, workspace_id: str, conversation_id: str)

    # --- Layer 2: Short-term (Postgres + Mem0 raw) ---
    async def store_short_term(self, workspace_id: str, content: str, metadata: dict)
    async def search_short_term(self, workspace_id: str, query: str, days: int = 7) -> List[MemoryItem]
    async def run_decay(self)  # Hourly job

    # --- Layer 3: Long-term (Mem0 with fact extraction) ---
    async def store_long_term(self, workspace_id: str, content: str, category: str)
    async def search_long_term(self, workspace_id: str, query: str, limit: int = 5) -> List[MemoryItem]
    async def get_user_profile(self, workspace_id: str) -> UserProfile

    # --- Cross-layer ---
    async def retrieve_context(self, workspace_id: str, agent_id: int, query: str,
                                conversation_id: str = None) -> ContextBundle
    async def store_exchange(self, workspace_id: str, agent_id: int,
                              user_msg: str, assistant_msg: str, conversation_id: str)
    async def promote_to_long_term(self, item: MemoryItem)  # L2 → L3
    async def consolidate(self, workspace_id: str)  # Daily job
```

### 5.2 User ID Strategy (Unified)

**ONE format, namespace prefixes:**

> [!WARNING]
> **Mem0 Namespace Isolation**: Wrap the Mem0 client so it strictly requires a typed `WorkspaceID` object, forcing developers to provide the scope at compile/type-check time rather than relying on string concatenation which could easily leak workspace A's long-term memory to workspace B with a single missing prefix.


```
mem:{workspace_id}                              → L3 global (workspace-wide facts)
mem:{workspace_id}:agent:{agent_id}             → L3 agent-specific
mem:{workspace_id}:recipe:{recipe_id}           → L3 recipe learnings
mem:{workspace_id}:daily                        → L2 daily activity logs
mem:session:{workspace_id}:{conversation_id}    → L1 session cache (Redis)
```

All consumers use these formats via `UnifiedMemoryService` methods — never construct user_ids directly.

### 5.3 Migration from Current State

| Current | New | Migration |
|---------|-----|-----------|
| SmartMemoryManager | UnifiedMemoryService | Wrapper, delegates internally |
| RecipeMemoryService | UnifiedMemoryService.store_short_term() + tags | Recipe-specific methods become thin wrappers |
| Widget widget_memory.py | UnifiedMemoryService with widget scope | Remove standalone API, use shared service |
| Platform executor search | UnifiedMemoryService.search_long_term() | Replace inline Mem0Client creation |
| Memory stats browse | UnifiedMemoryService.get_all() | Replace lazy getter |
| MemoryInjector (deprecated) | DELETE | Already dead code |

---

## 6. Performance at Scale

### 6.1 Request Path Latency

| Step | 1 user | 10K users | Strategy |
|------|--------|-----------|----------|
| L1 Redis session lookup | <5ms | <10ms | Redis scales horizontally, key-partitioned |
| L3 cached in Redis | <5ms | <10ms | Cache L3 results for 5min in Redis |
| L3 Mem0 cold fetch | 200-500ms | 200-500ms | Only on cache miss (~10% of requests) |
| L2 Postgres search | 50ms | 100ms | B-tree index on (workspace_id, created_at) |
| Context Router logic | <10ms | <10ms | In-process, no I/O |
| **Total pre-LLM** | **~60ms (cached)** | **~120ms (cached)** | **Current: 500-1000ms (always cold)** |

### 6.2 Caching Strategy

```
L3 Mem0 results → cached in Redis (5 min TTL)
  Key: mem:cache:{workspace_id}:{agent_id}:{query_hash}
  Value: serialized MemoryResult

L2 user profile → cached in Redis (10 min TTL)
  Key: mem:profile:{workspace_id}
  Value: {name, preferences, facts}

L1 session → always in Redis (24hr TTL)
  Key: mem:session:{workspace_id}:{conversation_id}
  Value: {summary, decisions, last_updated}
```

**Cache invalidation:** On store_exchange(), invalidate the cache key for that workspace+agent.

### 6.3 Storage Projections

Per workspace (active, 1 user):
- L1: ~10 session keys × 2KB = 20KB Redis
- L2: ~200 rows/month × 1KB = 200KB Postgres
- L3: ~50 facts (Mem0 deduplicates) × 500B = 25KB vector store

At 10K workspaces:
- L1: 200MB Redis (trivial)
- L2: 2GB Postgres/month (partition by workspace_id, archive after 30 days)
- L3: 250MB Mem0 (pgvector handles this easily)

---

## 7. Background Jobs

### 7.1 Session Consolidation (Hourly)

```
For each expired L1 session (last_activity > 1hr ago):
  1. Extract key decisions and action items
  2. Store in L2 (short-term) with importance score
  3. Delete L1 session key
```

### 7.2 Decay & Promotion (Daily)

```
For each L2 item:
  1. Calculate retention: exp(-0.1 * hours) * (1 + 0.5*importance + 0.1*min(access_count, 10))
  2. If retention < 0.3: archive or delete
  3. If importance > 0.7 AND access_count > 3: promote to L3
     - Store via Mem0 with infer=True (fact extraction)
     - Mark L2 item as promoted
```

### 7.3 Consolidation (Weekly)

> [!IMPORTANT]
> **Scale constraint (10K Users)**: Scanning a monolithic `memory_short_term` table for 10K workspaces in a single loop will eventually timeout. Partition the table by `workspace_id` and dispatch parallel or batched queue tasks (`consolidate_workspace(id)`) to background workers instead of looping sequentially.

```
For each workspace L3 memories:
  1. Find duplicates (embedding similarity > 0.95)
  2. Merge via LLM summarization
  3. Flag stale memories (no access in 30+ days)
  4. Generate workspace memory health report
```

Uses the existing `consolidation.py` engine (705 lines, already built for PRD-05).

---

## 8. NL2SQL Integration (New Capability)

### 8.1 Why

Agents need to answer questions about users' **external, connected databases** ("What's our current MRR?" or "How many users signed up last week?") without pre-loading all business data. This operates on user-supplied databases, not the Automatos core system DB.

### 8.2 Design

When a user connects a database, Automatos syncs, indexes the schema, and heavily caches its metadata in Redis. You cannot afford to introspect their schema on every request. The Context Router dynamically injects awareness of these specific external databases (as outlined in 4.3).

Extend PRD-21's safe SQL execution with natural language:

```
User: "How many active users do we have?"
  ↓
Agent calls: query_data(question="How many active users do we have?")
  ↓
NL2SQL Service:
  1. Load cached schema metadata (from PRD-21 DatabaseIntrospectionService)
  2. LLM generates SQL: SELECT COUNT(*) FROM users WHERE last_active > NOW() - INTERVAL '30 days'
  3. SQLValidator validates (SELECT-only, no DDL/DML)
  4. Execute with timeout
  5. Return formatted result
  ↓
Agent: "You currently have 1,247 active users (last 30 days)."
```

**Safety:**
- Read-only (SQLValidator enforces SELECT only)
- Schema allowlist (only expose permitted tables)
- Query audit trail (existing `database_query_audit` table)
- Per-query timeout (5 seconds)
- Row limit (1000 rows max)

### 8.3 Tool Definition

```python
{
    "name": "query_data",
    "description": "Query business data using natural language. Use this when users ask about metrics, counts, trends, or any data that lives in the application database.",
    "parameters": {
        "question": {"type": "string", "description": "Natural language question about business data"},
        "database_id": {"type": "integer", "description": "Database source ID (optional, uses default)"}
    }
}
```

---

## 9. Phased Rollout

### Phase 1: Foundation (Week 1-2)
**Goal:** Single memory service, Redis session layer, fix fragmentation

- [ ] Create `UnifiedMemoryService` singleton with shared Mem0Client
- [ ] Implement L1 Redis session store (store/retrieve/expire)
- [ ] Migrate SmartMemoryManager to delegate to UnifiedMemoryService
- [ ] Migrate RecipeMemoryService to use shared service
- [ ] Migrate platform executor, widget, memory stats to shared service
- [ ] Delete deprecated `MemoryInjector` class
- [ ] Standardize user_id format across all consumers
- [ ] Add Redis caching of L3 Mem0 results (5min TTL)

**Outcome:** All consumers use ONE service. Session continuity works. 5x faster repeated queries.

### Phase 2: Context Router (Week 3)
**Goal:** Intelligent pre-fetching, knowledge awareness

- [ ] Build Context Router (temporal detection, session hydration, knowledge awareness)
- [ ] Replace hardcoded memory retrieval in SmartChatOrchestrator with Context Router
- [ ] Add capability map injection to system prompt
- [ ] Wire daily logs through Context Router (not separate injection)
- [ ] Add context budget allocation (configurable per model)

**Outcome:** Agents know what they can look up. Context is assembled intelligently.

### Phase 3: Layered Storage (Week 4-5)
**Goal:** Graduated importance, decay, promotion

- [ ] Create `memory_short_term` Postgres table (L2)
- [ ] Wire L2 storage for all exchanges (raw, `infer=False`)
- [ ] Implement Ebbinghaus decay scoring (hourly job)
- [ ] Implement L2→L3 promotion pipeline (daily job)
- [ ] Wire consolidation engine (existing 705-line implementation)
- [ ] Add memory health dashboard metrics

**Outcome:** Save everything short-term, promote what matters to long-term.

### Phase 4: NL2SQL + Knowledge Graph (Week 6-7)
**Goal:** Agents can query live data

- [ ] Build NL2SQL service (wraps PRD-21 safe SQL + LLM generation)
- [ ] Register `query_data` tool in ActionRegistry
- [ ] Add schema caching and query audit
- [ ] Wire into Context Router L4 awareness
- [ ] Integration tests with sample business queries

**Outcome:** Agents answer data questions from live database.

### Phase 5: Scale & Optimize (Week 8+)
**Goal:** Production-ready for 10K users

- [ ] Redis cluster for L1 (if single instance insufficient)
- [ ] Postgres partitioning for L2 by workspace_id
- [ ] Mem0 connection pooling and request batching
- [ ] Context Router A/B testing (measure retrieval quality)
- [ ] Memory dashboard showing layer distribution and health
- [ ] Performance benchmarks: p50/p95/p99 for context assembly

**Outcome:** Proven at scale with measurable quality metrics.

---

## 10. What Gets Deleted / Deprecated

| File | Action | Reason |
|------|--------|--------|
| `modules/memory/operations/injection.py` | DELETE | Dead code, replaced by SmartChatOrchestrator |
| `modules/memory/service.py` (HierarchicalMemoryManager) | ABSORB | Decay/promotion logic moves into UnifiedMemoryService |
| `modules/memory/types/memory_types.py` | ABSORB | MemoryLevel enum → L0-L4 constants |
| `api/widget_memory.py` standalone client | REFACTOR | Use UnifiedMemoryService instead of own Mem0Client |
| `SmartMemoryManager` class | REFACTOR → thin wrapper | Delegates to UnifiedMemoryService |
| `RecipeMemoryService` class | REFACTOR → thin wrapper | Delegates to UnifiedMemoryService |
| All inline `Mem0Client()` instantiation | DELETE | Use singleton from UnifiedMemoryService |

**No code gets orphaned.** Every deletion is replaced by the unified service.

---

## 11. Conflicts Resolved

| Conflict | Resolution |
|----------|------------|
| PRD-05 (hierarchical) vs PRD-39 (Mem0 flat) | **Both.** L2 uses Postgres (PRD-05 decay logic), L3 uses Mem0 (PRD-39 fact extraction). Not competing — complementary layers. |
| PRD-03 (knapsack) vs PRD-69 (iterative retrieval) | **Context Router Phase 2.** Single-pass by default, iterative for MULTI_STEP intents only. |
| PRD-08 (cognitive formatting) vs PRD-69 (phase-aware compaction) | **Both.** PRD-08 formats retrieved chunks. PRD-69 D.2 compacts conversation history. Different concerns. |
| 5 user_id formats | **ONE format** with namespace prefixes via UnifiedMemoryService. |
| 5 Mem0Client instances | **ONE singleton** with shared circuit breaker and connection pool. |
| Double memory injection | **Eliminated.** Context Router is the single injection point. |

---

## 12. Success Metrics

| Metric | Current | Target (Phase 1) | Target (Phase 5) |
|--------|---------|-------------------|-------------------|
| Context assembly latency (p50) | 500ms | 60ms | 30ms |
| Context assembly latency (p95) | 1000ms | 200ms | 100ms |
| Session continuity | 0% (lost on close) | 100% (24hr) | 100% (configurable) |
| Memory retrieval relevance | Unknown | Baseline measured | >0.7 cosine similarity |
| Mem0 requests per chat message | 2-3 (cold) | 0.2 (cached) | 0.1 |
| Concurrent users supported | ~100 | ~1,000 | ~10,000 |
| Memory consumers using shared service | 0/6 | 6/6 | 6/6 |
| Cross-session context accuracy | 0% | 70% | 90% |

---

## 13. Dependencies

| Dependency | Status | Blocker? |
|------------|--------|----------|
| Redis on Railway | Deployed | No |
| Mem0 on Railway | Deployed | No |
| Postgres on Railway | Deployed | No |
| PRD-21 (Database Knowledge) | MVP built | No (Phase 4 only) |
| PRD-08 (RAG v2) | Complete | No |
| PRD-77 Phase 4 (memory bugs) | Partial | Yes — fix before Phase 1 |
| PRD-69 (instincts) | Design only | No (Phase 3+ integration) |
| Consolidation engine | Built (705 lines) | No — ready to wire |
| HierarchicalMemoryManager decay | Built (401 lines) | No — ready to absorb |

---

## 14. Risk Register

| Risk | Impact | Mitigation |
|------|--------|------------|
| Redis becomes SPOF for all memory | High | Graceful degradation: if Redis down, skip L1/cache, hit Mem0 directly (current behavior) |
| Mem0 fact extraction quality varies | Medium | Store raw in L2 always; L3 extraction is bonus, not sole source |
| NL2SQL generates unsafe queries | High | SQLValidator + schema allowlist + audit trail + timeouts |
| Singleton DB Session Leak | Critical | Ensure `UnifiedMemoryService` acquires DB sessions per-request from an async pool, never holding a single session globally. |
| Synchronous L2/L3 Writes | High | Push L2/L3 storage operations to a background queue (Celery/ARQ) so TTFT (Time To First Token) doesn't suffer during the main chat request cycle. |
| Temporal Detection Latency | Medium | Keep Context Router temporal checks fast and regex-driven to stay under the 10ms budget; avoid heavy NLP models here. |
| Migration breaks existing memory | High | Phase 1 is additive — UnifiedMemoryService wraps existing code first, replaces later |
| Over-caching stale context | Medium | Cache invalidation on write + short TTLs (5min L3 cache) |
| Consolidation job runs too long at scale | Medium | Partition by workspace, process in batches, configurable concurrency |

---

## 15. Open Questions

1. **Should L2 short-term also use Mem0?** Or is Postgres + time-based queries sufficient without vector search?
2. **Memory export/import** — should users be able to download their agent's memories? GDPR compliance?
3. **Cross-workspace memory** — should an enterprise org share certain memories across workspaces?
4. **Memory quotas** — at what point do we limit storage per workspace/plan tier?
5. **Agent-to-agent memory transfer** — when Agent B is created from Agent A's template, copy memories?
