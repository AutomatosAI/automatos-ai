# UnifiedMemoryService

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docs/PRDS/55-AUTONOMOUS-ASSISTANT-PLATFORM.md](docs/PRDS/55-AUTONOMOUS-ASSISTANT-PLATFORM.md)
- [frontend/components/auth/sign-up-form.tsx](frontend/components/auth/sign-up-form.tsx)
- [orchestrator/alembic/versions/20260215_add_heartbeat_and_channels.py](orchestrator/alembic/versions/20260215_add_heartbeat_and_channels.py)
- [orchestrator/alembic/versions/prd123_checkpoint_count.py](orchestrator/alembic/versions/prd123_checkpoint_count.py)
- [orchestrator/api/channels.py](orchestrator/api/channels.py)
- [orchestrator/api/heartbeat.py](orchestrator/api/heartbeat.py)
- [orchestrator/api/missions.py](orchestrator/api/missions.py)
- [orchestrator/channels/base.py](orchestrator/channels/base.py)
- [orchestrator/channels/discord_adapter.py](orchestrator/channels/discord_adapter.py)
- [orchestrator/channels/google_chat_adapter.py](orchestrator/channels/google_chat_adapter.py)
- [orchestrator/channels/line_adapter.py](orchestrator/channels/line_adapter.py)
- [orchestrator/channels/manager.py](orchestrator/channels/manager.py)
- [orchestrator/channels/slack_adapter.py](orchestrator/channels/slack_adapter.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/consumers/chatbot/smart_memory.py](orchestrator/consumers/chatbot/smart_memory.py)
- [orchestrator/core/context_guard.py](orchestrator/core/context_guard.py)
- [orchestrator/core/models/channels.py](orchestrator/core/models/channels.py)
- [orchestrator/core/models/orchestration.py](orchestrator/core/models/orchestration.py)
- [orchestrator/core/models/orchestration_enums.py](orchestrator/core/models/orchestration_enums.py)
- [orchestrator/core/services/plugin_security_scanner.py](orchestrator/core/services/plugin_security_scanner.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/modules/agents/__init__.py](orchestrator/modules/agents/__init__.py)
- [orchestrator/modules/agents/factory/__init__.py](orchestrator/modules/agents/factory/__init__.py)
- [orchestrator/modules/coordination/dispatcher.py](orchestrator/modules/coordination/dispatcher.py)
- [orchestrator/modules/coordination/planner.py](orchestrator/modules/coordination/planner.py)
- [orchestrator/modules/coordination/reconciler.py](orchestrator/modules/coordination/reconciler.py)
- [orchestrator/modules/memory/context_router.py](orchestrator/modules/memory/context_router.py)
- [orchestrator/modules/memory/integrations/mem0_client.py](orchestrator/modules/memory/integrations/mem0_client.py)
- [orchestrator/modules/memory/unified_memory_service.py](orchestrator/modules/memory/unified_memory_service.py)
- [orchestrator/modules/tools/discovery/action_registry.py](orchestrator/modules/tools/discovery/action_registry.py)
- [orchestrator/modules/tools/execution/concurrency.py](orchestrator/modules/tools/execution/concurrency.py)
- [orchestrator/services/checkpoint_service.py](orchestrator/services/checkpoint_service.py)
- [orchestrator/services/coordinator_service.py](orchestrator/services/coordinator_service.py)
- [orchestrator/services/orchestration_state.py](orchestrator/services/orchestration_state.py)
- [orchestrator/tests/test_budget_gate.py](orchestrator/tests/test_budget_gate.py)
- [orchestrator/tests/test_dispatcher_parallel.py](orchestrator/tests/test_dispatcher_parallel.py)
- [orchestrator/tests/test_unified_memory.py](orchestrator/tests/test_unified_memory.py)
- [scripts/ralph/IMPLEMENTATION_PLAN.md](scripts/ralph/IMPLEMENTATION_PLAN.md)
- [scripts/ralph/progress.txt](scripts/ralph/progress.txt)

</details>



The `UnifiedMemoryService` is the centralized memory management service that provides a single entry point for all memory operations across the Automatos AI platform. It replaces fragmented `Mem0Client` instances with a shared service managing a 5-layer memory stack (L0–L4), ensuring consistent workspace scoping and preventing the `user_id` format inconsistencies that previously led to cross-tenant data leaks. [orchestrator/modules/memory/unified_memory_service.py:1-21]().

**Scope**: This page covers the `UnifiedMemoryService` API, the `MemoryNamespace` helper, memory tier operations (L1 sessions, L2 short-term, L3 long-term), and integration patterns.

---

## Architecture Overview

The `UnifiedMemoryService` orchestrates five memory tiers, each serving a distinct purpose in the agent's cognitive architecture:

**Five-Layer Memory Stack**

```mermaid
graph TB
    subgraph L0["L0: Focus (Context Window)"]
        [CurrentConv] -- "No storage - lives in LLM prompt" --> [L0]
    end
    
    subgraph L1["L1: Working Memory (Redis)"]
        [SessionMemory] -- "24hr TTL (MEMORY_SESSION_TTL_SECONDS)" --> [Redis]
        [SessionOps] -- "get_session / update_session" --> [L1]
    end
    
    subgraph L2["L2: Short-term Memory (Postgres)"]
        [memory_items] -- "Ebbinghaus decay (MEMORY_DECAY_RATE)" --> [PostgreSQL]
        [STOps] -- "store_short_term / search_short_term" --> [L2]
    end
    
    subgraph L3["L3: Long-term Memory (Mem0)"]
        [Mem0Client] -- "Fact extraction + semantic search" --> [Mem0API]
        [Mem0Cache] -- "Redis cache (MEMORY_CACHE_TTL_SECONDS)" --> [L3]
        [Mem0Ops] -- "store_long_term / search_long_term" --> [L3]
    end
    
    subgraph L4["L4: Org Knowledge (RAG/NL2SQL)"]
        [Tools] -- "search_knowledge / query_database" --> [L4]
    end
    
    [L1] -- "consolidate on end_session" --> [L2]
    [L2] -- "promote on importance" --> [L3]
    [L3] -- "Key: mem:cache:ws:scope:hash" --> [Mem0Cache]
```

**Sources**: [orchestrator/modules/memory/unified_memory_service.py:1-21](), [orchestrator/config.py:82-118]()

---

## MemoryNamespace: Standardized Scoping

The `MemoryNamespace` class is a frozen dataclass used to build standardized, scoped `user_id` strings for Mem0 and Redis keys. All memory consumers MUST use this helper to prevent the naming inconsistencies documented in PRD-79. [orchestrator/modules/memory/unified_memory_service.py:38-46]().

### Namespace Formats

| Scope | Method | Format | Example |
|-------|--------|--------|---------|
| Workspace-wide | `workspace()` | `mem:{workspace_id}` | `mem:ws-abc` |
| Agent-specific | `agent(agent_id)` | `mem:{workspace_id}:agent:{agent_id}` | `mem:ws-abc:agent:42` |
| Recipe learnings | `recipe(recipe_id)` | `mem:{workspace_id}:recipe:{recipe_id}` | `mem:ws-abc:recipe:10` |
| Workflow | `workflow(workflow_id)` | `mem:{workspace_id}:workflow:{workflow_id}` | `mem:ws-abc:workflow:99` |
| Daily logs | `daily()` | `mem:{workspace_id}:daily` | `mem:ws-abc:daily` |
| L1 session | `session(conversation_id)` | `mem:session:{workspace_id}:{conversation_id}` | `mem:session:ws-abc:c1` |
| L3 cache | `cache_key(agent_id, hash)` | `mem:cache:{workspace_id}:{scope}:{hash}` | `mem:cache:ws-abc:42:abc` |

**Sources**: [orchestrator/modules/memory/unified_memory_service.py:34-117](), [orchestrator/tests/test_unified_memory.py:133-178]()

---

## L3: Long-term Memory (Mem0)

L3 stores facts extracted from conversations via Mem0's LLM-powered fact extraction. The `UnifiedMemoryService` manages a shared `Mem0Client` and handles result caching in Redis. [orchestrator/modules/memory/unified_memory_service.py:178-182]().

### Mem0 Client Integration
The `Mem0Client` interacts with the Mem0 server using an internal circuit breaker and exponential backoff to ensure reliability. [orchestrator/modules/memory/integrations/mem0_client.py:66-141](). It converts message lists into a single text string for fact extraction. [orchestrator/modules/memory/integrations/mem0_client.py:158-165]().

**Sources**: [orchestrator/modules/memory/integrations/mem0_client.py:66-141](), [orchestrator/modules/memory/integrations/mem0_client.py:143-165]()

### search_long_term
This method performs a semantic search against Mem0. It computes a `query_hash` to check the Redis cache before making an external API call. [orchestrator/modules/memory/unified_memory_service.py:363-423]().

```python
# Implementation detail: cache lookup before Mem0 search
query_hash = hashlib.sha256(query.lower().encode()).hexdigest()[:16]
cache_key = ns.cache_key(agent_id, query_hash)
cached = await redis.get(cache_key)
if cached:
    return json.loads(cached)
```

**Sources**: [orchestrator/modules/memory/unified_memory_service.py:363-423](), [orchestrator/tests/test_unified_memory.py:159-165]()

### store_long_term
Stores content as facts in Mem0. After storage, it invalidates the workspace search cache using a pattern match `mem:cache:{workspace_id}:*` to ensure subsequent searches reflect the new data. [orchestrator/modules/memory/unified_memory_service.py:311-362]().

**Sources**: [orchestrator/modules/memory/unified_memory_service.py:311-362]()

---

## L2: Short-term Memory (Postgres)

L2 uses the `memory_items` table in PostgreSQL with `pgvector` for semantic retrieval of recent exchanges. [orchestrator/modules/memory/unified_memory_service.py:11-12]().

### Data Model: MemoryItem
- `importance`: Float (0.0 to 1.0) used for promotion logic. [orchestrator/config.py:106-107]().
- `decay_rate`: Float (default 0.1) used for Ebbinghaus forgetting curve. [orchestrator/config.py:100-101]().
- `access_count`: Integer tracking how often a memory is retrieved. [orchestrator/config.py:109]().

**Sources**: [orchestrator/config.py:100-111](), [orchestrator/modules/memory/unified_memory_service.py:11-12]()

---

## L1: Working Memory (Redis Sessions)

L1 maintains per-conversation session state in Redis. This allows agents to maintain context across browser refreshes or within a long-running chat session. [orchestrator/modules/memory/unified_memory_service.py:123-130]().

### SessionMemory Dataclass
Stores the current state of a conversation:
- `summary`: A running summary of the chat.
- `decisions`: A list of agreed-upon points.
- `action_items`: Tasks identified during the session.
- `exchange_count`: Number of messages processed.

**Sources**: [orchestrator/modules/memory/unified_memory_service.py:123-148]()

### Lifecycle Methods
- `update_session`: Increments exchange count and refreshes the Redis TTL. [orchestrator/modules/memory/unified_memory_service.py:1019-1120]().
- `end_session`: Marks the session as ended and reduces TTL to the consolidation window (default 1 hour), allowing background jobs to move data to L2. [orchestrator/config.py:86-87]().

**Sources**: [orchestrator/modules/memory/unified_memory_service.py:1019-1120](), [orchestrator/config.py:86-87]()

---

## Integration with Coordination Services

The `CoordinatorService` for missions uses a shared context backend (vector field or Redis) for inter-agent context. [orchestrator/services/coordinator_service.py:84-105](). When a task completes, its output is injected into the shared field to be accessible by other agents in the mission. [orchestrator/services/coordinator_service.py:176-198]().

**Sources**: [orchestrator/services/coordinator_service.py:84-105](), [orchestrator/services/coordinator_service.py:176-198]()

---

## Context Routing and Budgeting

Retrieval is budget-constrained per the `Config` settings. Tokens are allocated to different memory sources to prevent prompt overflow. [orchestrator/config.py:90-97]().

### Token Budget Allocation

| Memory Layer | Config Variable | Default Tokens |
|--------------|-----------------|----------------|
| L1 Session | `CONTEXT_BUDGET_SESSION` | 500 |
| L3 Long-term | `CONTEXT_BUDGET_LONG_TERM` | 800 |
| L2 Temporal | `CONTEXT_BUDGET_TEMPORAL` | 600 |
| Daily Logs | `CONTEXT_BUDGET_DAILY` | 400 |
| Awareness | `CONTEXT_BUDGET_AWARENESS` | 200 |

**Sources**: [orchestrator/config.py:90-97](), [orchestrator/modules/memory/unified_memory_service.py:1226-1415]()

---

## Platform Tools for Memory

The platform exposes memory management capabilities to agents via the `ActionRegistry`. High-value actions like `platform_search_memory` and `platform_store_memory` are promoted to first-class OpenAI tool schemas. [orchestrator/modules/tools/discovery/action_registry.py:119-135]().

**Sources**: [orchestrator/modules/tools/discovery/action_registry.py:119-135](), [scripts/ralph/progress.txt:68-76]()

---