# Memory Integration

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [orchestrator/alembic/versions/prd206_chat_summary.py](orchestrator/alembic/versions/prd206_chat_summary.py)
- [orchestrator/consumers/chatbot/integration.py](orchestrator/consumers/chatbot/integration.py)
- [orchestrator/consumers/chatbot/smart_memory.py](orchestrator/consumers/chatbot/smart_memory.py)
- [orchestrator/consumers/chatbot/smart_orchestrator.py](orchestrator/consumers/chatbot/smart_orchestrator.py)
- [orchestrator/modules/context/sections/memory.py](orchestrator/modules/context/sections/memory.py)
- [orchestrator/modules/memory/context_router.py](orchestrator/modules/memory/context_router.py)
- [orchestrator/modules/memory/recall_ranking.py](orchestrator/modules/memory/recall_ranking.py)
- [orchestrator/modules/memory/thread_checkpoint.py](orchestrator/modules/memory/thread_checkpoint.py)
- [orchestrator/modules/memory/unified_memory_service.py](orchestrator/modules/memory/unified_memory_service.py)
- [orchestrator/modules/tools/discovery/handlers_search.py](orchestrator/modules/tools/discovery/handlers_search.py)
- [orchestrator/services/memory_archival_job.py](orchestrator/services/memory_archival_job.py)
- [orchestrator/services/memory_jobs.py](orchestrator/services/memory_jobs.py)
- [orchestrator/tests/test_l3_distill_input.py](orchestrator/tests/test_l3_distill_input.py)
- [orchestrator/tests/test_memory_restart_and_isolation.py](orchestrator/tests/test_memory_restart_and_isolation.py)
- [orchestrator/tests/test_memory_single_write_path.py](orchestrator/tests/test_memory_single_write_path.py)
- [orchestrator/tests/test_memory_stored_sse.py](orchestrator/tests/test_memory_stored_sse.py)
- [orchestrator/tests/test_p2w1_semantic_l2_recall.py](orchestrator/tests/test_p2w1_semantic_l2_recall.py)
- [orchestrator/tests/test_prd197_substrate.py](orchestrator/tests/test_prd197_substrate.py)
- [orchestrator/tests/test_prd206_recall_ranking.py](orchestrator/tests/test_prd206_recall_ranking.py)
- [orchestrator/tests/test_prd206_thread_checkpoint.py](orchestrator/tests/test_prd206_thread_checkpoint.py)
- [orchestrator/tests/test_recall_relevance_floor.py](orchestrator/tests/test_recall_relevance_floor.py)
- [orchestrator/tests/test_smart_orchestrator_store_exchange.py](orchestrator/tests/test_smart_orchestrator_store_exchange.py)
- [orchestrator/tests/test_unified_memory.py](orchestrator/tests/test_unified_memory.py)
- [orchestrator/tests/test_us011_context_budgets.py](orchestrator/tests/test_us011_context_budgets.py)

</details>



This page documents how the chat interface integrates with the 5-layer memory system during message processing. It covers memory retrieval (pre-LLM context assembly), storage (post-response persistence), and the flow of data between `SmartChatOrchestrator`, `ContextService`, and `UnifiedMemoryService`.

For the broader memory architecture and L0-L4 layer definitions, see **3. Memory System**. For context assembly mechanics and token budgets, see **4. Context Service**.

---

## Overview

Memory integration in the chat interface operates in two core phases:

1.  **Retrieval Phase** — Before the LLM call, relevant memories are fetched and injected into the system prompt via `MemorySection` or `SmartMemoryManager`. The `ContextRouter` analyzes the query to determine which memory layers (L1 session, L2 temporal, L3 long-term) should be consulted [orchestrator/modules/context/sections/memory.py:52-80]().
2.  **Storage Phase** — After the LLM response completes, the user-assistant exchange is stored across multiple layers: L1 Redis session, L2 Postgres short-term, and L3 long-term store. `SmartMemoryManager` applies classification logic to determine if a memory is `global` (workspace-wide identity) or `agent`-specific (tool usage patterns) [orchestrator/consumers/chatbot/smart_memory.py:109-137]().

Sources: [orchestrator/consumers/chatbot/integration.py:1-21](), [orchestrator/consumers/chatbot/smart_memory.py:63-74](), [orchestrator/modules/memory/unified_memory_service.py:8-21]()

---

## Memory Retrieval Architecture

### Dual-Path Retrieval Strategy and Code Entity Mapping

The chat system uses a prioritized retrieval path through the `ContextService`. The `SmartChatOrchestrator` coordinates this by checking intent and complexity [orchestrator/consumers/chatbot/smart_orchestrator.py:161-188](). If the advanced `ContextRouter` is available, it inspects query semantics via regex signals; otherwise, it falls back to the `SmartMemoryManager` [orchestrator/modules/context/sections/memory.py:75-84]().

**Title: Natural Language Space to Code Entity Space: Memory Retrieval Pipeline**
```mermaid
graph TB
    NLSpace[""Natural Language Space<br/>'What did we discuss last week?'""] --> ContextRouter[""ContextRouter.retrieve_context()<br/>orchestrator/modules/memory/context_router.py""]
    ContextRouter --> AnalyzeQuery[""analyze_query()<br/>-> ContextSignals""]
    AnalyzeQuery --> FetchLayers[""Fetch L1/L2/L3 Layers<br/>based on signals""]
    FetchLayers --> ContextBundle[""ContextBundle<br/>assembled bundle""]
    
    ContextBundle --> MemorySection[""MemorySection.render()<br/>orchestrator/modules/context/sections/memory.py""]
    MemorySection --> SmartMemoryManager[""SmartMemoryManager.retrieve_memories()<br/>orchestrator/consumers/chatbot/smart_memory.py""]
    SmartMemoryManager --> UnifiedService[""UnifiedMemoryService.search_long_term()<br/>orchestrator/modules/memory/unified_memory_service.py""]
```
Sources: [orchestrator/modules/memory/context_router.py:1-24](), [orchestrator/modules/context/sections/memory.py:52-126](), [orchestrator/modules/memory/unified_memory_service.py:1-21]()

---

## ContextService Integration

When `ContextService` is invoked, memory retrieval is encapsulated in the `MemorySection` class.

### MemorySection Render Flow

The `MemorySection` handles the complexity of checking for `skip_memory` flags and coordinating with the `UnifiedMemoryService` [orchestrator/modules/context/sections/memory.py:52-80]().

*   **Skip Logic**: If `skip_memory=True` is passed to the context builder, the section returns an empty string immediately [orchestrator/modules/context/sections/memory.py:61-63]().
*   **Token Budget**: Memory is assigned priority 6 (`priority: int = 6`) [orchestrator/modules/context/sections/memory.py:48-49](). If prompts exceed token limits, `TokenBudgetManager` may trim this section before critical components like `Identity` or `Tools`.
*   **Stashing**: The raw memory text is stashed in the context's `kwargs` as `_memory_context` so it can be exposed via SSE data streams [orchestrator/modules/context/sections/memory.py:95-96]().

Sources: [orchestrator/modules/context/sections/memory.py:1-100]()

---

## Two-Tier Memory Retrieval

`SmartMemoryManager` implements a parallel fetching strategy separating general user facts from agent-specific context.

**Title: Natural Language Space to Code Entity Space: Two-Tier Memory Fetching**
```mermaid
graph TB
    Query[""User Query / Intent<br/>'Remember my preferences'""] --> SmartMemory[""SmartMemoryManager.retrieve_memories()<br/>orchestrator/consumers/chatbot/smart_memory.py""]
    SmartMemory --> WidgetCheck{"widget_mode?"}
    
    WidgetCheck -->|True| AgentOnly[""Agent-Only Retrieval<br/>mem:ws:agent:ID""]
    WidgetCheck -->|False| TwoTier[""Parallel Two-Tier Fetch""]
    
    TwoTier --> GlobalTask[""UnifiedMemoryService.search_long_term(agent_id=None)<br/>Workspace-wide facts""]
    TwoTier --> AgentTask[""UnifiedMemoryService.search_long_term(agent_id=42)<br/>Agent-specific patterns""]
    
    GlobalTask --> Gather[""asyncio.gather()""]
    AgentTask --> Gather
    Gather --> Format["""_format_memories_for_llm()""]
    AgentOnly --> Format
```
Sources: [orchestrator/consumers/chatbot/smart_memory.py:151-195](), [orchestrator/modules/memory/unified_memory_service.py:52-78]()

**Widget Mode Isolation**: When `widget_mode` is active, the system strictly isolates memory to the agent-specific namespace to prevent leaking sensitive workspace-wide information into public-facing widgets [orchestrator/consumers/chatbot/smart_memory.py:157-192]().

Sources: [orchestrator/consumers/chatbot/smart_memory.py:151-195](), [orchestrator/modules/memory/unified_memory_service.py:52-78]()

---

## Memory Storage Flow & Write-Once Contract (G12)

After an LLM response completes, the system initiates a multi-layered persistence pipeline. To prevent double-writing rows to L2, the system enforces a strict write-once-per-layer invariant (PRD-142 W3-S7 / G12) [orchestrator/tests/test_memory_single_write_path.py:1-16]().

### Storage Pipeline Implementation

1.  **Fire-and-Forget Background Execution**: All L1/L2/L3 persistence runs asynchronously via `_spawn_background()` to prevent blocking the streaming response [orchestrator/consumers/chatbot/smart_orchestrator.py:36-53]().
2.  **L1 Working (Redis)**: `SessionMemory` updates current session state in Redis via `_unified_memory.update_session()` [orchestrator/modules/memory/unified_memory_service.py:127-143](), [orchestrator/tests/test_memory_single_write_path.py:99-102]().
3.  **L2 Short-Term (Postgres)**: Transcripts are written exactly once via `memory_manager.store_conversation()` [orchestrator/tests/test_memory_single_write_path.py:143-166](). The legacy direct `store_exchange` L2 write path has been collapsed.
4.  **L3 Long-Term (Durable Store)**: Extracted facts are persisted via `UnifiedMemoryService` [orchestrator/modules/memory/unified_memory_service.py:185-195]().

Sources: [orchestrator/consumers/chatbot/smart_orchestrator.py:36-53](), [orchestrator/tests/test_memory_single_write_path.py:1-166](), [orchestrator/tests/test_smart_orchestrator_store_exchange.py:1-163]()

---

## Daily Summaries & Maintenance

Background scheduling managed by `MemoryJobScheduler` handles long-term maintenance, temporal aggregation, and decay scoring [orchestrator/services/memory_jobs.py:1-22]().

| Task | Frequency | Implementation & Role |
| :--- | :--- | :--- |
| **Consolidation** | Periodic (Configurable) | Contradiction-based merging of near-duplicate L3 memories [orchestrator/services/memory_jobs.py:67-75](). |
| **Decay Scoring** | Hourly | Ebbinghaus retention scoring on L2 rows, archiving items below threshold [orchestrator/services/memory_jobs.py:77-85](). |
| **L2→L3 Promotion** | Daily (Cron) | Promotes high-signal L2 items into the durable L3 store [orchestrator/services/memory_jobs.py:87-96](). |

Sources: [orchestrator/services/memory_jobs.py:1-120]()

---