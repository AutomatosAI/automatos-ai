# PRD-64: Unified Action Discovery & Platform Self-Awareness

**Version:** 2.0 (Revised)
**Status:** Draft — Revised
**Priority:** P1
**Author:** Automatos AI Platform Team
**Created:** 2026-02-23
**Revised:** 2026-02-23
**Dependencies:** PRD-17 (Dynamic Tool Assignment — COMPLETE), PRD-36 (Composio Integration — COMPLETE)
**Pre-requisite:** Dead code cleanup (DONE), API endpoint audit (`docs/API-AUDIT.md` — DONE)

---

## Changes from v1.0

| What Changed | Why |
|---|---|
| Corrected "530 endpoints" → **564 endpoints across 76 files** | Actual count from codebase scan |
| Corrected "~10 internal tools always loaded" → **19 registered tools** in ToolRegistry | Actual count from `tool_registry.py` |
| Documented the **5-layer intent system** that already exists (was not mentioned in v1.0) | AutoBrain + SmartIntentClassifier + SmartToolRouter + PromptAnalyzer + ComposioToolService already do intent-based filtering |
| Changed `api_call` executor from HTTP calls → **service-layer calls** | Calling localhost HTTP from within the same process is wasteful; call service methods directly |
| Dropped composite actions | Let the LLM chain individual tools; add composites later only if specific patterns prove unreliable |
| Dropped Phase 4 (Playbooks) | Separate PRD scope |
| Dropped API cleanup from PRD scope | Now a separate pre-requisite (completed in API-AUDIT.md) |
| Corrected embedding config | Model: `qwen/qwen3-embedding-8b` via OpenRouter, 2048 dimensions |
| Fixed file paths | Several paths in v1.0 were wrong (e.g., embedding manager, vector store) |
| Reduced from 4 phases → **2 phases** | More focused, less speculative |

---

## Executive Summary

Auto discovers external tools (Gmail, Slack, GitHub) dynamically via Composio SDK semantic search. Internal tools (RAG, CodeGraph, Memory, NL2SQL) are already filtered by intent through the SmartToolRouter, but use keyword matching, not semantic search. Automatos's own 564 API endpoints — agent creation, recipe management, analytics — are invisible to Auto.

This PRD extends the **existing SmartToolRouter** with embedding-based ranking for internal tools, and adds a **Platform Action Registry** so Auto can perform platform operations (create agents, query analytics, manage recipes) through direct service-layer calls.

### What This Is NOT

- NOT a rewrite of Composio integration — Composio SDK remains the external tool discovery path
- NOT a new tool execution engine — `UnifiedToolExecutor` and `AgentPlatformTools` stay as-is
- NOT a parallel "UnifiedActionDiscovery" service — we extend SmartToolRouter
- NOT exposing all 564 endpoints — we curate ~50-60 high-value actions

---

## 1. Current Architecture (What Actually Exists)

### 1.1 The 5-Layer Intent System

The chatbot pipeline already has significant tool-routing intelligence. The v1.0 PRD understated this:

```
User Message
    │
    ▼
[Layer 1] AutoBrain (orchestrator/consumers/chatbot/auto.py)
    │  Complexity assessment: Atom → Molecule → Cell → Organ → Organism
    │  Agent selection: matches required tools to workspace agents
    │  Decides: RESPOND (Auto handles) | DELEGATE (sub-agent) | WORKFLOW
    │
    ▼
[Layer 2] SmartIntentClassifier (orchestrator/consumers/chatbot/intent_classifier.py)
    │  9 intent categories with regex-based classification:
    │  GREETING, CHITCHAT, MEMORY_RECALL, DATA_QUERY, SEARCH,
    │  EXTERNAL_ACTION, CREATION, MULTI_STEP, FACTUAL
    │  Returns: requires_tools (bool), suggested_tools (list), confidence (float)
    │
    ▼
[Layer 3] SmartToolRouter (orchestrator/consumers/chatbot/smart_tool_router.py)
    │  Maps intent → tool categories → filtered tool list
    │  CORE_TOOLS always included: search_knowledge, semantic_search, etc.
    │  Limits to 15 tools max. Sets tool_choice: auto | required | none
    │  *** THIS IS THE INSERTION POINT FOR EMBEDDING-BASED RANKING ***
    │
    ▼
[Layer 4] PromptAnalyzer (orchestrator/consumers/chatbot/prompt_analyzer.py)
    │  Legacy: rank_tools_for_query() uses token-overlap scoring
    │  Also: is_simple_message(), detect_tool_intent(), extract_search_terms()
    │  Overlaps with SmartIntentClassifier — candidate for consolidation
    │
    ▼
[Layer 5] ComposioToolService (orchestrator/modules/tools/services/composio_tool_service.py)
    3-tier Composio discovery:
    ├── Tier 1: Explicit actions from agent config
    ├── Tier 2: SDK semantic search (find_actions_by_use_case, limit=30)
    └── Tier 3: Ranked cache fallback
```

### 1.2 Tool Registry (19 Tools)

`orchestrator/modules/tools/registry/tool_registry.py` — single source of truth for internal tools:

| Category | Tools | Count |
|----------|-------|-------|
| Research | search_knowledge, semantic_search, search_codebase, search_tables, search_images, search_formulas, search_multimodal | 7 |
| Database | query_database, smart_query_database | 2 |
| File Operations | read_file, write_file, delete_file, list_directory, create_directory, generate_document | 6 |
| Shell | execute_command | 1 |
| API/HTTP | http_request, composio_execute | 2 |
| SSH | ssh_execute | 1 |

Each tool has: `ToolSpec` with name, category, description, parameters (OpenAI format), security level, executor class/method.

### 1.3 Tool Execution Chain (Unchanged by This PRD)

```
LLM returns tool_call
    │
    ├── Composio action prefix? → ComposioToolService.execute() → external API
    │
    ├── In ToolRegistry? → UnifiedToolExecutor routes to:
    │   ├── AgentPlatformTools.execute_tool()
    │   │   ├── search_knowledge → RAGService.retrieve_context()
    │   │   ├── search_codebase → CodeGraphService.search_symbols()
    │   │   ├── generate_document → DocumentGenerationService
    │   │   └── etc.
    │   ├── _execute_database_tool() → NL2SQLService
    │   ├── _execute_http_request() → internal HTTP
    │   └── _execute_ssh() → SSH
    │
    └── Unknown → error response
```

### 1.4 Context Filtering (Already Working)

The `ToolRegistry.validate_tool_access()` method (line 1198) already implements:
- **Context map**: `general`, `coding`, `ops`, `communication`, `research` → allowed tool categories
- **Composio gating**: Only expose `composio_execute` if agent has connected external apps
- **Agent 1 (Chatbot) forced to `general`**: Prevents shell/SSH tool exposure in general chat

### 1.5 Embedding Infrastructure

`orchestrator/core/llm/embedding_manager.py`:
- Model: `qwen/qwen3-embedding-8b` via OpenRouter
- Dimensions: 2048
- Used by: RAG ingestion, semantic search, document chunking

### 1.6 Key File Locations

| Component | File |
|-----------|------|
| AutoBrain (complexity + agent routing) | `orchestrator/consumers/chatbot/auto.py` |
| SmartIntentClassifier (9 intents) | `orchestrator/consumers/chatbot/intent_classifier.py` |
| SmartToolRouter (intent → tool filtering) | `orchestrator/consumers/chatbot/smart_tool_router.py` |
| PromptAnalyzer (legacy ranking) | `orchestrator/consumers/chatbot/prompt_analyzer.py` |
| ToolRegistry (19 tools, OpenAI format) | `orchestrator/modules/tools/registry/tool_registry.py` |
| UnifiedToolExecutor (32 execution routes) | `orchestrator/modules/tools/execution/unified_executor.py` |
| AgentPlatformTools (RAG/Code/Doc execution) | `orchestrator/modules/agents/services/agent_platform_tools.py` |
| ComposioToolService (3-tier discovery) | `orchestrator/modules/tools/services/composio_tool_service.py` |
| EmbeddingManager (qwen3, 2048d) | `orchestrator/core/llm/embedding_manager.py` |
| App mount (79 routers) | `orchestrator/main.py` |

---

## 2. Problem Statement

### What Works Well (Don't Touch)

- Composio tool execution: Gmail, Slack, Flights all working end-to-end
- The live pipeline: AutoBrain → SmartIntentClassifier → SmartToolRouter → ComposioToolService → LLM
- Context filtering (shell/SSH tools denied in general context)
- `is_simple_message()` gate (greetings skip tool loading entirely)
- 19 registered tools with OpenAI function-calling schemas

### What's Missing

1. **SmartToolRouter uses keyword matching, not semantic matching.** `_filter_tools_by_categories()` checks hardcoded category lists and `_tool_matches_query()` uses substring matching. A query like "help me find information about our deployment process" won't match `search_knowledge` unless it contains the exact keyword "search" or "knowledge."

2. **PromptAnalyzer's `rank_tools_for_query()` duplicates SmartToolRouter.** Both do keyword-based tool matching. Neither uses embeddings. This is wasted computation and potential ranking conflicts.

3. **Platform operations are invisible to Auto.** A user asking "create me an agent" or "what's my token usage?" gets nothing. The APIs exist (564 endpoints), the services exist, but Auto has no tool definitions for them.

4. **No semantic ranking for tool relevance.** When multiple tools match, there's no confidence-scored ranking — just category inclusion. The LLM gets a flat list of all matching tools with no priority signal.

---

## 3. Proposed Changes

### Phase 1: Semantic Tool Ranking in SmartToolRouter

**Goal:** Replace keyword matching with embedding similarity in SmartToolRouter. Same interface, smarter matching.

#### 3.1 Pre-compute Tool Embeddings at Startup

Add to `SmartToolRouter.__init__()`:

```python
class SmartToolRouter:
    def __init__(self):
        self.classifier = get_intent_classifier()
        self._tool_embeddings: Dict[str, List[float]] = {}
        self._embedding_manager = None  # Lazy init

    async def initialize_embeddings(self, tools: List[Dict[str, Any]]):
        """Pre-compute embeddings for all registered tools. Call once at startup."""
        from core.llm.embedding_manager import get_embedding_manager
        self._embedding_manager = get_embedding_manager()

        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            text = f"{name}: {desc}"
            self._tool_embeddings[name] = await self._embedding_manager.generate_embedding(text)
```

#### 3.2 Replace `_filter_tools_by_categories` with Semantic Ranking

```python
async def _rank_tools_by_similarity(
    self,
    query: str,
    all_tools: List[Dict[str, Any]],
    intent: IntentResult,
    max_tools: int = 15
) -> List[Dict[str, Any]]:
    """Rank tools by embedding similarity to query, with intent-based boosting."""
    if not self._tool_embeddings:
        # Fallback to keyword matching if embeddings not initialized
        return self._filter_tools_by_categories(all_tools, ...)

    query_embedding = await self._embedding_manager.generate_embedding(query)

    scored = []
    for tool in all_tools:
        name = tool.get("function", {}).get("name", "")
        if name not in self._tool_embeddings:
            continue

        score = cosine_similarity(query_embedding, self._tool_embeddings[name])

        # Boost suggested tools from intent classifier
        if name in intent.suggested_tools:
            score += 0.15

        # Boost core tools slightly
        if name in self.CORE_TOOLS:
            score += 0.05

        scored.append((tool, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [tool for tool, score in scored[:max_tools]]
```

#### 3.3 Consolidate PromptAnalyzer Overlap

`PromptAnalyzer.rank_tools_for_query()` becomes redundant once SmartToolRouter uses embeddings. The consolidation:

| PromptAnalyzer Method | Action |
|---|---|
| `is_simple_message()` | Keep — used by streaming service as fast gate |
| `detect_tool_intent()` | Remove — SmartIntentClassifier supersedes |
| `rank_tools_for_query()` | Remove — SmartToolRouter embeddings supersede |
| `extract_search_terms()` | Keep — used for RAG query refinement |
| `convert_to_llm_messages()` | Keep — message formatting |
| `parse_explicit_tool_call()` | Keep — explicit tool syntax |

#### 3.4 Files Changed

| File | Change |
|------|--------|
| `orchestrator/consumers/chatbot/smart_tool_router.py` | Add embedding initialization + semantic ranking |
| `orchestrator/consumers/chatbot/prompt_analyzer.py` | Remove `rank_tools_for_query()`, `detect_tool_intent()` |
| `orchestrator/consumers/chatbot/service.py` (or caller) | Call `smart_tool_router.initialize_embeddings()` at startup |

#### 3.5 Success Criteria

- "help me find information about deployments" → `search_knowledge` ranked #1 (not missed due to no keyword match)
- "analyze the codebase structure" → `search_codebase` ranked above `search_knowledge`
- "send an update to the team" → `composio_execute` ranked #1 (Composio actions benefit too)
- Discovery latency: <200ms (embeddings pre-computed, only query embedding is live)
- Zero regression in existing tool execution

---

### Phase 2: Platform Action Registry

**Goal:** Auto can perform platform operations via curated service-layer actions.

#### 3.6 ActionDefinition Dataclass

```python
@dataclass
class ActionDefinition:
    """A discoverable platform action."""
    name: str                          # "create_agent"
    description: str                   # Semantic description for embedding search
    category: str                      # "agents", "analytics", "recipes", etc.
    parameters_schema: Dict[str, Any]  # OpenAI function-calling format

    # Execution routing — service-layer, NOT HTTP
    service_class: str                 # "AgentService"
    service_method: str                # "create_agent"
    parameter_adapter: Optional[str]   # Function to map tool params → service params

    # Access control
    permission_level: str              # "read" | "write" | "destructive"
    requires_confirmation: bool        # True for write/destructive
    workspace_scoped: bool = True

    # Discovery metadata
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
```

#### 3.7 Service-Layer Executor (Not HTTP)

The `api_call` executor from v1.0 called FastAPI endpoints via HTTP — wasteful within the same process. Instead, each platform action maps to a **service method call**:

```python
class PlatformActionExecutor:
    """Execute platform actions via direct service calls."""

    async def execute(
        self,
        action: ActionDefinition,
        params: Dict[str, Any],
        workspace_id: str,
        db: Session
    ) -> Dict[str, Any]:
        # Import and instantiate the service
        service = self._get_service(action.service_class, db, workspace_id)

        # Adapt parameters if needed
        if action.parameter_adapter:
            params = self._adapt_params(action.parameter_adapter, params)

        # Call the service method
        method = getattr(service, action.service_method)
        result = await method(**params) if asyncio.iscoroutinefunction(method) else method(**params)

        return self._serialize_result(result)
```

This follows the pattern already used by `AgentPlatformTools`, which calls `RAGService`, `CodeGraphService`, etc. directly.

#### 3.8 Curated Platform Actions

Actions are clearly marked as **existing** (wrapping current services) vs **new** (requires new service work):

**Knowledge & Search** (existing — already in ToolRegistry, now also in action registry for semantic ranking)

| Action | Service | Method | Status |
|--------|---------|--------|--------|
| `search_knowledge` | RAGService | `retrieve_context()` | Existing |
| `semantic_search` | RAGService | `semantic_search()` | Existing |
| `search_codebase` | CodeGraphService | `search_symbols()` | Existing |
| `query_database` | NL2SQLService | `query()` | Existing |

**Agent Management** (new actions wrapping existing API services)

| Action | Service | Method | Permission |
|--------|---------|--------|------------|
| `create_agent` | AgentService | `create_agent()` | write |
| `list_agents` | AgentService | `list_agents()` | read |
| `get_agent` | AgentService | `get_agent()` | read |
| `update_agent` | AgentService | `update_agent()` | write |
| `assign_tools_to_agent` | AgentAppAssignment | `assign_apps()` | write |

**Analytics & Reporting** (new actions wrapping existing API logic)

| Action | Service | Method | Permission |
|--------|---------|--------|------------|
| `get_llm_usage_stats` | LLMAnalyticsService | `get_usage_stats()` | read |
| `get_cost_breakdown` | LLMAnalyticsService | `get_cost_breakdown()` | read |
| `get_agent_ranking` | AnalyticsService | `get_agent_ranking()` | read |
| `get_system_health` | (direct) | `health_check()` | read |

**Recipes & Workflows** (new actions wrapping existing API logic)

| Action | Service | Method | Permission |
|--------|---------|--------|------------|
| `list_recipes` | RecipeService | `list_recipes()` | read |
| `create_recipe` | RecipeService | `create_recipe()` | write |
| `execute_recipe` | RecipeService | `execute_recipe()` | write |
| `get_execution_history` | ExecutionHistoryService | `get_history()` | read |

**Document Management** (existing + new)

| Action | Service | Method | Permission |
|--------|---------|--------|------------|
| `generate_document` | DocumentGenerationService | `generate()` | write (existing) |
| `list_documents` | DocumentService | `list_documents()` | read (new) |
| `upload_document` | DocumentService | `upload()` | write (new) |

**Memory** (new actions wrapping existing API logic)

| Action | Service | Method | Permission |
|--------|---------|--------|------------|
| `recall_memory` | MemoryService | `search()` | read |
| `store_memory` | MemoryService | `add()` | write |
| `get_memory_stats` | MemoryService | `get_stats()` | read |

**Workspace** (new actions wrapping existing API logic)

| Action | Service | Method | Permission |
|--------|---------|--------|------------|
| `get_workspace_info` | WorkspaceService | `get_current()` | read |
| `list_team_members` | TeamService | `list_members()` | read |

Total: ~35-40 curated actions (not 60-80 as v1.0 estimated — start smaller, expand based on usage).

#### 3.9 Integration with SmartToolRouter

Platform actions get embedded alongside internal tools in Phase 1's semantic index. The SmartToolRouter treats them identically:

```python
# At startup: embed both ToolRegistry tools AND platform actions
all_tools = registry_tools + platform_action_tools
await smart_tool_router.initialize_embeddings(all_tools)

# At query time: single semantic search returns both types
ranked = await smart_tool_router.route(query, all_tools)
# ranked may contain: search_knowledge, create_agent, GMAIL_SEND_EMAIL
```

#### 3.10 Permission Model

| Level | Behavior | Example |
|-------|----------|---------|
| `read` | Execute immediately, return results | list_agents, get_llm_usage_stats |
| `write` | Auto describes action, then executes | create_agent, upload_document |
| `destructive` | Auto asks for explicit user confirmation | delete_agent, clear_memory |

Implementation: Check `action.permission_level` before execution. For `write`, include a brief description in the response. For `destructive`, return a confirmation prompt and wait for user approval.

#### 3.11 Files Changed

| File | Change |
|------|--------|
| New: `orchestrator/modules/tools/discovery/action_registry.py` | `ActionDefinition` dataclass + registry |
| New: `orchestrator/modules/tools/discovery/platform_actions.py` | Curated action definitions |
| New: `orchestrator/modules/tools/discovery/platform_executor.py` | Service-layer executor |
| Modified: `orchestrator/consumers/chatbot/smart_tool_router.py` | Include platform actions in embedding index |
| Modified: `orchestrator/modules/tools/execution/unified_executor.py` | Route `platform_action` type to PlatformActionExecutor |

#### 3.12 Success Criteria

- "Create me an agent" → `create_agent` discovered and executed
- "What's my token usage this month?" → `get_llm_usage_stats` discovered
- "List my recipes" → `list_recipes` discovered
- All actions scoped to user's workspace
- Write/destructive actions properly gated
- <200ms discovery latency

---

## 4. Security Model

### 4.1 Workspace Isolation

Every platform action executes with the workspace_id from the current session. Service methods already enforce workspace scoping (same as API endpoints).

### 4.2 Permission Enforcement

- Read actions: immediate execution
- Write actions: Auto describes intent, then executes
- Destructive actions: explicit user confirmation required
- Rate limit: max 10 write actions per conversation turn

### 4.3 Audit Trail

All platform action executions log to existing analytics:
- User/session context
- Action name and parameters (credentials redacted)
- Result status
- Workspace context

---

## 5. Success Metrics

### Phase 1 (Semantic Ranking)
- Tool relevance: correct tool in top-3 for >90% of test queries
- Discovery latency: <200ms (query embedding only; tool embeddings pre-computed)
- Zero regression in Composio tool execution
- PromptAnalyzer deduplication complete

### Phase 2 (Platform Actions)
- 35-40 platform actions registered and discoverable
- Auto creates agents, queries stats, manages recipes via actions
- Write actions properly gated with descriptions
- Destructive actions require confirmation
- All actions workspace-scoped

---

## 6. Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Embedding quality for short tool descriptions | Medium | Concatenate name + description + tags + examples for richer text |
| Startup time for embedding 50+ tools | Low | ~2-3 seconds for 50 embeddings; async, non-blocking |
| Service methods may not exist for all curated actions | Medium | Audit services before registering; mark actions as `requires_new_service` |
| Platform actions competing with Composio for same intent | Low | Context boost for workspace-scoped tools; clear description differentiation |
| Write/destructive actions executed without awareness | High | Permission levels + confirmation; audit logging |
| Action registry becomes stale as services evolve | Medium | Startup validation that registered service methods exist |

---

## 7. Implementation Order

1. **Phase 1: Semantic Ranking** (~10-15 hours)
   - Pre-compute tool embeddings at startup
   - Replace keyword matching in SmartToolRouter with cosine similarity
   - Remove duplicate methods from PromptAnalyzer
   - Test with existing tool set

2. **Phase 2: Platform Actions** (~15-20 hours)
   - `ActionDefinition` dataclass + registry
   - Service-layer executor
   - Curate first 20 actions (agents, analytics, recipes)
   - Permission model
   - Expand to full 35-40 actions
   - Integration testing

**Total: 25-35 hours** (down from 70-90 in v1.0 — focused scope, no speculative phases)

---

## 8. Relationship to Existing PRDs

| PRD | Relationship |
|-----|-------------|
| **PRD-17** (Dynamic Tool Assignment) | Foundation — ToolRegistry stays, discovery extends it |
| **PRD-36** (Composio Integration) | Parallel — Composio SDK remains external tool path, unchanged |
| **PRD-12** (Playbooks) | Future — separate PRD if needed |
| **PRD-40** (Dynamic Tool Suggestions) | UI layer — can use ranked discovery results |
| **PRD-63** (Document Generation) | Existing tool — `generate_document` already in registry |

---

**Status:** Draft — Revised, Awaiting Review
**Next Steps:** Review revised PRD, then begin Phase 1 with embedding integration in SmartToolRouter
