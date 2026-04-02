# Composio Tool Regression Review

**Date:** 2026-04-02
**Branch:** `task/testCoverage`
**Reviewer:** Claude Opus 4.6
**Status:** For external review (GPT Codex, Gemini)
**Scope:** Why agents stopped using web search and Composio tools; what broke, when, and how to fix it

---

## Executive Summary

Automatos agents have become "dumb" — they no longer use web search (Tavily), Composio integrations (Slack, Gmail, GitHub, etc.), or external tools for research tasks. Instead, they default to internal RAG search only.

**Root cause:** A cascading regression across three PRDs (AgentFactory rewrite, PRD-122, PRD-123) that removed working per-action Composio schemas, doubled the tool count drowning the remaining generic `composio_execute` tool, and rewrote prompts to discourage tool exploration — all without adding web search guidance.

**Critical discovery during review:** The chat streaming service (`service.py`) already has `ComposioToolService` wired in correctly (lines 858-927). The agent factory (`agent_factory.py`) does NOT — it still uses the inferior `ComposioHintService`. Missions, heartbeats, and tasks flow through the agent factory, meaning they are all affected.

**Critical discovery #2 (routing blind spot):** Auto is a router/delegator, not an executor. The real problem is that the **UniversalRouter cannot see which agents have Composio tools**. Router logs show `apps=[none]` for ALL agents because `_build_agent_descriptions()` (engine.py:634-680) queries `AgentAppAssignment` directly but has **no workspace inheritance fallback**. Both `ComposioToolService` and `ComposioHintService` have this fallback (auto-inherit workspace-connected apps when no per-agent assignments exist), but the router does not. Result: "research competitors" routes to PROSPECT (name match) instead of an agent that actually HAS Tavily/web search tools.

---

## Table of Contents

1. [Timeline of the Regression](#1-timeline-of-the-regression)
2. [Architecture: How Tools Reach the LLM](#2-architecture-how-tools-reach-the-llm)
3. [The Four Compounding Failures](#3-the-three-compounding-failures)
4. [File-by-File Evidence](#4-file-by-file-evidence)
5. [What Works vs What Doesn't](#5-what-works-vs-what-doesnt)
6. [PRD Cross-Reference Analysis](#6-prd-cross-reference-analysis)
7. [Composio SDK State of the Art](#7-composio-sdk-state-of-the-art)
8. [The Fourth Failure: Routing Blind Spot](#8-the-fourth-failure-routing-blind-spot)
9. [Proposed Fix Architecture (Updated)](#9-proposed-fix-architecture-updated)
10. [Risk Assessment](#10-risk-assessment)
11. [Open Questions for Reviewers](#11-open-questions-for-reviewers)

---

## 1. Timeline of the Regression

### Before: What Worked (pre-March 11)

Agents had per-action Composio schemas. The LLM saw tools like:
```
COMPOSIO_SEARCH_WEB(query: str)
GMAIL_LIST_EMAILS(label: str, limit: int)
SLACK_SEND_MESSAGE(channel: str, text: str)
```

Each tool had typed parameters. The LLM knew exactly what it could do.

### The Regression Sequence

| Date | Commit | What Changed | Impact |
|------|--------|-------------|--------|
| **Mar 11** | `0ab9e0c6c` | AgentFactory clean rewrite — "ONE path: `get_tools_for_agent()`" | **Deleted** per-action Composio SDK fetch from agent_factory. All agent execution paths (missions, heartbeats, tasks) lost per-action schemas. |
| **Mar 12** | `667df25bd` | Per-action Composio schemas added (different branch) | Added `get_all_schemas_for_apps()` to agent_factory + `composio_actions` dict to unified_executor. **This was the working code.** |
| **Mar 13** | `46f990cf7` | ContextService migration — agent_factory rewired to use ContextService | ContextService calls `get_tools_for_agent()` which returns `composio_execute` (generic), not per-action schemas. Per-action code from Mar 12 was overwritten. |
| **Mar 13** | `82b5a44b2` | ComposioHintService added as compensating patch | Adds action names as prose in system prompt. Inferior to real function schemas — LLM reads hints instead of seeing callable tools. |
| **Mar 24** | `f95693b8c`→`17afce988` | PRD-122: 13 platform actions promoted to first-class schemas | Tool count doubled (~18 → ~33). `composio_execute` now competes with 32 well-described tools. |
| **Mar 24** | `588b4bfa0` | PRD-123: Prompt patterns rewrite | Tool guidance became restrictive ("one tool at a time", "don't spray tool calls"). **No mention of web search.** LLM avoids `composio_execute`. |
| **Mar 24** | `ceb62c543` | PRD-123: Tool tier stratification | System/platform tools bypass access checks. Composio unaffected (neutral). |
| **Apr 1** | `3a1ede4b3` | PRD-125 Phase 0: "Unbreak Chat" attempt | Added web search guidance to personality prompt + intent classifier. Partial fix — chat streaming was already using ComposioToolService. |

### Key Insight: The Mar 12 Code Was Correct

Commit `667df25bd` ("fix: replace generic composio_execute with SDK per-action tool schemas") implemented the correct pattern:

```python
# DELETED CODE from agent_factory.py (commit 667df25bd)
# This was the working implementation:
sdk_schemas = _client.get_all_schemas_for_apps(
    app_names=app_names,
    entity_id=entity_id,
)
for schema in sdk_schemas:
    action_name = schema.get("function", {}).get("name", "")
    if action_name:
        composio_action_set.add(action_name)
        tool_schemas.append(schema)
```

This code was deleted two days later when the ContextService migration (`46f990cf7`) overwrote the file.

---

## 2. Architecture: How Tools Reach the LLM

### Two Execution Paths (Critical Distinction)

```
PATH A: Chat Streaming (service.py)
═══════════════════════════════════
api/chat.py
  → StreamingChatService.stream_response_with_agent()
    → _get_tools()                          # get_tools_for_agent() → base tools
    → _prepare_messages()                   # system prompt + tool orchestration
    → _inject_composio_tools() [LINE 858]   # ← ComposioToolService (PER-ACTION)
      ├─ ComposioToolService.get_tools_for_step()  # SDK semantic search
      ├─ Strips composio_execute, adds per-action schemas
      └─ Falls back to ComposioHintService if SDK returns empty
    → generate_response(tools=use_tools)    # LLM sees per-action tools
    → _run_tool_loop()
      └─ _execute_composio_action() [LINE 1188]  # Direct execution

PATH B: Agent Factory (agent_factory.py) — MISSIONS, HEARTBEATS, TASKS
══════════════════════════════════════════════════════════════════════════
coordinator_service.py / heartbeat_service.py / recipe_executor.py
  → AgentFactory.execute_with_prompt()
    → ContextService.build_context()
      → ToolsSection._load_full()
        → get_tools_for_agent()             # Returns composio_execute (GENERIC)
    → _inject_composio_hints() [LINE 1120]  # ← ComposioHintService (HINTS ONLY)
      ├─ ComposioHintService.build_hints()  # 3-tier hint matching
      ├─ Adds system message with action names (prose, not schemas)
      └─ Constrains composio_execute enum (still generic tool)
    → generate_response(tools=tool_schemas) # LLM sees composio_execute + hints
    → tool loop (max 10 iterations)
      └─ unified_executor.execute_tool()    # Standard dispatch

PATH C: Recipe Executor (recipe_executor.py) — ALREADY CORRECT
══════════════════════════════════════════════════════════════════
  → ContextService.build_context()          # Base tools
  → ComposioToolService.get_tools_for_step() [LINE 169]  # Per-action schemas
  → Strips composio_execute, adds per-action tools
  → Falls back to ComposioHintService
```

### The Problem is Path B

- **Path A (Chat):** Has `ComposioToolService` wired in at `service.py:858-927`. Per-action schemas work.
- **Path B (Agent Factory):** Uses `ComposioHintService` only. Generic `composio_execute` + prose hints.
- **Path C (Recipes):** Has `ComposioToolService` wired in at `recipe_executor.py:169`. Per-action schemas work.

**Missions, heartbeats, and tasks all use Path B.** They get the inferior hint-based approach.

---

## 3. The Four Compounding Failures

### Failure 1: Enrichment Gap in `tool_router.py` (Pre-existing)

**File:** `orchestrator/modules/tools/tool_router.py:178-236`

The `composio_execute` tool description is enriched with available action names. But the enrichment queries `AgentAppAssignment` filtered by `agent_id` only:

```python
# tool_router.py:184-192
assignments = (
    session_used.query(AgentAppAssignment)
    .filter(
        AgentAppAssignment.agent_id == agent_id,
        AgentAppAssignment.is_active == True,
        AgentAppAssignment.app_type == "EXTERNAL"
    )
    .all()
)

if assignments:  # Line 194: NO FALLBACK if empty
    app_names = [a.app_name for a in assignments if a.app_name]
```

**The bug:** Most agents don't have direct `AgentAppAssignment` rows — they inherit workspace-connected apps. But `tool_router.py` has **no workspace inheritance fallback**. When `assignments` is empty, enrichment is skipped entirely. The LLM gets a bare `composio_execute` with no clue what actions exist.

**Contrast with other files that DO have inheritance:**

| File | Line | Has Workspace Inheritance? |
|------|------|---------------------------|
| `tool_registry.py` | 1290-1299 | YES — falls back to `connected_apps` |
| `composio_hint_service.py` | 255-263 | YES — falls back to workspace apps |
| `composio_tool_service.py` | 324-332 | YES — falls back to workspace apps |
| **`tool_router.py`** | **184-194** | **NO — skips enrichment if no assignments** |

### Failure 2: PRD-122 Doubled Tool Count

**Commits:** `f95693b8c` through `17afce988`
**PRD:** `docs/PRDS/122-TOOL-ROUTING-PROMOTION-FIRST-CLASS-SCHEMAS.md`

13 platform actions promoted from `platform_execute` dispatcher to first-class OpenAI function schemas:
- `platform_list_agents`, `platform_get_agent`, `platform_create_agent`, `platform_update_agent`
- `platform_browse_marketplace_agents`, `platform_browse_marketplace_skills`, `platform_browse_marketplace_plugins`
- `platform_install_skill`, `platform_install_plugin`
- `platform_get_system_health`, `platform_get_activity_feed`
- `platform_search_memory`, `platform_store_memory`

**Impact:** Tool count went from ~18 to ~33. Each promoted tool has a detailed description (50-100 tokens). An unenriched `composio_execute` with a vague 20-token description is now buried among 32 competitors. The LLM statistically picks the tools it understands best.

**Files changed:**
- `modules/tools/discovery/action_registry.py` — Added `promoted` field, `get_promoted()`, `to_first_class_schemas()`
- `modules/tools/discovery/actions_agents.py` — Marked 4 agent actions `promoted=True`
- `modules/tools/discovery/actions_marketplace.py` — Marked 3 marketplace actions `promoted=True`
- `modules/tools/discovery/actions_monitoring.py` — Marked 1 action `promoted=True`, 6 actions `admin_only=True`
- `modules/tools/discovery/actions_memory.py` — Marked 2 memory actions `promoted=True`
- `modules/tools/tool_router.py:263-291` — Appends promoted schemas to tool list

### Failure 3: PRD-123 Restrictive Prompts Without Web Search Guidance

**Commit:** `588b4bfa0`
**PRD:** `docs/PRDS/123-HARNESS-PATTERN-ADOPTION.md`

Four prompt sections rewritten in `consumers/chatbot/personality.py`:

1. **`get_tool_guidance_prompt()` (Pattern B):** Expanded from ~40 words to ~600 words. Added:
   - "One tool at a time unless the task clearly requires multiple"
   - "Don't spray tool calls hoping something sticks"
   - "Use tools only when they genuinely help"
   - "Search the knowledge base to answer 'how are you?'" → don't do this
   - **MISSING:** No mention of web search tools (Tavily, Composio Search)
   - **MISSING:** No mention that agents have internet access via Composio

2. **`get_platform_skill()` (Pattern H):** Expanded from ~400 to ~600 tokens. Goal-oriented capability map. **Does not mention web search or Composio tools.**

3. **`get_self_learning_instruction()` (Pattern E):** Memory decision framework. Expanded from ~60 to ~800 words. Not related to tool routing.

4. **`get_anti_patterns()` (Pattern F):** Added anti-pattern section. Includes "Over-researching simple requests" which may suppress legitimate research tool use.

**The compounding effect:** The LLM now sees 33+ tools, is told to be conservative about tool use ("one at a time", "don't spray"), and is never told it has web search capabilities. When it encounters "research competitors in our market", it:
1. Sees `search_knowledge` (internal RAG) — clear, well-described
2. Sees `composio_execute` — vague, unenriched (no actions listed)
3. Reads prompt guidance: "Don't spray tool calls", "Use tools only when they genuinely help"
4. **Conclusion:** Use `search_knowledge` (safe, understood) instead of `composio_execute` (risky, unclear)

### Failure 4: Router Cannot See Agent Tool Capabilities

**File:** `orchestrator/core/routing/engine.py:634-680`

The UniversalRouter's Tier 3 LLM classification builds an agent list for the routing prompt. Each agent includes an `Apps: [...]` field. But the app lookup queries `AgentAppAssignment` directly with **no workspace inheritance fallback**:

```python
# engine.py:643-651
app_assignments = (
    self._db.query(AgentAppAssignment)
    .filter(
        AgentAppAssignment.agent_id == agent.id,
        AgentAppAssignment.is_active.is_(True),
    )
    .all()
)
app_names = [a.app_name for a in app_assignments]
# When empty → apps=[none]. No fallback to workspace connections.
```

**Impact:** The routing LLM sees `Apps: [none]` for every agent. It routes by name/description/tags only. "Research competitors" goes to PROSPECT (because "prospect" and "research" are in its tags) instead of an agent with TAVILY or web search tools.

**This is the most critical failure** because even if per-action schemas are loaded correctly after routing, the **wrong agent** was selected. A sales prospecting agent doing research gets a different personality, prompt, and approach than a dedicated research agent.

| File | Line | Has Workspace Inheritance? |
|------|------|---------------------------|
| `tool_registry.py` | 1290-1299 | YES |
| `composio_hint_service.py` | 255-263 | YES |
| `composio_tool_service.py` | 324-332 | YES |
| `tool_router.py` | 184-194 | NO |
| **`engine.py`** | **643-651** | **NO** |

---

## 4. File-by-File Evidence

### Core Files Involved

#### `orchestrator/modules/tools/tool_router.py`
- **Lines 129-309:** `get_tools_for_agent()` — single source of truth for tool lists
- **Lines 178-236:** Composio enrichment code — **HAS THE ENRICHMENT GAP** (no workspace inheritance)
- **Lines 263-291:** PRD-122 promoted schemas appended here
- **Role:** Returns `[core_tools + composio_execute + platform_tools + promoted_tools]`
- **Bug:** Agents without direct `AgentAppAssignment` get bare `composio_execute`

#### `orchestrator/consumers/chatbot/service.py`
- **Lines 573-591:** `_get_tools()` — calls `get_tools_for_agent()`
- **Lines 858-927:** `_inject_composio_tools()` — **ALREADY HAS ComposioToolService** wired in correctly
- **Lines 1182-1191:** Per-action Composio execution in tool loop
- **Lines 1427-1431:** `_execute_composio_action()` — direct SDK execution
- **Status:** Chat streaming path works correctly (when conditions are met)

#### `orchestrator/modules/agents/factory/agent_factory.py`
- **Lines 686-996:** `execute_with_prompt()` — main agent execution
- **Lines 791-803:** Tool loading via ContextService or direct `get_tools_for_agent()`
- **Lines 805-818:** Composio detection and hint injection routing
- **Lines 1120-1162:** `_inject_composio_hints()` — uses `ComposioHintService` (INFERIOR)
- **Lines 1087-1118:** `_inject_composio_recipe_hints()` — recipe-specific hint injection
- **Status:** Uses hint-based approach only. **Does NOT use ComposioToolService.**

#### `orchestrator/modules/tools/services/composio_tool_service.py`
- **Lines 63-350:** `ComposioToolService` class
- **Lines 97-255:** `get_tools_for_step()` — semantic search → per-action schemas
- **Lines 294-338:** `_resolve_allowed_apps()` — **HAS workspace inheritance** (line 325)
- **Lines 20-21:** Comment says: `Consumers: Recipe executor (Phase 1), Chatbot / external API (Phase 2 — PRD-50)`
- **Status:** Phase 2 partially done (chat has it, agent_factory doesn't)

#### `orchestrator/modules/tools/services/composio_hint_service.py`
- **Lines 89-674:** `ComposioHintService` class
- **Lines 103-212:** `build_hints()` — 3-tier hint matching
- **Lines 217-275:** `_resolve_allowed_apps()` — **HAS workspace inheritance** (line 255-263)
- **Status:** Fallback mechanism. Works but inferior to per-action schemas.

#### `orchestrator/consumers/chatbot/personality.py`
- **Lines 245-296:** `get_tool_guidance_prompt()` — restrictive tool guidance, no web search mention
- **Lines 298-345:** `get_platform_skill()` — capability map, no web search mention
- **Lines 400-415:** `get_anti_patterns()` — includes "over-researching" anti-pattern
- **Status:** PRD-123 changes are architecturally sound but missing web search guidance

#### `orchestrator/modules/tools/execution/unified_executor.py`
- **Line 102:** `self.composio_actions: dict = {}` — initialized empty
- **Lines 414-426:** Per-action Composio routing logic — checks `composio_actions` dict
- **Line 524-525:** `_execute_composio_execute()` — delegates to `exec_composio` module
- **Status:** Per-action routing exists but `composio_actions` dict is **never populated**. Dead code path.

#### `orchestrator/api/recipe_executor.py`
- **Lines 166-204:** ComposioToolService integration — semantic search + hint fallback
- **Lines 240-262:** Tool merging — strips `composio_execute`, adds per-action schemas
- **Status:** Working correctly. Reference implementation.

#### `orchestrator/modules/context/sections/tools.py`
- **Lines 120-134:** `_load_full()` — calls `get_tools_for_agent()` (prompt-agnostic)
- **Lines 136-193:** `_load_filtered()` — calls `get_tools_for_agent()` + SmartToolRouter filtering
- **Status:** No ComposioToolService integration. Returns generic `composio_execute`.

#### `orchestrator/modules/context/modes.py`
- **Lines 35-134:** ContextMode configurations with tool_loading strategies
- **CHATBOT:** `tool_loading="filtered"` (SmartToolRouter filtering)
- **TASK_EXECUTION:** `tool_loading="full"` (all assigned tools)
- **HEARTBEAT_AGENT:** `tool_loading="full"`
- **RECIPE:** `tool_loading="full"`
- **COORDINATOR:** `tool_loading="full"`
- **Status:** All modes use `get_tools_for_agent()` which returns `composio_execute`

### Database Tables

| Table | Role | Status |
|-------|------|--------|
| `AgentAppAssignment` | Maps agents → Composio apps (direct assignment) | Many agents have NO rows |
| `ComposioActionCache` | Cached action schemas per app | Populated by sync job |
| `ComposioAppCache` | App metadata (name, logo, categories) | Populated by sync job |
| `composio_entities` | Maps workspace → Composio entity_id | One per workspace |
| `composio_connections` | OAuth connection status per entity | Tracks active/pending |

---

## 5. What Works vs What Doesn't

### Working Paths

| Path | Uses ComposioToolService? | Per-Action Schemas? | Workspace Inheritance? |
|------|--------------------------|--------------------|-----------------------|
| **Chat streaming** (`service.py:858`) | YES | YES | YES (via ComposioToolService) |
| **Recipe executor** (`recipe_executor.py:169`) | YES | YES | YES (via ComposioToolService) |

### Broken Paths

| Path | Uses ComposioToolService? | Per-Action Schemas? | Workspace Inheritance? |
|------|--------------------------|--------------------|-----------------------|
| **Missions** (via agent_factory) | NO — uses ComposioHintService | NO — generic `composio_execute` | PARTIAL (hints have it, enrichment doesn't) |
| **Heartbeats** (via agent_factory) | NO — uses ComposioHintService | NO — generic `composio_execute` | PARTIAL |
| **Tasks** (via agent_factory) | NO — uses ComposioHintService | NO — generic `composio_execute` | PARTIAL |
| **Voice chat** (via streaming service) | YES (inherits from streaming) | YES | YES |

### Edge Case: Chat Streaming Conditions

Chat streaming calls `_inject_composio_tools()` at line 1873 only when:
```python
if _complexity != Complexity.ATOM:  # line 1873
    # AND inside _inject_composio_tools:
    if latest_text and agent_id and self.workspace_id and not skip_composio:  # line 879
```

`skip_composio` is True when `complexity_assessment.action == Action.RESPOND` (simple greetings).
For DELEGATE and MISSION actions, ComposioToolService is called.

**Potential failure:** If `self.workspace_id` is None/empty, the entire Composio injection is skipped silently. The `except` at line 924 catches all errors and logs a warning — a SDK failure or type mismatch would cause silent fallback to no Composio tools.

---

## 6. PRD Cross-Reference Analysis

### PRDs That Did NOT Break Things (82-108)

| PRD | Scope | Tool Routing Impact |
|-----|-------|-------------------|
| **82** | Research assessment | None — document only |
| **82A** | Sequential Mission Coordinator | None — uses existing `AgentFactory.execute_with_prompt()` |
| **82B** | Mission Intelligence Layer | None — agent matching and templates |
| **82C** | Parallel Execution + Budget | None — parallel dispatch, budget gates |
| **100** | Research: Autonomous Operating Layer | None — strategic document |
| **101** | Mission Schema Data Model | None — new tables, additive |
| **102** | Coordinator Architecture | None — uses existing tool pipeline |
| **103** | Verification & Quality | None — post-execution judging |
| **104** | Ephemeral Agents | Minor — adds `_resolve_explicit_tools()` to tool_router, but doesn't change main path |
| **105** | Budget Governance | None — admission gate before LLM call |
| **106** | Outcome Telemetry | None — logging, no execution changes |
| **107** | Context Interface Abstraction | None — port/adapter wrapping, transparent |
| **108** | Memory Field Prototype | None — Qdrant vector field, parallel to tool pipeline |

### PRDs That DID Break Things (122, 123)

#### PRD-122: Tool Routing Promotion (CONTRIBUTED TO REGRESSION)

**What it did right:**
- Promoted high-value platform actions to first-class schemas (better UX)
- Added admin-only gating (security improvement)
- Added permission enforcement (correct architecture)

**What it broke (indirectly):**
- Doubled tool count from ~18 to ~33
- `composio_execute` (already unenriched for most agents) now competes with 32 well-described tools
- SmartToolRouter's FILTERED strategy was updated to include promoted tools, but no equivalent update for Composio tool prominence

**Files modified that matter:**
- `tool_router.py:263-291` — Appends promoted schemas (increases tool count)
- `action_registry.py` — Added `promoted` field and schema generation
- `consumers/chatbot/smart_tool_router.py` — Added `ALWAYS_INCLUDE` set for promoted tools

#### PRD-123: Harness Pattern Adoption (CONTRIBUTED TO REGRESSION)

**What it did right:**
- Memory decision framework (better memory hygiene)
- Anti-patterns (prevents over-engineering)
- Platform awareness prompt (goal-oriented)
- Tool tier stratification (access control)

**What it broke (directly):**
- `personality.py:get_tool_guidance_prompt()` — Restrictive guidance with NO web search mention
- `personality.py:get_anti_patterns()` — "Over-researching simple requests" may suppress legitimate research
- Combined with PRD-122's doubled tool count, the restrictive prompts made `composio_execute` nearly invisible

**The missing paragraph:** The tool guidance prompt should include:
```
### Internal vs External Information
- Internal questions (about this workspace) → Use search/knowledge/platform tools
- External questions (competitors, market, news) → Use Composio web search tools
  (TAVILY_TAVILY_SEARCH, COMPOSIO_SEARCH_WEB, etc.)
- If web search tools are in your available tools, use them for external research
```

This was added in the PRD-125 Phase 0 fix (`3a1ede4b3`), but only for the chat path.

### PRD-125: Workflow Decoupling (IN PROGRESS)

**Relevance:** Phase 0 "Unbreak Chat" is the immediate context for this review.

**What Phase 0 did:**
- Added `Action.MISSION` to AutoBrain (route complex → mission suggestion)
- Error-guarded `_stream_workflow_bridge()` with timeout
- Added web search patterns to intent classifier
- Added web search category to SmartToolRouter
- Added web search guidance to personality prompt

**What Phase 0 did NOT do:**
- Did not fix agent_factory's missing ComposioToolService integration
- Did not fix `tool_router.py`'s enrichment gap
- Did not address the fact that missions/heartbeats/tasks still use hint-based approach

---

## 7. Composio SDK State of the Art

### Current Automatos SDK Version

```
# requirements.txt:103
composio-openai==0.11.1
```

### SDK v3 Architecture (from Composio docs)

Composio's v3 SDK recommends **per-action schemas**, not a generic dispatcher:

```python
# Composio v3 recommended pattern:
composio = Composio()
session = composio.create(user_id="workspace-uuid")
tools = session.tools()  # Returns OpenAI-format per-action schemas

# Or filtered by toolkit:
tools = composio.tools.get(
    user_id="workspace-uuid",
    toolkits=["TAVILY", "SLACK"],
    limit=30
)
```

Each action becomes its own OpenAI function tool:
```json
{
  "type": "function",
  "function": {
    "name": "TAVILY_TAVILY_SEARCH",
    "description": "Search the web using Tavily",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {"type": "string", "description": "Search query"}
      },
      "required": ["query"]
    }
  }
}
```

### What Automatos Already Has (But Doesn't Fully Use)

| Component | File | Status |
|-----------|------|--------|
| `ComposioToolService` | `modules/tools/services/composio_tool_service.py` | WORKS — used by recipes + chat streaming |
| `ComposioClient.get_app_actions()` | `core/composio/client.py:669-737` | WORKS — uses `composio.tools.get(toolkits=[app])` |
| `ToolRouterManager` | `core/composio/tool_router_manager.py` | EXISTS — uses `session.tools()`, not integrated into main pipeline |
| `ComposioToolRouter` | `modules/tools/composio_tool_router.py` | EXISTS — per-agent scoping, not integrated into main pipeline |
| `ComposioToolRouterExecutor` | `modules/tools/execution/composio_router_executor.py` | EXISTS — meta-tool executor, not integrated |
| `unified_executor.composio_actions` | `modules/tools/execution/unified_executor.py:102` | EXISTS — per-action routing dict, **always empty** |

### The `composio_execute` Meta-Tool Problem

Automatos's `composio_execute` is a generic dispatcher:
```json
{
  "name": "composio_execute",
  "description": "Execute a Composio integration action",
  "parameters": {
    "action": "string (action name to execute)",
    "params": "object (action-specific parameters)",
    "app_name": "string (optional app name)"
  }
}
```

The LLM must:
1. Know what actions exist (from enrichment or hints)
2. Guess the correct `action` name string
3. Guess the correct `params` structure (no schema visible)

Per-action schemas eliminate all three problems — the LLM sees typed function signatures.

---

## 8. The Fourth Failure: Routing Blind Spot

### The Architectural Insight

Auto is a **router/delegator**, not an executor. When a user says "research competitors," Auto's job is to pick the right agent — not to use Composio tools itself. The problem isn't just that agents lack per-action schemas; it's that **the router picks the wrong agent entirely**.

### Evidence from Production Logs

```
[router] Tier 3 agent: id=312 name='PROSPECT' apps=[none] desc=''
[router] Tier 3 agent: id=185 name='SCOUT' apps=[none] desc='Lead intelligence agent...'
[router] Tier 3: LLM response: {"route":"PROSPECT","confidence":0.83,
  "reasoning":"PROSPECT is the strongest match because its tags include prospect and research"}
```

Every agent shows `apps=[none]`. The router LLM selects based on name/description/tags only — it has **zero visibility into which agents have web search or Composio tools**.

### Root Cause: Missing Workspace Inheritance in Router

`_build_agent_descriptions()` in `engine.py:634-680`:

```python
# Lines 643-651 — Router's app lookup (NO INHERITANCE)
app_assignments = (
    self._db.query(AgentAppAssignment)
    .filter(
        AgentAppAssignment.agent_id == agent.id,
        AgentAppAssignment.is_active.is_(True),
    )
    .all()
)
app_names = [a.app_name for a in app_assignments]
# When empty → apps=[none] in the prompt. NO FALLBACK.
```

Compare with `ComposioToolService._resolve_allowed_apps()` at line 294-338:

```python
# Lines 324-331 — ComposioToolService (HAS INHERITANCE)
if not assigned_apps:
    if connected_apps:
        logger.info(
            "[ComposioToolService] Agent %s has no app assignments — "
            "inheriting %d workspace apps", agent_id, len(connected_apps)
        )
        return connected_apps
    return []
```

And `ComposioHintService._resolve_allowed_apps()` at line 217-275 — same pattern.

**Both Composio services auto-inherit workspace-connected apps when no per-agent assignments exist. The router does not.**

### Impact

The LLM routing prompt includes `Apps: [none]` for every agent. The routing LLM:
- Cannot factor tool capability into its decision
- Routes "research competitors" to PROSPECT (name contains "prospect"/"research")
- Ignores agents that actually have Tavily, web search, or research tools connected
- Even if the selected agent later gets ComposioToolService tools via inheritance, the WRONG agent was selected — a sales prospecting agent doing research gets a different personality, prompt, and approach than a dedicated research agent

### Why This Wasn't Caught

1. **Testing gap:** Router unit tests check tier logic, not app enrichment accuracy
2. **Logs misleading:** `apps=[none]` looks like "no apps assigned" which seems correct if you don't know about workspace inheritance
3. **Partial success:** Some requests work because the LLM routing happens to pick an agent that can handle it (coincidence, not capability matching)

### The Fix

The router's `_build_agent_descriptions()` needs the same workspace inheritance as `ComposioToolService._resolve_allowed_apps()`:

1. Query `AgentAppAssignment` for the agent (existing code)
2. If empty, query `EntityManager.get_entity_connections()` for workspace-connected apps
3. Include the inherited apps in the `Apps: [...]` field of the LLM routing prompt
4. The routing LLM can now see: "SCOUT has [TAVILY, COMPOSIO_SEARCH_WEB, GITHUB]" and correctly route research tasks to SCOUT

This is a **single-method change** in `engine.py:_build_agent_descriptions()` — ~15 lines of code, following a proven pattern from two existing services.

---

## 9. Proposed Fix Architecture (Updated)

### Design Principle: Auto Routes, Agents Execute

The fix has two independent parts:
1. **Fix routing** — Router sees agent tool capabilities → picks the right agent
2. **Fix execution** — Agent factory gets per-action Composio schemas → agent can use tools

Both are needed. Fixing routing alone means the right agent is picked but still can't use tools efficiently. Fixing execution alone means agents have tools but the wrong agent is selected.

### What Needs to Change

The fix is completing `ComposioToolService` Phase 2 — the same service that already makes recipes and chat streaming work — into the agent factory path. **AND** adding workspace inheritance to the router.

### Approach: Two-Phase Tool Loading

```
CURRENT (broken for agent_factory):
  get_tools_for_agent() → [platform + composio_execute] → ComposioHintService → LLM

PROPOSED (matches chat streaming + recipe pattern):
  get_tools_for_agent() → [platform tools] (base)
  + ComposioToolService.get_tools_for_step(prompt) → [per-action schemas]
  = merged tool list → LLM
  fallback → ComposioHintService (if SDK returns empty)
```

### Where to Inject

**Option A: In `agent_factory.py` (Direct — matches chat streaming pattern)**
- Replace `_inject_composio_hints()` at line 1120 with `_inject_composio_tools()` that uses `ComposioToolService`
- Keep `ComposioHintService` as fallback (identical to `service.py:909-923`)
- Agent factory already has the user prompt at line 789: `original_user_prompt = prompt`
- **Pros:** Minimal change, proven pattern from service.py
- **Cons:** Duplicates logic between service.py and agent_factory.py

**Option B: In `ContextService` / `ToolsSection` (Architectural — single source)**
- Add a new `ToolLoadingStrategy.COMPOSIO_AWARE` that takes a prompt parameter
- ToolsSection calls `get_tools_for_agent()` for base tools, then `ComposioToolService` for Composio tools
- All callers (chat, missions, heartbeats, tasks) get the same behavior
- **Pros:** Single source of truth, no duplication
- **Cons:** ContextService currently doesn't have the user prompt at tool-loading time; requires interface change

**Option C: In `get_tools_for_agent()` directly (Deepest — changes the function)**
- Add optional `prompt` parameter to `get_tools_for_agent()`
- When prompt is provided, use `ComposioToolService` instead of `composio_execute`
- **Pros:** Truly single source, all callers benefit automatically
- **Cons:** Changes a widely-used interface, mixes prompt-dependent and prompt-independent logic

### Recommended Fix Order

**Phase 0: Fix Routing (HIGHEST PRIORITY — ~15 lines)**
Add workspace inheritance to `engine.py:_build_agent_descriptions()`. Without this, the wrong agent is selected regardless of tool availability. This is the smallest change with the biggest impact.

**Phase 1: Fix Execution (Option A — matches chat streaming pattern)**
Wire `ComposioToolService` into `agent_factory.py:_inject_composio_hints()`, following the exact pattern from `service.py:858-927`. This fixes missions, heartbeats, and tasks immediately.

**Phase 2: Fix Enrichment Gap**
Add workspace inheritance to `tool_router.py:184-192` for the `composio_execute` enrichment code. Even though per-action schemas are the primary path, `composio_execute` remains as a fallback and should work correctly.

**Phase 3: Architectural Unification (Option B)**
Extract the ComposioToolService integration into ContextService so all paths share one implementation. This prevents future drift between service.py and agent_factory.py.

**Phase 4: Dead Code Cleanup**
Either populate `unified_executor.composio_actions` dict when per-action tools are resolved, or remove the dead routing logic at `unified_executor.py:414-426`.

### Gotchas and Risks

#### 1. Tool Count Explosion
`ComposioToolService` caps at `_MAX_TOOLS = 30` and uses semantic search to pick relevant actions. Without semantic search, an agent with 10 apps × 50 actions = 500 tools. The cap is critical.

#### 2. Latency
`ComposioToolService` calls the Composio SDK for semantic search. Recipe steps and missions run once — acceptable. Chat messages are frequent — the chat path already handles this. Agent factory paths (missions, heartbeats) are one-shot, so latency is acceptable.

#### 3. SDK Failure = No Tools
If Composio SDK is down, `ComposioToolService.get_tools_for_step()` returns empty. The fallback to `ComposioHintService` (which uses the DB cache, not SDK) prevents total tool loss. Both service.py and recipe_executor.py already implement this fallback.

#### 4. Prompt Availability
Agent factory has the user prompt at `execute_with_prompt()` line 789. Mission tasks have the task description. Heartbeats have the heartbeat prompt. All execution paths have a prompt available for semantic search.

#### 5. unified_executor.composio_actions Dict
The agent factory execution path uses `unified_executor.execute_tool()` which checks `self.composio_actions` at line 414. But the dict is empty, so per-action tools fall through to the generic tool registry lookup (line 428). This still works (the registry has a catch-all), but it's inefficient and may miss the optimized Composio execution path.

**Fix:** Either populate `composio_actions` when per-action tools are resolved, or add per-action detection logic in the executor (similar to `service.py:1182-1191` which checks `composio_result.action_set`).

---

## 10. Risk Assessment

### What Will NOT Break

| Component | Why |
|-----------|-----|
| **Recipe executor** | Already uses ComposioToolService. No change needed. |
| **Chat streaming** | Already uses ComposioToolService. No change needed. |
| **Platform tools** | Different namespace (`platform_*`). Untouched. |
| **PRD-122 promoted tools** | Different namespace. Appended separately. |
| **ToolRegistry core tools** | `search_knowledge`, `generate_document`, etc. Untouched. |
| **Composio validation** | `ComposioToolExecutor.execute()` still validates app assignment + connection. |
| **Admin gating** | PRD-122 permission checks are in ActionRegistry, not affected. |

### What Could Break (Must Verify)

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Composio SDK down during mission | Medium | Fallback to ComposioHintService (proven pattern) |
| Too many per-action tools (context overflow) | Low | `_MAX_TOOLS = 30` cap in ComposioToolService |
| Type mismatch on workspace_id (string vs UUID) | Low | SQLAlchemy PGUUID handles conversion; service.py already works with string |
| `unified_executor.composio_actions` empty → wrong execution path | Medium | Per-action tools will route through registry fallback; still works but less efficient |
| Heartbeat agents calling Composio tools they shouldn't | Low | ComposioToolService respects `AgentAppAssignment` + workspace connections |

### What the Fix Does NOT Address (Future Work)

1. **Composio SDK v3 migration:** Automatos uses `composio-openai==0.11.1`. The v3 SDK has breaking changes (entity_id → user_id, Apps → Toolkits). Migration is separate work.
2. **Tool Router meta-tools:** `ToolRouterManager` and `ComposioToolRouter` exist but aren't wired in. These provide SEARCH + EXECUTE meta-tools (intermediate pattern). Not needed if per-action schemas work.
3. **ContextService architectural unification:** Chat streaming and agent factory currently duplicate the ComposioToolService integration logic. Should be unified in ContextService.
4. **SmartToolRouter Composio awareness:** The FILTERED tool loading strategy uses SmartToolRouter to filter tools by intent. It doesn't know about Composio per-action tools. May over-filter.

---

## 11. Open Questions for Reviewers

### Architecture Questions

1. **Should `ComposioToolService` be integrated into `get_tools_for_agent()` directly, or remain a separate injection step?** The current pattern (separate injection after base tools) works for chat and recipes but requires each caller to wire it in manually. Integrating into `get_tools_for_agent()` would require a prompt parameter on a currently prompt-agnostic function.

2. **Should `composio_execute` be removed entirely, or kept as a fallback?** Recipe executor keeps it as fallback (strips only when per-action tools are found). This seems correct — some edge cases may need the generic tool when SDK search misses.

3. **Should `unified_executor.composio_actions` be populated, or should execution routing detect per-action tools differently?** The chat streaming path doesn't use this dict — it has its own `_execute_composio_action()` method that checks `composio_result.action_set`. The agent factory path uses `unified_executor.execute_tool()` which checks the empty dict. These are two different execution routing strategies for the same problem.

4. **Is the `_MAX_TOOLS = 30` cap in ComposioToolService appropriate for all execution paths?** Missions may need more tools (broader scope). Heartbeats may need fewer (focused task). Should this be configurable per ContextMode?

### Process Questions

5. **Was the AgentFactory clean rewrite (commit `0ab9e0c6c`) reviewed before merge?** It deleted working per-action Composio code. The commit message says "single tool source via get_tools_for_agent()" which is architecturally correct but missed that `get_tools_for_agent()` doesn't provide per-action schemas.

6. **Should PRD-122 promoted tools have been accompanied by a Composio prominence fix?** Adding 13 new platform tools without ensuring Composio tools remain discoverable was an oversight.

7. **Should PRD-123 prompt rewrites have included web search guidance?** The pattern adoption was modeled on Claude Code's patterns, which has different tool architecture. Copying Claude Code's restrictive tool guidance without adapting it for Composio was a mismatch.

### Data Questions

8. **How many agents have direct `AgentAppAssignment` rows vs inheriting workspace apps?** This determines how many agents are affected by the enrichment gap in `tool_router.py`.

9. **What is the actual SDK search latency for `ComposioToolService.get_tools_for_step()`?** Log data from recipe executions would show this. If >500ms, caching may be needed for high-frequency paths.

10. **Are there agents that should NOT have Composio tools but would get them via workspace inheritance?** The workspace inheritance pattern assumes all agents in a workspace should access all connected apps. This may not be desirable for specialized agents (e.g., a "writing assistant" shouldn't send Slack messages).

---

## Appendix A: Git Commits Referenced

| Commit | Date | Description |
|--------|------|-------------|
| `0ab9e0c6c` | 2026-03-11 | AgentFactory clean rewrite — deleted per-action Composio code |
| `667df25bd` | 2026-03-12 | Per-action Composio schemas added (overwritten next day) |
| `46f990cf7` | 2026-03-13 | ContextService migration — overwrote per-action code |
| `82b5a44b2` | 2026-03-13 | ComposioHintService added as compensating patch |
| `f95693b8c` | 2026-03-24 | PRD-122 Phase 1: promoted field + first-class schemas |
| `17afce988` | 2026-03-24 | PRD-122: Replace hardcoded field schemas with promoted |
| `588b4bfa0` | 2026-03-24 | PRD-123: Prompt patterns H, E, B, F (restrictive tool guidance) |
| `ceb62c543` | 2026-03-24 | PRD-123: Tool tier stratification |
| `3a1ede4b3` | 2026-04-01 | PRD-125 Phase 0: Unbreak chat + web search intelligence |

## Appendix B: File Reference Index

| File | Lines | Role |
|------|-------|------|
| `core/routing/engine.py` | 634-680 | UniversalRouter agent descriptions — MISSING WORKSPACE INHERITANCE |
| `modules/tools/tool_router.py` | 129-309 | Central tool assembly — has enrichment gap |
| `consumers/chatbot/service.py` | 858-927 | Chat ComposioToolService integration (WORKS) |
| `consumers/chatbot/service.py` | 1182-1191 | Chat per-action execution routing (WORKS) |
| `modules/agents/factory/agent_factory.py` | 1120-1162 | Agent factory ComposioHintService (INFERIOR) |
| `modules/tools/services/composio_tool_service.py` | 97-255 | Per-action schema service (CORRECT SOLUTION) |
| `modules/tools/services/composio_hint_service.py` | 103-212 | Hint-based service (FALLBACK) |
| `api/recipe_executor.py` | 166-262 | Recipe ComposioToolService integration (WORKS) |
| `consumers/chatbot/personality.py` | 245-296 | Tool guidance prompt (MISSING WEB SEARCH) |
| `modules/tools/execution/unified_executor.py` | 102, 414-426 | Per-action routing dict (ALWAYS EMPTY) |
| `modules/context/sections/tools.py` | 120-134, 136-193 | ContextService tool loading (NO COMPOSIO AWARENESS) |
| `modules/context/modes.py` | 35-134 | ContextMode configs (ALL USE get_tools_for_agent) |
| `core/composio/entity_manager.py` | 26-39, 74-90 | Workspace → entity mapping |
| `core/composio/tool_executor.py` | 141-523 | Composio action execution + validation |
| `core/composio/client.py` | 669-737 | SDK `composio.tools.get()` usage |
| `core/composio/tool_router_manager.py` | 38-184 | Tool Router session management (NOT WIRED IN) |
| `modules/tools/composio_tool_router.py` | 25-205 | Per-agent Tool Router (NOT WIRED IN) |

## Appendix C: Composio SDK References

- Migration guide: `https://docs.composio.dev/docs/migration-guide/new-sdk`
- OpenAI provider: `https://docs.composio.dev/providers/openai`
- Tool usage patterns: `https://docs.composio.dev/patterns/tools/use-tools`
- Current package: `composio-openai==0.11.1` (v2 API, pre-v3 migration)
