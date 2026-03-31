# PRD-122: Tool Routing Promotion, Permission Enforcement & First-Class Schemas

**Status:** Draft — Awaiting Review
**Date:** 2026-03-31
**Authors:** Gerard Kavanagh + Claude
**Priority:** P1
**Dependencies:** PRD-64 (Action Discovery — COMPLETE), PRD-108 (Memory Field — COMPLETE), PRD-17 (Dynamic Tool Assignment — COMPLETE)
**Supersedes:** None (extends PRD-64 dispatcher pattern)

---

## TL;DR

Two problems, one fix:

1. **Auto can't reliably call platform tools** — they're hidden behind a dispatcher indirection. The LLM must mentally bridge markdown descriptions to `platform_execute(action=..., params={...})`, which it frequently fails to do. Promote ~13 high-value actions to first-class OpenAI tool schemas.

2. **`permission_level` is declared on all 91 actions but never enforced** — every action has a permission level (`read`/`write`/`destructive`) but the executor only checks `requires_confirmation` (4 destructive deletes) and rate-limits writes. There's no role-based gating. Six infrastructure tools (`platform_get_system_health`, `platform_get_logs`, `platform_query_loki_logs`, `platform_query_prometheus`, `platform_get_alerts`, `platform_list_services`) expose system internals to every user. Before promoting tools to first-class schemas — giving users shiny direct buttons — we must enforce who can press them.

---

## 1. Problem Statement

### 1.1 Auto Reports "I Can See Tools But Can't Call Them"

When asked to browse marketplace skills or list agents, Auto responds:

> "I can see the names of the platform tools, but I don't have executable tool bindings exposed here. I understand the commands. I know the order to execute them. I do not currently have live access to invoke them in this session."

This is technically incorrect — Auto CAN call the tools via the `platform_execute` dispatcher. But the LLM doesn't reliably make this cognitive leap from "described in prompt text" to "wrap in `platform_execute(action=..., params=...)`."

### 1.2 Evidence: Field Tools Work, Dispatcher Doesn't

In PRD-108, we faced the same problem with field memory tools. The fix: register `platform_field_query` and `platform_field_inject` as **first-class tool schemas** (tool_router.py:259-316). After this change, agents called them reliably.

The platform_execute dispatcher was designed for token efficiency (PRD-64: collapse 60 schemas into 1). But the tradeoff — LLM must learn tool names from markdown, not from callable schemas — is too steep for frequently-used actions.

### 1.3 Permission Level Is Decorative — Not Enforced

Every `ActionDefinition` carries a `permission_level` field (`read` | `write` | `destructive`). The executor (`platform_executor.py:286-334`) does two things:
1. **`requires_confirmation`** — returns early asking the LLM to confirm (4 destructive deletes only)
2. **Rate limiting** — write/destructive actions get `check_rate_limit()` (10/min)

It does NOT:
- Check the user's role against the action's permission level
- Restrict admin-only tools to admin users
- Differentiate between workspace-owner, member, or agent callers

**Consequence:** Any authenticated user (or any agent in any workspace) can call `platform_get_logs`, `platform_query_prometheus`, `platform_list_services` — tools that expose Railway deployment logs, Grafana metrics, and infrastructure inventory.

#### 1.3.1 Tools That Should Be Admin-Only

| Tool | Current Level | Risk |
|------|--------------|------|
| `platform_get_system_health` | read | Exposes DB, Redis, API health for all services |
| `platform_get_logs` | read | Reads deployment logs from ANY Railway service |
| `platform_query_loki_logs` | read | Queries application logs (Grafana/Loki) |
| `platform_query_prometheus` | read | Queries infrastructure metrics |
| `platform_get_alerts` | read | Shows all firing/resolved alerts |
| `platform_list_services` | read | Enumerates all Railway services |

#### 1.3.2 `search_chat_history` Has a Multi-Tenancy Bug

`handlers_search.py:33` contains:
```sql
WHERE c.user_id = (SELECT id FROM users LIMIT 1)
```
This queries by the **first user in the database**, not the caller's workspace. A data leakage risk — user A's search could return user B's chat history.

### 1.4 External Validation: Claude Code's Architecture

Research of Anthropic's Claude Code source (via instructkr/claude-code, ~512K LOC) reveals their production pattern:

1. **Every tool gets its own typed function schema** — no dispatcher indirection
2. **Tool Search/Deferral** for scale — when tool count is too high, tools are deferred (name only in prompt) and loaded on demand via `ToolSearch`
3. **Feature-flag conditional inclusion** — not all tools appear for all agent types
4. **MCP tools dynamically wrapped** with proper `inputJSONSchema`

Their `assembleToolPool()` function merges built-in + MCP tools with deduplication. No dispatcher pattern at all.

---

## 2. Current Architecture (What Exists)

### 2.1 The Three Tool Systems

```
┌─────────────────────────────────────────────────────────┐
│             get_tools_for_agent()                        │
│             (tool_router.py:129)                         │
│             SINGLE SOURCE OF TRUTH                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. ToolRegistry (core tools)                           │
│     → search_knowledge, read_file, shell_exec, etc.    │
│     → Each gets its own OpenAI function schema          │
│     → ~10-15 first-class schemas                        │
│                                                         │
│  2. ActionRegistry (platform actions)                   │
│     → 91 actions (agents, marketplace, monitoring...)   │
│     → ALL collapsed into 1 dispatcher: platform_execute │
│     → Action names described as markdown in prompt      │
│                                                         │
│  3. Composio (external tools)                           │
│     → Per-agent app assignments via AgentAppAssignment  │
│     → composio_execute meta-tool + per-action schemas   │
│     → SDK-provided schemas (COMPOSIO_SEARCH_WEB, etc.)  │
│                                                         │
│  4. First-class field tools (PRD-108 exception)         │
│     → platform_field_query, platform_field_inject       │
│     → Hardcoded schemas in tool_router.py:259-316       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 The Dispatcher Pattern (PRD-64)

```
LLM's tool list:
  [search_knowledge, read_file, ..., platform_execute, platform_field_query, ...]

System prompt:
  "## Available Platform Actions
   Use platform_execute(action, params) to call these.
   - platform_list_agents: List agents in workspace...
   - platform_browse_marketplace_skills: Browse skills...
   ... (91 actions as markdown text)"

LLM must: Read markdown → call platform_execute(action="...", params={...})
LLM actually: "I can see the names but can't execute them"
```

### 2.3 Execution Routing (Already Handles Direct Calls)

`unified_executor.py:392` already routes any `platform_*` call directly to `PlatformActionExecutor`:

```python
# Line 392 — existing code, no changes needed
if tool_name.startswith("platform_"):
    return await self._execute_platform_action(
        tool_name, parameters, workspace_id=workspace_id, trace_id=trace
    )
```

This means **promoting actions to first-class schemas requires NO changes to the execution layer**.

---

## 3. Impact Analysis — What Touches Tools

### 3.1 All Callers of `get_tools_for_agent()`

| Caller | File | Context | Impact of Adding Schemas |
|--------|------|---------|-------------------------|
| Chatbot Service | `consumers/chatbot/service.py:584` | Chat sessions | **PRIMARY BENEFICIARY** — Auto sees callable schemas |
| Context ToolsSection | `modules/context/sections/tools.py:129` | FULL strategy | Sees new schemas, passes to LLM |
| Context ToolsSection | `modules/context/sections/tools.py:150` | FILTERED strategy | SmartToolRouter must categorize new tools |
| Agent Factory | `modules/agents/factory/agent_factory.py:799` | Mission/task execution | Mission agents see new schemas |
| Chatbot LLM API | `api/chatbot_llm.py:248,584` | Direct API endpoint | Sees new schemas |

### 3.2 Tool Loading Strategies (ContextService)

| Strategy | Behaviour | Impact |
|----------|-----------|--------|
| **FULL** | Returns everything from `get_tools_for_agent()` | Sees promoted schemas immediately |
| **FILTERED** | `get_tools_for_agent()` → SmartToolRouter filters by intent | **Must update SmartToolRouter** to categorize promoted tools |
| **DISPATCHER_ONLY** | Returns only `platform_execute` | **No impact** — promoted tools don't appear here |
| **NONE** | Empty list | No impact |

### 3.3 Consumer Interaction Map

| Consumer | Uses Tools? | How? | Promotion Impact |
|----------|------------|------|-----------------|
| **Chatbot (Auto)** | Yes | FULL or FILTERED strategy | Primary beneficiary |
| **Missions/Coordinator** | Yes | Via ContextService → FULL | Mission agents can call promoted tools directly |
| **Channels (Slack, etc)** | Yes | Via ContextService | Channel agents see promoted tools |
| **Playbooks/Recipes** | Yes | Via recipe_executor.py | Recipe agents see promoted tools |
| **Heartbeat** | Indirect | Schedules agent execution | Heartbeat agents would see promoted tools |
| **Universal Router** | No | Routes to agents; doesn't build tool lists | **No impact** |
| **Composio** | Separate | Own routing path (composio_execute, per-action) | **No impact** — different namespace |
| **Workspace tools** | Separate | workspace_* prefix routing | **No impact** — different namespace |

### 3.4 What Does NOT Break

- **Composio routing** — Composio tools use `composio_execute` or SDK schema names (`COMPOSIO_SEARCH_WEB`). No namespace collision with `platform_*` promoted tools.
- **Workspace routing** — `workspace_*` tools are routed by prefix at unified_executor.py:399. No collision.
- **Existing platform_execute** — The dispatcher stays for non-promoted actions. LLMs can still use it. Promoted actions route via the direct `platform_*` path instead.
- **Execution layer** — `unified_executor.py:392` already handles direct `platform_*` calls. No changes needed.
- **PlatformActionExecutor** — Handler dispatch is name-based. Doesn't care whether the call came from `platform_execute` dispatcher or a direct schema.

### 3.5 What DOES Need Updating

| Component | Change Required | Risk |
|-----------|----------------|------|
| **ActionRegistry** | Add `promoted` field, `to_first_class_schemas()` method | LOW — additive |
| **tool_router.py** | Append promoted schemas after dispatcher | LOW — additive |
| **platform_actions.py section** | Exclude promoted from markdown catalog | LOW — reduces duplication |
| **SmartToolRouter** | Categorize promoted tools in INTENT_TO_TOOLS | **MEDIUM** — if missed, FILTERED strategy won't suggest them |
| **actions_*.py files** | Mark ~13 actions as `promoted=True` | LOW — flag addition |

---

## 4. Proposed Solution

### Phase 0: Permission Enforcement (P0 — Must Ship Before Promotion)

Before promoting tools to first-class schemas, enforce the permission model that already exists but is never checked.

#### 4.0.1 Add `admin_only` Flag to ActionDefinition

```python
@dataclass
class ActionDefinition:
    # ... existing fields ...
    permission_level: str = "read"       # read | write | destructive
    admin_only: bool = False             # NEW — workspace admin required
    requires_confirmation: bool = False
```

Mark 6 infrastructure tools as `admin_only=True`:
- `platform_get_system_health`
- `platform_get_logs`
- `platform_query_loki_logs`
- `platform_query_prometheus`
- `platform_get_alerts`
- `platform_list_services`

#### 4.0.2 Enforce `admin_only` in PlatformExecutor

The permission infrastructure already exists but is unused by the executor:
- `workspace_members.role` column — values: `owner` | `admin` | `editor` | `viewer` | `member`
- `ROLE_PERMISSIONS` dict in `orchestrator/core/workspaces/permissions.py` — full matrix
- `@require_permission()` decorator — used by API endpoints (team, billing) but NOT by tool execution
- `RequestContext.user.system_role` — `"admin"` from Clerk `publicMetadata.role` for system-wide admin
- `UserContext.role` — workspace-level role from `workspace_members`

**No migration needed.** The role column and permission system are already built. The gap is that `platform_executor.py` never checks them.

Add a check before the existing `requires_confirmation` block in `platform_executor.py:286`:

```python
# NEW: Admin-only gate — uses existing workspace_members.role
if action_def and action_def.admin_only:
    is_admin = await self._check_caller_is_admin(workspace_id, caller_context)
    if not is_admin:
        return {
            "success": False,
            "error": f"Action '{action_name}' requires workspace admin or owner role.",
            "permission_denied": True,
        }
```

`_check_caller_is_admin()` queries `workspace_members` for the caller and checks `role in ("owner", "admin")`. Also returns `True` if `system_role == "admin"` (system-wide admin). Fail-closed: if lookup fails, deny access.

**Caller context:** The executor currently receives `workspace_id` but not `user_id`. The `RequestContext` (from `orchestrator/core/auth/dependencies.py`) carries both. We need to thread `RequestContext` (or at minimum `user_id` + `system_role`) through the execution chain:
- `unified_executor.py` → `_execute_platform_action()` → `PlatformActionExecutor.execute()`
- Currently only passes `workspace_id`. Must also pass caller identity.

**Implementation note (from review):** Attach `RequestContext` to the existing `tool_args` or `trace_id` metadata flowing through the execution chain, rather than having `PlatformActionExecutor` rebuild auth state from scratch. The context is already resolved upstream — just pass it through.

#### 4.0.3 Enforce `permission_level` in PlatformExecutor

Extend the existing rate-limit block (`platform_executor.py:321`) to also validate caller permissions:

```python
# EXISTING: Rate limit write/destructive
if action_def and action_def.permission_level in ("write", "destructive"):
    await check_rate_limit(str(self.workspace_id), "platform_write")

# NEW: Validate write permission for non-agent callers
if action_def and action_def.permission_level == "destructive":
    if not action_def.requires_confirmation:
        # Destructive action without confirmation gate — reject
        logger.error(f"Destructive action {action_name} missing requires_confirmation=True")
        return {"success": False, "error": "Internal: destructive action misconfigured"}
```

This ensures any new destructive action MUST have `requires_confirmation=True` or the executor rejects it. Defense in depth.

#### 4.0.4 Exclude `admin_only` Tools from Non-Admin Tool Lists

In `tool_router.py` → `get_tools_for_agent()`, filter out `admin_only` actions when building schemas for non-admin callers. Admin tools shouldn't appear in the LLM's tool list at all — not just be blocked at execution time.

```python
promoted_schemas = action_registry.to_first_class_schemas(
    exclude_admin=not is_workspace_admin
)
```

This prevents the LLM from attempting admin calls it can't make, reducing wasted tokens and confusing error responses.

`get_tools_for_agent()` currently receives `agent_id`, `workspace_id`, `trace_id`. Must also receive caller role context. Two options:
- **(A) Pass `RequestContext`** — clean but couples tool_router to auth
- **(B) Pass `is_admin: bool`** — minimal, decoupled

Recommend **(B)** — the caller (chatbot service, agent factory, etc.) already has access to `RequestContext` and can resolve the boolean before calling.

#### 4.0.5 Fix `search_chat_history` Multi-Tenancy Bug

Replace `handlers_search.py:33`:
```sql
-- BEFORE (broken):
WHERE c.user_id = (SELECT id FROM users LIMIT 1)

-- AFTER (workspace-scoped):
WHERE c.workspace_id = :workspace_id
```

Or if `chats` doesn't have `workspace_id`, join through `users` → `workspace_members` to scope by workspace.

### Phase 1: Promote High-Value Actions (P1)

Add a `promoted: bool` flag to `ActionDefinition`. Actions marked `promoted=True` get their own OpenAI function schema in `get_tools_for_agent()`, alongside the dispatcher.

**Actions to promote (~13):**

| Action | Category | Rationale |
|--------|----------|-----------|
| `platform_list_agents` | agents | Most common user query |
| `platform_get_agent` | agents | Agent inspection |
| `platform_create_agent` | agents | Onboarding flow |
| `platform_update_agent` | agents | Configuration |
| `platform_browse_marketplace_agents` | marketplace | Discovery |
| `platform_browse_marketplace_skills` | marketplace | Discovery |
| `platform_browse_marketplace_plugins` | marketplace | Discovery |
| `platform_install_skill` | marketplace | Setup |
| `platform_install_plugin` | marketplace | Setup |
| `platform_get_system_health` | monitoring | Status (admin-only — promoted but gated) |
| `platform_get_activity_feed` | monitoring | Activity |
| `platform_search_memory` | memory | Knowledge retrieval |
| `platform_store_memory` | memory | Knowledge storage |

**Token impact:** ~13 x 300 tokens = ~3,900 tokens additional. Current tool payload ~5.5K. New total ~9.4K — well within 128K context budget.

**Field tool consolidation:** Remove the hardcoded `_FIELD_TOOL_SCHEMAS` block from tool_router.py:259-316. Instead, mark `platform_field_query` and `platform_field_inject` as `promoted=True` in their ActionRegistry registration. Consolidates all promotion logic in one place.

### Phase 2: Dispatcher Enum (P2) — Clean Break

Add an `enum` of valid action names to the `platform_execute` dispatcher's `action` parameter. **Promoted actions are excluded from the enum** — they only exist as first-class schemas. No dual paths, no tech debt. If the LLM wants to call `platform_list_agents`, there's exactly one way: the direct schema.

The enum gives the LLM autocomplete-style guidance for the ~78 remaining non-promoted actions.

```json
"action": {
  "type": "string",
  "enum": ["platform_configure_agent_heartbeat", "platform_assign_tool", ...],
  "description": "The platform action name"
}
```

Token impact: ~78 names x ~10 tokens = ~780 tokens. Acceptable.

### Phase 3: Tool Discovery (Future — separate PRD)

Inspired by Claude Code's `ToolSearch`. A `platform_discover_actions` tool that searches the ActionRegistry by keyword and returns full schemas on demand. Valuable when action count exceeds 100+. Not needed now — Phase 1+2 solves the immediate problem.

---

## 5. Design Details

### 5.1 ActionDefinition Changes

```python
@dataclass
class ActionDefinition:
    name: str
    description: str
    category: str
    parameters: Dict[str, Any]
    permission_level: str = "read"       # read | write | destructive (NOW ENFORCED)
    requires_confirmation: bool = False
    workspace_scoped: bool = True
    admin_only: bool = False             # NEW — workspace admin required
    tags: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    promoted: bool = False               # NEW — first-class schema when True
```

### 5.2 ActionRegistry New Methods

```python
def get_promoted(self) -> List[ActionDefinition]:
    """Actions that get first-class OpenAI tool schemas."""
    self._ensure_initialized()
    return [a for a in self._actions.values() if a.promoted]

def to_first_class_schemas(self) -> List[Dict[str, Any]]:
    """OpenAI function schemas for promoted actions."""
    return [a.to_openai_schema() for a in self.get_promoted()]

def build_prompt_summary(self, exclude_promoted: bool = True) -> str:
    """Markdown catalog, excluding promoted actions (they have first-class schemas).
    No dual paths — promoted actions don't appear in markdown OR dispatcher enum."""
    # ... existing logic, skip actions where a.promoted if exclude_promoted
```

### 5.3 tool_router.py Changes

After the dispatcher append (line 247), add:

```python
promoted_schemas = action_registry.to_first_class_schemas()
openai_tools.extend(promoted_schemas)
logger.info(f"[tool-trace {trace_id}] Added {len(promoted_schemas)} promoted action schemas")
```

Remove the hardcoded `_FIELD_TOOL_SCHEMAS` block (lines 259-316) — field tools become promoted actions instead.

### 5.4 SmartToolRouter Updates

**Review finding:** If SmartToolRouter's intent classifier doesn't detect the right intent, promoted tools won't be suggested under FILTERED strategy. Critical promoted tools (agent info, field tools) should bypass intent filtering entirely.

Add an `ALWAYS_INCLUDE` set for promoted tools that must be available regardless of detected intent:

```python
# Tools that bypass intent filtering — always included when promoted
ALWAYS_INCLUDE = {
    "platform_list_agents", "platform_get_agent",
    "platform_search_memory", "platform_store_memory",
    "platform_field_query", "platform_field_inject",
}
```

Also add promoted tools to the appropriate intent categories for non-critical promoted tools:

```python
TOOL_CATEGORIES = {
    # ... existing categories ...
    "platform_management": [
        "platform_list_agents", "platform_get_agent",
        "platform_create_agent", "platform_update_agent",
    ],
    "marketplace": [
        "platform_browse_marketplace_agents",
        "platform_browse_marketplace_skills",
        "platform_browse_marketplace_plugins",
        "platform_install_skill", "platform_install_plugin",
    ],
    "monitoring": [
        "platform_get_system_health", "platform_get_activity_feed",
    ],
}

INTENT_TO_TOOLS = {
    # ... existing mappings ...
    Intent.DATA_QUERY: [..., "platform_management", "monitoring"],
    Intent.EXTERNAL_ACTION: [..., "marketplace"],
    Intent.CREATION: [..., "platform_management", "marketplace"],
}
```

### 5.5 platform_actions.py Section Changes

```python
def _build(self) -> str:
    registry = get_action_registry()
    content = registry.build_prompt_summary(exclude_promoted=True)  # Skip promoted
    if self.max_tokens:
        content = self.truncate(content, self.max_tokens)
    return content
```

---

## 6. Files to Modify

### Phase 0: Permission Enforcement

| File | Change | Lines |
|------|--------|-------|
| `modules/tools/discovery/action_registry.py` | Add `admin_only` field to `ActionDefinition` | 28-49 |
| `modules/tools/discovery/platform_executor.py` | Add `admin_only` gate + destructive safety check | 286-334 |
| `modules/tools/discovery/actions_monitoring.py` | Mark 6 infrastructure tools `admin_only=True` | TBD |
| `modules/tools/discovery/handlers_search.py` | Fix `search_chat_history` to filter by workspace_id | 33 |
| `modules/tools/tool_router.py` | Filter `admin_only` tools from non-admin callers | 242+ |

### Phase 1-2: Promotion & Dispatcher Enum

| File | Change | Lines |
|------|--------|-------|
| `modules/tools/discovery/action_registry.py` | Add `promoted` field, `get_promoted()`, `to_first_class_schemas()`, update `build_prompt_summary()` | 28-183 |
| `modules/tools/tool_router.py` | Append promoted schemas, remove hardcoded field tool schemas | 242-317 |
| `modules/context/sections/platform_actions.py` | Pass `exclude_promoted=True` | 44-55 |
| `consumers/chatbot/smart_tool_router.py` | Add promoted tools to TOOL_CATEGORIES and INTENT_TO_TOOLS | TBD |
| `modules/tools/discovery/actions_agents.py` | Mark 4 actions `promoted=True` | TBD |
| `modules/tools/discovery/actions_marketplace.py` | Mark 5 actions `promoted=True` | TBD |
| `modules/tools/discovery/actions_monitoring.py` | Mark 2 actions `promoted=True` | TBD |
| `modules/tools/discovery/actions_memory.py` | Mark 2 actions `promoted=True` | TBD |

**Files NOT modified:**
- `unified_executor.py` — already routes `platform_*` directly (line 392)
- `exec_platform.py` — transparent pass-through
- `exec_composio.py` — different namespace entirely
- `exec_workspace.py` — different namespace entirely
- `core/routing/engine.py` — doesn't build tool lists

---

## 7. Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|-----------|
| **Admin gate breaks Auto for admin users** — if `_check_caller_is_admin()` query is wrong, admins lose infra tools | HIGH | Fail-open with logging in initial rollout, fail-closed after validation. Test with known admin workspace. |
| **Caller identity not threaded to executor** — `PlatformExecutor.execute()` only receives `workspace_id`, not user identity | MEDIUM | Must thread `user_id` + `system_role` through `unified_executor` → `platform_executor`. Additive change, doesn't break existing callers (default to non-admin). |
| SmartToolRouter doesn't categorize promoted tools → FILTERED strategy never suggests them | MEDIUM | Explicitly add to TOOL_CATEGORIES and INTENT_TO_TOOLS |
| Token budget increase (~4K) affects small-context models | LOW | 4K is <4% of 128K context. Monitor with token telemetry. |
| Promoted tools appear for mission agents who don't need them | LOW | Already true for all tools in FULL strategy. No regression. |
| Promoted actions removed from dispatcher but LLM still tries `platform_execute(action="platform_list_agents")` | LOW | Promoted actions are removed from the dispatcher enum — LLM won't see them there. Clean break, no dual-path tech debt. |
| Field tool consolidation breaks mission agents | MEDIUM | Test mission execution before/after. ActionRegistry schema must match current hardcoded schema exactly. |
| Namespace collision with Composio per-action schemas | NONE | Composio uses `COMPOSIO_*` prefix, platform uses `platform_*` |
| **`search_chat_history` fix breaks existing searches** | LOW | Current behavior is already broken (queries wrong user). Any fix improves it. |

---

## 8. Verification Plan

### 8.1 Unit Tests

**Phase 0 — Permission enforcement:**
- `test_platform_executor.py`: Verify `admin_only` actions return `permission_denied` for non-admin callers
- `test_platform_executor.py`: Verify `admin_only` actions succeed for admin callers
- `test_platform_executor.py`: Verify destructive actions without `requires_confirmation=True` are rejected
- `test_action_registry.py`: Verify `admin_only` actions excluded from `to_first_class_schemas(exclude_admin=True)`

**Phase 1-2 — Promotion:**
- `test_action_registry.py`: Verify `get_promoted()`, `to_first_class_schemas()`, `build_prompt_summary(exclude_promoted=True)`
- `test_tool_router.py`: Verify promoted schemas appear in `get_tools_for_agent()` output
- `test_smart_tool_router.py`: Verify promoted tools are suggested for relevant intents

### 8.2 Integration Tests

**Phase 0:**
1. **Admin test:** Log in as workspace admin → call `platform_get_system_health` → should succeed
2. **Non-admin test:** Log in as workspace member → call `platform_get_system_health` → should return `permission_denied`
3. **Multi-tenant search isolation test (from review):** Create conversations in Workspace A and Workspace B. Query `search_chat_history` as Workspace A. Assert zero results from Workspace B. This is a regression test — the bug must never recur.
4. **System admin override test:** Verify `system_role == "admin"` (Clerk publicMetadata) bypasses workspace-level admin_only checks

**Phase 1-2:**
5. **Chat test:** Ask Auto "What agents do I have?" → verify it calls `platform_list_agents` directly (not `platform_execute`)
6. **Chat test:** Ask Auto "Browse marketplace skills" → verify it calls `platform_browse_marketplace_skills` directly
7. **Chat test:** Ask Auto "Configure agent 147 heartbeat" → verify it still uses `platform_execute(action='platform_configure_agent_heartbeat', ...)` for non-promoted actions
8. **Mission test:** Run a field memory benchmark trial → verify `platform_field_query` still works after consolidation
10. **Channel test:** Send a Slack message asking for agent list → verify tool routing works through channel consumer

### 8.3 Observability

Check Railway logs for `[tool-trace]` entries:
- `Routing to PlatformActionExecutor: platform_list_agents` → direct path works
- `platform_execute -> platform_configure_agent_heartbeat` → dispatcher still works for non-promoted

---

## 9. Alternatives Considered

### 9.1 Promote ALL 91 Actions (Rejected)

**Why not:** 91 schemas x ~300 tokens = ~27K tokens. Too expensive. Also overwhelms the LLM's tool selection with 100+ options.

### 9.2 Remove the Dispatcher Entirely (Rejected)

**Why not:** Same token budget issue. Also removes the single-schema elegance for rarely-used actions.

### 9.3 Improve Prompt Engineering Only (Rejected)

**Why not:** The current dispatcher description already includes examples and explicit instructions. The problem isn't the prompt — it's the fundamental gap between "tool described in text" and "tool in the callable schema list." LLMs are trained to use function-calling schemas, not to parse markdown for tool names and mentally construct dispatcher calls.

### 9.4 Claude Code's ToolSearch Pattern (Deferred)

**Why deferred:** Good pattern for 100+ tools. At 91 actions with 13 promoted, the remaining 78 behind the dispatcher + enum is manageable. Revisit if action count grows past 120.

---

## 10. Implementation Order

### Phase 0: Permission Enforcement (deploy first, independently)

No migration needed — `workspace_members.role` column already exists with values `owner`|`admin`|`editor`|`viewer`|`member`. Permission matrix in `permissions.py` already defined. Just need to wire it into the executor.

1. Add `admin_only: bool = False` field to `ActionDefinition`
2. Mark 6 infrastructure tools as `admin_only=True` in `actions_monitoring.py`
3. Thread caller identity (`user_id`, `system_role`) through `unified_executor` → `platform_executor`
4. Add `_check_caller_is_admin()` in `platform_executor.py` — queries `workspace_members.role` for `owner`/`admin`, also checks `system_role == "admin"`
5. Add admin gate before `requires_confirmation` block
6. Add destructive safety check: reject destructive actions missing `requires_confirmation=True`
7. Pass `is_admin` to `get_tools_for_agent()` → filter `admin_only` tools from non-admin callers
8. Fix `search_chat_history` workspace scoping in `handlers_search.py`
9. Deploy and verify admin/non-admin access patterns

### Phase 1: Promote High-Value Actions

9. Add `promoted: bool = False` field to `ActionDefinition`
10. Add `get_promoted()`, `to_first_class_schemas()` to `ActionRegistry`
11. Update `build_prompt_summary()` to support `exclude_promoted`
12. Mark ~13 actions as `promoted=True` in `actions_*.py` files
13. Update `get_tools_for_agent()` to include promoted schemas
14. Consolidate field tools: remove hardcoded `_FIELD_TOOL_SCHEMAS`, mark field actions as `promoted=True`
15. Update SmartToolRouter categories and intent mappings
16. Update platform_actions.py section to exclude promoted

### Phase 2: Dispatcher Enum

17. Add enum to `to_dispatcher_schema()` for remaining actions
18. Deploy and verify with chat + mission + channel tests

---

## 11. Open Questions

1. **Agent callers vs user callers:** When a mission agent calls `platform_get_system_health`, should it be allowed? Agents act on behalf of the workspace owner. Options: (a) agents inherit workspace owner permissions, (b) agents always get `read` level only, (c) agent-level permission config. Recommendation: **(a)** for now — agents run in the owner's workspace context. Restrict later if multi-tenant orgs need it.

3. **Which 13 actions to promote?** The list above is a best guess based on expected usage. Should we instrument dispatcher calls first to see which actions are actually attempted most often?

4. **SmartToolRouter overhaul?** The intent-to-tool mapping is hardcoded. Should we make it data-driven (action tags → intent matching) instead of maintaining a static dict?

5. **Per-agent promotion?** Should different agents see different promoted actions? E.g., mission agents only see field tools, chatbot sees marketplace + agent management. This would require `promoted` to be context-dependent rather than a static flag.

---

## Appendix A: Research — Claude Code Tool Routing

Source: `instructkr/claude-code` (exposed TypeScript source, ~1,900 files, 512K LOC)

### Key Patterns

**Centralized tool registry with per-tool schemas:**
```typescript
export function getAllBaseTools(): Tools {
  return [AgentTool, BashTool, FileReadTool, FileEditTool, ...]
}
```

**Feature-flag conditional inclusion:**
```typescript
const SleepTool = feature('PROACTIVE') ? require('./SleepTool').SleepTool : null
```

**Tool Search/Deferral:**
When tool count exceeds threshold, tools are deferred — name only in prompt, loaded on demand via `ToolSearchTool`. This prevents context explosion while keeping tools discoverable.

**MCP Tool Wrapper:**
MCP tools are dynamically instantiated with proper `inputJSONSchema`:
```typescript
return { ...MCPTool, name: fullyQualifiedName, inputJSONSchema: tool.inputSchema }
```

**Permission system decoupled from availability:**
Tools appear in prompts even if denied. Permission checks happen at invocation time, not discovery time.

### Applicable Lessons

1. First-class schemas > dispatcher indirection for LLM reliability
2. Deferral/search pattern handles tool count scaling
3. Feature flags enable context-dependent tool inclusion
4. Execution routing should be name-based, not schema-format-dependent (our system already does this)

### Not Applicable

1. Their MCP wrapper pattern — we don't use MCP for platform tools
2. Their permission mode system — we have our own via `validate_tool_access()`
3. Their coordinator mode — different architecture than our mission coordinator

---

## Appendix B: Full Dependency Map

### Tool Assembly Chain
```
get_tools_for_agent() [tool_router.py:129]
  ├── ToolRegistry.get_all() → core tool schemas
  ├── ActionRegistry.to_dispatcher_schema() → platform_execute
  ├── ActionRegistry.to_first_class_schemas() → promoted schemas [NEW]
  ├── _FIELD_TOOL_SCHEMAS → field tools [TO BE REMOVED, consolidated into promoted]
  └── ComposioActionCache enrichment → composio_execute params
```

### Consumers
```
get_tools_for_agent()
  ├── consumers/chatbot/service.py:584 (chat sessions)
  ├── modules/context/sections/tools.py:129 (FULL strategy)
  ├── modules/context/sections/tools.py:150 (FILTERED → SmartToolRouter)
  ├── modules/agents/factory/agent_factory.py:799 (mission/task agents)
  └── api/chatbot_llm.py:248,584 (API endpoint)
```

### Execution Routing
```
UnifiedToolExecutor.execute_tool() [unified_executor.py:310]
  ├── "platform_execute" → unwrap action+params → PlatformActionExecutor
  ├── "platform_*" → direct → PlatformActionExecutor [handles promoted tools]
  ├── "workspace_*" → WorkspaceClient proxy
  ├── "composio_execute" → Composio meta-tool
  ├── COMPOSIO_* per-action → Composio SDK executor
  └── other → ToolRegistry handler
```
