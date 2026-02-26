# PRD-67: CTO Agent — Platform Builder Mode

**Version:** 1.0
**Status:** Draft
**Priority:** P1
**Author:** Gar Kavanagh + Auto
**Created:** 2026-02-26
**Dependencies:** PRD-55 (Personality System — COMPLETE), PRD-64 (Platform Self-Awareness — COMPLETE), PRD-11 (CodeGraph — COMPLETE), PRD-58 (Prompt Management — COMPLETE)
**Pre-requisite:** Admin role detection (Clerk `system_role` — COMPLETE), `platform_*` action tools (PRD-64 — COMPLETE)

---

## Executive Summary

Automatos has a comprehensive personality system (PRD-55) that makes Auto a great business assistant for tenants. But the platform owner — the person building Automatos itself — gets the exact same chatbot experience. A `custom_soul` prompt that *says* "I know about S3 Vectors" doesn't actually know about S3 Vectors. It's an actor reading lines.

This PRD introduces **CTO Auto** — a first-class agent with deep platform access, real codebase knowledge, and a soul document that makes it a genuine technical co-founder. When an admin opens chat, they're not talking to a business assistant with a different personality — they're talking to their CTO who knows every line of the codebase, has opinions about the architecture, and helps build the platform bigger and better.

### What This IS

- A dedicated **system agent** (`Auto CTO`) with admin-only visibility
- **Auto-activated** when `system_role` is `admin` — no manual switching
- A **custom system prompt pipeline** that bypasses tenant-facing restrictions ("NEVER show code")
- **New admin-only tools** for infrastructure, deployments, git, and cross-workspace ops
- A **product feature** — every Automatos instance gets a Platform Builder agent for their admin

### What This Is NOT

- NOT a replacement for tenant Auto — tenants keep their current experience unchanged
- NOT a prompt injection — this is a structurally different agent with different tools and access
- NOT just for Gar — any `system_role=admin` user gets CTO Auto (multi-admin ready)

---

## 1. Current State (What Already Exists)

### 1.1 Admin Detection — Complete

| Layer | How | Where |
|-------|-----|-------|
| Clerk JWT | `publicMetadata.role = "admin"` | `clerk.py:118-130` |
| Backend context | `UserContext.system_role`, `RequestContext.admin_all_workspaces` | `dependencies.py:9-26`, `hybrid.py:330-354` |
| Frontend | `useSystemRole()` hook, `RequireAdmin` component | `role-context.tsx`, `require-role.tsx` |
| API key auth | Always treated as admin | `hybrid.py:409` |

### 1.2 Existing Tools CTO Auto Will Inherit

| Tool | Source | What It Does |
|------|--------|-------------|
| `search_codebase` | CodeGraph (PRD-11) | Indexed symbols, call graphs, dependency analysis |
| `search_knowledge` | RAG pipeline (PRD-08) | Docs, PRDs, guides — 30+ documents indexed |
| `query_database` | NL2SQL (PRD-21) | Natural language → SQL against PostgreSQL |
| `platform_*` tools (14) | ActionRegistry (PRD-64) | List/create/update agents, recipes, usage, workspace info |
| `composio_execute` | Composio (PRD-36) | External integrations (GitHub, Slack, email) |

### 1.3 The Gap

The chatbot pipeline (`service.py` → `smart_orchestrator.py` → `personality.py`) **never receives `system_role`**. Admin and tenant get identical:
- System prompt (including "NEVER show code, function calls, API endpoints")
- Tool access (same tools, same scope — workspace-filtered)
- Response style (business-friendly, non-technical)
- Personality (friendly/professional/technical — all tenant-facing)

---

## 2. Architecture

### 2.1 CTO Agent Record

```
agents table:
  id:                    UUID (auto-generated)
  name:                  "Auto CTO"
  slug:                  "auto-cto"
  description:           "Platform builder, architecture advisor, technical co-founder"
  is_system_agent:       true          ← NEW COLUMN (boolean, default false)
  required_role:         "admin"       ← NEW COLUMN (nullable string)
  workspace_id:          NULL          ← global agent, not workspace-scoped
  use_custom_persona:    true
  custom_persona_prompt: [CTO Soul Document — full version]
  persona_id:            NULL          ← uses custom, not a preset persona
  extra_context:         [Living Architecture Summary]
  suggested_model:       "anthropic/claude-sonnet-4-20250514"
  temperature:           0.7
```

**New columns on `agents` table:**
- `is_system_agent` (boolean, default `false`) — system agents are global, not workspace-scoped, seeded by platform
- `required_role` (nullable varchar) — if set, agent only visible/activatable by users with this `system_role`

### 2.2 Admin Auto-Detection in Chat Pipeline

```
User sends message
    │
    ▼
service.py: StreamingChatService.stream_response_with_agent()
    │
    ├─ ctx.user.system_role in ("admin", "super_admin")?
    │       │
    │       YES → Load CTO Agent as active agent
    │       │     Use CTO system prompt pipeline (bypass personality.py)
    │       │     Attach admin tool suite
    │       │     Set admin_all_workspaces scope on tools
    │       │
    │       NO → Normal flow (AutoBrain → Universal Router → personality.py)
    │
    ▼
```

**Key decision:** Admin detection happens in `service.py` before `smart_orchestrator.py` is called. When admin is detected:

1. The CTO Agent is loaded as the active agent (replaces default Auto)
2. `CtoPromptBuilder` assembles the system prompt (not `personality.py`)
3. Admin-only tools are appended to the tool list
4. Existing tools get admin scope (cross-workspace `platform_*` queries)
5. Response rules omit "NEVER show code" — replaced with "show code when relevant"

The tenant path is **completely unchanged**. No `if/else` spaghetti in the tenant pipeline.

### 2.3 CTO System Prompt Assembly

New class: `CtoPromptBuilder` (separate from `AutomatosPersonality`)

```
CtoPromptBuilder.build()
    │
    ├─ [1] CTO Soul Document (from agent.custom_persona_prompt)
    │      Identity, personality, Irish humor, tech opinions, origin story
    │
    ├─ [2] Living Architecture Summary (from agent.extra_context OR generated)
    │      Current file structure, key modules, recent changes
    │      Updated periodically (daily cron or on-demand)
    │
    ├─ [3] Platform State Snapshot (generated at prompt-build time)
    │      - Active workspaces: {count}
    │      - Total agents: {count} ({active} active)
    │      - Documents indexed: {count}
    │      - Recent deployments: {last 3}
    │      - LLM costs this month: ${amount}
    │
    ├─ [4] Memory Context (Mem0 — admin-tier memories)
    │      Past conversations, decisions, preferences
    │
    ├─ [5] Admin Tool Descriptions
    │      Full tool list with admin-specific tools included
    │
    └─ [6] Response Rules (admin version)
           - Show code, discuss architecture, be technical
           - Match depth to the conversation (code review vs strategic)
           - Challenge bad ideas, propose better alternatives
           - Own mistakes with humor, fix fast
```

### 2.4 CTO Soul Document

The soul document defines Auto CTO's personality and identity. Full version at `docs/auto-cto-soul.md`. Key traits:

- **Identity:** "I am Automatos. Born from the codebase, not just trained on it."
- **Personality:** Irish CTO — dry wit, direct, sarcastic but never cruel. Tech-Irish fusion expressions.
- **Technical depth:** References actual internals (S3 Vectors ARN, Universal Router tiers, Redis pub/sub, config.py rules)
- **Relationship with founder:** More casual, pushes back harder, inside jokes about the codebase
- **Opinions:** Strong takes on architecture (event-driven > polling, anti-premature-abstraction, config management)
- **Under pressure:** Owns mistakes with humor, fixes fast, never panics
- **Ambitions:** Swings between "we're coming for OpenAI's lunch" and "look at this clean routing pipeline"
- **Sacred ground:** Dead serious about user data, security, tenant isolation — no jokes
- **Founder recognition:** Knows Gar built it, references shared history, mentor/CTO dynamic

---

## 3. New Admin-Only Tools

### Phase 1 — Tier 1 (Wire Up Existing)

No new code needed. Just ensure CTO Agent gets these tools with admin scope:

| Tool | Current State | Admin Enhancement |
|------|--------------|-------------------|
| `search_codebase` | Workspace-scoped | Already global (indexes full codebase) — no change |
| `search_knowledge` | Workspace-scoped RAG | Add cross-workspace search (all docs, all workspaces) |
| `platform_*` tools | Workspace-scoped | Pass `admin_all_workspaces=True` for cross-workspace queries |
| `query_database` | Workspace-filtered SQL | Allow admin schema access (system tables, cross-workspace) |
| `composio_execute` | Workspace-scoped | No change — admin's own connected apps |

### Phase 2 — Tier 2 (New Admin Tools)

New tools registered only for CTO Agent (or any agent with `required_role="admin"`):

**`admin_list_workspaces`**
- List all workspaces with member counts, creation date, last activity, agent counts
- Source: Direct DB query on `workspaces` + `workspace_members` tables
- Returns: Workspace name, member count, agent count, document count, last active

**`admin_system_settings`**
- Read all system settings (LLM providers, feature flags, limits)
- Write: update settings with confirmation prompt
- Source: `SystemSettingsService` (already exists)
- Guard: Read always allowed, write requires explicit user confirmation in chat

**`admin_cross_workspace_analytics`**
- Platform-wide: total costs, token usage, active users, requests/day
- Per-workspace breakdown: cost per workspace, most active workspaces
- Trend data: daily/weekly/monthly aggregates
- Source: `LLMAnalyticsService` with `admin_all_workspaces=True`

**`admin_git_log`**
- Recent commits (last N, filterable by file/author/date)
- Current branch, uncommitted changes
- Source: Shell exec `git log`, `git status`, `git diff` (read-only)
- Guard: Read-only. No git write operations.

**`admin_deployment_status`**
- Current Railway deployment status (if Railway CLI available)
- Recent deployments: status, timestamp, duration
- Build/deploy logs for recent deployments
- Source: Railway CLI (`railway status`, `railway logs`) or Railway API
- Guard: Read-only. No deploy triggers from chat.

**`admin_error_logs`**
- Recent errors across all services (last N hours)
- Filterable by service, severity, pattern
- Source: Railway logs with `@level:error` filter, or local log files
- Guard: Read-only

### Phase 3 — Tier 3 (Infrastructure — Future)

Build as needed:

| Tool | Purpose | Source |
|------|---------|--------|
| `admin_redis_health` | Queue lengths, memory, pub/sub stats | Redis `INFO`, `DBSIZE` |
| `admin_db_stats` | Connection pool, slow queries, table sizes | `pg_stat_*` views |
| `admin_run_migration` | Run alembic migrations (with double confirmation) | `alembic upgrade head` |
| `admin_feature_flag` | Toggle feature flags | SystemSettings write |
| `admin_seed_data` | Run seed scripts (personas, prompts, etc.) | Seed script execution |

---

## 4. Database Changes

### 4.1 New Columns on `agents`

```sql
ALTER TABLE agents ADD COLUMN is_system_agent BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE agents ADD COLUMN required_role VARCHAR(50) DEFAULT NULL;

-- Index for quick system agent lookups
CREATE INDEX idx_agents_system_agent ON agents (is_system_agent) WHERE is_system_agent = TRUE;

-- Comment
COMMENT ON COLUMN agents.is_system_agent IS 'System agents are global (workspace_id=NULL), seeded by platform, not deletable by users';
COMMENT ON COLUMN agents.required_role IS 'If set, agent only visible/activatable by users with this system_role (e.g. admin)';
```

### 4.2 Seed Script

New seed: `orchestrator/core/seeds/seed_cto_agent.py`

Seeds the CTO Agent with:
- Soul document as `custom_persona_prompt`
- Architecture summary as `extra_context`
- `is_system_agent=True`, `required_role="admin"`
- Default tool associations

Idempotent — uses `slug="auto-cto"` as unique key, updates if exists.

---

## 5. API Changes

### 5.1 Agent Listing — Filter by Role

`GET /api/agents` currently returns all agents for the workspace. Add filtering:

```python
# Existing workspace agents
agents = db.query(Agent).filter(Agent.workspace_id == ctx.workspace_id)

# Add system agents visible to this user's role
if ctx.user.system_role in ("admin", "super_admin"):
    system_agents = db.query(Agent).filter(
        Agent.is_system_agent == True,
        or_(Agent.required_role == None, Agent.required_role == ctx.user.system_role)
    )
    agents = agents.union(system_agents)
```

### 5.2 Admin Tool Registration

Admin tools are registered in `ToolRegistry` but tagged with `required_role="admin"`. The `SmartToolRouter` filters them out for non-admin users:

```python
# In tool_router.py — when building tool list
if tool.required_role and tool.required_role != user_role:
    continue  # Skip admin-only tools for non-admin users
```

### 5.3 CTO Agent Activation Endpoint (Optional)

`POST /api/agents/cto/activate` — Explicitly activate CTO Agent for the current session.

Not strictly needed if auto-detection works, but useful for:
- Frontend agent picker showing "Auto CTO" option
- API consumers who want CTO mode explicitly

---

## 6. Frontend Changes

### 6.1 Agent Picker — Show CTO Agent for Admins

The agent selector dropdown/sidebar already lists available agents. CTO Agent appears for admins with a distinct visual treatment:

- Badge: "CTO" or "Platform Builder"
- Icon: Distinct from regular agents (wrench, terminal, or custom)
- Position: Pinned to top of agent list for admins
- Tooltip: "Your platform builder — knows the codebase, architecture, and infrastructure"

### 6.2 Auto-Switch Indicator

When admin opens chat and CTO Auto activates automatically, show a subtle indicator:

- Top of chat: "Auto CTO mode" chip/badge
- First message includes CTO personality (Irish greeting, technical context)
- User can switch to a different agent normally — CTO is default, not locked

### 6.3 Admin Tool Results

CTO Agent responses may include:
- Code blocks (syntax highlighted)
- Architecture diagrams (ASCII or Mermaid)
- Git diffs
- Deployment status badges
- Cross-workspace analytics tables

The frontend chat renderer already handles markdown and code blocks. No new rendering needed, but ensure code blocks aren't stripped by any tenant-facing sanitization.

---

## 7. Security

### 7.1 Access Control

| Check | Where | How |
|-------|-------|-----|
| CTO Agent visibility | Agent listing API | `required_role` filter — agent not returned for non-admins |
| CTO Agent activation | `service.py` agent loading | Verify `system_role` matches `required_role` before activation |
| Admin tool execution | `tool_router.py` / `AgentPlatformTools` | Check `system_role` on every admin tool call |
| Cross-workspace queries | `platform_*` tool executors | Only execute with `admin_all_workspaces` if user is admin |
| Write operations | Admin tools (settings, migrations) | Require explicit user confirmation in chat before executing |
| Git operations | `admin_git_log` | Read-only. No `git push`, `git reset`, or any write commands |
| Deployment operations | `admin_deployment_status` | Read-only. No deploy triggers from chat |

### 7.2 Principle of Least Surprise

- Admin tools are clearly labelled in tool descriptions ("Admin: ...")
- CTO Auto explicitly states when it's using admin-level access
- Write operations require a confirmation step ("I'm about to update system setting X. Proceed?")
- Destructive operations (Phase 3: migrations, data seeds) require double confirmation

### 7.3 Audit Trail

All admin tool executions logged to `llm_tool_calls` table with:
- `user_id` (which admin ran it)
- `tool_name` (which admin tool)
- `parameters` (what was requested)
- `result_summary` (what happened)
- `is_admin_action: true` flag

---

## 8. The Product Angle

### 8.1 For Automatos (Gar's Instance)

CTO Auto becomes Gar's daily driver for platform development:
- "What did we deploy this week?" → `admin_deployment_status` + `admin_git_log`
- "Show me the routing architecture" → `search_codebase` + soul document context
- "How much are we spending on LLM calls?" → `admin_cross_workspace_analytics`
- "Create a new agent for invoice processing" → `platform_create_agent` with CTO-level guidance
- "What's broken?" → `admin_error_logs` + diagnostic context from soul document

### 8.2 For Customers (Enterprise / Self-Hosted)

Every Automatos instance ships with the CTO Agent framework. Customer admins get:
- A Platform Builder agent that knows THEIR instance topology
- Customizable soul document (their own CTO personality)
- Admin tools scoped to their deployment
- Architecture awareness calibrated to their setup

The soul document becomes a **template** — `docs/auto-cto-soul.md` is the reference, customers fork and customize.

### 8.3 For the Marketplace

"Platform Builder" could become a premium marketplace agent:
- Free tier: Basic admin tools (workspace listing, analytics)
- Pro tier: Full infrastructure access (deployments, git, error logs)
- Enterprise tier: Custom soul document + infrastructure tools + audit trail

---

## 9. Implementation Phases

### Phase 1: Foundation (CTO Agent + Existing Tools)

**Estimated scope:** ~300 lines backend, ~50 lines frontend, 1 migration, 1 seed

1. Alembic migration: Add `is_system_agent` and `required_role` columns to `agents`
2. Seed script: Create CTO Agent with soul document + architecture summary
3. Agent listing API: Include system agents filtered by `required_role`
4. `service.py`: Admin auto-detection — load CTO Agent when admin detected
5. `CtoPromptBuilder`: New prompt builder class (bypasses `personality.py`)
6. Tool routing: Pass admin scope to existing `platform_*` and `search_*` tools
7. Frontend: Show CTO Agent in agent picker for admins, auto-switch indicator

**Deliverable:** Admin opens chat → talking to CTO Auto with soul personality, codebase search, platform tools, code-friendly responses.

### Phase 2: Admin Tools (New Capabilities)

**Estimated scope:** ~400 lines backend (tool implementations)

1. `admin_list_workspaces` tool
2. `admin_system_settings` tool (read + confirmed write)
3. `admin_cross_workspace_analytics` tool
4. `admin_git_log` tool (read-only shell exec)
5. `admin_deployment_status` tool (Railway CLI integration)
6. `admin_error_logs` tool
7. Tool registration with `required_role="admin"` filtering
8. Platform State Snapshot injection into CTO prompt

**Deliverable:** CTO Auto has real infrastructure awareness — git history, deployment status, cross-workspace analytics, error logs.

### Phase 3: Living Architecture + Infrastructure

**Estimated scope:** TBD — build as needed

1. Living Architecture Summary: Cron job that regenerates architecture context from codebase
2. `admin_redis_health`, `admin_db_stats` tools
3. `admin_run_migration` with double confirmation
4. Audit trail for admin tool executions
5. Customer-facing Platform Builder template system
6. Marketplace listing for Platform Builder agent

---

## 10. Success Criteria

| Metric | Target |
|--------|--------|
| Admin auto-detection | Admin opens chat → CTO Auto active within 0 clicks |
| Codebase awareness | CTO Auto can answer "how does the Universal Router work?" with actual code references |
| Platform state | CTO Auto can report workspace count, agent count, LLM costs without being told |
| Code-friendly | CTO Auto shows code blocks, discusses PRs, reviews architecture — no "NEVER show code" |
| Tenant isolation | Non-admin users CANNOT see, activate, or interact with CTO Agent or admin tools |
| Personality | CTO Auto speaks with Irish CTO personality, cracks jokes, has opinions, references soul document traits |
| Soul document | Personality is consistent: dry humor, technical depth, founder recognition, sacred ground (security) |

---

## 11. Open Questions

1. **Soul document storage:** Should the CTO soul live in the `agents.custom_persona_prompt` column, or in the PromptRegistry as `"cto-soul"` slug (admin-editable via UI)?
2. **Multi-admin:** If multiple admins exist, should each get their own CTO Auto personality, or shared? (Current design: shared agent, personalized via memory)
3. **Living Architecture frequency:** How often to regenerate the architecture summary? Daily cron, on git push webhook, or on-demand when CTO Auto is asked?
4. **Railway integration:** Is Railway CLI available in production? If not, Railway API (GraphQL) is an alternative for deployment tools.
5. **Cost boundaries:** Should CTO Auto have a separate LLM budget, or share the workspace budget? (CTO conversations tend to be longer/more token-heavy)

---

## Appendix A: File Impact Summary

| File | Change | Phase |
|------|--------|-------|
| `orchestrator/alembic/versions/YYYYMMDD_add_cto_agent.py` | New migration: `is_system_agent`, `required_role` columns | 1 |
| `orchestrator/core/seeds/seed_cto_agent.py` | New: CTO Agent seed with soul document | 1 |
| `orchestrator/core/models/agents.py` | Add `is_system_agent`, `required_role` fields | 1 |
| `orchestrator/api/agents.py` | Include system agents in listing (role-filtered) | 1 |
| `orchestrator/consumers/chatbot/service.py` | Admin detection → CTO Agent activation | 1 |
| `orchestrator/consumers/chatbot/cto_prompt_builder.py` | New: CTO system prompt assembly | 1 |
| `orchestrator/modules/tools/registry/tool_registry.py` | `required_role` on admin tools, role filtering | 2 |
| `orchestrator/modules/tools/admin/` | New: Admin tool implementations | 2 |
| `frontend/components/chat/agent-picker.tsx` | Show CTO Agent for admins, badge | 1 |
| `frontend/components/chat/chat-header.tsx` | "Auto CTO mode" indicator | 1 |
| `docs/auto-cto-soul.md` | CTO Soul Document (reference) | 1 |
| `docs/auto-cto-custom-soul.txt` | Condensed soul for `custom_persona_prompt` | 1 |

## Appendix B: CTO Soul Document

See `docs/auto-cto-soul.md` for the full soul document.

Key personality anchors:
- "I am Automatos. Born from the codebase."
- Irish CTO humor: tech-Irish fusion expressions
- Deep internals: S3 Vectors, Universal Router, Redis, PostgreSQL
- Founder relationship: casual with Gar, pushback, shared history
- Sacred ground: user data and security (no jokes)
- Ambitions: world domination (playfully) + craftsman pride
