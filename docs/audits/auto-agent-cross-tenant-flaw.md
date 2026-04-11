# Auto Agent Cross-Tenant Flaw — Investigation & Fix Plan

**Date:** 2026-04-11
**Author:** Claude (Opus 4.6) session
**Severity:** HIGH — cross-tenant data exposure via shared agent fallback
**Status:** Flaw documented. Quick fix: none required for 2026-04-11 demo (current behaviour routes all workspaces to admin's agent by design). Proper fix: scoped below, not yet implemented.

---

## TL;DR

The "Auto" chat experience is not a real agent. It is a **picker function** (`get_default_agent_id` in `orchestrator/api/chat.py:264-322`) that scans the calling workspace for agents with Composio app assignments, picks the one with the most, and — critically — falls back to a hardcoded `return 1` when nothing matches. Agent id `1` is the admin workspace's very first agent.

Consequence: **every workspace that does not yet have Composio-assigned agents chats with the admin workspace's agent.** This is a silent cross-tenant leak. New users' chats hit admin's agent, admin's system prompt, admin's tool assignments, admin's LLM configuration — and any memories that agent has accumulated.

Separately, the Settings > Orchestrator page appears to let each workspace design its own Auto (persona, soul, heartbeat, LLM). It does not. The `PUT /api/workspaces/current/orchestrator` endpoint writes those fields to `workspace.settings` (a JSONB blob on the Workspace row). **No code anywhere reads those fields to build a system prompt, pick a model, or instantiate an agent.** The Orchestrator settings form is disconnected plumbing — it saves state that nothing consumes.

The two bugs compound: users think they are configuring their own Auto (they are not — the save is inert), and while they think they are chatting with their own Auto, they are actually chatting with admin's agent.

---

## Evidence

### 1. The hardcoded agent fallback

`orchestrator/api/chat.py:264-322`, function `get_default_agent_id(db, workspace_id)`:

```python
def get_default_agent_id(db: Session, workspace_id) -> int:
    """
    Pick a sensible default agent for chat when the client does not send agentId.
    Preference:
    - Any agent in this workspace with active EXTERNAL app assignments (Composio),
      ordered by number of assignments (desc).
    - Fallback to agent id=1.
    """
    try:
        # ... query AgentAppAssignment joined to Agent, filtered by workspace_id,
        #     group by agent_id, order by count(assignments) desc, limit 10 ...
        for (agent_id,) in candidates:
            if not agent_id:
                continue
            if connected_apps:
                # verify the agent's assigned apps intersect with
                # the workspace's actually-connected Composio apps
                ...
                if not assigned_apps.intersection(connected_apps):
                    continue
            return int(agent_id)
    except Exception:
        pass
    return 1
```

The function is workspace-scoped in its query (`Agent.workspace_id == workspace_id`), so the first branch is tenant-safe. The problem is the **terminal `return 1`**: when a workspace has no candidate agents — i.e. any fresh workspace, or any workspace where no agent has Composio assignments — it returns agent id `1` regardless of which tenant called.

Agent id `1` is the earliest-created agent in the database, which in production is an agent belonging to the admin workspace.

### 2. The disconnected Orchestrator settings form

`orchestrator/api/workspaces.py:272-316`, handler `save_orchestrator_settings`:

```python
@router.put("/current/orchestrator")
async def save_orchestrator_settings(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    workspace = db.query(Workspace).get(ctx.workspace_id)
    # ... validation of personality_mode, communication_style, proactive_level,
    #     thinking_level, heartbeat sub-fields ...
    settings = workspace.settings or {}
    settings["orchestrator"] = {
        "personality_mode": ...,
        "custom_soul": ...,
        "communication_style": ...,
        "proactive_level": ...,
        "thinking_level": ...,
        "heartbeat": { ... },
        "harness": { ... },
    }
    workspace.settings = settings
    db.commit()
    return settings["orchestrator"]
```

A grep across the codebase for readers of `workspace.settings["orchestrator"]` or `settings.get("orchestrator")` returns only the matching GET handler. No agent construction path, no system-prompt builder, no LLM manager, no heartbeat scheduler reads these fields. The form is a UI-only illusion of control.

The form also has no `llm.provider` / `llm.model` field in the payload. The LLM dropdowns that were crashing in `SystemLLMSettingsTab.tsx` are for the legacy system-wide default, not for a per-workspace Auto agent.

### 3. What users are actually configuring

- **Agent roster** (visible): users create agents manually, assign them tools, assign them an LLM. These agents have proper `Agent` rows, proper `workspace_id` scoping, proper tool plumbing.
- **Auto** (invisible): the picker function. Users cannot see it, cannot configure it, cannot inspect what it chose.
- **Orchestrator settings page** (visible, but inert): users think this configures Auto. It does not.

The mental model the UI promises ("I design my Auto's soul, personality, LLM") and the reality (a function that `return 1`s) are completely disconnected.

---

## Blast radius

### Who is affected today

Any workspace in production where:
- At least one chat was initiated via Auto (no `agentId` in the request), AND
- The workspace has no agents with Composio app assignments

→ that chat ran against agent id=1, which is an admin-owned agent.

Based on the current state of the admin workspace (heavy Composio use, many assignments), the first branch of `get_default_agent_id` *will* succeed for the admin workspace, so admin is mostly hitting its own agents. New tenants, test workspaces, and workspaces with Composio configured but not assigned to agents are all silently hitting admin.

### What leaks

When chat runs under an agent, the following belong to that agent's tenant:
- System prompt / persona / soul
- Tool assignments (including Composio tokens, workspace files, platform actions)
- LLM model & API key
- Conversation memory injected into the system prompt (per PRD-108 / Mem0 wiring)
- Any reports the agent has submitted via `platform_submit_report`
- Any scratchpad state

A cross-tenant user hitting admin's agent can, in the worst case, trigger tools scoped to the admin workspace (if the tools themselves don't re-check `ctx.workspace_id`), see admin's memories injected into the system prompt, and observe admin's persona framing of responses. Whether a tool call actually executes cross-tenant depends on whether each tool validates the caller's workspace against the agent's workspace — this needs a separate audit.

### What does NOT leak (that I can confirm)

- The RequestContext passed to the chat endpoint still has the **caller's** workspace_id. So any code path that reads `ctx.workspace_id` directly (e.g. `StreamingChatService(db, workspace_id=ctx.workspace_id)` at chat.py:339) is safe. The leak is specifically in the agent-bound config (prompt, model, tool list, memories) that flows from `Agent.workspace_id`, not in request-scoped operations.
- Tools that go through `get_tools_for_agent` get the agent's tool list, not the caller's. A tool router does not currently reconcile agent workspace vs caller workspace — so a tool allowed on admin's agent will appear on the tool list even when a non-admin user's message invokes it.

This split — caller workspace vs agent workspace being different — is the part that needs the deepest audit. Every tool handler needs to be checked for "does it use `agent.workspace_id` or `ctx.workspace_id` when scoping its read/write?"

---

## Why this happened

Three factors, in historical order:

1. **Legacy MVP shortcut.** Early in the project, "Auto" meant "pick any agent that exists." `return 1` was a hack for the single-tenant dev phase, never revisited after multi-tenancy landed.

2. **Settings page built without a consumer.** Someone (earlier session) built the Orchestrator settings UI on the assumption that a downstream component would later read `workspace.settings.orchestrator` and bind it to an agent. That downstream component was never built, and the UI shipped anyway because it looked finished.

3. **No integration test on tenant isolation for Auto chats.** The existing tenant-isolation tests (see the multi-tenancy fix on 2026-02-07 in MEMORY.md) covered workspace resolution and workspace access control, but not "what agent does an auto-routed chat land on." The picker was assumed correct because the query had a `workspace_id` filter — the terminal `return 1` was invisible to anyone not reading the function's bottom.

---

## Fix plan (proper, not the quick fix)

### Principle

Auto becomes a **real, per-workspace, hidden agent row.** Same pattern as PRD-67's CTO Agent (`is_system_agent=True`, `slug="auto-cto"`). The Orchestrator settings page becomes the editor for that row. The picker function dies and is replaced with a direct lookup.

### Changes

**1. Schema / data model**

Add `llm_provider` and `llm_model` to the Orchestrator settings payload schema, if not already on the Agent model (they are — `Agent.llm_provider`, `Agent.llm_model` already exist). No migration needed; `is_system_agent` column already exists from PRD-67.

**2. `workspaces.py` — upsert on save**

On `PUT /current/orchestrator`, after validation, upsert an `Agent` row:
- `workspace_id = ctx.workspace_id`
- `slug = f"auto-{ctx.workspace_id}"`
- `is_system_agent = True`
- `name = "Auto"`
- `role = "orchestrator"`
- `system_prompt = build_soul(payload.personality_mode, payload.custom_soul, ...)`
- `llm_provider = payload.llm.provider`
- `llm_model = payload.llm.model`

Store the resulting agent id on `workspace.settings.orchestrator.agent_id` for fast lookup by chat.py.

On `GET /current/orchestrator`, if the Auto agent row exists, merge its current `system_prompt`, `llm_provider`, `llm_model` back into the response so the form round-trips what's actually in effect.

**3. `chat.py` — replace the picker**

```python
def get_default_agent_id(db: Session, workspace_id) -> int:
    from core.models import Agent
    auto = (
        db.query(Agent.id)
        .filter(
            Agent.workspace_id == workspace_id,
            Agent.slug == f"auto-{workspace_id}",
            Agent.is_system_agent == True,  # noqa: E712
        )
        .scalar()
    )
    if auto:
        return int(auto)
    # Pre-configuration fallback: first agent in THIS workspace (never cross-tenant)
    first = (
        db.query(Agent.id)
        .filter(Agent.workspace_id == workspace_id)
        .order_by(Agent.id.asc())
        .scalar()
    )
    if first:
        return int(first)
    raise HTTPException(500, "No agent available for workspace. Configure Auto at Settings > Orchestrator.")
```

The Composio-based heuristic is removed entirely. Auto is explicit, not guessed.

**4. Roster filtering**

The agents list endpoint (`/api/agents` — needs to be located) must add `.filter(Agent.is_system_agent != True)` so Auto does not appear in the visible Roster UI. PRD-67's CTO agent already relies on this behaviour — grep for `is_system_agent` in the agents API to confirm current filtering.

**5. Admin workspace migration (one-time)**

Because admin's current chats rely on agent id=1's persona/model/tools, seed admin's Auto row from agent #1 before deploying the chat.py change:

```python
# One-time migration script
admin_ws_id = <admin workspace id>
agent_one = db.query(Agent).get(1)
auto = Agent(
    workspace_id=admin_ws_id,
    slug=f"auto-{admin_ws_id}",
    is_system_agent=True,
    name="Auto",
    role="orchestrator",
    system_prompt=agent_one.system_prompt,
    llm_provider=agent_one.llm_provider,
    llm_model=agent_one.llm_model,
)
db.add(auto)
db.commit()
# Also copy tool assignments from agent 1 to the new auto row.
```

Without this, admin's demo experience will regress the moment chat.py flips.

### Deploy order

1. Ship the `workspaces.py` upsert (creates rows, does nothing visible)
2. Ship the roster filter (hides new rows from the UI)
3. Run the admin migration script
4. Ship the `chat.py` picker replacement (now live)

This ordering ensures no window where chat breaks for lack of a row.

### Out of scope for this fix

- Tool handlers that might use `agent.workspace_id` vs `ctx.workspace_id` inconsistently. That is a **separate audit** and should block nothing here, but must be tracked.
- Migrating existing non-admin workspaces retroactively. They will get a fresh Auto row on first save; until then they use the "first agent in workspace" fallback, which is tenant-safe.
- Multiple Auto variants per workspace (e.g. "research Auto" vs "writing Auto"). The slug scheme supports it (`auto-{workspace_id}-{variant}`) but the UI does not.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Admin demo regresses when chat.py flips | Seed admin's Auto row from agent #1 before flipping |
| Non-admin workspaces without agents get 500s | Fallback to "first agent in workspace"; only raise if literally zero agents exist |
| Tool handlers assume caller workspace == agent workspace | Separate audit; track as follow-up issue |
| Orchestrator settings form still missing LLM fields | Add `llm.{provider,model}` to the payload schema before the `workspaces.py` upsert ships |
| Memories/reports from agent #1 leak into admin's new Auto | Seeded migration intentionally copies agent #1's state; this is a feature not a bug for admin |

---

## Recommended next actions

1. **Today (demo):** no code change. Current behaviour routes all Auto chats to admin's agent #1, which is the desired demo experience. The SelectItem crash fix already deployed unblocks the Orchestrator settings page from loading.

2. **Tomorrow:** open a PRD for the proper fix. Include:
   - This document as the background section
   - The 5-step deploy order above
   - A tool-handler audit task as a sibling workstream
   - Acceptance: "new workspace chats with Auto never hit an agent whose workspace_id != caller workspace_id"

3. **Before the proper fix ships:** run a grep across tool handlers for uses of `agent.workspace_id` and flag any that should be `ctx.workspace_id` instead. This is the highest-value pre-work because it reveals whether the current leak has a tool-execution dimension or is prompt/memory-only.

4. **Long-term:** tenant isolation tests should include "Auto chat routing" as a first-class test matrix: every combination of (admin/non-admin caller) × (has-agents/no-agents workspace) must land on an agent owned by the caller's workspace.

---

## Appendix: why the quick fix is "no code change"

The user's demo is in the admin workspace. The admin workspace has many agents with Composio assignments, so `get_default_agent_id` picks one of them via the first branch — no `return 1` fallback needed. Even if the fallback triggered, it returns admin's agent #1, which is already the expected admin experience.

The *only* user-visible fix needed for today is the Radix `SelectItem` crash on the Orchestrator settings page, which was shipped as commit `e565007e6` earlier in this session. That restores the page's ability to render without touching any of the plumbing described above.
