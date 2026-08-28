# PRD-228 — Fleet State: Situational Awareness for Auto and Humans

> **Status:** Draft for rollout planning — written 2026-08-27, not yet scheduled.
> **Origin:** Munder Difflin deep review (2026-08-27) — their `fleet.json` (per-agent tokens · cost · status · breaker level · last tool · inbox backlog, refreshed continuously) is the orchestrator's situational awareness; we have every input and no aggregate. Review artifact:
> https://claude.ai/code/artifact/f31677a8-f2cb-47fe-b7dd-f705d764418b
> **Type (per CLAUDE.md §3):** Extension — a read-model over existing state; one route, one tool, one view.

## 1. Overview

"Auto, how's the team doing?" should get a grounded answer, and the Agents page should show what each agent is doing *right now*. Today neither is possible: agent detail shows aggregate counters and the last five historical log rows; roster badges are the human-toggled lifecycle (`active | idle | maintenance`); and Auto has no tool that returns live floor state. PRD-226's "awareness" doctrine stays aspirational until this read-model exists.

## 2. Current reality (grounded)

- **No live agent view.** Roster badge is human-set lifecycle (`frontend/components/agents/agent-roster.tsx:324-329`, toggled via `agent-status-control-modal.tsx`); the details modal's `workload` tab shows totals plus 5 historical log rows (`agent-details-modal.tsx:529-598`) — retrospective, not current. No per-agent page exists (`frontend/app/agents/page.tsx` → management component only).
- **Every input exists, scattered:**
  - Board: `assigned_agent_id`, `status`, `lease_until`, `attempts`, SLA fields on `board_tasks` (`core/models/core.py:1558-1643`); claim/lease logic in `services/board_dispatcher.py`.
  - Missions: `orchestration_tasks` state per agent (`core/models/orchestration.py:156`); dispatcher busy-tracking already feeds the matcher's busy filter (`agent_matcher.py:459`).
  - Heartbeats: `services/heartbeat_service.py` per-agent cadence/state.
  - Watches: active supervision per target (`core/models/watches.py`).
  - Notifications/asks: pending grants by subject (`approval_grants` hot index).
  - Cost/tokens: the platform's existing usage/cost telemetry lanes (exact source tables to be pinned during build — see open question 2; do **not** invent a new cost store).
- **Closest existing aggregates:** Command Center summary counts (`command-center-shell.tsx:96-111`) and the KPI endpoints (`api/kpi_api.py`) — task-centric, not agent-centric.
- Auto's action surface has list tools (`platform_list_agents`, `platform_list_tasks`, `platform_list_missions`, `platform_list_watches`) but composing floor state from four list calls per question is token-expensive and lossy.

## 3. Goals

- G1: One read-model answers, per agent: current work item (board task or mission task + title), execution status, active leases, pending asks, active watches, last activity timestamp, and period tokens/cost.
- G2: Auto can read it in one call: `platform_fleet_status` — grounding for PRD-226 doctrine §1 and honest answers to "what's everyone doing?".
- G3: Humans see it live: the Agents surface gains a fleet view with a real "working on X / idle / blocked" line per agent, replacing guesswork.
- G4: Cheap enough to poll: single query set, no N+1 across agents, workspace-scoped.

## 4. Non-goals

- No new state writes, no new lifecycle semantics, no scheduler — this is a **read-model** only.
- No per-agent live pages in v1 (the fleet view + existing modal suffice; deep pages are a follow-on if the view earns it).
- No breaker-ladder construct (Munder Difflin's `steer→constrain→stop`) — guardrails are covered by the existing Guardrails/Blueprints/SDLC stack (Gerard, 2026-08-27); the read-model *surfaces* existing budget/watch states, it does not add enforcement.
- No SSE push for fleet in v1 — the view polls like missions do today; push rides the board lane later if needed.

## 5. Design

- **Component A — the read-model.** `services/fleet_state.py`: one function `get_fleet_state(workspace_id)` composing, per active agent: current assignment (running board task via lease + status, else running orchestration task via state), queue depth (assigned-not-started count), blocked count + open asks (grants pending by subject → agent), active watches touching the agent's work, last-activity (max of task activity timestamps), and period usage (tokens/cost from the canonical telemetry source once pinned). Deterministic shape, documented, versioned.
- **Component B — the route.** `GET /api/v1/fleet` returning Component A for the caller's workspace. Full route-manifest procedure: `RouterSpec` in `orchestrator/router_manifest.py`, regenerate + commit `orchestrator/reports/route-manifest.json` with bumped `route_count`, add the `api-client.ts` call — route-contract CI stays green.
- **Component C — the tool.** `platform_fleet_status` (3-file pattern) returning a compact rendering of Component A (top-line per agent + anomalies: stalled, over-budget, blocked-with-ask). Registered like every platform action; included in the heartbeat orchestrator context (`ContextMode.HEARTBEAT_ORCHESTRATOR`) so the standing loop can flag anomalies proactively at its existing autonomy dial.
- **Component D — the fleet view.** In the Agents surface (`agent-management.tsx` tab set): per-agent row = name/avatar, live line ("working: <task title>", "idle", "blocked: awaiting answer"), queue depth, cost this period, watch badges. Poll interval matching missions detail (10s). Reuse roster components; no duplicate hooks (extend the existing agents hook family).

## 6. Waves & acceptance criteria

**Wave 0 — read-model + route + tool.**
- [ ] `get_fleet_state` unit-tested against seeded fixtures (agent with running board task; agent with running mission task; idle agent; blocked agent with open ask); no N+1 (query-count assertion).
- [ ] Route manifest updated per procedure; route-contract CI green.
- [ ] `platform_fleet_status` returns the compact form; "how's the team doing" in chat produces a grounded per-agent answer (eval case added to the local gold set).

**Wave 1 — the view.**
- [ ] Fleet view renders live lines for the fixture states; updates within one poll interval of an agent move (agent-move SSE from PRD-227 may arrive sooner — the view listens to the existing board stream where mounted).
- [ ] No duplicate hook introduced (house rule) — existing agents hooks extended.

## 7. Technical considerations

- Workspace isolation and permissions mirror the board endpoints (`require_workspace_permission` read scope).
- The read-model must tolerate partial sources (e.g., cost source unavailable) by omitting fields, not failing — same fail-soft posture as the channels sender.
- PRD-224/225 enrich this view (tickets and asks appear as they ship) but are not dependencies — the read-model reads whatever exists.

## 8. Open questions (Gerard)

1. Where should the fleet view live: a tab inside the Agents page (proposal) or a Command Center tab beside Board/Watchlist?
2. Canonical cost/token source for "this period" — the cost ledger the credit/usage surfaces already read, or the telemetry lane? (Pin during Wave 0; both exist, one must be canonical — no new store.)
3. Period definition for cost display: rolling 24h, current billing period, or both?
