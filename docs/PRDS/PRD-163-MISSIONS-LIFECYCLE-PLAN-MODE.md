# PRD-163 — Missions Lifecycle & Plan Mode for Auto (WS-8)

**Chain:** Block A, branch `ralph/prd-163-missions-planmode` from main after Night-1. Size **M**. **Blocks PRD-164.**
**Source:** report §2.9; PRD-154 S5 fixed `platform_get_mission`, context_messages, duplicate route, honest create reply.

## Overview

Auto plans like Claude Code plans: explore read-only, produce a structured plan, get approval (human or policy), then execute with honest lifecycle control. Mission stalls in `awaiting_approval` end — either a human sees a card in chat, or workspace policy auto-approves under a dollar ceiling.

## Binding amendments

D9 (policy enum + $ ceiling + per-request override + countdown card), D11 ($ ceilings are the budget mechanism — replace token-estimate pause), Q52 implementation: OpenHands-style policy levels + Devin-style countdown, Q53 default: AWAITING_HUMAN final-review stays retired (delete remnants), Q54 default: chat-approved plans execute verbatim via plan-import (planner validates, doesn't re-decompose), Q55: kill MIN_TASKS=3 floor + the 8K upstream stuffing (PRD-164/166 replace with field digest), Q56: `created_by` = the chatting user; notifications resolve to them, Q57 default: build the apply-path for approval-time task/agent edits (LangGraph-style), Q63: executor auto-attaches recent chat as context_messages.

## User Stories

### S1: Lifecycle tools for Auto
Register approve/reject/pause/resume/cancel/replan as platform tools over the existing coordinator methods, each declaring `needs_approval` semantics; `mission_plan_ready` + `awaiting_approval` notifications to the creating USER (Q56 attribution fix).
**Acceptance:**
- [ ] Auto can drive a seeded mission through pause→resume→cancel in an integration test
- [ ] Notification lands for the chatting user, not the agent (test)
- [ ] Reachability gate green; tool schemas documented

### S2: Plan mode (read-only) + structured plan handoff
`plan_only` mission create: planner runs with the dispatcher schema filtered to `permission_level=read` tools, returns the full plan (tasks, agents, deps, cost estimate) WITHOUT execution; `platform_update_mission_plan` + plan-import endpoint accept (possibly edited) planner JSON verbatim — planner validates structure only (Q54).
**Acceptance:**
- [ ] Plan-mode run makes zero write-tool calls (tool-audit test)
- [ ] Imported plan executes exactly the given DAG (no re-decomposition; test diffs plan vs executed tasks)
- [ ] MIN_TASKS=3 floor removed — a 1-task plan is legal (test)

### S3: Approval policy engine
Workspace policy enum `always_ask | auto_below_budget | full_auto` + `approval_dollar_ceiling` + optional `auto_proceed_after_seconds` (countdown); per-request override from chat ("auto-approve this one"); ties into the §12.3 autonomy gate posture — `full_auto` requires the gate flag; every auto-approval writes a distinct audit event.
**Acceptance:**
- [ ] Policy matrix test: each enum × under/over ceiling × gate on/off → expected approval path
- [ ] Countdown auto-proceed fires + is cancelable (test)
- [ ] Audit event with policy + ceiling snapshot on every auto-approval

### S4: In-chat approval card
AG-UI-shaped interrupt: plan summary, cost estimate vs ceiling, task/agent edit affordances (Q57 apply-path: edits PATCH the plan before approval), approve/reject/auto-proceed countdown — rendered via the widget router (stale-names fix is PRD-164 S5; this story registers the mission-approval widget correctly from the start).
**Acceptance:**
- [ ] Auto-created mission surfaces a card in chat; approve from the card starts execution — dev-browser verify
- [ ] Edited agent override persists into execution (test)
- [ ] Reject returns structured feedback to Auto's context (test)

### S5: Budget + execution honesty
Dollar-ceiling budget policy replaces token-estimate pause (`test_budget_gate.py` updated); timeout scaling per power mode; real dependency edges in `TaskResponse` + DAG canvas reflects them; async planning (create returns immediately, plan lands via notification); delete dead AWAITING_HUMAN panel/endpoints + plan-modification stubs that don't apply (Q53/Q57 resolution).
**Acceptance:**
- [ ] Budget pause triggers at $ ceiling in a metered test, resumable after raise
- [ ] DAG shows real edges for a seeded multi-dep plan — dev-browser verify
- [ ] Dead paths deleted; contract green; `test_budget_gate.py` updated not skipped

## Non-Goals

Planning context pack + semantic agent matching (164), field-digest context (164/166), board task internals (161).

## Success Metrics

- Auto: "plan a mission to X" → plan card → approve → execution, end-to-end in one chat session — demoable.
- Zero missions stuck in `awaiting_approval` > 24h in the pilot week (policy or notification catches all).
- Plan-mode runs provably read-only (audit).

## Testing

New policy-matrix suite, plan-mode tool-audit test, plan-import diff test; updated `test_budget_gate.py`, coordinator tests; protected recipe suite green (mission↔playbook adjacency). Full suite + contract green.
