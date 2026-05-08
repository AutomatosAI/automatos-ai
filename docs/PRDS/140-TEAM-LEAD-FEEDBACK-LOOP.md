# PRD-140 — Team Lead Feedback Loop

**Status:** Draft — needs review before implementation
**Type:** Architecture / Feature
**Priority:** P2 (after Fixes 1+2 land and pilot validates org chart)
**Owner:** Platform
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-05-08
**Related PRDs:** PRD-76 (Agent Reporting), PRD-77 (Self-Scheduling), PRD-82A (Sequential Coordinator), PRD-121 (HARNESS — workspace-level self-optimisation), PRD-67 (CTO Agent)

---

## 1. Purpose

The org chart now expresses hierarchy (Auto → ATLAS / SENTINEL / ORACLE / VECTOR / etc., with team-leads managing sub-teams like Growth, Engineering, Knowledge). Today that hierarchy is **descriptive** — it shows on a chart but team leads have no special authority over their teams' work.

This PRD closes that gap. Team-lead agents (e.g. VECTOR for Growth, ATLAS for Platform, ORACLE for Knowledge) get a structured way to:

1. Review their team's output (already possible via reports).
2. Identify gaps and propose changes.
3. Communicate suggestions to the team.
4. **Optionally** apply changes themselves within scoped permissions.
5. Operate on a recurring cadence so improvements compound.

It is the **team-level analogue** of PRD-121 HARNESS (which is workspace-level, owned by Auto). Where HARNESS optimises the whole org weekly, the team-lead loop optimises one team continuously, with the team's own lead in the driver's seat.

---

## 2. Background

### 2.1 What we have today

Building blocks already shipped:

| Capability | Tool / Mechanism | Source |
|---|---|---|
| Read team output | `platform_browse_reports`, `platform_get_latest_report` | PRD-76 |
| Raise work for an agent | `platform_create_task` (BoardTask with `assigned_agent_id` + `created_by_id`) | PRD-72 |
| Edit an agent's metadata | `platform_update_agent` (job_title, description, team, tools) | platform tools |
| Edit playbooks | `platform_update_playbook`, `platform_run_playbook` | PRD-12 / PRD-71 |
| Configure heartbeats | heartbeat config tools | PRD-77 |
| Publish reports | `platform_submit_report` | PRD-76 |
| Self-scheduled execution | Heartbeat scheduler | PRD-77 |
| Mission decomposition | MissionPlanner → tasks with delegation (Fix 2) | PRD-82A + Fix 2 |
| Hierarchy data | `Agent.reports_to_id`, `Agent.team`, `Agent.job_title` | PRD existing |

What is missing:

1. **Hierarchy-aware permissions.** Any agent with the right tool can edit any agent in the workspace. There is no check that "VECTOR can only modify agents that ultimately report to VECTOR."
2. **Standard team-review pattern.** No template, no recurring trigger, no canonical artefact for "team lead reviewed their team this week."
3. **Inter-agent communication primitive beyond reports.** A team lead can publish a report; team members read reports through `platform_get_latest_report`, but there is no targeted "here is feedback for *you*" channel. BoardTasks are the closest existing primitive but they are work, not feedback.
4. **Authority boundary configuration.** No way to express per-agent autonomy ("VECTOR may auto-apply Tier 1 changes; Tier 3 requires Gerard's nod").

### 2.2 Why this is its own PRD

PRD-121 HARNESS already exists for workspace-wide self-optimisation. It is Auto's job: read everything, decide cross-team, propose, apply, baseline.

Team-lead loops are smaller, more frequent, and locally scoped:

- **HARNESS:** weekly, cross-workspace, owned by Auto, optimises models / tools / heartbeats / playbooks across all teams.
- **Team-lead loop:** daily/weekly, single-team, owned by the lead, optimises team output quality and execution.

They are complementary. HARNESS sees the forest; team leads work the trees. A clean implementation should let team leads own their team's improvements and surface anything cross-team to HARNESS or to a human.

### 2.3 Why not just hand-roll it per team

We could write a one-off "VECTOR weekly review" playbook today. The reasons not to:

- It bakes the pattern into one playbook instead of the platform — every other team lead has to copy it.
- Permission gaps remain untouched: hand-rolled playbooks rely on the team lead having generic platform tools, with no hierarchy check.
- No standard artefact (a "team review" report) means the platform can't learn from these reviews over time (e.g. surface "VECTOR has flagged the same thing 3 reviews running — escalate").
- It misses the chance to make this a marketplace primitive: every workspace template (Shopify, SaaS, etc.) should ship with team-level loops out of the box.

---

## 3. Goals

| # | Goal |
|---|---|
| G1 | A team lead agent can run a structured review of its team's output on a recurring schedule, producing a tracked artefact ("team review report"). |
| G2 | A team lead can raise BoardTasks against its team members with a paper trail (created_by, reason, source review). |
| G3 | Team leads can edit their team's agents/playbooks/heartbeats — but only within their hierarchy subtree, enforced by a permission helper, not by trust. |
| G4 | The platform supports configurable autonomy per team lead (advisor / manager / autopilot). Default is conservative. |
| G5 | Cross-team or high-risk changes escalate: either to Auto (workspace-wide arbitration) or to a human (BoardTask flagged for review). |
| G6 | The pattern is reusable: any agent flagged as a team lead inherits the loop. Workspace templates can ship team-level loops out of the box. |

### Non-goals

- Replacing HARNESS. HARNESS keeps its workspace remit; team loops are local.
- Building inter-agent direct messaging. Communication remains through reports + tasks (existing primitives) until proven insufficient.
- Letting team leads spawn/delete agents autonomously. Hiring/firing is workspace-level (Auto + human).
- Team leads modifying budgets, billing, or auth configuration. Those stay admin-only forever.

---

## 4. Concepts

### 4.1 Team Lead

An agent flagged as a team lead. Initially identified by:
- Has at least one direct report (other agents have `reports_to_id = lead.id`).
- Optionally explicit field: `is_team_lead` boolean, OR `Agent.role_type = 'team_lead'`. To decide in design.

### 4.2 Hierarchy Subtree

Given a team lead `L`, the subtree is the set of agents reachable by traversing `reports_to_id` upward and finding `L` as an ancestor. VECTOR's subtree includes PULSE, GA ANALYST, RALLY, SCOUT, QUILL, CANVAS, SOCIAL OPS, SOCIAL PUBLISHER.

### 4.3 Autonomy Tier

A per-team-lead setting:

| Tier | Read team data | Raise BoardTasks | Edit own subtree | Apply changes auto |
|---|:-:|:-:|:-:|:-:|
| **Advisor** (default) | ✓ | ✓ | — | — |
| **Manager** | ✓ | ✓ | ✓ | ✓ for low-risk; flagged for human on medium/high |
| **Autopilot** | ✓ | ✓ | ✓ | ✓ for low+medium; high still flagged |

Risk tiers borrow from PRD-121 HARNESS's 5-tier framework — keep one risk model, not two.

### 4.4 Team Review

A scheduled execution where a team lead:
1. Browses team reports for the period.
2. Browses team task throughput / quality / cost.
3. Diagnoses gaps or improvement opportunities.
4. Produces a "team review" report (`report_type=summary`, `metrics.trigger=team_review`).
5. Optionally raises BoardTasks for issues found.
6. Optionally applies low-risk changes if Manager/Autopilot.
7. Optionally escalates high-risk findings to Auto via report or BoardTask.

---

## 5. Architecture

### 5.1 Permission helper (foundational)

Single new module: `core/security/hierarchy_permissions.py`.

```python
def can_actor_modify_target(
    db: Session,
    actor_agent_id: int,
    target_agent_id: int,
) -> bool:
    """True if actor is target's manager (transitively) OR system agent."""
```

Used inside `platform_update_agent`, `platform_update_playbook`, heartbeat config tools, etc. Auto / system agents bypass. Other actors must own the target via `reports_to_id` ancestry.

Without this helper, all the rest is unsafe — any agent with the tool today can edit any agent in the workspace. This is the **gating** piece of the PRD.

### 5.2 Team Review playbook (canonical)

A built-in playbook `team-review` that any team-lead heartbeat invokes. Steps:

1. **Collect** — call `platform_browse_reports(agent_team=<lead.team>, period=<since last review>)`, plus task / cost / latency rollups.
2. **Diagnose** — LLM step. Identify gaps (e.g. "GA ANALYST hasn't reported in 5 days", "SOCIAL OPS' posts have low engagement").
3. **Decide** — emit a list of structured "intent" objects: `raise_task`, `update_agent`, `update_playbook`, `update_heartbeat`, `escalate_to_auto`. Each carries a risk tier.
4. **Apply / Queue** — based on autonomy tier, either auto-apply ≤ tier or convert to a `pending_change` BoardTask for human review.
5. **Report** — produce the team review report. Includes: scope, findings, intents (applied + queued), next steps.

Reuses PRD-121 risk framework end-to-end. The lead's autonomy tier decides where the cut-off sits.

### 5.3 Team-lead heartbeat config

Extend the heartbeat config to support a "team review" cadence:

```jsonc
{
  "heartbeat_type": "team_review",
  "interval": "1w",
  "playbook": "team-review",
  "autonomy_tier": "advisor"  // advisor | manager | autopilot
}
```

The default for any newly-flagged team lead is `advisor` running weekly.

### 5.4 Escalation channel

When a team lead's review produces findings outside its remit (e.g. cross-team coordination, budget impact, agent removal), the loop produces an `escalate` intent. Escalations:

- File a BoardTask flagged `for_human=True` if the workspace has no active Auto loop, OR
- Submit a structured report to Auto's queue (a new lightweight inbox; or piggyback on `platform_get_latest_report` filtered by `audience=auto`).

Concrete mechanism is part of design review; the simplest version is "BoardTask with `for_human=true, escalation_reason=...`" — no new mechanism.

### 5.5 Telemetry

Each team review run records:
- Findings count
- Intents (by type, by risk tier)
- Auto-applied vs queued
- Duration, cost, model used

Feeds into Activity / Command Centre. Useful both for the user (is VECTOR actually doing useful work?) and for HARNESS (workspace-wide trends).

---

## 6. Phased Plan

### Phase 1 — Foundation (advisor only)

1. Implement `hierarchy_permissions.can_actor_modify_target` and apply to all `platform_update_agent`, `platform_update_playbook`, heartbeat config writes.
2. Backfill: every existing tool gets a permission check. Tests cover Auto-bypass, in-subtree, cross-subtree denial.
3. Add `Agent.is_team_lead` (computed: has direct reports) OR explicit field. Decide in review.
4. Build `team-review` playbook as a builtin (read-only diagnose + report; no apply yet).
5. Default schedule: weekly heartbeat for any team lead. UI toggle to disable.
6. Team review report becomes a first-class report type (`report_type=summary` with `trigger=team_review`).

**Exit:** VECTOR runs weekly, files a review, raises tasks. No edits applied automatically. Human reviews tasks.

### Phase 2 — Manager autonomy

1. Add `autonomy_tier` setting per team lead (`advisor` default).
2. Wire `team-review` decide step to emit risk-tagged intents.
3. Auto-apply intents at tier ≤ Manager threshold (low-risk only). Queue medium/high as BoardTasks.
4. UI: settings page per team lead surfaces the autonomy slider + recent applied/queued history.
5. Audit log row per applied change (`who, what, why, source review`).

**Exit:** VECTOR can update team agent descriptions / heartbeat intervals / non-destructive playbook edits without human review. Anything else still queued.

### Phase 3 — Autopilot + escalation

1. Autopilot tier flips medium-risk to auto-apply.
2. Escalation channel formalised — BoardTask flagged for human OR routed to Auto.
3. Cross-team findings detected and escalated automatically.
4. Convergence detection (à la PRD-121 §3): when reviews stop producing new findings, slow the cadence.

**Exit:** Long-running team leads operate at near-zero human touch for routine work, escalate the right things, and dial down their own cadence as they converge.

### Phase 4 — Marketplace

1. Workspace templates (Shopify etc.) ship with team leads pre-flagged + appropriate autonomy defaults.
2. Skills marketplace publishes "team-review" variants by team type (Growth, Engineering, Sales).
3. Documentation pattern: "How to add a new team lead in your workspace."

**Exit:** Onboarding into a vertical template gets the loop on day one.

---

## 7. Open Questions for Review

| # | Question | Default if unanswered |
|---|---|---|
| Q1 | Should `is_team_lead` be a derived property (has direct reports) or an explicit boolean column? | Derived — keeps the org chart as source of truth. |
| Q2 | What counts as "low-risk" for auto-apply at Manager tier? `update_agent.description`? `heartbeat.interval`? Disabling a tool? | Borrow PRD-121 risk model: T1 = description / cosmetic, T2 = heartbeat tuning / tool re-assignment within team. T3+ queues. |
| Q3 | Does VECTOR get permission to *create* BoardTasks even at Advisor tier, or only flag them for the human? | Advisor can create tasks. The task itself is suggestion, not action — humans / agents still have to execute it. |
| Q4 | Where does the team review report appear in UI? Reports tab, Activity feed, or a dedicated team page? | Reports tab + Activity feed for visibility. Team page is Phase 4. |
| Q5 | Do we need an explicit "team lead inbox" so escalations from below land somewhere VECTOR will read? Or is browsing reports enough? | Browsing reports is enough for v1. Re-evaluate after Phase 2 telemetry. |
| Q6 | Conflict resolution — what if HARNESS proposes one change and a team lead proposes another for the same agent in the same week? | HARNESS wins for now; team-lead change deferred. Document explicitly. |
| Q7 | Cost guardrail — cap the team-review run's tokens/cost? | Yes — borrow `_POWER_MODE_CAPS` from coordinator (default `standard`). Configurable per heartbeat. |
| Q8 | Synthesis model override (Fix 1) — does the `decide` step in `team-review` qualify as synthesis? | Probably yes — confirm during build. Reduces cost without quality hit. |
| Q9 | Should team leads be able to *refuse* a HARNESS-applied change to their team (revert it)? | Out of scope for v1. Either Auto or human reverts. |
| Q10 | Audit retention — how long do we keep applied-change history? | At least 90 days. Reuse existing audit infra. |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Permission helper bug → escalating privileges** | Default-deny, exhaustive tests for actor/target combinations, ALWAYS apply on ALL platform_update_* tools (audit grep gate), no opt-out. |
| **Team-lead LLM hallucinates urgent fix** | Default Advisor tier means no auto-apply. Manager tier limited to T1 changes. Audit log per change. |
| **Loop runs cost-amok with no convergence detection** | Phase 1 caps cadence (weekly), Phase 3 adds convergence detection. Token/cost caps via `_POWER_MODE_CAPS`. |
| **Conflicting changes from HARNESS + team lead** | Q6: document HARNESS-wins precedence; surface conflicts in team review report. |
| **User confused about who changed what** | Audit log surfaces actor + reason on agent / playbook detail pages. Team review report explicitly lists applied changes. |
| **Team lead ages out of sync with current org** | Reviews run against live data; no stale snapshots. Convergence detection slows but doesn't stop the loop. |

---

## 9. Success Criteria

End of Phase 1 (Advisor):
- VECTOR runs weekly team-review heartbeats without human prompting.
- Team review reports appear in Reports tab.
- BoardTasks raised by VECTOR show `created_by_id=VECTOR.id` and reference the source review.
- No `platform_update_*` call from a non-Auto agent succeeds outside its hierarchy subtree (test).

End of Phase 2 (Manager):
- ≥ 50% of T1 intents auto-applied without human touch.
- Audit log shows every change with reason + source review.
- No regression in team output quality vs Phase 1 baseline (defined per team).

End of Phase 3 (Autopilot):
- VECTOR (or another team lead) operates a full week with zero human edits to its team config.
- Escalation channel produces ≥ 1 valid escalation in a month (something genuinely needed a human).
- Convergence detected: cadence drops from weekly to bi-weekly automatically.

End of Phase 4 (Marketplace):
- New Shopify-template workspace ships with at least one active team lead loop, runs first review within 7 days, no errors.

---

## 10. Out of Scope (Park)

- Team lead **between** workspaces (cross-tenant peer review).
- Team lead **negotiating** with another team lead (e.g. VECTOR ↔ ATLAS over shared agents). Coordinated changes go through Auto.
- Replacing HARNESS or merging the two loops.
- Letting team leads modify billing / quotas / spending caps.
- Promoting a non-team-lead agent to team lead automatically.

---

## 11. Decision Required Before Build

This PRD ships nothing on its own. The next concrete step is a review session to:

1. Resolve Open Questions Q1–Q10.
2. Validate the autonomy tier model against pilot expectations.
3. Confirm Phase 1 is the right first cut (vs starting at Manager).
4. Confirm risk model alignment with PRD-121 HARNESS.
5. Approve permission helper as the gating piece — no build of Phase 1 features starts until that helper is in.

Once those are settled, file Phase 1 implementation tickets and proceed.

---

**Last updated:** 2026-05-08
