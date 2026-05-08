# PRD-140 — Team Lead Feedback Loop

**Status:** Draft v2 — direction approved by Auto, gated on permission helper + Phase 1 telemetry before Manager/Autopilot
**Type:** Architecture / Feature
**Priority:** P2 (after Fixes 1+2 land and pilot validates org chart)
**Owner:** Platform
**Author:** Gerard Kavanagh + Claude
**Reviewer:** Auto (CTO)
**Date:** 2026-05-08
**Related PRDs:** PRD-76 (Agent Reporting), PRD-77 (Self-Scheduling), PRD-82A (Sequential Coordinator), PRD-121 (HARNESS — workspace-level self-optimisation), PRD-67 (CTO Agent), PRD-72 (Activity / Command Centre)

---

## Operating Principle

> **Team leads may optimise execution inside their team, but Auto remains the workspace authority for structural changes, cross-team arbitration, skill/tool governance, and escalation until scoped autonomy has been explicitly enabled and audited.**

This is the rule the rest of the PRD enforces.

---

## 1. Purpose

The org chart now expresses hierarchy (Auto → ATLAS / SENTINEL / ORACLE / VECTOR / etc., team-leads managing sub-teams). Today that hierarchy is **descriptive** — it shows on a chart but team leads have no special authority over their teams' work.

This PRD turns the org chart into **operational authority** without turning it into chaos. Team-lead agents (e.g. VECTOR for Growth, ATLAS for Platform, ORACLE for Knowledge) get a structured way to:

1. Review their team's output (already possible via reports).
2. Identify gaps and propose changes.
3. Communicate suggestions to the team via tracked artefacts (reports + tasks).
4. **Optionally** apply scoped changes themselves within hard permission boundaries.
5. Operate on a recurring cadence so improvements compound — without thrashing.

It is the **team-level analogue** of PRD-121 HARNESS (workspace-level, owned by Auto). Where HARNESS optimises the whole org weekly, the team-lead loop optimises one team continuously, with the team's own lead in the driver's seat — but capped by Auto's ultimate authority.

---

## 2. Authority Stack

The full chain of command, top to bottom:

```
Human / Gerard
  > Auto / HARNESS                   (workspace-wide arbiter)
    > ATLAS / SENTINEL / ORACLE      (platform/reliability/knowledge stewards)
      > Team-lead loop               (e.g. VECTOR over Growth)
        > Individual agent           (heartbeat/task output)
```

Concrete rules:

- Human can override anyone.
- Auto can override any team lead. HARNESS-applied changes outrank team-lead changes for the same target.
- ATLAS / SENTINEL / ORACLE outrank team leads on platform-architecture / reliability / knowledge concerns respectively.
- A team lead directs its sub-tree but cannot direct other team leads or agents outside its subtree.
- Individual agents cannot override their manager.

Conflicts resolve upward, never sideways. VECTOR ↔ ATLAS disputes go to Auto, not direct.

---

## 3. Background

### 3.1 What we have today

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
| Hierarchy data | `Agent.reports_to_id`, `Agent.team`, `Agent.job_title` | existing |

### 3.2 What is missing

1. **Hierarchy-aware permissions** for *every* mutating tool, not one-off. Currently any agent with the tool can mutate any object in the workspace.
2. **Managed Resource Scope** — clear ownership rules for non-agent objects (playbooks, skills, tools, heartbeats).
3. **Standard team-review pattern** — template, recurring trigger, canonical artefact.
4. **Configurable autonomy** per team lead with hard gating on each tier.
5. **Pending-change pipeline** with rollback data, audit log, and cooldowns.

### 3.3 Why not just hand-roll a VECTOR playbook today

Reasons not to:

- Bakes the pattern into one playbook; every other team lead has to copy it.
- Permission gaps remain — generic platform tools have no hierarchy check.
- No standard artefact means the platform can't learn from these reviews over time.
- Misses the marketplace opportunity: every workspace template should ship with team-level loops.

---

## 4. Goals

| # | Goal |
|---|---|
| G1 | A team lead agent can run a structured review of its team's output on a recurring schedule, producing a tracked artefact ("team review report"). |
| G2 | A team lead can raise BoardTasks against agents *in its own subtree* with paper trail (created_by, reason, source review). |
| G3 | Team leads can edit team-owned resources within their hierarchy subtree — but only via a permission helper applied to *every* mutating tool, not by trust. |
| G4 | The platform supports configurable autonomy per team lead. Default is conservative (Advisor). Higher tiers gated on observed clean history. |
| G5 | Cross-team or high-risk changes escalate to Auto (or human if no Auto loop active) — never silently dropped. |
| G6 | Every auto-applied change carries rollback data, audit log entry, source review reference, and cooldown enforcement. |
| G7 | The pattern is reusable: any agent flagged as a team lead inherits the loop. Workspace templates can ship team-level loops out of the box. |

### 4.1 Non-goals

- Replacing HARNESS. HARNESS keeps its workspace remit.
- Building inter-agent direct messaging. Reports + tasks are sufficient for v1.
- Letting team leads spawn / delete agents autonomously.
- Letting team leads modify budgets, billing, auth, or model-provider config.
- Letting team leads edit skills directly in any tier (always escalates to Auto/ATLAS).

---

## 5. Concepts

### 5.1 Team Lead — capability vs enabled

Two separate flags:

- **`is_team_lead_derived`** (computed): the agent has at least one direct report (`reports_to_id` chain). Read-only, derived from the org chart.
- **`team_lead_enabled`** (explicit setting on Agent): whether the feedback loop is active for this lead. Default `false` even when `is_team_lead_derived=true`. Operator must opt in.

Why both: an agent may have direct reports without you wanting the loop active yet (e.g. team is being assembled, schedule conflict).

### 5.2 Hierarchy Subtree

Given a team lead `L`, the subtree is the set of agents whose `reports_to_id` chain includes `L`. VECTOR's subtree includes PULSE, GA ANALYST, RALLY, SCOUT, QUILL, CANVAS, SOCIAL OPS, SOCIAL PUBLISHER. Computed via recursive CTE on `agents.reports_to_id`.

### 5.3 Managed Resource Scope

A team lead `L` can act on a resource `R` only when:

| Resource | In-scope condition |
|---|---|
| **Agent** | `R` is in `L`'s subtree. |
| **Heartbeat** | `R.agent_id` is in `L`'s subtree. |
| **Task (create / assign)** | `R.assigned_agent_id` is in `L`'s subtree. |
| **Playbook** | `R.owner_agent_id ∈ L's subtree`, OR `R.team == L.team`, OR every executing agent is in `L`'s subtree. *AND* the playbook is not shared cross-team. |
| **Skill** | **Out of scope at every tier.** Always escalates to Auto/ATLAS. Skill edits affect behaviour far beyond one agent. |
| **Tool assignment** | Out of scope unless the tool is on a per-team pre-approved list and the target agent is in subtree. Otherwise queue. |
| **Cross-team object** | Out of scope. Escalate to Auto. |

This is the gating definition. Without it, "team lead can edit playbooks" is unsafe — playbooks aren't always cleanly owned by one agent.

### 5.4 Autonomy Tiers

Per-team-lead setting, default `Advisor`:

| Tier | Read team data | Raise tasks (in subtree) | Edit own subtree (T1) | Edit own subtree (T2) | Edit own subtree (T3+) |
|---|:-:|:-:|:-:|:-:|:-:|
| **Advisor** (default) | ✓ | ✓ | recommend only | recommend only | recommend only |
| **Manager** | ✓ | ✓ | auto-apply | queue for human | queue for human |
| **Autopilot** | ✓ | ✓ | auto-apply | auto-apply (gated) | queue for human |

Autopilot is **not** a slider users can flip on day one. Promotion from Manager → Autopilot requires:

- ≥ 4 successful Manager reviews
- Zero applied changes reverted in last 30 days
- Zero high-risk changes denied at the queue gate in last 30 days
- Explicit human enable (workspace admin)

### 5.5 Risk Tiers (defined here, not borrowed)

| Tier | Definition | Examples |
|---|---|---|
| **T1 — Cosmetic / descriptive** | Surface-level, no behaviour change | Agent description copy, job title wording, report formatting preference, task title/priority/deadline, heartbeat report title wording |
| **T2 — Behavioural but reversible, scoped** | Changes execution but reversible and confined to owned resources | Heartbeat interval (within approved bounds, e.g. 1h–7d), heartbeat prompt wording that doesn't change authority, playbook prompt wording inside a team-owned playbook, task assignee within subtree |
| **T3 — Operationally significant** | Visible operational impact, harder to reverse | Tool assignment / removal, playbook structural edits (steps added/removed), heartbeat outside approved bounds, agent deactivation |
| **T4 — Skill / behaviour** | Affects multiple agents or core behaviour | Skill edits (any), skill assignment / removal, agent creation, model changes |
| **T5 — Workspace-level** | Cross-team or platform impact | Cross-team ownership changes, publishing automation, outreach automation, budget/cost/quota changes, auth/billing |

Auto-apply ceiling per tier:

- **Advisor:** none (recommend only).
- **Manager:** T1 only.
- **Autopilot:** T1 + T2 (gated on cooldowns + rollback).
- T3+ always queues for human or escalates to Auto, regardless of tier.

### 5.6 Team Review

A scheduled execution where a team lead:

1. **Collect** — browse team reports (`agent_team=lead.team`, period since last review), task throughput, cost, latency.
2. **Diagnose** — LLM step. Identify gaps.
3. **Decide** — emit structured **Intents** (see §6.3). Each carries risk_tier and target.
4. **Apply / Queue** — based on autonomy tier + Managed Resource Scope, either auto-apply (with rollback data + audit log + cooldown check) or convert to a `pending_change` BoardTask.
5. **Detect conflicts** — flag anything overlapping with HARNESS pending changes or other recent edits.
6. **Report** — produce the team review report listing scope, findings, intents (applied + queued + escalated), conflicts.

---

## 6. Architecture

### 6.1 Permission helper (the gating piece)

Single new module: `core/security/hierarchy_permissions.py`.

```python
def can_actor_modify(
    db: Session,
    actor_agent_id: int,
    target_type: str,                 # "agent" | "heartbeat" | "playbook" | "task" | "skill" | "tool_assignment"
    target_id: int | str,
    change_type: str,                 # "create" | "update" | "delete" | "assign"
) -> PermissionDecision:
    """Return permit/deny + reason. System agents (Auto/CTO) bypass."""
```

`PermissionDecision` carries: `allowed: bool`, `reason: str`, `escalation_target: Optional[int]` (e.g. Auto's id when this requires arbitration).

Applied to **every** mutating platform_* tool: `platform_update_agent`, `platform_update_playbook`, `platform_create_task`, heartbeat config tools, tool assignment tools, etc.

Default-deny for anything not explicitly allowed. Skill edits return `allowed=False, escalation_target=auto.id` for any non-system actor regardless of scope.

**An audit-grep CI gate** ensures every `platform_update_*` / `platform_assign_*` call routes through this helper. No opt-out.

### 6.2 Pending Change pipeline

For v1, `pending_change` is implemented as a BoardTask with a structured `metadata.pending_change` payload. Designed so it can migrate to a first-class `PendingChange` table later without breaking callers.

Payload schema (v1):

```jsonc
{
  "kind": "pending_change",
  "source_agent_id": 192,          // the lead that proposed
  "source_review_id": "...",       // team review report id
  "target_type": "agent",
  "target_id": 446,
  "change_type": "update",
  "proposed_patch": { "job_title": "Web & Attribution Analyst" },
  "current_value": { "job_title": "Analyst" },
  "risk_tier": "T1",
  "reason": "Job title aligns with org chart cleanup; no other agent affected",
  "evidence": ["report:abc123", "org_chart:vector_subtree"],
  "cooldown_until": "2026-05-15T00:00:00Z",
  "requires_approval_from": "human" | "auto" | null,
  "rollback_data": { "agent_id": 446, "snapshot": {...} }
}
```

Phase 2+ promotes this to a `PendingChange` table with the same fields plus `status`, `applied_at`, `approved_by`, `applied_by`, etc.

### 6.3 Structured Intent schema

The `Decide` step emits strict JSON. No free-form NL parsing.

```json
{
  "review_period_start": "2026-05-01T00:00:00Z",
  "review_period_end": "2026-05-08T00:00:00Z",
  "intents": [
    {
      "type": "raise_task",
      "target_agent": "QUILL",
      "risk_tier": "T1",
      "reason": "No long-form content report submitted this week",
      "action": "Draft one founder-led essay brief by Friday",
      "evidence": ["report_id:123", "missing_report:quill:7d"],
      "requires_approval": false
    },
    {
      "type": "update_heartbeat",
      "target_agent": "PULSE",
      "risk_tier": "T2",
      "reason": "Daily cadence too frequent — 60% of reports are 'no change'",
      "proposed_patch": { "interval": "3d" },
      "current_value": { "interval": "1d" },
      "evidence": ["heartbeat_history:pulse:30d"],
      "requires_approval": true
    },
    {
      "type": "escalate",
      "target": "auto",
      "risk_tier": "T5",
      "reason": "SOCIAL OPS publishing volume dropped 70% — possibly tool/auth issue, needs cross-team look",
      "evidence": ["report_id:..."]
    }
  ]
}
```

Valid `type` values: `raise_task`, `update_agent`, `update_heartbeat`, `update_playbook`, `escalate`. Anything else is rejected.

### 6.4 Cooldowns / anti-thrash

Per-resource minimum interval between changes by team-lead loops:

| Resource type | Cooldown |
|---|---|
| Heartbeat config | 7 days |
| Playbook prompt edit | 7 days |
| Agent description / job title | 14 days |
| Tool assignment (Phase 3+ only) | 30 days |
| Skill changes | n/a (never team-lead, always Auto) |

A change inside its cooldown queues instead of auto-applying, regardless of tier.

### 6.5 Per-review rate limits

Caps on a single team-review run:

| Limit | Default |
|---|---|
| `max_reports_read` | 50 |
| `max_tasks_created_per_run` | 10 |
| `max_pending_changes_per_run` | 5 |
| `max_auto_applied_per_run` | 3 |
| `max_playbooks_touched_per_run` | 1 |

Hitting a cap surfaces in the review report so it's visible, not silent.

### 6.6 Audit log + rollback

Every auto-applied change creates an entry:

- `actor_agent_id`
- `target_type`, `target_id`
- `change_type`
- `previous_value`
- `new_value`
- `reason`
- `source_review_id`
- `risk_tier`
- `applied_at`
- `rollback_data` (everything needed to revert)

**Phase 2 hard requirement:** no auto-apply without a stored rollback. If rollback can't be captured, the intent queues.

Audit retention:

- Standard changes: **180 days**
- High-risk / destructive: **indefinite (archived)**
- Minimum: 90 days

### 6.7 Conflict detection (Phase 1 onward)

Even at Advisor tier the review report should surface:

- Same target object has a pending HARNESS recommendation
- Same agent has another pending change from a previous review
- Same playbook was modified within cooldown window
- Same heartbeat has an open change request

Auto-apply waits until Phase 2, but visibility starts in Phase 1.

### 6.8 Team-lead heartbeat config

Extension to heartbeat config:

```jsonc
{
  "heartbeat_type": "team_review",
  "interval": "1w",                    // 1d / 1w / 2w
  "playbook": "team-review",
  "team_lead_enabled": true,
  "autonomy_tier": "advisor",          // advisor | manager | autopilot
  "limits": {                          // overrides §6.5 defaults
    "max_tasks_created_per_run": 10,
    "max_pending_changes_per_run": 5,
    "max_auto_applied_per_run": 3
  }
}
```

Default for any newly-flagged team lead: `autonomy_tier=advisor`, `interval=1w`.

### 6.9 Escalation channel

For v1: BoardTask flagged `for_human=true` with `escalation_reason` and `escalation_target` (`"auto"` or `"human"`). Surfaces in the existing BoardTask UI plus the team review report.

No new inbox primitive in v1.

### 6.10 Telemetry

Each team review run records (feeds into Activity / Command Centre + HARNESS workspace trends):

- Scope (team, agents reviewed, period)
- Findings count
- Intents by type, by risk tier
- Auto-applied vs queued vs escalated counts
- Conflicts detected
- Cooldowns hit
- Rate-limit caps hit
- Duration, tokens, cost, model used (Fix 1 synthesis-override eligible — see §7 Q8)

---

## 7. Open Questions — answered

Resolved during Auto's review (2026-05-08). Logged here so the build picks them up.

| # | Question | Answer |
|---|---|---|
| Q1 | Is `is_team_lead` derived or explicit? | **Both.** `is_team_lead_derived` (computed) is the capability; `team_lead_enabled` (boolean) is the config. Loop runs only when both are true. |
| Q2 | What counts as low-risk for Manager auto-apply? | T1 only. T2 auto-applies at Autopilot. Tools / skills / deletions never. Definitions in §5.5. |
| Q3 | Does Advisor get to create BoardTasks? | Yes — but only for agents in its subtree. Cross-subtree task creation routes via escalate intent. |
| Q4 | Where does the team review report live? | Reports tab + Activity feed in v1. Dedicated team page is Phase 4. |
| Q5 | Inbox for escalations from below? | No. Reports + BoardTasks suffice for v1. Re-evaluate after Phase 2 telemetry. |
| Q6 | HARNESS vs team lead conflict? | HARNESS wins. Team-lead recommendation **queues** rather than being discarded — Auto can decide later. |
| Q7 | Cost guardrail? | Yes — §6.5 per-run rate limits + token/cost caps via `_POWER_MODE_CAPS` (default `standard`). |
| Q8 | Is `Decide` step synthesis (Fix 1 eligible)? | Yes — qualifies as synthesis. Use Gemini Flash via Fix 1 override. Output is structured JSON, so reasoning load is moderate. |
| Q9 | Can team leads refuse HARNESS changes? | Out of scope for v1. They can flag concerns via escalate intent. No auto-revert. |
| Q10 | Audit retention? | Standard: 180 days. High-risk: indefinite. Minimum: 90 days. |
| Q11 | Can a team lead edit skills assigned to its own team? | **No.** Skills always route through Auto/ATLAS regardless of tier. Skill edits affect behaviour beyond one agent. |
| Q12 | Can team leads create or remove agents from their team? | No. Hiring / firing is workspace-level (Auto + human). Team leads can request via escalate. |

---

## 8. Phased Plan

### Phase 1 — Foundation (Advisor only)

**Permission helper first. Nothing else builds until it's in.**

1. `core/security/hierarchy_permissions.py` with `can_actor_modify(...)`. Default-deny.
2. Apply helper to every existing mutating `platform_*` tool. Tests cover Auto-bypass, in-subtree allow, cross-subtree deny, skill always-deny.
3. CI audit-grep gate: any new `platform_update_*` / `platform_assign_*` without the helper fails CI.
4. Add `Agent.team_lead_enabled` boolean column. `is_team_lead_derived` exposed via API as computed field.
5. Build `team-review` builtin playbook — Collect + Diagnose + Report (no Apply, no Decide-with-side-effects).
6. Team review report becomes a first-class report type (`report_type=summary`, `metrics.trigger=team_review`, structured intents in `metrics.intents`).
7. Conflict detection (§6.7) surfaces in the review report.
8. UI:
   - Reports tab filter for `trigger=team_review`.
   - Per-agent detail page surfaces team-lead capability + enable toggle.
   - Per-task detail page shows `created_by_id` + source review link when present.

**Exit:** VECTOR runs weekly, files a review with structured intents, raises BoardTasks within its subtree only. No platform mutations applied automatically. Conflicts visible. Permission helper proven across all tools.

### Phase 2 — Manager autonomy

1. Add `autonomy_tier` setting per team lead (`advisor` default).
2. `Decide` step emits risk-tagged intents per §6.3 schema.
3. Manager tier auto-applies T1 only. T2 / T3+ queue as pending-change BoardTasks.
4. Cooldowns (§6.4) and per-run rate limits (§6.5) enforced.
5. Audit log (§6.6) — required: rollback_data captured for every applied change. No rollback = no apply.
6. UI: settings page per team lead surfaces autonomy slider (locked above Manager), recent applied / queued history, audit log.

**Exit:** Lead can update T1-only resources (descriptions, task priorities, etc.) within subtree. Anything else queued. Every applied change has rollback. No regression in team output quality vs Phase 1 baseline.

### Phase 3 — Autopilot + escalation

1. Autopilot tier flips T2 to auto-apply, gated on cooldowns and clean Manager history (§5.4 preconditions).
2. Escalation channel formalised — BoardTask routed to Auto or human based on `escalation_target`.
3. Cross-team findings always escalate.
4. Convergence detection (à la PRD-121 §3): when reviews stop producing new findings, slow the cadence.

**Exit:** Long-running team leads operate near-zero human touch for routine work, escalate the right things, dial down their own cadence as they converge.

### Phase 4 — Marketplace

1. Workspace templates (Shopify etc.) ship with team leads pre-flagged + appropriate autonomy defaults.
2. Skills marketplace publishes "team-review" variants by team type (Growth, Engineering, Sales).
3. Documentation pattern: "How to add a new team lead in your workspace."

**Exit:** Onboarding into a vertical template gets the loop on day one.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| **Permission helper bug → privilege escalation** | Default-deny, exhaustive tests, CI audit-grep gate, applied to every mutating tool, never opt-out. Skills always-deny irrespective of scope. |
| **Team-lead LLM hallucinates urgent fix** | Default Advisor (no apply). Manager limited to T1. Cooldowns. Rollback data required. Audit log per change. |
| **Loop runs cost-amok with no convergence** | Phase 1 caps cadence at weekly. Phase 3 adds convergence detection. Per-run rate limits + token / cost caps via `_POWER_MODE_CAPS`. |
| **Conflicting changes from HARNESS + team lead** | Q6: HARNESS wins, team-lead change queues. Conflict detection (§6.7) surfaces in review report from Phase 1. |
| **Team lead thrashes the same prompt every run** | §6.4 cooldowns: heartbeat / playbook prompt 7d, agent description 14d. Cooldown hit queues instead of applying. |
| **User confused about who changed what** | Audit log surfaces actor + reason on agent / playbook detail pages. Team review report explicitly lists applied + queued + escalated changes. |
| **Autopilot enabled too early** | Promotion gated on §5.4 preconditions: clean Manager history, zero reverts, explicit admin enable. Not a slider. |
| **Skills edited by team leads cause drift across the workspace** | Skills always escalate (§5.3, Q11). No tier permits direct skill edit. |
| **Cross-team task creation chaos** | Q3: BoardTasks from team leads constrained to subtree. Cross-subtree work routes via `escalate` intent. |
| **Rollback data missing for a Manager-tier change** | §6.6: no rollback = no apply. Intent queues instead. |

---

## 10. Success Criteria

End of Phase 1 (Advisor):
- VECTOR runs weekly team-review heartbeats without human prompting.
- Team review reports appear in Reports tab + Activity feed.
- BoardTasks raised by VECTOR show `created_by_id=VECTOR.id` and reference the source review.
- Conflict detection produces accurate flags vs HARNESS pending changes (manual spot-check).
- No `platform_update_*` call from a non-system agent succeeds outside its hierarchy subtree (test-enforced).
- CI audit-grep gate passes — every mutating tool routes through `can_actor_modify`.

End of Phase 2 (Manager):
- ≥ 50% of T1 intents auto-applied without human touch.
- 100% of auto-applied changes have rollback data.
- Audit log shows every change with actor + reason + source review.
- No regression in team output quality vs Phase 1 baseline (defined per team).
- Cooldowns measurable in telemetry — no resource modified within its cooldown.

End of Phase 3 (Autopilot):
- VECTOR operates a full week with zero human edits to its team config.
- Escalation channel produces ≥ 1 valid escalation in a month.
- Convergence detected: cadence drops from weekly to bi-weekly automatically for at least one team lead.
- Promotion gating works: at least one workspace tries Autopilot without meeting preconditions and is denied.

End of Phase 4 (Marketplace):
- New Shopify-template workspace ships with at least one active team lead loop, runs first review within 7 days, no errors.

---

## 11. Out of Scope (Park)

- Cross-tenant peer review between team leads.
- Team lead negotiating with another team lead directly (always via Auto).
- Replacing or merging with HARNESS.
- Letting team leads modify billing / quotas / spending caps.
- Promoting a non-team-lead agent to team lead automatically.
- Direct skill edits in any tier (Q11).
- Auto-revert of HARNESS changes by team leads (Q9).
- Inter-agent direct messaging primitive (re-evaluate after Phase 2).

---

## 12. Build Order (Approved Sequence)

1. **Hierarchy permission helper** + CI audit-grep gate.
2. **Managed Resource Scope checks** for agent / task / playbook / heartbeat / tool-assignment / skill (skill = always deny for non-system).
3. **Team review report schema** (PRD-76 extension — new `trigger=team_review`, structured `metrics.intents`).
4. **Built-in `team-review` playbook** (Advisor-only — Collect + Diagnose + Report, no Apply).
5. **VECTOR pilot** — enable on Growth team only. Telemetry for one month.
6. **Telemetry + audit visibility** — Activity feed, agent detail page, audit log UI.
7. **Manager-mode design review** (gate before Phase 2) — review pilot data, confirm risk tiers, confirm rollback design.
8. **Phase 2 build** (Manager).
9. **Phase 3 build** (Autopilot, only after preconditions met).
10. **Phase 4** (marketplace templates).

---

## 13. Decision Required Before Build

This PRD ships nothing on its own. Concrete next steps:

1. **Approve the operating principle** (top of this doc).
2. **Approve the permission helper as gating** — no other Phase 1 work starts until it lands and passes the CI audit gate.
3. **Confirm risk tier definitions** (§5.5) match operator expectations.
4. **Confirm Phase 1 scope** (Advisor only, no Apply).
5. **Confirm autonomy precondition gating** (§5.4) for Phase 3 — including the human enable step.

Once approved, file Phase 1 implementation tickets. Phase 2 stays gated on a design-review session against pilot data.

---

**Last updated:** 2026-05-08 (v2 — incorporates Auto's review feedback)
**Reviewer notes:** Auto approved direction, requested Managed Resource Scope as a first-class concept, skills carve-out, structured intent schema, rate limits, cooldowns, rollback data, Autopilot precondition gating. All incorporated above.
