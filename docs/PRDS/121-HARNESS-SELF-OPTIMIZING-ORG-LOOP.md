# PRD-121 — HARNESS: Self-Optimizing Organization Loop

**Version:** 1.0
**Type:** Implementation
**Status:** Draft
**Priority:** P1
**Research Base:** Meta-Harness (arXiv 2603.28052v1), Mission Zero (Mission-0.1), PRDs 82A (Coordinator), 76 (Reports), 64 (Action Discovery), 12 (Playbook Patterns), 59 (Workflow Engine V2)
**Author:** Gerard Kavanagh + Claude
**Date:** 2026-03-31

---

## 1. Goal

**User-facing:** Auto continuously tunes your team — you see better agent performance, lower costs, and occasional board suggestions for bigger changes. No setup required.

**Technical:** Close the optimization loop on Mission Zero. Today the organization is built once and drifts. HARNESS is a system playbook that runs weekly, invisible to the user. It collects org-wide metrics, diagnoses regressions against a stored baseline, prescribes configuration changes with risk scores, auto-applies safe ones, queues risky ones as board tasks for human review, and snapshots a new baseline for next week's comparison.

The result: a Meta-Harness-style iterative optimization loop applied to organizational configuration. Agents, models, heartbeats, tools, costs, and quality converge toward optimal over 4-6 weekly runs.

---

## 2. Background — Why This Matters

### 2.1 The Meta-Harness Paper

Stanford's Meta-Harness (Lee et al., March 2026) proves three things relevant to Automatos:

1. **The harness matters more than the model.** Changing the code/config around a fixed LLM produces a 6x performance gap. Automatos' tool routing, memory injection, agent configuration, and mission coordination IS the harness.
2. **Full diagnostic traces beat summaries.** Giving an optimizer access to raw execution traces (not compressed summaries) improved accuracy from 34.6% to 50.0% — a 44% gain.
3. **Iterative search converges fast.** Meta-Harness matches competitors in 4 evaluations vs 60, because the proposer sees everything.

### 2.2 Mission Zero Gap

Mission Zero (Mission-0.1) is a one-shot organizational build:
- Research marketplace → Build agent specs → Execute configuration → Validate

There is no closed loop. After the initial build:
- Agent models may be cost-inefficient for their actual workload
- Heartbeat intervals may be too frequent (wasting tokens) or too rare (stale data)
- Tools get assigned but never called
- Success rates drift without anyone noticing
- Cost creeps without attribution

HARNESS closes this loop.

---

## 3. What Ships

| Component | Description |
|-----------|-------------|
| HarnessService | Orchestrator-level service. Registered at startup like the coordinator tick. Runs weekly cron. 5-phase pipeline hardcoded in service (Collect → Diagnose → Prescribe → Apply → Baseline). Same for every workspace. |
| 3 platform tools | `platform_harness_status`, `platform_harness_trigger`, `platform_harness_history` — called by Auto, not user-facing |
| Workspace file layout | `/harness/` directory with baselines, traces, changelogs, prescriptions |
| Risk framework | 5-tier risk scoring for prescriptions (auto-apply ≤ 2, queue ≥ 3) |
| Convergence detection | Track delta magnitude across runs, detect when org config stabilizes |

## 4. What Does NOT Ship (Deferred)

| Deferred | Target | Why |
|----------|--------|-----|
| Prompt optimization / A/B testing | v2 | Needs shadow-mode execution infrastructure |
| Tool/skill assignment optimization | v2 | Needs deeper impact analysis before auto-applying |
| Blueprint rule modifications | v2 | Governance changes need more human oversight initially |
| Explicit rollback mechanism | v2 | Baseline diff is sufficient for v1; next run can prescribe reversions |
| Auto-cadence switching (weekly ↔ biweekly) | v2 | Convergence detection ships in v1, but cadence change is manual |
| Frontend dashboard for HARNESS | v2 | Existing Reports tab + Board are sufficient |
| Cross-workspace pattern sharing | v3 | Requires marketplace-level analytics |
| Agent creation/retirement | v3 | High-risk structural changes |
| HARNESS self-prompt-tuning | v3 | Needs safeguards around modifying hardcoded phase prompts |
| Grafana integration for infra metrics | v3 | Platform tools cover application-level metrics for now |

---

## 5. User Experience

### 5.1 The User Never Sees HARNESS

HARNESS is **orchestrator infrastructure** — same level as the coordinator tick or the heartbeat scheduler. It is:
- **Registered at server startup** alongside the coordinator's 5s tick and heartbeat scheduler
- **Not in any UI** — no playbook list, no settings page, no toggle
- **Same for every workspace** — hardcoded phases, hardcoded schedule, no per-workspace config
- **Cannot be deleted, disabled, or modified** by users

The user's experience is:
1. They sign up, create a workspace, add agents
2. HARNESS runs silently in the background (dormant until ≥ 3 agents with ≥ 7 days of data)
3. First meaningful run produces a baseline and an audit report in the Reports tab
4. Subsequent runs may produce board tasks like `[HARNESS] Suggest: reduce SCOUT heartbeat to 180min` — the user approves or dismisses like any other task
5. Over time, agent configs quietly improve. Costs trend down. Success rates trend up.

### 5.2 Auto Surfaces HARNESS Naturally

When a user asks Auto:
- "How's the team performing?" → Auto calls `platform_harness_status`, references latest report
- "Any optimization ideas?" → Auto calls `platform_harness_history`, summarizes recent prescriptions
- "Why did SCOUT's model change?" → Auto reads `/harness/changelog/` for the applied change + rationale

The user never needs to know the word "HARNESS." It's just Auto being a good CTO.

### 5.3 Lifecycle

| Stage | Trigger | Behavior |
|-------|---------|----------|
| **Dormant** | < 3 agents OR < 7 days of data | Cron fires but Step 2 detects insufficient data, writes baseline-only, no prescriptions |
| **Exploring** | First 3 runs with sufficient data | Full optimization pass, learning the org's profile |
| **Converging** | Runs 4-6+ | Deltas shrinking, prescriptions fewer and more targeted |
| **Converged** | Delta magnitude < 2.0 for 2+ runs | Monitoring mode — only flags regressions, suggests biweekly cadence |
| **Diverging** | External change (new agents, model swap, etc.) | Re-enters exploring mode, weekly cadence resumes |

### 5.4 System-Level Registration

HARNESS registers at server startup in `main.py` lifespan, same pattern as:
- `CoordinatorService.start()` — 5s mission tick
- `HeartbeatService.start()` — agent/orchestrator heartbeats
- `PlaybookSchedulerService.start()` — cron playbooks

```python
harness_service = HarnessService()
await harness_service.start(scheduler)  # Registers weekly cron for all workspaces
```

No per-workspace provisioning. No seed migration. The service queries all active workspaces and registers a job for each. New workspaces get picked up on the next scheduler reload.

---

## 6. Design Decisions

### 6.1 Orchestrator Heartbeat Schedule, NOT Playbook

Playbooks are user-visible — they appear in the playbook/recipe list, can be deleted, renamed, or misconfigured by non-technical users. HARNESS must be invisible infrastructure.

HARNESS registers as a **named orchestrator heartbeat schedule** (`job_id = f"harness_{workspace_id}"`), same pattern as the existing orchestrator heartbeat but with its own weekly cron. It's configured in code, not in the database.

- Agent heartbeats: one per agent (Auto's is taken) — not suitable
- Playbooks: user-visible, deletable — not suitable
- Orchestrator heartbeat schedule: system infrastructure, registered at startup — correct

**Schedule:** `0 2 * * 0` (Sunday 2AM UTC weekly)

### 6.2 Hardcoded Service, NOT Configurable Template

HARNESS has a fixed 5-phase pipeline. The phases, prompts, and risk thresholds are hardcoded in `HarnessService`, not stored in a DB template. This means:
- No seed migration needed
- No user can modify, delete, or misconfigure it
- Same behavior for every workspace
- Updates ship with code deploys
- Missions are for user goals with LLM-planned DAGs — wrong primitive for a fixed system pipeline

### 6.3 Workspace Files, NOT New DB Tables

Meta-Harness stores full traces on filesystem for LLM readability. Full diagnostic traces are too large for JSONB columns. Workspace files are:
- Agent-readable via `workspace_read_file`
- Human-browsable in the UI
- Naturally organized by date
- No schema migration required

### 6.4 Platform Tools + NL2SQL Hybrid

Platform analytics tools provide pre-computed aggregations (good for dashboards). But `platform_query_data` (NL2SQL) can query raw data from `heartbeat_results`, `recipe_executions`, `orchestration_runs`, `llm_usage`, and `agent_reports` for richer cross-table analysis that isn't exposed via platform tools. Using both maximizes coverage.

### 6.5 Board Tasks for Human Review

Prescriptions with risk ≥ 3 become `board_tasks` with `tags: ["harness"]`. The human moves them to "done" (approved) or "blocked" (rejected). The next HARNESS run reads board task statuses to close the loop. No new approval mechanism needed — the existing board lifecycle is sufficient.

---

## 7. The 5-Phase Pipeline

### 6.1 Phase 1: COLLECT

**Agent:** Auto (CTO)
**Mode:** Read-only
**Purpose:** Gather comprehensive raw metrics across all agents and systems.

**Tool calls:**

| # | Tool | What It Returns |
|---|------|-----------------|
| 1 | `platform_list_agents` | All 14 agents with configs, models, tools, skills |
| 2 | `platform_get_agent_ranking` | Composite scoring: success, speed, volume |
| 3 | `platform_get_success_rate` | Overall success rate + 7-day trend |
| 4 | `platform_get_error_rates` | Failures by agent type, severity (30d) |
| 5 | `platform_get_cost_breakdown` | Cost by agent, provider, model (7d + 30d) |
| 6 | `platform_get_sla_compliance` | Completion rate + response time vs targets |
| 7 | `platform_get_efficiency_score` | Composite 0-100 score |
| 8 | `platform_get_bottlenecks` | Failure rates, queue buildup, slow executions |
| 9 | `platform_get_llm_usage` | Token counts by model |
| 10 | `platform_board_summary` | Tasks by status, priority, busiest agents |
| 11 | `workspace_read_file` | `/harness/baseline_latest.json` (previous baseline) |
| 12 | `platform_query_data` | Per-agent heartbeat costs (7d) |
| 13 | `platform_query_data` | Per-playbook success rates (7d) |
| 14 | `platform_query_data` | Recent mission outcomes (7d) |
| 15 | `platform_list_tasks` | Prior HARNESS prescriptions (tags=["harness"]) |

**Design principle:** Preserve full diagnostic data, not summaries. This is the Meta-Harness insight — raw traces enable the proposer to do causal reasoning about failures.

**Output:** Raw metrics JSON → scratchpad → Step 2

### 6.2 Phase 2: DIAGNOSE

**Agent:** Auto (CTO)
**Mode:** Read-only
**Purpose:** Compare current metrics against previous baseline, produce per-agent health cards.

**Per-agent health card:**
- success_rate_delta, cost_delta, efficiency_delta, error_rate_delta, token_usage_delta
- Classification: `REGRESSION` (>10% worse), `IMPROVEMENT` (>10% better), `ANOMALY` (zero activity, cost spike, new error patterns), `STABLE`

**Cross-cutting analysis:**
- Department-level aggregate performance
- Model cost-efficiency: are we paying premium prices for budget-tier output?
- Tool utilization: tools assigned but never called in 7d
- Heartbeat health: missed beats, excessive cost per heartbeat
- Playbook quality trends: declining `quality_score` on any playbook

**Root cause classification for each issue:**
`model_mismatch` | `prompt_drift` | `tool_gap` | `overload` | `underutilized` | `config_stale` | `cost_inefficient`

**Self-optimization:** Also reads the HARNESS playbook's own `learning_data` via `platform_get_recipe` to factor in patterns from prior HARNESS runs (e.g., "Step 1 COLLECT is slow due to NL2SQL timeout").

**Output:** Structured diagnosis JSON → scratchpad → Step 3

### 6.3 Phase 3: PRESCRIBE

**Agent:** Auto (CTO)
**Mode:** Read-only (generates prescriptions but does not apply)
**Purpose:** Generate prioritized, risk-scored configuration change proposals.

**Prescription schema:**

```json
{
  "prescription_id": "rx-2026-04-06-001",
  "target_type": "agent",
  "target_id": 147,
  "target_name": "SCOUT",
  "change_type": "heartbeat_tune",
  "current_value": {"interval_minutes": 120},
  "proposed_value": {"interval_minutes": 180},
  "risk_score": 2,
  "expected_improvement": "Save ~$0.80/week in heartbeat token costs with minimal staleness impact",
  "rationale": "SCOUT's heartbeat cost increased 28% while task volume dropped 36%. Extending interval reduces cost without impacting low activity."
}
```

**Risk scoring rules:**

| Score | Auto-apply? | Scope |
|-------|-------------|-------|
| 1 | Yes | Description, tags, team, job_title updates |
| 2 | Yes | Heartbeat interval ±30min, temperature ±0.1, active_hours shift |
| 3 | Queue for human | Model change within same cost tier, tool addition |
| 4 | Queue for human | Model tier change (haiku→sonnet), prompt rewrite, skill changes |
| 5 | Queue for human | Agent deactivation, proactive_level→autonomous, deletion |

**Pareto filter:** Prefer accuracy improvements over cost cuts when success_rate < 85%. Only optimize cost when quality is healthy. This prevents the optimizer from racing to the cheapest model at the expense of output quality.

**Rejected-change awareness:** Skip prescriptions matching prior "blocked" board tasks to avoid re-prescribing what the human already rejected.

**Output:** Prescriptions JSON array → scratchpad → Step 4

### 6.4 Phase 4: APPLY

**Agent:** Auto (CTO)
**Mode:** Write (conditional on risk score)
**Purpose:** Execute safe changes, queue risky ones for human review.

**Auto-apply (risk ≤ 2):**

| Change Type | Platform Tool |
|-------------|---------------|
| model_change (same tier) | `platform_update_agent` with model_config |
| temperature_adjust | `platform_update_agent` with model_config.temperature |
| heartbeat_tune | `platform_configure_agent_heartbeat` |
| tag_update | `platform_update_agent` with tags |
| description_update | `platform_update_agent` with description |

**Queue for review (risk ≥ 3):**

Creates `board_task` via `platform_create_task` with:
- Title: `[HARNESS] {change_type} for {agent_name}`
- Description: Full prescription details + rationale + risk score
- Tags: `["harness", "org-review", "risk-{N}"]`
- Priority: `"high"` if risk ≥ 4, else `"medium"`

**Also applies:** Previously approved board tasks (status=done, tag=harness) that weren't yet applied from prior runs.

**Output:** Changelog (applied + queued + failed) → scratchpad → Step 5

### 6.5 Phase 5: BASELINE

**Agent:** Auto (CTO)
**Mode:** Write
**Purpose:** Snapshot new state for next week's comparison, publish artifacts, submit audit report.

**Writes:**
1. `workspace_write_file` → `/harness/baseline_latest.json` (overwrite — current state)
2. `workspace_write_file` → `/harness/baselines/{YYYY-MM-DD}.json` (archive — append-only)
3. `workspace_write_file` → `/harness/traces/{YYYY-MM-DD}_trace.json` (full diagnostic trace)
4. `workspace_write_file` → `/harness/changelog/{YYYY-MM-DD}.md` (human-readable changes)
5. `workspace_write_file` → `/harness/prescriptions/{YYYY-MM-DD}_rx.json` (all prescriptions)
6. `platform_submit_report` → type=`"audit"`, title=`"HARNESS Weekly Org Review — Run #{N}"`

**Convergence tracking** (stored in baseline):

```json
{
  "convergence": {
    "iteration_count": 4,
    "total_delta_magnitude": 3.2,
    "prev_delta_magnitude": 7.8,
    "metric_variance": 0.04,
    "status": "converging"
  }
}
```

---

## 8. Baseline Schema

```json
{
  "version": 1,
  "created_at": "2026-04-06T02:15:00Z",
  "iteration": 4,
  "per_agent": {
    "147": {
      "name": "AUTO",
      "model": "anthropic/claude-sonnet-4-20250514",
      "success_rate": 92.3,
      "cost_7d": 12.45,
      "efficiency_score": 78,
      "error_rate": 3.2,
      "token_usage_7d": 450000,
      "heartbeat_interval": 15,
      "proactive_level": "act_notify",
      "tools_assigned": 12,
      "tools_used_7d": 8,
      "skills_assigned": 3
    }
  },
  "org_level": {
    "total_cost_7d": 87.50,
    "avg_success_rate": 88.1,
    "active_agents": 12,
    "total_missions_7d": 8,
    "total_heartbeats_7d": 1200,
    "total_playbook_runs_7d": 15
  },
  "convergence": {
    "iteration_count": 4,
    "total_delta_magnitude": 3.2,
    "prev_delta_magnitude": 7.8,
    "metric_variance": 0.04,
    "status": "converging"
  },
  "applied_changes": [],
  "queued_changes": []
}
```

---

## 9. Workspace File Layout

```
/harness/
  baseline_latest.json            # Current baseline (overwritten each run)
  baselines/
    2026-04-06.json               # Historical baselines (append-only)
    2026-04-13.json
  traces/
    2026-04-06_trace.json         # Full diagnostic trace per run
    2026-04-13_trace.json
  changelog/
    2026-04-06.md                 # Human-readable change log
    2026-04-13.md
  prescriptions/
    2026-04-06_rx.json            # All prescriptions per run
```

---

## 10. Risk Framework

| Risk | Auto-apply? | Examples | Rationale |
|------|-------------|----------|-----------|
| 1 | Yes | Description, tags, team | Cosmetic, zero operational impact |
| 2 | Yes | Heartbeat interval ±30min, temperature ±0.1 | Low impact, easily reversible |
| 3 | Queue | Model change within tier, add tool | Could affect output quality |
| 4 | Queue | Model tier change, prompt rewrite, skill change | Significant behavioral change |
| 5 | Queue | Agent deactivation, proactive→autonomous | Irreversible or high-risk |

**Rollback strategy (v1):** If an auto-applied change causes regression (detected in next HARNESS run), the DIAGNOSE step will identify it via delta computation, and PRESCRIBE will generate a reversion prescription. The baseline diff serves as the rollback reference.

---

## 11. Convergence Detection

| State | Condition | Behavior |
|-------|-----------|----------|
| `exploring` | iteration_count < 3 | Full optimization pass, weekly cadence |
| `converging` | total_delta_magnitude decreasing run-over-run | Continue weekly, standard risk thresholds |
| `converged` | delta < 2.0 AND variance < 0.02 for 2+ consecutive runs | Note in report, suggest biweekly switch |
| `diverging` | total_delta_magnitude increasing | Keep weekly, flag in report for human attention |

`total_delta_magnitude` = sum of absolute deltas across all agents across all metrics (success_rate, cost, efficiency, error_rate). When this approaches zero, the org configuration is stable.

---

## 12. Self-Learning

### 12.1 Heartbeat Results as Learning Data

Each HARNESS run stores its results in `heartbeat_results` (same as any orchestrator heartbeat):
- `findings` JSONB — diagnosis summary, prescription count, convergence state
- `actions_taken` JSONB — applied changes, queued changes, failures
- `tokens_used`, `cost` — resource consumption per run

Phase 2 (DIAGNOSE) reads prior HARNESS `heartbeat_results` via `platform_query_data` to detect multi-run patterns:
- Is COLLECT getting slower? (tokens_used trending up)
- Are prescriptions being rejected repeatedly? (same board tasks reappearing as blocked)
- Is the cost of running HARNESS itself growing?

### 12.2 Convergence as Self-Assessment

The convergence signal (`total_delta_magnitude`) IS the quality metric. If HARNESS runs are producing changes but metrics aren't improving (delta not shrinking), the system is thrashing. The DIAGNOSE phase detects this and reduces prescription aggressiveness.

### 12.3 Trace History as Memory

The `/harness/traces/` workspace files are HARNESS's long-term memory. Phase 2 can read all prior traces (non-Markovian access) to identify:
- "We tried upgrading SCOUT to sonnet in run 3 but it was reverted in run 4 — don't retry"
- "Heartbeat tuning for NEXUS has stabilized at 150min across 3 runs — leave it alone"

This mirrors Meta-Harness's key finding: the proposer reads 82+ files of prior history, not just the last run.

---

## 13. Feedback Loop — How Run N+1 Reads Run N

| Channel | Mechanism | What It Provides |
|---------|-----------|------------------|
| Baseline file | Step 1 reads `/harness/baseline_latest.json` | Previous week's per-agent metrics + convergence state |
| Board task resolution | Step 1 queries `platform_list_tasks(tags=["harness"])` | Which prescriptions were approved (done) or rejected (blocked) |
| Playbook learning_data | Step 2 reads via `platform_get_recipe` | Patterns from prior HARNESS runs |
| Trace history | Step 2 reads `/harness/traces/` | Multi-week trend detection (e.g., 3-week decline) |
| Report history | `platform_get_latest_report` for Auto | Prior audit report summaries |

---

## 14. Platform Tools

### 13.1 `platform_harness_status` (read)

Returns current HARNESS state: last run date, convergence status, quality score, iteration count, next scheduled run.

### 13.2 `platform_harness_trigger` (write)

Manually trigger a HARNESS run outside the weekly cron schedule. Useful for post-incident optimization or after major org changes.

### 13.3 `platform_harness_history` (read)

List past HARNESS runs with dates, prescription counts, applied/queued counts, convergence state per run.

---

## 15. Implementation — Files

### 14.1 New Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `orchestrator/services/harness_service.py` | 150 | `HarnessService`: `ensure_playbook_exists()` (called at workspace provisioning), `get_status()`, `trigger_now()` |
| `orchestrator/modules/tools/discovery/actions_harness.py` | 80 | 3 ActionDefinitions registered via `register_harness_actions()` |
| `orchestrator/modules/tools/discovery/handlers_harness.py` | 120 | Handler functions for the 3 platform tools |

### 15.2 Modified Files

| File | Change |
|------|--------|
| `orchestrator/modules/tools/discovery/platform_actions.py` | Add `from .actions_harness import register_harness_actions` + call in `register_all_actions()` |
| `orchestrator/modules/tools/execution/unified_executor.py` | Add 3 handler entries to `_handlers` dict |
| `orchestrator/consumers/chatbot/auto.py` | Add HARNESS keywords to `_PLATFORM_KEYWORDS` |
| `orchestrator/main.py` | Add `HarnessService.start(scheduler)` to lifespan startup |

### 15.3 3-File Pattern (Platform Tools)

Every platform tool in Automatos follows the same pattern:
1. **`actions_*.py`** — ActionDefinition registration (name, description, parameters, permission_level)
2. **`handlers_*.py`** — Handler function that does the work
3. **`platform_actions.py`** — Wires registrar into `register_all_actions()`

Plus `unified_executor.py` gets the handler entry in its `_handlers` dict. The 3 HARNESS platform tools follow this exactly.

### 15.4 What's NOT a Platform Tool

The 5-phase pipeline itself is NOT platform tools. It's hardcoded methods in `HarnessService`:
- `_phase_collect(workspace_id)` — gathers metrics via platform tool calls
- `_phase_diagnose(workspace_id, metrics, baseline)` — LLM-powered analysis
- `_phase_prescribe(workspace_id, diagnosis)` — LLM-powered prescription generation
- `_phase_apply(workspace_id, prescriptions)` — executes safe changes, queues risky ones
- `_phase_baseline(workspace_id, metrics, changelog)` — writes workspace files + submits report

These run sequentially inside a single `_harness_tick(workspace_id)` method, called by the APScheduler cron job.

---

## 16. Migration

**No migration needed.** No new tables, no new columns, no seed data.

HARNESS uses only existing infrastructure:
- `heartbeat_results` — stores execution results per run
- `board_tasks` — queued prescriptions for human review
- `agent_reports` — weekly audit report
- Workspace files — baselines, traces, changelogs

The service registers itself at startup. Deploy the code, restart the server, HARNESS is live.

---

## 17. Phasing

### v1 — Core Loop (this PRD)
- Orchestrator heartbeat schedule, weekly cron, hardcoded 5-phase pipeline
- Metrics collection via platform tools + NL2SQL
- Per-agent diagnosis with delta computation
- Prescriptions for: model_change, heartbeat_tune, temperature_adjust, tag/description
- Auto-apply risk ≤ 2, board_task queue risk ≥ 3
- Workspace file storage (baselines, traces, changelog)
- Convergence detection (basic)
- 3 platform tools for Auto (status, trigger, history)
- No migration, no seed data, no user-facing config

### v2 — Expanded Optimization Surface
- Prompt optimization: generate variants, A/B test via shadow mode
- Tool/skill assignment optimization
- Blueprint rule suggestions
- Explicit rollback mechanism with change_id tracking
- Convergence auto-cadence switching (weekly ↔ biweekly)
- Pareto frontier visualization in frontend

### v3 — Autonomous Adaptation
- Cross-workspace pattern sharing (marketplace-level insights)
- Agent creation/retirement recommendations
- HARNESS self-prompt-tuning from learning_data
- Cost budget enforcement (auto-downgrade models when budget exceeded)
- Grafana integration for infrastructure-level metrics

---

## 18. Meta-Harness Mapping

How HARNESS maps to the Stanford Meta-Harness architecture:

| Meta-Harness Component | HARNESS Equivalent |
|------------------------|-------------------|
| Proposer agent (Claude Code Opus) | Auto (CTO agent, Opus model) |
| Filesystem of prior candidates | `/harness/` workspace files (baselines, traces, changelogs) |
| Execution traces | `metrics.json` + per-agent health cards + heartbeat results |
| Score function | Agent rankings + SLA compliance + cost delta |
| Pareto frontier | Risk tiers (auto-apply vs human-review) |
| Code-space search | Config-space search (models, heartbeats, temperatures, tools) |
| Non-Markovian access | Step 2 reads ALL prior traces, not just last run |
| Convergence detection | total_delta_magnitude tracking across runs |
| Self-learning | PlaybookLearningService (Stage 6) + PlaybookQualityService (Stage 7) |

**Key difference:** Meta-Harness operates in a sandbox (evaluation set). HARNESS operates on live agents with real consequences. The risk framework ensures only safe changes are auto-applied.

---

## 19. Verification Plan

1. Seed the playbook via `seed_harness_playbook.py` against dev workspace
2. Manual trigger via `platform_harness_trigger` in Auto chat
3. Verify COLLECT: `/harness/traces/{date}_trace.json` has all metric categories
4. Verify DIAGNOSE: trace has per-agent health cards with deltas (first run: all baselines are "new")
5. Verify PRESCRIBE: prescriptions JSON has risk scores and rationale
6. Verify APPLY: board has queued tasks with `harness` tag; agents have auto-applied changes
7. Verify BASELINE: `/harness/baseline_latest.json` exists with correct schema
8. Verify REPORT: Reports tab shows "HARNESS Weekly Org Review" audit report
9. Verify self-learning: HARNESS playbook's `learning_data` and `quality_score` updated
10. Run a second time: Step 1 reads previous baseline, Step 2 computes real deltas
11. Cron test: playbook appears in `PlaybookSchedulerService._load_cron_playbooks()` on restart

---

## 20. Success Criteria

- [ ] HARNESS playbook created with 5 steps and weekly cron schedule
- [ ] Manual trigger works via `platform_harness_trigger`
- [ ] First run produces baseline, trace, changelog, prescriptions, and audit report
- [ ] Second run reads prior baseline and computes meaningful deltas
- [ ] Risk ≤ 2 changes are auto-applied (verified via `platform_get_agent`)
- [ ] Risk ≥ 3 changes appear as board tasks with `harness` tag
- [ ] Convergence state tracks correctly across 3+ runs
- [ ] PlaybookLearningService generates patterns from HARNESS executions
- [ ] PlaybookQualityService scores HARNESS executions on 5 dimensions
- [ ] Auto can answer "What's the HARNESS status?" via `platform_harness_status`
