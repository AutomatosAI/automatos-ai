# Platform Readiness Report — Mission Zero Enablement

**Date:** 2026-03-28
**Status:** Phases 1-3 Complete, Deployed, Migration Applied
**Commits:** eff58ec, 758faa7, d3ab771

---

## What Was Done

Mission Zero identified critical platform gaps preventing agents from self-managing. This report summarises the 3-phase build that addresses those gaps.

### Phase 1: Agent Analytics Tools (Complete)

**Problem:** 40+ analytics API endpoints existed but only 5 were exposed as platform tools. Agents like ATLAS and SENTINEL had no way to self-monitor.

**Solution:** 11 new platform tools wrapping real database queries:

| Tool | What It Returns |
|---|---|
| `platform_get_success_rate` | Overall agent success rate + 7-day trend |
| `platform_get_completion_time` | Avg task completion time + 24h comparison |
| `platform_get_error_rates` | Error breakdown by agent type with severity |
| `platform_get_queue_depth` | Pending/running task counts |
| `platform_get_efficiency_score` | Composite 0-100 score with A-D grade |
| `platform_get_cost_per_execution` | Cost trends with daily breakdown |
| `platform_get_peak_hours` | 24h usage pattern with peak/medium/low |
| `platform_get_bottlenecks` | Failure rates, queue buildup, slow executions |
| `platform_get_predictive_alerts` | Cost spikes, capacity warnings, rate limits |
| `platform_get_agent_ranking` | Agents ranked by composite performance score |
| `platform_get_sla_compliance` | Task completion + response time vs SLA targets |

**Impact:** Agents can now answer "how are we performing?", "what's costing the most?", "any problems?" without human intervention.

---

### Phase 2: KPI Dashboard Widgets (Complete)

**Problem:** The command centre had 8 operational widgets but zero KPI visibility for cost tracking, agent performance, playbook health, or approval status.

**Solution:**

**4 new widgets:**
- **Cost Tracker** — Total spend, daily sparkline chart, top 3 agent spenders, period-over-period change %
- **Agent Performance** — Color-coded horizontal bars (green >90%, yellow 70-90%, red <70%) ranked by success rate
- **Playbook Metrics** — Table with runs, success %, avg duration per playbook
- **Approval Gates** — Pending count, avg approval time, waiting mission list

**Widget Picker (new capability):**
- Dashboard refactored from static list to a **widget registry pattern**
- Click Customize -> "Widgets" popover shows all 12 widgets with show/hide toggles
- Drag to reorder still works
- Hidden state persists in localStorage
- New widgets auto-appear on existing installs (merge logic)
- Ready for a future widget marketplace

**Backend:** New `/api/kpi` router with 4 endpoints, all workspace-scoped.

---

### Phase 3: Governance & Blueprints (Complete)

**Problem:** `workspace.plan_limits` JSONB exists but was NEVER READ. `Agent.is_approved` exists but was NEVER ENFORCED. Zero budget tracking on missions. No quality standards.

**Solution:**

**New table: `agent_blueprints`**
- Per-workspace governance rules
- Configurable: `min_tools`, `require_system_prompt`, `max_budget_per_run`, `required_tags`, `allowed_models`
- One blueprint can be set as workspace default

**New columns on `orchestration_runs`:**
- `budget_config` — {max_cost, max_tokens, alert_at_pct}
- `budget_spent` — {cost, tokens, api_calls}

**6 new platform tools:**

| Tool | Permission | Purpose |
|---|---|---|
| `platform_list_blueprints` | read | List workspace blueprints |
| `platform_get_blueprint` | read | Get blueprint details |
| `platform_create_blueprint` | write | Create governance rules |
| `platform_update_blueprint` | write | Update rules |
| `platform_validate_agent` | read | Check agent against blueprint (pass/fail + reasons) |
| `platform_check_budget` | read | Mission budget status (ok/warning/exceeded) |

**Blueprint Validator Service:**
- Validates agents against configurable rules
- Returns specific failures ("Agent has 0 tools, minimum is 2") and warnings ("Missing recommended tags")
- Budget checker returns remaining cost/tokens with alert thresholds

**Frontend: AgentReadinessBadge**
- Green shield = all checks passed
- Yellow shield = warnings
- Red shield = failures
- Tooltip shows specific issues

---

## Platform Tool Inventory

**Total platform tools: 97** across 19 domains:

| Domain | Tools | Examples |
|---|---|---|
| Agents | 6 | list, get, create, update, delete |
| Playbooks/Recipes | 10 | CRUD + steps + execute |
| Analytics (base) | 5 | LLM usage, costs, workspace stats |
| Analytics (enhanced) | 11 | Success rate, bottlenecks, SLA, rankings |
| Governance | 6 | Blueprints, validation, budgets |
| Board Tasks | 6 | CRUD + assign + status |
| Marketplace | 9 | Browse agents/plugins/skills, install |
| Monitoring | 5 | Logs, Loki, Prometheus, alerts, health |
| Documents | 3 | List, delete, reprocess |
| Workspace | 7 | Info, memory, connected apps, store |
| Tools/LLMs | 3 | List tools, LLMs, datasources |
| Search | 2 | Chat history, memory search |
| Scheduling | 3 | Schedule, list, cancel |
| Reports | 2 | Submit, get latest |
| Assignments | 3 | Tool, skill, plugin assignment + heartbeat |
| Field | 3 | Query, inject, stability |
| Blog | 4 | Publish, list, get, update |
| Missions | 3 | Create, list, get |
| Workspace I/O | 6 | read_file, write_file, list_dir, grep, exec, git |

---

## Mission Zero Gaps — Status

| Gap Identified | Status | Notes |
|---|---|---|
| Analytics not exposed to agents | FIXED | 11 new tools (Phase 1) |
| No KPI visibility for humans | FIXED | 4 widgets + picker (Phase 2) |
| No governance/quality standards | FIXED | Blueprints + validator (Phase 3) |
| Budget fields exist but never read | FIXED | budget_config/budget_spent + check tool |
| is_approved never enforced | PARTIAL | Blueprint validation warns, doesn't block yet |
| Tool discovery for 880+ Composio actions | ALREADY DONE | `platform_list_tools` with search exists |
| Channel abstraction | ALREADY DONE | Templates use types, not specific channels |
| Agent config gaps (0 tools, no prompts) | NOW DETECTABLE | `platform_validate_agent` flags these |

---

## What Auto Can Now Do

1. **Self-monitor:** "What's our success rate?" -> real data from `platform_get_success_rate`
2. **Investigate problems:** "Any bottlenecks?" -> failure rates, queue buildup, slow execution detection
3. **Track costs:** "Are costs going up?" -> daily breakdown with trend analysis
4. **Rank agents:** "Who's the top performer?" -> composite scoring across success, speed, volume
5. **Check SLAs:** "Are we meeting targets?" -> completion rate + response time vs thresholds
6. **Set standards:** "Create a blueprint requiring prompts and 2 tools" -> governance enforcement
7. **Validate agents:** "Is SCOUT ready?" -> specific pass/fail with actionable reasons
8. **Monitor budgets:** "How much budget is left on this mission?" -> remaining cost/tokens with alerts
9. **Predict issues:** "Any upcoming capacity problems?" -> cost spikes, agent capacity, rate limits

---

## Verification Checklist

- [ ] Ask ATLAS: "What's our agent success rate?" — should return real % from executions
- [ ] Ask SENTINEL: "Any bottlenecks?" — should detect queue/failure/speed issues
- [ ] Open Activity -> Command Centre -> Customize -> Widgets — should show 12 widgets with toggles
- [ ] Cost Tracker widget shows real spend data
- [ ] Agent Performance widget shows color-coded bars
- [ ] Ask Auto: "Create a blueprint requiring system prompts and at least 2 tools"
- [ ] Ask Auto: "Validate agent SCOUT against the blueprint" — should flag missing tools/prompt
- [ ] Ask Auto: "Do you have everything you need to self-manage?" — the real test

---

## Next Steps

1. **Google Analytics integration** — GA4 added to workspace, needs frontend snippet + platform tool + KPI widget
2. **Budget enforcement** — Currently advisory (check_budget returns status). Next: auto-pause missions when exceeded
3. **Blueprint enforcement in coordinator** — Currently tools-only. Next: coordinator checks before dispatch, logs warnings
4. **Widget marketplace** — Registry pattern is ready for user-installed widgets
