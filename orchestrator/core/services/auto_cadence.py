"""Auto cadence loops — Wave 4.

Day-of-week-aware checklist additions injected into the orchestrator
heartbeat prompt. Pure prompt assembly — no new scheduler, no new tables.

Reads ``workspace.settings.orchestrator.heartbeat.cadence``:

    {
      "daily_brief":        { "enabled": false },
      "weekly_org_review":  { "enabled": false, "day": "mon" },
      "harness_review":     { "enabled": false, "day": "mon" },
      "post_change_validation": { "enabled": false },
      "incident_review":    { "enabled": false }
    }

All loops default OFF — workspaces opt in by setting the flag(s). Existing
heartbeats keep their current behaviour until the user enables a cadence.

Day codes follow APScheduler's ``day_of_week`` convention: mon, tue, wed,
thu, fri, sat, sun. Default for weekly cadences is ``mon``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ----------------------------------------------------------------- defaults

CADENCE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "daily_brief": {"enabled": False},
    "weekly_org_review": {"enabled": False, "day": "mon"},
    "harness_review": {"enabled": False, "day": "mon"},
    "post_change_validation": {"enabled": False},
    "incident_review": {"enabled": False},
    # PRD-140 Phase 1 — team-lead weekly review (Advisor only, no apply).
    # Fires on the configured day for agents whose ``team_lead_enabled``
    # flag is True. The block is injected into the team-lead's own
    # heartbeat (via _agent_tick), not Auto's orchestrator heartbeat.
    "team_review": {"enabled": False, "day": "mon"},
}


# ----------------------------------------------------------------- prompts
# Kept in this module so the heartbeat prompt builder stays small and so
# the prompt text itself is reviewable in one place.

_DAILY_BRIEF = """## Auto Daily Brief
Run through this every tick:
1. What changed in the last 24h? (`platform_get_activity_feed period=1d`)
2. What's blocked? (`platform_list_tasks status=blocked`)
3. What needs my call? (reports with requires_approval=true and not acknowledged; missions with escalation_level >= 2)
4. What did I handle for Gerard? — surface the autonomous actions (auto-applied HARNESS prescriptions, etc.)
5. What am I watching? — heartbeats with objective_met=False, agents trending toward warning
Reply with five short lines, one per item. If a section is empty, say "clear"."""


_WEEKLY_ORG_REVIEW = """## Weekly Org Review
Run this when today is the configured weekly review day:
1. `platform_list_agents` — audit skills_count, tools_count, heartbeat status. Flag thin/overloaded agents.
2. Manager review — pull each manager's latest report and surface the asks.
3. Skill drift — `platform_get_skill_content` for the role skills used most.
4. Duplicated responsibilities — cross-reference assignments. Recommend consolidation if overlap > 50%.
5. File a `summary` report with type=summary, status=ok|warning, recommendations + action_items populated."""


_HARNESS_REVIEW = """## Monday HARNESS Review
HARNESS runs Sunday 02:00 UTC. Review the audit:
1. `platform_harness_status` — confirm status=completed and artifacts.audit_report=ok.
   If status is failed, dormant_*, or any artifact failed: surface as a platform issue (NOT agent issue).
2. `platform_get_latest_report agent_name=Auto report_type=audit` — pull the audit.
3. Summarise for Gerard: convergence trend, top 3 issues, applied count, queued-for-review count, any failed artifacts.
4. `platform_list_tasks tag=harness status=todo` — list queued prescriptions with risk + rationale.
5. Send via `platform_send_notification` with severity=approval (if asks present) or info (if green).
Short message, not a wall of text."""


_POST_CHANGE_VALIDATION = """## Post-Change Validation
Triggered when significant platform changes shipped recently:
1. Identify the impacted agents/tools/playbooks from the change.
2. `platform_validate_agent` for each impacted agent — confirm governance still passes.
3. Verify heartbeats still fire correctly — read recent heartbeat_results.
4. Re-run any playbooks that reference changed components.
5. File a delivery report capturing what was validated and what regressed."""


_INCIDENT_REVIEW = """## Incident Review
Triggered after a failure pattern (failed HARNESS run, repeated agent_error events, mission stalls):
1. What happened — pull the failing event, error, run.
2. Impact — which agents/missions/users were affected?
3. Fix — what was done to recover?
4. Prevention — what stops this recurring? (config change, board task, monitoring rule)
5. Owner — who watches this going forward?
File an incident report with type=incident, status=warning|critical, action_items populated."""


# PRD-140 Phase 1 — Advisor-only team review.
# Fires inside a team-lead agent's own heartbeat. Read + diagnose + report;
# NEVER apply. Edits to team agents/playbooks/heartbeats route as
# action_items + linked_task_ids on the report; Auto/Gerard decide.
_TEAM_REVIEW = """## Team Review (Advisor — no edits)
You manage a team. Today is your review day. Read first, recommend second, never edit:
1. `platform_browse_reports period=7d agent_team=<your team>` — pull every report your team filed this week.
2. For each member, surface: did they hit their objective (`heartbeat_results.objective_met`)? are they over/under-tasked? any errors?
3. `platform_list_tasks` — check open tasks assigned to your team. Flag stalls, missed deadlines, gaps.
4. Compare with last week — what got better, what got worse? (Pull the previous team-review report.)
5. Identify gaps — missing skills, overdue work, agents drifting from their job_title.
6. `platform_submit_report report_type=summary` with these fields populated:
     - title:           "<Your team> Weekly Review — <YYYY-MM-DD>"
     - status:          ok|warning|critical
     - recommendations: structured list of suggested changes (target agent, change_type, reason, risk_tier)
     - action_items:    concrete next steps with owner agent_id
     - escalation_level: 0..4 (use 2=APPROVAL when changes need Gerard, 0=FYI when team is healthy)
     - requires_approval: true when any action_item is non-trivial
7. For each action_item that maps to existing work, `platform_create_task` for the owning agent INSIDE YOUR SUBTREE only.
   Cross-team or out-of-subtree actions go in recommendations with escalation_target=auto.
8. NEVER call platform_update_agent / platform_update_playbook / platform_update_skill / platform_assign_*
   in Advisor mode. The hierarchy gate will reject those anyway. Recommend, don't apply."""


_PROMPT_MAP: Dict[str, str] = {
    "daily_brief": _DAILY_BRIEF,
    "weekly_org_review": _WEEKLY_ORG_REVIEW,
    "harness_review": _HARNESS_REVIEW,
    "post_change_validation": _POST_CHANGE_VALIDATION,
    "incident_review": _INCIDENT_REVIEW,
    "team_review": _TEAM_REVIEW,
}


# ----------------------------------------------------------------- API


def get_cadence_config(hb_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Merge stored cadence settings onto defaults."""
    user_cfg = (hb_config or {}).get("cadence") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for key, default in CADENCE_DEFAULTS.items():
        merged = dict(default)
        merged.update(user_cfg.get(key) or {})
        out[key] = merged
    return out


def build_cadence_block(
    hb_config: Dict[str, Any],
    *,
    now: Optional[datetime] = None,
) -> str:
    """Return the cadence section for today's tick (empty when nothing fires).

    Always-on loops (daily_brief, post_change_validation, incident_review)
    fire every tick when enabled. Day-gated loops (weekly_org_review,
    harness_review) fire only when today matches their configured day.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    today_code = _DAY_CODES[now.weekday()]
    cadence = get_cadence_config(hb_config)
    blocks: list[str] = []

    # Always-on cadences
    for key in ("daily_brief", "post_change_validation", "incident_review"):
        if cadence[key].get("enabled"):
            blocks.append(_PROMPT_MAP[key])

    # Day-gated cadences
    for key in ("weekly_org_review", "harness_review", "team_review"):
        cfg = cadence[key]
        if cfg.get("enabled") and cfg.get("day", "mon") == today_code:
            blocks.append(_PROMPT_MAP[key])

    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks)


_DAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
