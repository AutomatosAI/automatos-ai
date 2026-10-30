"""Auto cadence loops — Wave 4.

Day-of-week-aware checklist additions injected into the orchestrator
heartbeat prompt. Pure prompt assembly — no new scheduler, no new tables.

Reads ``workspace.settings.orchestrator.heartbeat.cadence``:

    {
      "weekly_org_review":  { "enabled": false, "day": "mon" },
      "harness_review":     { "enabled": false, "day": "mon" },
      "post_change_validation": { "enabled": false },
      "incident_review":    { "enabled": false }
    }

All loops default OFF — workspaces opt in by setting the flag(s). Existing
heartbeats keep their current behaviour until the user enables a cadence.

PRD-268 §12-Q11 retired the cadence "Daily Brief": the day's brief is the one
posted into each person's Auto thread (the scheduled digest, or the brief on
open, ``services.brief``). It has no default here, so a stored setting for it
is never merged and adds no block.

Day codes follow APScheduler's ``day_of_week`` convention: mon, tue, wed,
thu, fri, sat, sun. Default for weekly cadences is ``mon``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

# ----------------------------------------------------------------- defaults

CADENCE_DEFAULTS: Dict[str, Dict[str, Any]] = {
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

_WEEKLY_ORG_REVIEW = """## Weekly Org Review
Run this when today is the configured weekly review day:
1. `platform_list_agents` — audit skills_count, tools_count, heartbeat status. Flag thin/overloaded agents.
2. Manager review — pull each manager's latest report and surface the asks.
3. Skill drift — `platform_get_skill_content` for the role skills used most.
4. Duplicated responsibilities — cross-reference assignments. Recommend consolidation if overlap > 50%.
5. File a `summary` report with type=summary, status=ok|warning, recommendations + action_items populated."""


# PRD-263 US-023: HARNESS is report-only and runs on the installation's weekday cron, so
# this text names no day or time and asks for no queued cards. {judged_week} is filled
# in when the block is built: the ISO week this week's run judges.
_HARNESS_REVIEW = """## Weekly HARNESS Review
HARNESS reviews the team once a week and only reports: it changes nothing and queues nothing. This week's run judges {judged_week}.
1. `platform_harness_status`. If status is scheduled_not_run_yet, or cur_window is not {judged_week}, this week's run has not happened yet: tell the owner the review waits for it, and stop.
   If status is failed, or artifacts.audit_report is not ok: surface it as a platform issue (NOT an agent issue).
   If status is dormant_insufficient_agents: say HARNESS waits for more active agents, and stop.
2. `platform_get_latest_report agent_name=Auto report_type=audit` — pull the audit.
3. Summarise for the owner: coverage (judged agents out of active agents) and the INSUFFICIENT_DATA, WATCH and REGRESSION counts, naming the agents behind each WATCH and REGRESSION. If convergence is insufficient_data, say most agents had too little work to judge; never call an unjudged agent stable.
4. Send via `platform_send_notification` with severity=info, status=warning if any agent is in REGRESSION, else ok.
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
     - escalation_level: 0..4 (use 2=APPROVAL when changes need the owner's call, 0=FYI when team is healthy)
     - requires_approval: true when any action_item is non-trivial
7. For each action_item that maps to existing work, `platform_create_task` for the owning agent INSIDE YOUR SUBTREE only.
   Cross-team or out-of-subtree actions go in recommendations with escalation_target=auto.
8. NEVER call platform_update_agent / platform_update_playbook / platform_update_skill / platform_assign_*
   in Advisor mode. The hierarchy gate will reject those anyway. Recommend, don't apply."""


_PROMPT_MAP: Dict[str, str] = {
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

    Always-on loops (post_change_validation, incident_review) fire every
    tick when enabled. Day-gated loops (weekly_org_review,
    harness_review) fire only when today matches their configured day.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    today_code = _DAY_CODES[now.weekday()]
    cadence = get_cadence_config(hb_config)
    blocks: list[str] = []

    # Always-on cadences
    for key in ("post_change_validation", "incident_review"):
        if cadence[key].get("enabled"):
            blocks.append(_prompt(key, now))

    # Day-gated cadences
    for key in ("weekly_org_review", "harness_review", "team_review"):
        cfg = cadence[key]
        if cfg.get("enabled") and cfg.get("day", "mon") == today_code:
            blocks.append(_prompt(key, now))

    if not blocks:
        return ""
    return "\n\n" + "\n\n".join(blocks)


def _prompt(key: str, now: datetime) -> str:
    """The cadence's prompt text; the HARNESS review names the ISO week this week's run judges."""
    if key != "harness_review":
        return _PROMPT_MAP[key]
    from services.harness_runs import last_complete_iso_week

    return _PROMPT_MAP[key].format(judged_week=last_complete_iso_week(now))


_DAY_CODES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
