"""HARNESS ActionDefinitions (PRD-121): the weekly report-only review of the team (PRD-263),
and platform_harness_apply, which makes one change its owner approved (PRD-263 US-052)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_harness_actions(registry: ActionRegistry) -> None:
    """Register HARNESS platform tools for Auto."""
    _register_status(registry)
    _register_trigger(registry)
    _register_history(registry)
    _register_apply(registry)


def _register_status(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_harness_status",
        description=(
            "Returns the state of HARNESS, the weekly report-only review of the "
            "team. Once a week, on the weekday cron set for this installation, "
            "HARNESS judges each agent's last complete ISO week against its own "
            "recent weeks and writes an audit report; it changes nothing. Status "
            "is one of: disabled (the workspace's mode is Disabled), running (a "
            "run is in flight), failed (the latest run failed or was interrupted "
            "— check `error`), dormant_insufficient_agents (too few active agents "
            "to judge — see `active_agents`), scheduled_not_run_yet (no run yet — "
            "see `next_scheduled_run`), or completed (with iteration_count, "
            "cur_window — the ISO week judged —, coverage — judged agents over "
            "active agents —, convergence, the insufficient_data, watch and "
            "regression counts, last_run_at and artifacts.audit_report). An agent "
            "with too little work is INSUFFICIENT_DATA, never stable. Use when "
            "asked about team performance, HARNESS status, or org health."
        ),
        category="harness",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["harness", "optimization", "org-health", "status"],
        examples=[
            "how is the team performing?",
            "what's the optimization status?",
            "harness status",
            "org health check",
            "why hasn't harness run yet?",
        ],
    ))


def _register_trigger(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_harness_trigger",
        description=(
            "Start a HARNESS review now instead of waiting for the weekly run on "
            "the weekday cron. Like every run it is report-only: it judges the "
            "last complete ISO week, reports coverage and each agent's verdict "
            "(INSUFFICIENT_DATA when an agent had too little work) in the audit "
            "report, and changes nothing. A rerun of a week already judged keeps "
            "that week's verdicts. Useful after major org changes (new agents) or "
            "after an incident. Does nothing when the workspace's mode is Disabled."
        ),
        category="harness",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["harness", "optimization", "trigger"],
        examples=[
            "run harness now",
            "trigger optimization",
            "optimize the team",
            "run the optimization loop",
        ],
    ))


def _register_history(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_harness_history",
        description=(
            "List past HARNESS runs, newest first, from the run table: for each, "
            "the ISO week it judged (cur_window), status, start and finish time, "
            "iteration, coverage, convergence, the insufficient_data, watch and "
            "regression counts, how many changes its report suggested "
            "(prescription_count) and how many it applied itself (applied_count: "
            "0 for every weekly report-only run; a run from before the report-only "
            "loop shows what that loop applied). A figure a run did not record is "
            "null, never 0. Use to review how the team's verdicts changed over time."
        ),
        category="harness",
        parameters={
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of past runs to return (default 10).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["harness", "optimization", "history"],
        examples=[
            "show harness history",
            "what optimizations have been made?",
            "past optimization runs",
            "harness changelog",
        ],
    ))


def _register_apply(registry: ActionRegistry) -> None:
    # PRD-263 US-052: no card, because the owner's click on the change's Approvals card is its consent.
    registry.register(ActionDefinition(
        name="platform_harness_apply",
        description=(
            "Makes one HARNESS change that the owner approved on its Approvals card "
            "(Command Center > Governance > Approvals, or the card's evidence ticket). "
            "It runs when the owner clicks Approve: never call it to make a change "
            "yourself. It acts only while that approval is granted and has not "
            "expired, and only if the setting still reads the value it had when "
            "HARNESS proposed the change; otherwise it changes nothing and says why. "
            "The change (an agent's heartbeat interval, a tool taken from an agent, "
            "or the undoing of an earlier HARNESS change) is made as the workspace's Auto."
        ),
        category="harness",
        parameters={
            "type": "object",
            "properties": {
                "rx_id": {
                    "type": "string",
                    "description": "The id of the HARNESS change the Approvals card is for (rx-<run>-<n>).",
                },
            },
            "required": ["rx_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["harness", "approval"],
    ))
