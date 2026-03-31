"""HARNESS ActionDefinitions (PRD-121) — Self-Optimizing Organization Loop."""

from .action_registry import ActionDefinition, ActionRegistry


def register_harness_actions(registry: ActionRegistry) -> None:
    """Register HARNESS platform tools for Auto."""

    registry.register(ActionDefinition(
        name="platform_harness_status",
        description=(
            "Returns current HARNESS optimization loop state: last run date, "
            "convergence status (exploring/converging/converged/diverging), "
            "iteration count, and next scheduled run. Use when asked about "
            "team performance, optimization status, or org health."
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
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_harness_trigger",
        description=(
            "Manually trigger a HARNESS optimization run outside the weekly "
            "cron schedule. Useful after major org changes (new agents, model "
            "swaps) or post-incident to re-evaluate configurations."
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

    registry.register(ActionDefinition(
        name="platform_harness_history",
        description=(
            "List past HARNESS optimization runs with dates, prescription "
            "counts, applied/queued counts, and convergence state per run. "
            "Use to review what optimizations have been made over time."
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
