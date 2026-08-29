"""Fleet-status ActionDefinition — PRD-228 US-003 (platform_fleet_status).

One read tool that answers "how's the team doing?" in a single call: the live
floor read-model (US-001) rendered compactly, with an anomalies section. It
takes no arguments — the workspace comes from the caller context — so the
``required[]`` list is empty (onboarding-wall lesson: never mark a field the
handler cannot default).
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_fleet_actions(registry: ActionRegistry) -> None:
    """Register the fleet-status read tool (PRD-228)."""

    registry.register(ActionDefinition(
        name="platform_fleet_status",
        description=(
            "Live floor state for the whole team in one call: per agent, what "
            "they're working on right now (or idle / blocked awaiting an answer), "
            "queue depth, and rolling-24h token/cost — plus an ANOMALIES section "
            "flagging stalled agents (working but no recent activity), watches "
            "that hit their action budget, and agents blocked on an open ask. "
            "Use to answer 'how's the team doing?', 'what is everyone working "
            "on?', 'is anyone stuck?', or before deciding who to assign work to."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["fleet", "read", "agents", "status", "situational-awareness"],
        examples=[
            "how's the team doing?",
            "what is everyone working on right now?",
            "is anyone stuck or stalled?",
            "who's free to take this on?",
        ],
    ))
