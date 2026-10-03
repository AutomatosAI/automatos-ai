"""The tool that resumes a mission (F247: it also retries a failed one).

Night 7: #0176 failed while the AI credit was out, and Resume was refused because
the mission was not paused. Resume now retries a failed mission with the same plan
(modules/coordination/mission_retry.py), and this tool says so. It left
actions_missions.py, whose one register function is past the length rule.
"""

from .action_registry import ActionDefinition, ActionRegistry

_MISSION_ID = {"mission_id": {"type": "string", "description": "The mission/run UUID."}}


def register_mission_resume_action(registry: ActionRegistry) -> None:
    """Register platform_resume_mission."""
    registry.register(ActionDefinition(
        name="platform_resume_mission",
        description=("Resume a paused mission (it goes back to running), or retry a failed one: its failed "
                     "steps run again from where it stopped, with the same plan."),
        category="missions",
        parameters={"type": "object", "properties": dict(_MISSION_ID), "required": ["mission_id"]},
        permission_level="write",
        requires_confirmation=False,
        tags=["missions", "write", "lifecycle", "resume"],
        examples=["resume that mission", "continue the paused mission", "retry the failed mission"],
    ))
