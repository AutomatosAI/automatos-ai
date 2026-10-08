"""platform_add_playbook_step: a step always has the agent that does it (F321, night 9b).

Chat 5247a359: Auto added both steps of "Monday green stock" with ``agent_id: null``,
because this tool said agent_id was "optional — uses default agent if not set". There
is no default agent; the run failed, "steps 1 and 2 have no agent". The tool now asks
for agent_id or agent_name and adds no step without one
(modules/tools/discovery/playbook_staffing.py). It left actions_playbooks.py, whose one
register function is past the length rule.
"""

from .action_registry import ActionDefinition, ActionRegistry
from .agent_refs import agent_id_property

_ADD_PLAYBOOK_STEP_PARAMETERS = {
    "type": "object",
    "properties": {
        "playbook_id": {
            "type": "integer",
            "description": "ID of the playbook to add the step to.",
        },
        "prompt_template": {
            "type": "string",
            "description": "The prompt template for this step. Supports {input.*} and {steps[N].*} variable substitution.",
        },
        "agent_id": agent_id_property("The agent that does this step. Give it, or agent_name: there is no "
                                      "default agent, and a step with none is not added"),
        "agent_name": {
            "type": "string",
            "description": ("The agent's name or job title as the owner said it (e.g. 'Inventory Watchdog'), "
                            "when agent_id isn't given. Exactly one agent must answer to it; if none or "
                            "several do, nothing is added and the agents are listed."),
        },
        "order": {
            "type": "integer",
            "description": "Position in the step list (0-based). Defaults to end of list.",
        },
        "error_handling": {
            "type": "string",
            "enum": ["stop", "skip", "retry"],
            "description": "What to do if this step fails. Defaults to 'stop'.",
        },
        "output_key": {
            "type": "string",
            "description": "Key name to store this step's output under (for referencing in later steps).",
        },
    },
    "required": ["playbook_id", "prompt_template"],
}


def register_add_playbook_step_action(registry: ActionRegistry) -> None:
    """platform_add_playbook_step."""
    registry.register(ActionDefinition(
        name="platform_add_playbook_step",
        description=(
            "Append a new step to an existing playbook. Each step has a prompt template and "
            "the agent that does it (agent_id or agent_name): a step with no agent can't "
            "run, so none is added without one. Steps execute sequentially."
        ),
        category="playbooks",
        parameters=_ADD_PLAYBOOK_STEP_PARAMETERS,
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "steps", "add", "write"],
        examples=[
            "add a step to playbook 3 that summarizes the results",
            "add a code review step to the bug triage playbook",
        ],
    ))
