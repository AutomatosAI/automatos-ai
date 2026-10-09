"""The playbook tools that carry the owner's "wait for me" (F242): create, update, execute.

Night 7: a run the owner asked to wait for them went straight to Done (#0112),
and a timer's runs did too (#0162, #0170), because a playbook had no such
setting: Auto wrote "wait for me" into a step's prompt, which nothing reads.
``wait_for_me`` is now the playbook's setting (``execution_config``) or one
run's request; services/playbook_wait.py honours it. These three tools left
actions_playbooks.py, whose one register function is past the length rule.
"""

from .action_registry import ActionDefinition, ActionRegistry
from .actions_playbook_schedule import register_playbook_schedule_action
from .actions_playbook_steps import register_add_playbook_step_action
from .actions_playbooks import _INPUTS_PARAM

_CREATE_PLAYBOOK_PARAMETERS = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Name for the new playbook.",
        },
        "description": {
            "type": "string",
            "description": "What the playbook does.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for categorization.",
        },
        "wait_for_me": {
            "type": "boolean",
            "description": (
                "true when the owner wants every run of this playbook to wait for their check: "
                "its card goes to Review, not Done, until they approve it. Use it whenever the "
                "owner says to wait for them, never a sentence in a step's prompt."
            ),
        },
        "inputs": dict(_INPUTS_PARAM),
    },
    "required": ["name", "description"],
}

_UPDATE_PLAYBOOK_PARAMETERS = {
    "type": "object",
    "properties": {
        "playbook_id": {
            "type": "integer",
            "description": "ID of the playbook to update.",
        },
        "name": {
            "type": "string",
            "description": "New name for the playbook.",
        },
        "description": {
            "type": "string",
            "description": "New description.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Replace the playbook's tags with this list.",
        },
        "execution_config": {
            "type": "object",
            "description": "Runtime config: { mode, max_retries, timeout_per_step, quality_threshold }.",
        },
        "schedule_config": {
            "type": "object",
            "description": ("Schedule config: { type: 'manual'|'cron'|'trigger', cron_expression, trigger_config, "
                            "timezone, enabled }. To switch a timer off send {enabled: false}: its time is kept, "
                            "and {enabled: true} switches it back on."),
        },
        "wait_for_me": {
            "type": "boolean",
            "description": (
                "true when the owner wants every run of this playbook to wait for their check: "
                "its card goes to Review, not Done, until they approve it. Use it whenever the "
                "owner says to wait for them, never a sentence in a step's prompt."
            ),
        },
        "inputs": dict(_INPUTS_PARAM),
    },
    "required": ["playbook_id"],
}

_EXECUTE_PLAYBOOK_PARAMETERS = {
    "type": "object",
    "properties": {
        "playbook_id": {
            "type": "integer",
            "description": "ID of the playbook to execute.",
        },
        "playbook_name": {
            "type": "string",
            "description": "Name of the playbook to execute (alternative to ID).",
        },
        "input_data": {
            "type": "object",
            "description": (
                "Input data to pass to the playbook (key-value pairs); its steps read "
                "each as {key}. A playbook that takes one text reads it as {input}: "
                'pass {"input": "<the text>"}.'
            ),
        },
        "wait_for_me": {
            "type": "boolean",
            "description": (
                "true when the owner wants this run to wait for their check: its card goes to "
                "Review, not Done, until they approve it (the default is the playbook's own setting)."
            ),
        },
    },
    "required": [],
}


def register_playbook_run_actions(registry: ActionRegistry) -> None:
    """Register the playbook tools that take the owner's "wait for me"."""
    _register_create_playbook(registry)
    _register_update_playbook(registry)
    _register_execute_playbook(registry)
    _register_get_playbook_execution(registry)  # F321: left actions_playbooks.py to take `step`
    register_playbook_schedule_action(registry)  # F266: the timer takes wait_for_me too; left actions_playbooks.py
    register_add_playbook_step_action(registry)  # F321: a step always has its agent; left actions_playbooks.py


def _register_create_playbook(registry: ActionRegistry) -> None:
    """platform_create_playbook."""
    registry.register(ActionDefinition(
        name="platform_create_playbook",
        description=(
            "Create a new playbook (automated workflow). Starts as a draft with no "
            "steps — add steps with platform_add_playbook_step after creating. "
            "For one-off tasks, use platform_create_task instead."
        ),
        category="playbooks",
        parameters=_CREATE_PLAYBOOK_PARAMETERS,
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "create", "write"],
        examples=[
            "create a playbook for daily standup summaries",
            "make an automation for code review",
        ],
        misplaced={
            "steps": (
                "a playbook is created with no steps: create it, then add each step "
                "with platform_add_playbook_step."
            ),
        },
    ))


def _register_update_playbook(registry: ActionRegistry) -> None:
    """platform_update_playbook."""
    registry.register(ActionDefinition(
        name="platform_update_playbook",
        description=(
            "Update a playbook's metadata — name, description, tags, the inputs each run "
            "needs, execution config, or schedule. Use when the user asks to rename, update, "
            "or reconfigure a playbook. To modify steps, use platform_update_playbook_step instead."
        ),
        category="playbooks",
        parameters=_UPDATE_PLAYBOOK_PARAMETERS,
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "update", "write"],
        examples=[
            "rename playbook 5 to Daily Digest",
            "update the bug triage playbook description",
            "set playbook 3 to run on a cron schedule",
        ],
        misplaced={
            "steps": (
                "steps change one at a time, with platform_update_playbook_step "
                "(step_index plus what changes), platform_add_playbook_step or "
                "platform_delete_playbook_step."
            ),
        },
    ))


def _register_execute_playbook(registry: ActionRegistry) -> None:
    """platform_execute_playbook."""
    registry.register(ActionDefinition(
        name="platform_execute_playbook",
        description=(
            "Trigger a playbook run asynchronously. Returns the run's card number at once "
            "('number', e.g. #0440): give the owner that number, as the board shows it. The "
            "execution_id is only for platform_get_playbook_execution. Pass input_data under the "
            "names platform_get_playbook lists in 'inputs'. "
            "For one-off agent tasks, use platform_create_task instead. "
            "Provide playbook_id or playbook_name."
        ),
        category="playbooks",
        parameters=_EXECUTE_PLAYBOOK_PARAMETERS,
        permission_level="write",
        promoted=True,  # PRD-256 US-006: a first-class tool, pinned
        requires_confirmation=False,
        tags=["playbooks", "execute", "run", "write"],
        examples=[
            "run the daily digest playbook",
            "execute playbook 5",
            "trigger the bug triage automation",
            "run my New Cafe Onboarding for a new café",   # F288 (night 8): a named playbook, no word "playbook"
        ],
        accepts=("inputs", "input"),
        # F182: night 6 nested the café's details under "params".
        misplaced={key: "input_data" for key in ("params", "parameters", "variables", "data")},
    ))


def _register_get_playbook_execution(registry: ActionRegistry) -> None:
    """F321 (night 9b): Auto asked for run #0102 as "0102" and was told it did not
    exist; by its id it got empty step previews and no final output. The run is
    found by its card number too, and ``step`` reads a step's whole output."""
    registry.register(ActionDefinition(
        name="platform_get_playbook_execution",
        description=(
            "Check status and results of a running or completed playbook execution. "
            "Returns the run's final output, each step's status and a short preview. "
            "Pass step to read that step's whole output and what it saved. "
            "Provide execution_id (or the run's card number, e.g. #0102) or playbook_id."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": ("The execution_id returned from platform_execute_playbook, "
                                    "or the run's card number (e.g. #0102)."),
                },
                "step": {
                    "type": "integer",
                    "description": "A step number (1 is the first): its whole output and what it saved.",
                },
                "playbook_id": {
                    "type": "integer",
                    "description": "Playbook ID to list recent executions for (if no execution_id).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["playbooks", "execution", "status", "results"],
        examples=[
            "what's the status of that playbook run?",
            "check playbook execution abc123",
            "did the playbook run successfully?",
            "show me everything step 2 of run #0102 produced",
        ],
    ))
