"""Playbook/workflow ActionDefinitions (list, get, steps, schedule, delete). Create, update
and execute, which take the owner's "wait for me", are in actions_playbook_runs.py (F242)."""

from .action_registry import ActionDefinition, ActionRegistry

# F182 (night 6): what each run needs, declared the way a function signature is.
_INPUTS_PARAM = {
    "type": "object",
    "description": (
        'What each run needs, by name: {"cafe_name": {"required": true, "description": "The café\'s '
        'name"}}. Steps read a value as {{cafe_name}}. A run started without a required one does '
        "not start: it asks the owner for it."
    ),
}


def register_playbooks_actions(registry: ActionRegistry) -> None:
    """Register all playbook-related platform actions."""

    # ── Read ─────────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_playbooks",
        description=(
            "List all playbooks (automated workflows) in the workspace with names, "
            "triggers, status, and step counts. Use when the user asks about their "
            "playbooks, workflows, or automations. For details of ONE playbook, "
            "use platform_get_playbook instead."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "enum": ["active", "inactive", "all"],
                    "description": "Filter playbooks by status. Defaults to 'all'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["playbooks", "workflows", "automations"],
        examples=[
            "what playbooks do I have?",
            "list my workflows",
            "show automations",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_playbook",
        description=(
            "Get full details of ONE playbook — steps, trigger config, execution "
            "history. Use when the user asks about a specific playbook's details. "
            "For listing all playbooks, use platform_list_playbooks instead. "
            "Provide playbook_name or playbook_id."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "playbook_name": {
                    "type": "string",
                    "description": "Name of the playbook to look up.",
                },
                "playbook_id": {
                    "type": "integer",
                    "description": "ID of the playbook (alternative to name).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["playbooks", "details", "steps"],
        examples=[
            "show me the Jira Bug Triage playbook",
            "what does playbook 3 do?",
        ],
    ))

    # ── Write ────────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_add_playbook_step",
        description=(
            "Append a new step to an existing playbook. Each step has a prompt "
            "template and the agent that does it: a step with no agent can't run, so "
            "the step is not added without one. Steps execute sequentially."
        ),
        category="playbooks",
        parameters={
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
                "agent_id": {
                    "type": "integer",
                    "description": ("ID of the agent that does this step (from platform_list_agents). Give it, "
                                    "or agent_name. There is no default agent (F321)."),
                },
                "agent_name": {
                    "type": "string",
                    "description": ("The agent's name or job title as the owner said it (e.g. 'Inventory "
                                    "Watchdog'), when agent_id isn't given. One agent must answer to it; "
                                    "if none or several do, nothing is added and the agents are listed."),
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
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "steps", "add", "write"],
        examples=[
            "add a step to playbook 3 that summarizes the results",
            "add a code review step to the bug triage playbook",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_playbook_step",
        description=(
            "Modify an existing playbook step by its 0-based index. Can change "
            "prompt, agent, order, or error handling. To change part of a step's "
            "prompt, pass find and replace: that one passage changes and the rest "
            "stays as written. prompt_template replaces the WHOLE prompt, and the "
            "reply lists every line it dropped."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "playbook_id": {
                    "type": "integer",
                    "description": "ID of the playbook containing the step.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "0-based index of the step to update.",
                },
                "prompt_template": {
                    "type": "string",
                    "description": "New prompt template for this step: replaces the whole prompt.",
                },
                "find": {
                    "type": "string",
                    "description": "Exact text in the step's prompt to change; it must appear exactly once.",
                },
                "replace": {
                    "type": "string",
                    "description": "What `find` becomes (an empty string removes it).",
                },
                "agent_id": {
                    "type": "integer",
                    "description": "New agent ID for this step.",
                },
                "order": {
                    "type": "integer",
                    "description": "New position in the step list.",
                },
                "error_handling": {
                    "type": "string",
                    "enum": ["stop", "skip", "retry"],
                    "description": "New error handling strategy.",
                },
                "output_key": {
                    "type": "string",
                    "description": "New output key name.",
                },
            },
            "required": ["playbook_id", "step_index"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "steps", "update", "write"],
        examples=[
            "update step 2 of playbook 5 to use agent 3",
            "change the prompt in step 1 of the bug fixer playbook",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_delete_playbook_step",
        description=(
            "Remove a step from a playbook by its 0-based index. Remaining steps "
            "are re-ordered automatically."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "playbook_id": {
                    "type": "integer",
                    "description": "ID of the playbook to remove the step from.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "0-based index of the step to delete.",
                },
            },
            "required": ["playbook_id", "step_index"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "steps", "delete", "write"],
        examples=[
            "delete step 3 from playbook 5",
            "remove the last step from the bug fixer playbook",
        ],
    ))

    # ── Execution ────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_playbook_execution",
        description=(
            "Check status and results of a running or completed playbook execution. "
            "Returns step-by-step results and timing. "
            "Provide execution_id or playbook_id."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The execution_id returned from platform_execute_playbook.",
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
        ],
    ))

    # ── Destructive ──────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_delete_playbook",
        description=(
            "Permanently delete a playbook with full cleanup (triggers, scheduler, "
            "memory). System playbooks cannot be deleted. Only use when explicitly asked. "
            "Provide playbook_id or playbook_name."
        ),
        category="playbooks",
        parameters={
            "type": "object",
            "properties": {
                "playbook_id": {
                    "type": "integer",
                    "description": "ID of the playbook to delete.",
                },
                "playbook_name": {
                    "type": "string",
                    "description": "Name of the playbook to delete (alternative to ID).",
                },
            },
            "required": [],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["playbooks", "delete", "destructive"],
        examples=[
            "delete the test playbook",
            "remove playbook 5",
            "delete automation 3",
        ],
    ))
