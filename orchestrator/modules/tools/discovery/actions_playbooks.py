"""Playbook/workflow ActionDefinitions (list, get, create, update, steps, execute, delete)."""

from .action_registry import ActionDefinition, ActionRegistry


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
            "For listing all playbooks, use platform_list_playbooks instead."
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
        name="platform_create_playbook",
        description=(
            "Create a new playbook (automated workflow). Starts as a draft with no "
            "steps — add steps with platform_add_playbook_step after creating. "
            "For one-off tasks, use platform_create_task instead."
        ),
        category="playbooks",
        parameters={
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
            },
            "required": ["name", "description"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "create", "write"],
        examples=[
            "create a playbook for daily standup summaries",
            "make an automation for code review",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_playbook",
        description=(
            "Update a playbook's metadata — name, description, tags, execution config, "
            "or schedule. Use when the user asks to rename, update, or reconfigure a "
            "playbook. To modify steps, use platform_update_playbook_step instead."
        ),
        category="playbooks",
        parameters={
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
                    "description": "Schedule config: { type: 'manual'|'cron'|'trigger', cron_expression, trigger_config }.",
                },
            },
            "required": ["playbook_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "update", "write"],
        examples=[
            "rename playbook 5 to Daily Digest",
            "update the bug triage playbook description",
            "set playbook 3 to run on a cron schedule",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_add_playbook_step",
        description=(
            "Append a new step to an existing playbook. Each step has a prompt "
            "template and optional agent assignment. Steps execute sequentially."
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
                    "description": "ID of the agent to execute this step (optional — uses default agent if not set).",
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
            "prompt, agent, order, or error handling."
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
                    "description": "New prompt template for this step.",
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
        name="platform_execute_playbook",
        description=(
            "Trigger a playbook run asynchronously. Returns an execution_id "
            "immediately — check status later with platform_get_playbook_execution. "
            "For one-off agent tasks, use platform_create_task instead."
        ),
        category="playbooks",
        parameters={
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
                    "description": "Input data to pass to the playbook (key-value pairs).",
                },
            },
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["playbooks", "execute", "run", "write"],
        examples=[
            "run the daily digest playbook",
            "execute playbook 5",
            "trigger the bug triage automation",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_playbook_execution",
        description=(
            "Check status and results of a running or completed playbook execution. "
            "Returns step-by-step results and timing."
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
            "memory). System playbooks cannot be deleted. Only use when explicitly asked."
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
