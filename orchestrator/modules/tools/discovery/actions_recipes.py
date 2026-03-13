"""Recipe/workflow ActionDefinitions (list, get, create, update, steps, execute, delete)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_recipes_actions(registry: ActionRegistry) -> None:
    """Register all recipe-related platform actions."""

    # ── Read ─────────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_list_recipes",
        description=(
            "List all workflow recipes in the current workspace. Returns recipe names, "
            "trigger types, status, and step counts. Use when the user asks about their "
            "recipes, workflows, or automations."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": "string",
                    "enum": ["active", "inactive", "all"],
                    "description": "Filter recipes by status. Defaults to 'all'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["recipes", "workflows", "automations"],
        examples=[
            "what recipes do I have?",
            "list my workflows",
            "show automations",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_recipe",
        description=(
            "Get detailed information about a specific recipe including its steps, "
            "trigger configuration, and execution history. Use when the user asks "
            "about a specific recipe's details."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_name": {
                    "type": "string",
                    "description": "Name of the recipe to look up.",
                },
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe (alternative to name).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["recipes", "details", "steps"],
        examples=[
            "show me the Jira Bug Triage recipe",
            "what does recipe 3 do?",
        ],
    ))

    # ── Write ────────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_create_recipe",
        description=(
            "Create a new workflow recipe in the workspace. Requires a name and "
            "description. The recipe starts as a draft with no steps — steps can "
            "be added later. Use when the user asks to create a new recipe or automation."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name for the new recipe.",
                },
                "description": {
                    "type": "string",
                    "description": "What the recipe does.",
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
        tags=["recipes", "create", "write"],
        examples=[
            "create a recipe for daily standup summaries",
            "make an automation for code review",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_recipe",
        description=(
            "Update an existing recipe's metadata. Can change name, description, tags, "
            "execution_config, or schedule_config. Use when the user asks to rename, "
            "update, or reconfigure a recipe."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe to update.",
                },
                "name": {
                    "type": "string",
                    "description": "New name for the recipe.",
                },
                "description": {
                    "type": "string",
                    "description": "New description.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Replace the recipe's tags with this list.",
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
            "required": ["recipe_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["recipes", "update", "write"],
        examples=[
            "rename recipe 5 to Daily Digest",
            "update the bug triage recipe description",
            "set recipe 3 to run on a cron schedule",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_add_recipe_step",
        description=(
            "Add a new step to an existing recipe. The step is appended to the end "
            "by default, or inserted at a specific order position. Use when the user "
            "asks to add a step to a recipe or workflow."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe to add the step to.",
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
            "required": ["recipe_id", "prompt_template"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["recipes", "steps", "add", "write"],
        examples=[
            "add a step to recipe 3 that summarizes the results",
            "add a code review step to the bug triage recipe",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_update_recipe_step",
        description=(
            "Update an existing step in a recipe. Specify the step by its 0-based index. "
            "Can change the prompt, agent, order, error handling, or output key. "
            "Use when the user asks to modify or edit a recipe step."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe containing the step.",
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
            "required": ["recipe_id", "step_index"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["recipes", "steps", "update", "write"],
        examples=[
            "update step 2 of recipe 5 to use agent 3",
            "change the prompt in step 1 of the bug fixer recipe",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_delete_recipe_step",
        description=(
            "Delete a step from a recipe by its 0-based index. Remaining steps are "
            "re-ordered automatically. Use when the user asks to remove a step."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe to remove the step from.",
                },
                "step_index": {
                    "type": "integer",
                    "description": "0-based index of the step to delete.",
                },
            },
            "required": ["recipe_id", "step_index"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["recipes", "steps", "delete", "write"],
        examples=[
            "delete step 3 from recipe 5",
            "remove the last step from the bug fixer recipe",
        ],
    ))

    # ── Execution ────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_execute_recipe",
        description=(
            "Trigger a recipe run asynchronously. Returns an execution_id immediately "
            "that can be used to check status later. Use when the user asks to run, "
            "execute, or trigger a recipe or automation."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe to execute.",
                },
                "recipe_name": {
                    "type": "string",
                    "description": "Name of the recipe to execute (alternative to ID).",
                },
                "input_data": {
                    "type": "object",
                    "description": "Input data to pass to the recipe (key-value pairs).",
                },
            },
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["recipes", "execute", "run", "write"],
        examples=[
            "run the daily digest recipe",
            "execute recipe 5",
            "trigger the bug triage automation",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_recipe_execution",
        description=(
            "Check the status and results of a recipe execution. Returns execution "
            "status, step results summary, and timing. Use when the user asks about "
            "a recipe run's status or results."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "execution_id": {
                    "type": "string",
                    "description": "The execution_id returned from platform_execute_recipe.",
                },
                "recipe_id": {
                    "type": "integer",
                    "description": "Recipe ID to list recent executions for (if no execution_id).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["recipes", "execution", "status", "results"],
        examples=[
            "what's the status of that recipe run?",
            "check recipe execution abc123",
            "did the recipe run successfully?",
        ],
    ))

    # ── Destructive ──────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_delete_recipe",
        description=(
            "Delete a recipe with full cleanup — trigger subscriptions, scheduler, "
            "and memory. System recipes cannot be deleted. This is permanent. "
            "Use only when the user explicitly asks to delete a recipe."
        ),
        category="recipes",
        parameters={
            "type": "object",
            "properties": {
                "recipe_id": {
                    "type": "integer",
                    "description": "ID of the recipe to delete.",
                },
                "recipe_name": {
                    "type": "string",
                    "description": "Name of the recipe to delete (alternative to ID).",
                },
            },
            "required": [],
        },
        permission_level="destructive",
        requires_confirmation=True,
        tags=["recipes", "delete", "destructive"],
        examples=[
            "delete the test recipe",
            "remove recipe 5",
            "delete automation 3",
        ],
    ))
