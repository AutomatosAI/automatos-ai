"""Tool/skill/plugin assignment ActionDefinitions."""

from .action_registry import ActionDefinition, ActionRegistry


def register_assignments_actions(registry: ActionRegistry) -> None:
    """Register tool, skill, and plugin assignment actions for agents."""

    registry.register(ActionDefinition(
        name="platform_assign_tool_to_agent",
        description=(
            "Assign a Composio tool/app to an agent. The agent will then be able to "
            "use this tool when processing requests. Accepts agent_id or agent_name, "
            "and app_name (Composio app identifier). Idempotent — re-activates if "
            "previously deactivated."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to assign the tool to.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent (alternative to agent_id).",
                },
                "app_name": {
                    "type": "string",
                    "description": "Composio app name to assign (e.g., 'GMAIL', 'GITHUB', 'SLACK').",
                },
            },
            "required": ["app_name"],
        },
        permission_level="write",
        tags=["agents", "tools", "assign", "composio"],
        examples=[
            "assign GMAIL to my email agent",
            "add GITHUB tool to the dev agent",
            "give agent 5 access to SLACK",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_assign_skill_to_agent",
        description=(
            "Assign a skill to an agent. The skill must be enabled in the workspace "
            "or be a global marketplace skill. Accepts agent_id or agent_name, and "
            "skill_id or skill_name. Idempotent."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to assign the skill to.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent (alternative to agent_id).",
                },
                "skill_id": {
                    "type": "integer",
                    "description": "ID of the skill to assign.",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to assign (alternative to skill_id).",
                },
            },
            "required": [],
        },
        permission_level="write",
        tags=["agents", "skills", "assign"],
        examples=[
            "assign the summarization skill to my research agent",
            "add code review skill to agent 3",
            "give the analysis skill to the data agent",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_assign_plugin_to_agent",
        description=(
            "Assign a marketplace plugin to an agent. The plugin must be enabled in "
            "the workspace first. Accepts agent_id or agent_name, and plugin_id or "
            "plugin_slug. Idempotent."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent to assign the plugin to.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent (alternative to agent_id).",
                },
                "plugin_id": {
                    "type": "string",
                    "description": "UUID of the plugin to assign.",
                },
                "plugin_slug": {
                    "type": "string",
                    "description": "Slug of the plugin to assign (alternative to plugin_id).",
                },
            },
            "required": [],
        },
        permission_level="write",
        tags=["agents", "plugins", "assign"],
        examples=[
            "assign the code review plugin to my dev agent",
            "add plugin testing to agent 5",
            "give the devops plugin to the deployment agent",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_unassign_skill_from_agent",
        description=(
            "Remove a skill from an agent. Idempotent — no-op if the skill "
            "wasn't assigned. Drops the row from agent_skills only; the skill "
            "itself stays in the workspace and can be re-assigned later. Use "
            "this when triaging agents that have stale or wrong skill picks."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "integer", "description": "Agent id (or use agent_name)."},
                "agent_name": {"type": "string", "description": "Agent name (case-insensitive partial match)."},
                "skill_id": {"type": "integer", "description": "Skill id to unassign (or use skill_name)."},
                "skill_name": {"type": "string", "description": "Skill name (case-insensitive partial match within the workspace)."},
            },
            "required": [],
        },
        permission_level="write",
        tags=["agents", "skills", "unassign", "remove"],
        examples=[
            "unassign the code-review skill from VECTOR",
            "remove skill 42 from agent 188",
            "drop the growth-hacker skill from ATLAS",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_unassign_tool_from_agent",
        description=(
            "Remove a tool/app from an agent. Defaults to deactivating the "
            "assignment (is_active=False) so the audit trail stays intact; "
            "pass hard_delete=true to drop the row entirely. Use this to "
            "tidy up agents with broad tool access they don't need."
        ),
        category="agents",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {"type": "integer", "description": "Agent id (or use agent_name)."},
                "agent_name": {"type": "string", "description": "Agent name (case-insensitive partial match)."},
                "app_name": {"type": "string", "description": "App/tool name to unassign (e.g. 'GMAIL', 'GITHUB'). Case-insensitive."},
                "tool_name": {"type": "string", "description": "Alias for app_name."},
                "assignment_id": {"type": "integer", "description": "Specific agent_app_assignment row id (used when app_name is ambiguous)."},
                "hard_delete": {"type": "boolean", "description": "If true, drop the row instead of deactivating. Defaults to false."},
            },
            "required": [],
        },
        permission_level="write",
        tags=["agents", "tools", "unassign", "remove"],
        examples=[
            "unassign GMAIL from agent 188",
            "remove the github tool from VECTOR",
            "deactivate composio_search on the marketing agent",
        ],
    ))
