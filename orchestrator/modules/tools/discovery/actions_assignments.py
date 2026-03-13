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
