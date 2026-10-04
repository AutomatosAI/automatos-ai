"""Analytics/usage ActionDefinitions (LLM usage, costs, workspace stats, activity feed, NL2SQL)."""

from .action_registry import ActionDefinition, ActionRegistry
from .actions_data_routes import register_query_data_action, registers_with


@registers_with(after=[register_query_data_action])  # F301/F302: the data route moved there
def register_analytics_actions(registry: ActionRegistry) -> None:
    """Register analytics and usage platform actions."""

    registry.register(ActionDefinition(
        name="platform_get_llm_usage",
        description=(
            "Get LLM token usage statistics over a time period — total requests, "
            "tokens consumed, model breakdown. For cost estimates, use "
            "platform_get_cost_breakdown instead."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Defaults to 30.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "usage", "tokens", "llm"],
        examples=[
            "what's my token usage?",
            "how many API calls this month?",
            "show LLM usage stats",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_cost_breakdown",
        description=(
            "Get cost breakdown by model, agent, or day. For raw token counts, "
            "use platform_get_llm_usage instead. For a quick composite score, "
            "use platform_get_efficiency_score."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to look back. Defaults to 30.",
                },
                "group_by": {
                    "type": "string",
                    "enum": ["model", "agent", "day"],
                    "description": "How to group the cost breakdown. Defaults to 'model'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "costs", "spending", "budget"],
        examples=[
            "what are my costs?",
            "how much am I spending on LLM?",
            "cost breakdown by agent",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_workspace_stats",
        description=(
            "Get a dashboard overview — LLM usage, top models, top agents, "
            "routing distribution, resource counts. Use for a quick summary. "
            "For deep-dive into specific metrics, use the specialized analytics tools."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["today", "7d", "30d"],
                    "description": "Time period for stats. Defaults to '7d'.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["stats", "analytics", "usage", "dashboard"],
        examples=[
            "show workspace stats",
            "platform usage summary",
            "what's been happening this week?",
            "show me agent activity",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_activity_feed",
        description=(
            "Get a unified activity feed — recent chats, recipe runs, and routines. "
            "Shows what's been happening in the workspace. Use when the user asks "
            "about recent activity, what's been running, or wants an activity log."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["1d", "7d", "30d", "90d"],
                    "description": "Time period to look back. Defaults to '7d'.",
                },
                "type": {
                    "type": "string",
                    "description": "Comma-separated activity types: 'chat', 'recipe', 'routine'. Defaults to all.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of items to return. Defaults to 20, max 50.",
                },
            },
            "required": [],
        },
        permission_level="read",
        promoted=True,
        tags=["activity", "feed", "analytics", "history"],
        examples=[
            "what's been happening?",
            "show recent activity",
            "activity feed for the last week",
            "what has been running?",
        ],
    ))
