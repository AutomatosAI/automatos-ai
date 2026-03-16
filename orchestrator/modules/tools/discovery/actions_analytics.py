"""Analytics/usage ActionDefinitions (LLM usage, costs, workspace stats, activity feed, NL2SQL)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_analytics_actions(registry: ActionRegistry) -> None:
    """Register analytics and usage platform actions."""

    registry.register(ActionDefinition(
        name="platform_get_llm_usage",
        description=(
            "Get LLM usage statistics for the workspace over a time period. "
            "Returns total requests, tokens consumed, and model breakdown. "
            "Use when the user asks about token usage, API calls, or LLM activity."
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
            "Get cost breakdown by model and agent for the workspace. "
            "Shows estimated costs based on token usage and model pricing. "
            "Use when the user asks about costs, spending, or budget."
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
            "Get workspace usage statistics — LLM usage, top models, top agents, "
            "routing distribution, and resource counts. Use when the user asks for "
            "a dashboard view, usage summary, or platform health check."
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
        tags=["activity", "feed", "analytics", "history"],
        examples=[
            "what's been happening?",
            "show recent activity",
            "activity feed for the last week",
            "what has been running?",
        ],
    ))

    # ── NL2SQL / Query Data ──────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_query_data",
        description=(
            "Query business data using natural language. Converts your question "
            "into SQL and executes it against a connected database. Use this when "
            "the user asks about metrics, counts, trends, revenue, users, or any "
            "data that lives in their connected databases. Returns results as a "
            "formatted table with row count and the generated SQL."
        ),
        category="database",
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "Natural language question about business data "
                        "(e.g. 'How many active users this month?', "
                        "'Top 10 customers by revenue')."
                    ),
                },
                "database_id": {
                    "type": "integer",
                    "description": (
                        "ID of the database source to query. If omitted, uses "
                        "the workspace's default (first active) database."
                    ),
                },
            },
            "required": ["question"],
        },
        permission_level="read",
        requires_confirmation=False,
        tags=["database", "query", "analytics", "metrics", "nl2sql", "data"],
        examples=[
            "how many active users do we have",
            "what's our current MRR",
            "show revenue trend for last 6 months",
            "top 5 products by sales",
            "how many users signed up last week",
            "query the database for average order value",
        ],
    ))
