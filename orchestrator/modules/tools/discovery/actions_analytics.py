"""Analytics/usage ActionDefinitions (LLM usage, costs, workspace stats, activity feed, NL2SQL)."""

from .action_registry import ActionDefinition, ActionRegistry


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

    # ── NL2SQL / Query Data ──────────────────────────────────────────
    # F301 (night 9): Auto asked "how many members cancelled April to September, and the
    # most common reason?" in one call and three times reported the top reason's 4 as the
    # total (11). The description now says how to ask (one figure per call, the owner's
    # qualifiers) and how to report (the figure the rows show, and what they count).

    registry.register(ActionDefinition(
        name="platform_query_data",
        description=(
            "Answer a question about the business's own records (orders, subscriptions, "
            "members, stock, sales, customers) from a connected database. It reads the "
            "schema itself, turns the question into SQL, runs it and returns the rows, "
            "the SQL and what they count: never ask the user for table, column or field "
            "names. Ask in the user's words, with every qualifier they gave (active, "
            "cancelled, a plan, a date range), and ask one figure per call: a total and "
            "a breakdown (how many cancelled, and the most common reason) are two calls. "
            "Report the figure the rows show and say what it counts; a group's count is "
            "never the total. With one database connected, pass only the question: that "
            "one is used. Name a database (database_id) only when several are connected."
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
                    "type": "string",
                    "description": (
                        "The database source to query: its name (e.g. 'sales_db') "
                        "or its numeric id. Omit it when the workspace has one "
                        "database — that one is used. With several, name one; "
                        "the error lists them."
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
