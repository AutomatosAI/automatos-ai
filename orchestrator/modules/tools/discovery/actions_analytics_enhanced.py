"""Enhanced analytics ActionDefinitions — exposes dashboard & performance metrics to agents."""

from .action_registry import ActionDefinition, ActionRegistry


def register_analytics_enhanced_actions(registry: ActionRegistry) -> None:
    """Register enhanced analytics platform actions (dashboard + performance)."""

    # ── Dashboard Metrics ──────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_success_rate",
        description=(
            "Get the overall agent success rate as a percentage, with 7-day trend. "
            "Use when checking how well agents are performing overall, or when "
            "monitoring operational health."
        ),
        category="analytics",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["analytics", "success", "performance", "health"],
        examples=[
            "what's our success rate?",
            "how are agents performing?",
            "check agent health",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_completion_time",
        description=(
            "Get average task completion time in minutes, with 24-hour comparison. "
            "Use when checking how fast tasks are being completed or looking for "
            "performance regressions."
        ),
        category="analytics",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["analytics", "performance", "speed", "completion"],
        examples=[
            "how long do tasks take?",
            "average completion time",
            "are tasks getting faster?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_error_rates",
        description=(
            "Get error rate breakdown by agent type. Shows total executions, "
            "failures, and error rate percentage per agent type. Use when "
            "investigating failures or identifying problematic agent categories."
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
        tags=["analytics", "errors", "failures", "debugging"],
        examples=[
            "which agents are failing?",
            "error rates by type",
            "show failure breakdown",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_queue_depth",
        description=(
            "Get real-time queue depth — how many tasks are pending or running. "
            "Shows high-priority vs normal breakdown. Use when checking system "
            "load or investigating task delays."
        ),
        category="analytics",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["analytics", "queue", "load", "pending"],
        examples=[
            "how many tasks are queued?",
            "what's the current backlog?",
            "check queue depth",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_efficiency_score",
        description=(
            "Get a composite efficiency score (0-100) with grade (A-D). "
            "Combines CPU, memory, agent utilization, and workflow completion "
            "into a single health metric. Use for quick platform health checks."
        ),
        category="analytics",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["analytics", "efficiency", "health", "score"],
        examples=[
            "what's our efficiency score?",
            "overall platform health",
            "system efficiency grade",
        ],
    ))

    # ── Performance Analytics ──────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_cost_per_execution",
        description=(
            "Get average cost per execution with 30-day daily breakdown. "
            "Shows cost trend (increasing/decreasing/stable) and per-day data. "
            "Use when analyzing cost efficiency or planning budgets."
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
        tags=["analytics", "costs", "efficiency", "budget"],
        examples=[
            "what's our cost per execution?",
            "are costs going up or down?",
            "cost efficiency trend",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_peak_hours",
        description=(
            "Get peak usage hours — 24-hour pattern showing when the platform "
            "is most active. Returns hourly API call counts, active agents, "
            "and peak/medium/low categories. Use for capacity planning."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to analyze. Defaults to 30.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "peak", "hours", "capacity"],
        examples=[
            "when is the platform busiest?",
            "peak usage hours",
            "when should I schedule heavy tasks?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_bottlenecks",
        description=(
            "Detect resource bottlenecks — CPU, memory, and database connection "
            "issues with severity, recommendations, and impact assessment. "
            "Use when investigating slowdowns or system health issues."
        ),
        category="analytics",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        tags=["analytics", "bottlenecks", "health", "performance"],
        examples=[
            "any bottlenecks?",
            "what's slowing things down?",
            "check for resource issues",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_predictive_alerts",
        description=(
            "Get predictive capacity alerts — storage, agent capacity, and API "
            "rate limit forecasts. Shows predicted issues before they happen. "
            "Use for proactive monitoring and capacity planning."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "threshold": {
                    "type": "number",
                    "description": "Alert confidence threshold (0-100). Defaults to 60.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "alerts", "predictive", "capacity"],
        examples=[
            "any upcoming capacity issues?",
            "predictive alerts",
            "will we run out of resources?",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_agent_ranking",
        description=(
            "Get agent performance ranking — composite score based on success rate, "
            "speed, and task volume. Returns ranked list with per-agent metrics. "
            "Use when evaluating agent performance or finding top/bottom performers."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "metric": {
                    "type": "string",
                    "enum": ["performance_score", "success_rate", "avg_response_time", "tasks_completed"],
                    "description": "Metric to rank by. Defaults to 'performance_score'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max agents to return. Defaults to 20.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "agents", "ranking", "performance"],
        examples=[
            "rank agents by performance",
            "who are the top performers?",
            "worst performing agents",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_sla_compliance",
        description=(
            "Get SLA compliance metrics — task completion rate vs 95% target, "
            "response time vs 120s target, overall compliance score and status. "
            "Use when checking if the platform meets service level agreements."
        ),
        category="analytics",
        parameters={
            "type": "object",
            "properties": {
                "days": {
                    "type": "integer",
                    "description": "Number of days to analyze. Defaults to 30.",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["analytics", "sla", "compliance", "quality"],
        examples=[
            "are we meeting SLAs?",
            "SLA compliance report",
            "check service level targets",
        ],
    ))
