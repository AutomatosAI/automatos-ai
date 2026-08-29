"""Monitoring/observability ActionDefinitions (Loki, Prometheus, alerts, Railway logs/services)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_monitoring_actions(registry: ActionRegistry) -> None:
    """Register monitoring, observability, and infrastructure actions."""

    # ── Loki Logs ────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_query_loki_logs",
        description=(
            "Search centralized application logs across ALL services (7-day retention). "
            "Much more powerful than Railway deploy logs. Use for investigating errors, "
            "tracing requests, and debugging production issues. "
            "Filter by service, severity, keyword, or time range. "
            "For log content from a single Railway deploy, use platform_get_logs instead."
        ),
        category="monitoring",
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Service to query logs from: 'automatos-backend', 'agent-opt-worker', "
                        "'workspace-worker', 'mem0-server'. "
                        "Defaults to 'automatos-backend' if omitted."
                    ),
                },
                "level": {
                    "type": "string",
                    "description": "Filter by log level: 'info', 'warning', 'error'.",
                },
                "search": {
                    "type": "string",
                    "description": (
                        "Free-text search within log messages. "
                        "Examples: 'timeout', 'agent_id=147', 'memory', 'heartbeat'."
                    ),
                },
                "minutes": {
                    "type": "integer",
                    "description": "How far back to search in minutes (default 60, max 10080 = 7 days).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max log entries to return (default 100, max 500).",
                },
            },
            "required": [],
        },
        permission_level="read",
        super_admin_only=True,
        tags=["logs", "monitoring", "loki", "observability", "debugging"],
        examples=[
            "show me error logs from the last hour",
            "search logs for 'heartbeat' in the backend",
            "get warning and error logs from agent-opt-worker",
            "find logs mentioning agent 147",
            "check for timeout errors in the last 30 minutes",
        ],
    ))

    # ── Prometheus ───────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_query_prometheus",
        description=(
            "Query real-time system metrics via PromQL. "
            "Use for checking uptime, error rates, response times, database connections, "
            "Redis memory. Supports raw PromQL or preset names ('health', 'error_rate', "
            "'latency', 'postgres', 'redis', 'all'). "
            "For log content, use platform_query_loki_logs instead."
        ),
        category="monitoring",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "PromQL query string OR a preset name. "
                        "Presets: 'health' (all services up/down), 'error_rate' (HTTP 5xx rate), "
                        "'latency' (p95 response time), 'postgres' (DB connections + cache hit), "
                        "'redis' (memory + clients + latency), 'all' (full health dashboard). "
                        "Or provide raw PromQL like 'rate(automatos_http_requests_total[5m])'. "
                        "Optional — defaults to the 'health' preset when omitted."
                    ),
                },
                "range_minutes": {
                    "type": "integer",
                    "description": "Time range for range queries in minutes (default 15).",
                },
            },
            # query defaults to the 'health' preset in the handler
            # (handlers_monitoring.query_prometheus) — omitting it is valid, so it
            # is not required. See the tool-schema walker guard.
            "required": [],
        },
        permission_level="read",
        super_admin_only=True,
        tags=["metrics", "monitoring", "prometheus", "health", "observability"],
        examples=[
            "check if all services are healthy",
            "what's the current error rate?",
            "show me Redis memory usage",
            "how many Postgres connections are active?",
            "what's the p95 response time?",
        ],
    ))

    # ── Alerts ───────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_alerts",
        description=(
            "Get infrastructure alerts — firing, resolved, and recent. "
            "Use to understand current system health issues. "
            "For deeper investigation, follow up with platform_query_loki_logs "
            "or platform_query_prometheus. Filter by status, severity, or time range."
        ),
        category="monitoring",
        parameters={
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Filter by alert status: 'firing', 'resolved', 'all' (default 'all').",
                },
                "severity": {
                    "type": "string",
                    "description": "Filter by severity: 'critical', 'warning', 'info'.",
                },
                "hours": {
                    "type": "integer",
                    "description": "Look back this many hours (default 24, max 168 = 7 days).",
                },
            },
            "required": [],
        },
        permission_level="read",
        super_admin_only=True,
        tags=["alerts", "monitoring", "infrastructure", "health"],
        examples=[
            "are there any firing alerts?",
            "show me critical alerts from the last 24 hours",
            "what alerts fired this week?",
            "any infrastructure issues right now?",
        ],
    ))

    # ── Railway Logs & Services ──────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_get_logs",
        description=(
            "Fetch Railway deployment logs for a specific service. "
            "For searching across ALL services, use platform_query_loki_logs instead "
            "(more powerful, 7-day retention). "
            "Use platform_list_services first to discover available service names. "
            "Supports keyword filtering (e.g. 'error', 'timeout', 'Exception')."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Railway service name to fetch logs from "
                        "(e.g. 'automatos-api', 'workspace-worker'). "
                        "Use 'list' to see all available services."
                    ),
                },
                "lines": {
                    "type": "integer",
                    "description": "Number of log lines to retrieve (default 200, max 1000).",
                },
                "filter": {
                    "type": "string",
                    "description": (
                        "Filter logs by keyword or severity. "
                        "Examples: 'error', 'Exception', 'timeout', 'WARNING'."
                    ),
                },
            },
            "required": ["service"],
        },
        permission_level="read",
        super_admin_only=True,
        tags=["logs", "infrastructure", "railway", "observability", "debugging"],
        examples=[
            "get error logs from the API",
            "fetch recent logs from automatos-api",
            "show me the last 100 warning logs from workspace-worker",
            "list available services",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_list_services",
        description=(
            "List all Railway services in the project with names and IDs. "
            "Use before platform_get_logs to discover available service names."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        super_admin_only=True,
        tags=["services", "infrastructure", "railway"],
        examples=[
            "what services are running?",
            "list railway services",
        ],
    ))
