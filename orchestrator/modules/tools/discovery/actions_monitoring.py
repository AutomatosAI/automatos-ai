"""Monitoring/observability ActionDefinitions (Loki, Prometheus, alerts, Railway logs/services)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_monitoring_actions(registry: ActionRegistry) -> None:
    """Register monitoring, observability, and infrastructure actions."""

    # ── Loki Logs ────────────────────────────────────────────────────

    registry.register(ActionDefinition(
        name="platform_query_loki_logs",
        description=(
            "Search application logs stored in Loki (the centralized log system). "
            "Query by service, severity level, keyword, or time range. Returns structured "
            "log entries with timestamps. Much more powerful than Railway deploy logs — "
            "this searches ALL log history (7-day retention) across all services."
        ),
        category="monitoring",
        parameters={
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": (
                        "Service to query logs from: 'automatos-backend', 'agent-opt-worker', "
                        "'log-relay', 'prometheus', 'grafana', 'loki', 'alertmanager'."
                    ),
                },
                "level": {
                    "type": "string",
                    "description": "Filter by log level: debug, info, warning, error, critical.",
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
            "Query Prometheus metrics for real-time system health. Supports PromQL queries "
            "or preset health checks. Use to check service uptime, error rates, response "
            "times, database connections, Redis memory, and more."
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
                        "Or provide raw PromQL like 'rate(automatos_http_requests_total[5m])'."
                    ),
                },
                "range_minutes": {
                    "type": "integer",
                    "description": "Time range for range queries in minutes (default 15).",
                },
            },
            "required": ["query"],
        },
        permission_level="read",
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
            "Get infrastructure alerts from the monitoring system. Shows firing, "
            "resolved, and recent alerts with severity, service, and details. "
            "Use to understand current system health issues."
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
            "Fetch deployment logs from a Railway service. Returns recent log lines "
            "with timestamps and severity levels. Use to investigate errors, capture "
            "server-side context for bug reports, or monitor service health. "
            "Supports filtering by keyword (e.g. 'error', 'timeout', 'Exception')."
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
            "List all Railway services in the project. Returns service names and IDs. "
            "Use to discover available services before fetching logs."
        ),
        category="infrastructure",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        permission_level="read",
        tags=["services", "infrastructure", "railway"],
        examples=[
            "what services are running?",
            "list railway services",
        ],
    ))
