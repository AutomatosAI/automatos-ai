"""Report ActionDefinitions (submit report, get latest report)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_report_actions(registry: ActionRegistry) -> None:
    """Register agent report actions (PRD-76)."""

    registry.register(ActionDefinition(
        name="platform_submit_report",
        description=(
            "Submit a report after completing a task or heartbeat cycle. Writes the "
            "report file to workspace storage and records metadata for tracking. "
            "Call this after every heartbeat run, research completion, or deliverable. "
            "The report will be visible on the Activity page and the agent's profile."
        ),
        category="reports",
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title for the report (e.g. 'Platform Health Check', 'Weekly Newsletter Draft').",
                },
                "content": {
                    "type": "string",
                    "description": "Full report content in markdown format.",
                },
                "report_type": {
                    "type": "string",
                    "enum": ["standup", "research", "incident", "summary", "delivery", "audit"],
                    "description": "Category: standup (routine check), research (deep-dive), incident (problem), summary (rollup), delivery (completed work), audit (compliance).",
                },
                "status": {
                    "type": "string",
                    "enum": ["ok", "warning", "critical", "info"],
                    "description": "Overall status. ok=nothing to worry about, warning/critical=needs attention, info=informational.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-line summary shown in activity feed cards (auto-generated from content if omitted).",
                },
                "metrics": {
                    "type": "object",
                    "description": "Structured metrics relevant to this report (e.g. { errors_found: 2, services_checked: 5, cost: 0.003 }).",
                },
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "file_path": {"type": "string", "description": "Relative workspace path to the attachment file."},
                            "file_type": {"type": "string"},
                        },
                    },
                    "description": "Additional files produced alongside this report (images, data files, etc.).",
                },
                "required_sections": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of section headings the report must contain (e.g. ['Summary', 'Metrics', 'Next Steps']). Submission fails if any are missing from the markdown content.",
                },
            },
            "required": ["title", "content", "report_type", "status"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["reports", "write", "heartbeat", "standup"],
        examples=[
            "submit a health check report",
            "file a status report after my heartbeat run",
            "create an incident report about the API errors",
            "submit research findings",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_latest_report",
        description=(
            "Get the most recent report from a specific agent. Returns the report "
            "metadata and full content. Use to read another agent's latest output "
            "before taking action (e.g. reading research before sending a newsletter)."
        ),
        category="reports",
        parameters={
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "description": "Name of the agent whose report to fetch.",
                },
                "agent_id": {
                    "type": "integer",
                    "description": "ID of the agent (alternative to agent_name).",
                },
                "report_type": {
                    "type": "string",
                    "enum": ["standup", "research", "incident", "summary", "delivery", "audit"],
                    "description": "Filter by report type (optional).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["reports", "read", "cross-agent"],
        examples=[
            "get the latest report from sentinel",
            "what did the researcher find?",
            "read the market researcher's latest research report",
            "get the most recent standup from the monitoring agent",
        ],
    ))
