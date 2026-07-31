"""Report ActionDefinitions (submit report, get latest report)."""

from .action_registry import ActionDefinition, ActionRegistry


def register_report_actions(registry: ActionRegistry) -> None:
    """Register agent report actions (PRD-76)."""

    registry.register(ActionDefinition(
        name="platform_submit_report",
        description=(
            "Save a structured report after completing work — research, heartbeat, "
            "audit, or deliverable. The report is stored persistently in the workspace "
            "and visible on the Activity page and agent profile. ALWAYS submit a report "
            "after finishing significant work. For ephemeral notes, use write_file instead."
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
                    "enum": ["standup", "research", "incident", "summary", "delivery", "audit", "onboarding"],
                    "description": "Category: standup (routine check), research (deep-dive), incident (problem), summary (rollup), delivery (completed work), audit (compliance), onboarding (Mission Zero founding-document summary).",
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
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "rationale": {"type": "string"},
                            "impact": {"type": "string", "description": "Expected outcome if adopted."},
                        },
                    },
                    "description": "Structured recommendations Auto can route or surface as decisions. Prefer these over burying recommendations in markdown — they make the report machine-readable.",
                },
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "owner": {"type": "string", "description": "Agent or person who owns this action."},
                            "due": {"type": "string", "description": "ISO timestamp or natural deadline."},
                            "priority": {"type": "string", "enum": ["urgent", "high", "medium", "low"]},
                        },
                    },
                    "description": "Concrete next steps. Auto can promote these into board tasks automatically.",
                },
                "linked_task_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Board task IDs this report references (origin task, follow-up tasks, etc.).",
                },
                "requires_approval": {
                    "type": "boolean",
                    "description": "Set to true when the report's recommendations need a human decision before any action. Surfaces in the 'Decisions Needed' queue.",
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
        name="platform_browse_reports",
        description=(
            "List agent reports across the workspace with optional filters. "
            "Use this to analyse cost / model / duration patterns across agents, "
            "tasks and playbooks — the system admin agent uses this to recommend "
            "model swaps and cost savings. For a single agent's most-recent report, "
            "use platform_get_latest_report instead."
        ),
        category="reports",
        parameters={
            "type": "object",
            "properties": {
                "agent_id": {
                    "type": "integer",
                    "description": "Optional agent filter.",
                },
                "agent_name": {
                    "type": "string",
                    "description": "Optional agent name filter (substring match).",
                },
                "report_type": {
                    "type": "string",
                    "enum": ["standup", "research", "incident", "summary", "delivery", "audit", "task"],
                    "description": "Optional report type filter.",
                },
                "status": {
                    "type": "string",
                    "enum": ["ok", "warning", "critical", "info"],
                    "description": "Optional status filter.",
                },
                "trigger": {
                    "type": "string",
                    "enum": ["heartbeat", "task", "playbook"],
                    "description": "Optional trigger filter — matches metrics.trigger field.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional model filter — matches metrics.model field (e.g. 'openai/gpt-5').",
                },
                "period": {
                    "type": "string",
                    "enum": ["1d", "7d", "30d", "90d", "all"],
                    "description": "Time window. Default 7d.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max rows to return (default 50, max 200).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["reports", "read", "analytics", "monitoring"],
        examples=[
            "list reports from the last 7 days",
            "show me playbook reports that ran on opus-4.7",
            "list critical reports across the workspace",
            "show task reports for agent FORGE this month",
        ],
    ))

    # Wave 3 — operating-signal lifecycle tools

    registry.register(ActionDefinition(
        name="platform_acknowledge_report",
        description=(
            "Mark a report as acknowledged (read + actioned) by the calling agent "
            "or a named user. Stamps acknowledged_by/acknowledged_at so the "
            "'Decisions Needed' queue can drop the row. Use after Auto has "
            "summarised the report for Gerard or routed its action_items into "
            "board tasks."
        ),
        category="reports",
        parameters={
            "type": "object",
            "properties": {
                "report_id": {
                    "type": "string",
                    "description": "UUID of the report to acknowledge.",
                },
                "user_id": {
                    "type": "integer",
                    "description": "Optional user id to attribute. Defaults to the workspace owner when omitted.",
                },
            },
            "required": ["report_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["reports", "ack", "decisions", "queue"],
        examples=[
            "acknowledge the latest HARNESS audit",
            "mark report xyz as actioned",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_link_report_to_task",
        description=(
            "Add a board-task id to a report's linked_task_ids. Use when Auto "
            "promotes a report's action_items into actual board tasks — keeps "
            "the trail from finding → ask → ticket intact."
        ),
        category="reports",
        parameters={
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "task_id": {"type": "integer"},
            },
            "required": ["report_id", "task_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["reports", "tasks", "linkage"],
        examples=[
            "link this report to task 42",
            "tie the audit findings to the new ticket",
        ],
    ))

    registry.register(ActionDefinition(
        name="platform_get_latest_report",
        description=(
            "Read the most recent report from a specific agent. Use to check another "
            "agent's latest output before taking action (e.g. reading research before "
            "writing a newsletter). For historical reports, this returns only the latest one."
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
