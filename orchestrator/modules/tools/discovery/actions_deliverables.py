"""Deliverable ActionDefinitions (PRD-164 S3 — list/get agent tools).

Gives Auto first-class access to the workspace deliverables it (and other
agents) produce: reports, generated documents, images, code, slides. Backed
by the existing ``DeliverableService`` over ``v_workspace_outputs`` — no new
query path.
"""

from .action_registry import ActionDefinition, ActionRegistry


def register_deliverables_actions(registry: ActionRegistry) -> None:
    """Register deliverable discovery actions (PRD-164 S3)."""
    registry.register(_list_deliverables_action())
    registry.register(_get_deliverable_action())


def _origin_filters() -> dict:
    """Where a deliverable came from: its origin, the run that made it, and origins to leave out."""
    return {
        "source_type": {
            "type": "string",
            "description": (
                "Filter by origin: chat, task, mission, heartbeat, "
                "playbook, trigger (optional)."
            ),
        },
        "source_id": {
            "type": "string",
            "description": (
                "Filter by the originating mission/task/heartbeat id "
                "(e.g. a mission id to list that mission's deliverables)."
            ),
        },
        "exclude_source_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Leave out deliverables of these origins (optional): "
                "social_post leaves out a Socials post's own images and videos."
            ),
        },
    }


def _list_parameters() -> dict:
    """platform_list_deliverables' filters."""
    return {
        "type": "object",
        "properties": {
            "artifact_type": {
                "type": "string",
                "enum": [
                    "report", "image", "document", "code", "slide",
                    "spreadsheet", "blog_post", "archive", "audio", "video",
                ],
                "description": "Filter by artifact type (optional).",
            },
            **_origin_filters(),
            "agent_id": {
                "type": "integer",
                "description": "Filter by producing agent id (optional).",
            },
            "mine": {
                "type": "boolean",
                "description": "true = only deliverables produced by you (the calling agent).",
            },
            "search": {
                "type": "string",
                "description": "Search in title/summary/file path (optional).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20, max 50).",
            },
        },
        "required": [],
    }


def _list_deliverables_action() -> ActionDefinition:
    return ActionDefinition(
        name="platform_list_deliverables",
        description=(
            "List deliverables (agent outputs) in the workspace: reports, "
            "generated documents, images, code, slides, spreadsheets. "
            "Filter by type, source, the producing agent, or a mission/task "
            "source_id. Pass mine=true to list your own outputs. Use this to "
            "answer 'what have you produced?', to find a mission's outputs, "
            "or to locate an earlier report/document."
        ),
        category="deliverables",
        parameters=_list_parameters(),
        permission_level="read",
        tags=["deliverables", "read", "outputs", "documents", "reports"],
        examples=[
            "what deliverables do we have",
            "list your deliverables",
            "show the outputs from that mission",
            "find the report you wrote yesterday",
        ],
    )


def _get_deliverable_action() -> ActionDefinition:
    return ActionDefinition(
        name="platform_get_deliverable",
        description=(
            "Fetch one deliverable by id — metadata plus (optionally) its "
            "text content. Use after platform_list_deliverables to read a "
            "report, generated document, or code file an agent produced."
        ),
        category="deliverables",
        parameters={
            "type": "object",
            "properties": {
                "deliverable_id": {
                    "type": "string",
                    "description": "ID of the deliverable (from platform_list_deliverables).",
                },
                "include_content": {
                    "type": "boolean",
                    "description": "true = also return the file's text content (default false).",
                },
            },
            "required": ["deliverable_id"],
        },
        permission_level="read",
        tags=["deliverables", "read", "outputs", "content"],
        examples=[
            "open that deliverable",
            "read the generated report",
            "show me the contents of deliverable X",
        ],
    )
