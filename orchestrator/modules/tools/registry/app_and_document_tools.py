"""Two core tools ToolRegistry registers right after its core set: composio_execute
(external apps through Composio) and generate_document (PRD-63)."""
from __future__ import annotations

from typing import List

from .tool_registry import (
    GENERATE_DOCUMENT_DESCRIPTION,
    GENERATE_DOCUMENT_FORMAT_DESCRIPTION,
    SecurityLevel,
    ToolCategory,
    ToolParameter,
    ToolSpec,
)

def _composio_execute_parameters() -> List[ToolParameter]:
    return [
        ToolParameter(
            name="app_name",
            type="string",
            description="App name (e.g., 'GMAIL', 'SLACK', 'GITHUB')",
            required=False,
        ),
        ToolParameter(
            name="action",
            type="string",
            description="Action name (e.g., 'GMAIL_LIST_EMAILS', 'SLACK_SEND_MESSAGE')",
            required=True,
        ),
        ToolParameter(
            name="params",
            type="object",
            description=(
                "Action-specific parameters as a JSON object: every field the "
                "action takes (e.g. issue_key, channel, text) goes here."
            ),
            required=False,
            default={},
        ),
    ]


_GENERATE_DOCUMENT_EXAMPLES = [
    {
        "action": "generate_document",
        "params": {
            "title": "Weekly Status Report",
            "format": "pdf",
            "data": {"sections": [{"title": "Summary", "content": "All tasks completed on time. The team delivered 5 features and resolved 12 bugs."}]},
        },
    },
    {
        "action": "generate_document",
        "params": {
            "title": "User Export",
            "format": "xlsx",
            "data": {"rows": [{"name": "Alice", "email": "alice@example.com"}]},
        },
    },
]


def app_and_document_specs() -> List[ToolSpec]:
    """The two specs, in registration order."""
    return [_composio_execute_spec(), _generate_document_spec()]


def _composio_execute_spec() -> ToolSpec:
    return ToolSpec(
        name="composio_execute",
        category=ToolCategory.API_TOOLS,
        description=(
            "Execute any action across 1000+ external app integrations via Composio — "
            "web search, email, messaging, GitHub, CRM, calendars, databases, and more. "
            "If your tool list includes per-action tools (with typed parameters), prefer those "
            "for accuracy. Use composio_execute for any action not already in your tool list. "
            "Check platform_list_connected_apps to see what's available. "
            "The action's own parameters go inside the `params` object, not at the top level."
        ),
        executor_class="ComposioToolExecutor",
        executor_method="execute",
        parameters=_composio_execute_parameters(),
        security_level=SecurityLevel.CAUTIOUS,
        permissions_required={"read": True, "execute": True},
        examples=[
            {
                "action": "composio_execute",
                "params": {
                    "action": "SLACK_SEND_MESSAGE",
                    "params": {"channel": "#general", "text": "Hello"}
                },
            },
        ],
        metadata={"integration_type": "composio"},
    )


def _generate_document_spec() -> ToolSpec:
    return ToolSpec(
        name="generate_document",
        category=ToolCategory.FILE_OPERATIONS,
        description=(
            f"{GENERATE_DOCUMENT_DESCRIPTION} "
            "Use it when the user needs a formatted, downloadable document. The document "
            "holds only what 'data' carries, so write the full content there; for PDFs, "
            "'sections' with full paragraphs. "
            "Returns a download URL."
        ),
        executor_class="AgentPlatformTools",
        executor_method="execute_tool",
        parameters=_generate_document_parameters(),
        returns="JSON with filename, format, download_url, and size_kb",
        security_level=SecurityLevel.CAUTIOUS,
        permissions_required={"read": True, "write": True},
        examples=_GENERATE_DOCUMENT_EXAMPLES,
        metadata={"added_in": "PRD-63"}
    )


def _generate_document_parameters() -> List[ToolParameter]:
    # PRD-251 US-117: the social formats render a social template through
    # media-render; every format generate() dispatches is offered.
    from core.models.core import DOCUMENT_TEMPLATE_FORMATS
    from services.deliverable_tag_schemas import tags_parameter

    return [
        ToolParameter(
            name="title",
            type="string",
            description="Document title (e.g., 'Monthly Sales Report', 'Invoice #1234')",
            required=True
        ),
        ToolParameter(
            name="format",
            type="string",
            description=GENERATE_DOCUMENT_FORMAT_DESCRIPTION,
            required=True,
            enum=list(DOCUMENT_TEMPLATE_FORMATS)
        ),
        ToolParameter(
            name="data",
            type="object",
            description=(
                "The document's content; a section left empty renders empty. "
                "For PDF/DOCX reports: {\"sections\": [{\"title\": \"Section Name\", \"content\": \"Write full "
                "paragraphs of text here — this is the body of the document.\"}], "
                "\"author\": \"...\", \"date\": \"...\"}. "
                "For tables/xlsx: {\"columns\": [\"col1\", \"col2\"], \"rows\": [[\"val1\", \"val2\"]]}. "
                "For a social template (social_image / social_video): its variables by name, "
                "as platform_get_template_schema lists them; one left out takes its default."
            ),
            required=True
        ),
        ToolParameter(
            name="template_name",
            type="string",
            description="Template to use (e.g., 'Basic Report', 'Invoice'). Omit for auto-selection.",
            required=False
        ),
        # P2-09 S2 (F031/J2): the handler already parses/validates template_id,
        # but only the chatbot's inline schema declared it — so id-driven
        # template generation worked in chat and nowhere else. Declaring it
        # here gives the non-chat autonomy lane (missions/board/scheduled)
        # parity, and makes platform_get_template_schema's "use before
        # generate_document" discovery flow followable. Wording mirrors the
        # inline chat schema (agent_platform_tools.get_available_tools).
        ToolParameter(
            name="template_id",
            type="string",
            description="UUID of a specific template to fill (from platform_list_templates). Takes precedence over template_name.",
            required=False
        ),
        # 7 Oct: the Deliverable's tags; the chat schema says the same (services/deliverable_tag_schemas.py).
        tags_parameter(),
    ]
