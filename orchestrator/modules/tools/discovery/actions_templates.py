"""The document-template tools: list the workspace's templates, read what one needs, look at a page.

F346 (night 10b): the list stopped at "Executive Summary", so Auto never saw the
owner's own templates, and the schema took only an id. Both tools changed, and they
left actions_documents.py, whose one register function is past the length rule.

PRD-255 US-012: platform_render_preview draws a page of a template or a Deliverable
into the calling session's folder, so the agent can open it and judge it.

PRD-255 US-013: platform_create_template and platform_update_template build the
owner's document templates in the studio's block format, through the studio's own
checks (handlers_template_writes); a starter is copied, never changed.
"""

from __future__ import annotations

import functools
from typing import Callable

from .action_registry import ActionDefinition, ActionRegistry

Register = Callable[[ActionRegistry], None]


def register_template_actions(registry: ActionRegistry) -> None:
    """platform_list_templates and platform_get_template_schema (PRD-167 S6, F346), platform_render_preview,
    platform_create_template and platform_update_template (PRD-255 US-013)."""
    _register_list_templates(registry)
    _register_get_template_schema(registry)
    _register_render_preview(registry)
    _register_create_template(registry)
    _register_update_template(registry)


def _register_list_templates(registry: ActionRegistry) -> None:
    # PRD-167 S6: document-template tools. Let agents discover the workspace's
    # templates and the data each one expects, then fill one via generate_document.
    registry.register(ActionDefinition(
        name="platform_list_templates",
        description=(
            "List every document template in this workspace (branded letters, reports, "
            "invoices, the owner's own), one line each: name | format | category | id. "
            "Social image and video templates are listed with format social_image or "
            "social_video. Use before generate_document to pick a template, then call "
            "platform_get_template_schema to learn what data it needs."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Optional filter — pdf, docx, xlsx, social_image or social_video.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter (e.g. 'report', 'invoice', 'letter').",
                },
                "name": {
                    "type": "string",
                    "description": "Optional: only templates whose name contains this text (any case).",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["documents", "templates", "generate"],
        examples=[
            "what document templates do we have?",
            "list invoice templates",
            "show me the branded report templates",
        ],
    ))



def _register_get_template_schema(registry: ActionRegistry) -> None:
    registry.register(ActionDefinition(
        name="platform_get_template_schema",
        description=(
            "Get the data a document template expects: its variable chips "
            "(user/company/brand/date), the data.* fields you must supply, each table's "
            "columns, which fields are required and which fill themselves (fallbacks), "
            "plus sample data. Name the template by id or by name. Use this after "
            "platform_list_templates and before generate_document so you fill it correctly."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "UUID of the template (from platform_list_templates).",
                },
                "template_name": {
                    "type": "string",
                    "description": "The template's name, instead of its id (e.g. 'Branded Invoice').",
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["documents", "templates", "schema", "generate"],
        examples=[
            "what fields does the Branded Letter template need?",
            "show the schema for that template",
        ],
    ))


def _register_render_preview(registry: ActionRegistry) -> None:
    # PRD-255 US-012: read-only apart from the PNG it writes into the calling session's
    # own folder (sessions/<ticket>/); outside a session it refuses and writes nothing.
    registry.register(ActionDefinition(
        name="platform_render_preview",
        description=(
            "Draw one page of a document template or of a Deliverable as a PNG in your session's "
            "folder and get its path, so you can open the picture and judge the page before you "
            "report or revise. A template is drawn with the brand kit and its own sample data; pass "
            "brand_kit (the fields platform_update_brand_kit takes) to draw it from a proposed kit "
            "instead, which is NOT saved. Document templates only (pdf, docx, xlsx). Runs only in a "
            "session working a ticket."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "template_id": {
                    "type": "string",
                    "description": "The template's id (from platform_list_templates). Give this or deliverable_id.",
                },
                "deliverable_id": {
                    "type": "string",
                    "description": "A Deliverable's id, to look at a document already made. Give this or template_id.",
                },
                "page": {
                    "type": "integer",
                    "description": "Which page to draw (default 1).",
                    "minimum": 1,
                },
                "brand_kit": {
                    "type": "object",
                    "description": (
                        "Optional, templates only: a proposed kit change (platform_update_brand_kit's "
                        "fields) laid over the stored kit for this drawing only. Never saved."
                    ),
                },
            },
            "required": [],
        },
        permission_level="read",
        tags=["documents", "templates", "preview", "brand"],
        examples=[
            "show me page 1 of the brand board",
            "render the invoice template with the proposed colours",
            "look at page 2 of the report I just made",
        ],
    ))


# The fields a template is made of, as create and update take them (PRD-255 US-013).
_TEMPLATE_BODY_FIELDS = {
    "name": {"type": "string", "description": "The template's name, unique in the workspace."},
    "description": {"type": "string", "description": "One line on what the template is for."},
    "category": {"type": "string", "description": "e.g. 'invoice', 'quote', 'letter', 'report' (default 'general')."},
    "blocks": {
        "type": "object",
        "description": (
            "The studio's block tree: {\"version\": 1, \"blocks\": [...]} (headings, text with chips, tables, "
            "data tables, images, sections, brand parts). Checked as the studio checks it; field-level errors "
            "come back when it is refused."
        ),
    },
    "sample_data": {"type": "object", "description": "Example data.* values the template shows in previews."},
    "tags": {"type": "array", "items": {"type": "string"}, "description": "Labels to find it by."},
}


def _register_create_template(registry: ActionRegistry) -> None:
    # PRD-255 US-013: the workspace's own content, saved like the studio's POST
    # /templates (documents:create); on the hierarchy gate's ALLOW_LIST.
    registry.register(ActionDefinition(
        name="platform_create_template",
        description=(
            "Make a new document template (pdf, docx or xlsx) in the studio's block format: a name, a "
            "format and the block tree, or copy_of a template's id to start from its layout (a platform "
            "starter is customised this way: copy it, then update the copy). Saved through the studio's "
            "own checks and listed in the studio like any template, tagged with who made it. Social "
            "templates are not made here."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                **_TEMPLATE_BODY_FIELDS,
                "format": {"type": "string", "enum": ["pdf", "docx", "xlsx"], "description": "The file it makes."},
                "copy_of": {
                    "type": "string",
                    "description": "A template's id (from platform_list_templates) to copy; fields you send replace its own.",
                },
            },
            "required": [],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["documents", "templates", "studio", "brand", "create"],
        examples=[
            "make a quote template in our brand",
            "create a price list template",
            "copy the invoice starter so we can customise it",
        ],
    ))


def _register_update_template(registry: ActionRegistry) -> None:
    # PRD-255 US-013: like the studio's PUT /templates/{id} (documents:update), but
    # a platform starter is refused (copy it first); on the gate's ALLOW_LIST.
    registry.register(ActionDefinition(
        name="platform_update_template",
        description=(
            "Change a document template of this workspace: send its template_id and only what changes "
            "(name, description, category, blocks, sample_data, tags). The block tree is checked as the "
            "studio checks it. A platform starter is never changed: copy it with platform_create_template "
            "(copy_of) and update the copy. Social templates are not changed here; nothing is deleted."
        ),
        category="documents",
        parameters={
            "type": "object",
            "properties": {
                "template_id": {"type": "string", "description": "The template's id, from platform_list_templates."},
                **_TEMPLATE_BODY_FIELDS,
            },
            "required": ["template_id"],
        },
        permission_level="write",
        requires_confirmation=False,
        tags=["documents", "templates", "studio", "brand", "edit"],
        examples=[
            "add a terms section to our quote template",
            "change the price list template's table columns",
            "rename the proposal template",
        ],
    ))


def with_template_actions(register: Register) -> Register:
    """``register``, then the template tools, so the documents actions stay one call."""

    @functools.wraps(register)
    def registers_both(registry: ActionRegistry) -> None:
        register(registry)
        register_template_actions(registry)

    return registers_both
