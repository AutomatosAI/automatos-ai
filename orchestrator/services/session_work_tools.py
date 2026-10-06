"""#942 part 2: the session tools for documents, playbooks, reports and missions.

A ``runtime: cli`` agent lost these the moment it left the API runtime (issue #942's
table): an API agent reaches them from ``get_tools_for_agent_async``, a session had
only PRD-245's fixed list. Each tool here runs the API agents' own action through
``UnifiedToolExecutor.execute_tool``, with the ticket's workspace and agent handed to
the executor by ``session_tools.call_tool`` (never taken from the call), and only the
fields its schema declares forwarded.

* ``generate_document`` is the one tool that is not a platform action: the executor
  routes it by NAME (``unified_executor`` → ``exec_document``), exactly as an API
  agent's ``generate_document`` call is routed, so it is dispatched by name like
  ``composio_execute``. It writes a Deliverable in the agent's own workspace (the
  document service resolves the workspace from the agent, which is the ticket's).
* ``list_templates`` and ``get_template_schema`` (brand kit at generation, prep for night
  10): the read-only template tools an API agent has before ``generate_document``
  (``platform_list_templates``, ``platform_get_template_schema``), so a session fills
  the owner's branded template instead of guessing its name and its fields.
* ``render_preview`` (PRD-255 US-012) draws a page of a template or a Deliverable into
  the ticket's folder, so the session opens what it made and judges it.
* ``create_template`` / ``update_template`` (PRD-255 US-013) build the owner's document
  templates in the studio's block format, through the studio's own checks; a starter
  is copied (``copy_of``), never changed. Two more writes.
* ``run_playbook`` starts a run in the session's workspace: the second write here.
* Missions: the read tools a mission's step agent uses. ``platform_list_missions``
  and ``platform_get_mission`` read the board's missions; ``platform_field_query``
  reads what the mission's other agents found, the field resolved from this
  ticket's run by ``call_tool``. The lifecycle actions (create, approve, reject,
  pause, cancel, replan, resume) are Auto's controls over the board, not a ticket
  agent's work, and stay out.

Plain specs, like ``session_data_tools``: ``session_tools`` builds its rows from
``GROUP_TOOL_SPECS`` (this module's and the data tools, in the catalogue's order).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Tuple

from services.session_data_tools import DATA_TOOL_SPECS, project_answer

# How ``generate_document`` reaches the executor (``session_tools.DISPATCH_TOOL_NAME``,
# repeated here because that module imports this one to build its list).
DISPATCH_BY_NAME = "tool_name"
REPORT_TYPES: Tuple[str, ...] = ("standup", "research", "incident", "summary", "delivery", "audit")
MISSION_STATES: Tuple[str, ...] = ("pending", "planning", "running", "paused", "completed", "failed")
PLAYBOOK_STATUSES: Tuple[str, ...] = ("active", "inactive", "all")
DOCUMENT_FORMATS_TEXT = (
    "pdf, docx or xlsx; social_image (a PNG) or social_video (an MP4) render one of the "
    "workspace's social templates and need its template_id or template_name."
)
SCOPED = Callable[[Dict[str, Any], Any], Dict[str, Any]]


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def forward(fields: Mapping[str, str], required: Tuple[str, ...] = (), needs: str = "",
            one_of: Tuple[str, ...] = ()) -> SCOPED:
    """A scope that forwards the session's ``fields`` under the action's names
    (``{session_name: action_name}``) and drops everything else. A missing
    ``required`` field, or none of ``one_of``, is refused with ``needs``, the
    sentence the model reads."""

    def scope(params: Dict[str, Any], ctx: Any) -> Dict[str, Any]:
        out = {fields[key]: value for key, value in params.items() if key in fields and not _blank(value)}
        missing = any(fields[key] not in out for key in required)
        if missing or (one_of and not any(fields[key] in out for key in one_of)):
            from services.session_tools import SessionToolRefused

            raise SessionToolRefused(needs)
        return out

    return scope


def _string(description: str) -> Dict[str, Any]:
    return {"type": "string", "description": description}


def _schema(properties: Dict[str, Any], required: Tuple[str, ...] = ()) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required)}


PLAYBOOK_REF = {
    "playbook_name": _string("The playbook's name, as list_playbooks shows it."),
    "playbook_id": {"type": "integer", "description": "The playbook's id (instead of its name)."},
}
NAME_A_PLAYBOOK = "Name the playbook: playbook_name, or playbook_id from list_playbooks."

GENERATE_DOCUMENT_SPEC: Dict[str, Any] = {
    "name": "generate_document",
    "action": "generate_document",
    "dispatch": DISPATCH_BY_NAME,
    "reads_only": False,
    "description": (
        "Make a finished document from data: a PDF, Word or Excel file, or a social image or video "
        "from one of the workspace's templates. It is saved to Deliverables, attributed to you, and "
        "a copy is put in your folder (the answer names it) so you can open and check it. Use "
        "it when the work's result is a document the owner will open, send or print; your report "
        "still goes through submit_report."
    ),
    "input_schema": _schema({
        "title": _string("The document's title."),
        "format": _string(f"The file type: {DOCUMENT_FORMATS_TEXT}"),
        "template_name": _string("A template to fill, e.g. 'Basic Report' or 'Invoice' (list_templates lists "
                                 "them, get_template_schema says what data each needs). Omit to let it choose."),
        "template_id": _string("A specific template's id (instead of its name)."),
        "data": {"type": "object", "description": "What goes in the document: the fields the template uses."},
    }, ("title", "format", "data")),
    "scope": forward({k: k for k in ("title", "format", "template_name", "template_id", "data")},
                     ("title", "format", "data"),
                     "generate_document needs a title, a format (pdf, docx or xlsx) and the data to fill it."),
    "project": project_answer,
    "tags": ("documents",),
}

TEMPLATE_FORMATS: Tuple[str, ...] = ("pdf", "docx", "xlsx", "social_image", "social_video")

LIST_TEMPLATES_SPEC: Dict[str, Any] = {
    "name": "list_templates",
    "action": "platform_list_templates",
    "description": (
        "List every document template in this workspace (branded letters, reports, invoices, the "
        "owner's own), one line each: name | format | category | id; social images and videos with "
        "their format. Use it before generate_document to pick the owner's own template, then "
        "get_template_schema for the data it needs."
    ),
    "input_schema": _schema({
        "format": {"type": "string", "enum": list(TEMPLATE_FORMATS), "description": "Only templates of this format."},
        "category": _string("Only this category, e.g. 'report', 'invoice' or 'letter'."),
        "name": _string("Only templates whose name contains this text (any case)."),
    }),
    "scope": forward({"format": "format", "category": "category", "name": "name"}),
    "project": project_answer,
    "tags": ("documents",),
}

GET_TEMPLATE_SCHEMA_SPEC: Dict[str, Any] = {
    "name": "get_template_schema",
    "action": "platform_get_template_schema",
    "description": (
        "Read what one document template needs: the data fields you fill in generate_document's "
        "data, each table's columns, which fields are required and which fill themselves (fallbacks, "
        "the brand, the company, the date), and sample data."
    ),
    "input_schema": _schema({
        "template_id": _string("The template's id, from list_templates."),
        "template_name": _string("The template's name, instead of its id."),
    }),
    "scope": forward({"template_id": "template_id", "template_name": "template_name"},
                     needs="get_template_schema needs the template: its template_id or template_name from list_templates.",
                     one_of=("template_id", "template_name")),
    "project": project_answer,
    "tags": ("documents",),
}

RENDER_PREVIEW_SPEC: Dict[str, Any] = {
    "name": "render_preview",
    "action": "platform_render_preview",
    "description": (
        "Draw one page of a document template, or of a Deliverable, as a PNG in your folder and get "
        "its path: open the picture to judge the page before you report or revise. A template is "
        "drawn with the brand kit and its sample data; brand_kit draws it from a proposed kit "
        "instead, which is NOT saved. Document templates only (pdf, docx, xlsx)."
    ),
    "input_schema": _schema({
        "template_id": _string("The template's id, from list_templates. Give this or deliverable_id."),
        "deliverable_id": _string("A Deliverable's id (a document already made). Give this or template_id."),
        "page": {"type": "integer", "minimum": 1, "description": "Which page to draw (default 1)."},
        "brand_kit": {"type": "object", "description": "Templates only: a proposed brand kit change (kit fields "
                      "such as palette, type_scale or accent_use), drawn for this picture only and never saved."},
    }),
    "scope": forward({k: k for k in ("template_id", "deliverable_id", "page", "brand_kit")},
                     needs="render_preview needs what to draw: a template_id from list_templates, or a deliverable_id.",
                     one_of=("template_id", "deliverable_id")),
    "project": project_answer,
    "tags": ("documents",),
}

TEMPLATE_BODY_FIELDS: Tuple[str, ...] = ("name", "description", "category", "blocks", "sample_data", "tags")
TEMPLATE_BODY_SCHEMA: Dict[str, Any] = {
    "name": _string("The template's name, unique in the workspace."),
    "description": _string("One line on what the template is for."),
    "category": _string("e.g. 'invoice', 'quote', 'letter', 'report' (default 'general')."),
    "blocks": {"type": "object", "description": "The studio's block tree, {\"version\": 1, \"blocks\": [...]}, "
               "checked as the studio checks it; field-level errors come back when it is refused."},
    "sample_data": {"type": "object", "description": "Example data.* values the template shows in previews."},
    "tags": {"type": "array", "items": {"type": "string"}, "description": "Labels to find it by."},
}

CREATE_TEMPLATE_SPEC: Dict[str, Any] = {
    "name": "create_template",
    "action": "platform_create_template",
    "reads_only": False,
    "description": (
        "Make a new document template (pdf, docx or xlsx) in the studio's block format: a name, a format and "
        "the block tree, or copy_of a template's id to start from its layout (customise a platform starter "
        "this way: copy it, then update_template the copy). Saved through the studio's checks, listed in the "
        "studio and tagged as made by you. Look at it with render_preview before you report. No social templates."
    ),
    "input_schema": _schema({
        **TEMPLATE_BODY_SCHEMA,
        "format": {"type": "string", "enum": ["pdf", "docx", "xlsx"], "description": "The file it makes."},
        "copy_of": _string("A template's id (from list_templates) to copy; the fields you send replace its own."),
    }),
    "scope": forward({k: k for k in (*TEMPLATE_BODY_FIELDS, "format", "copy_of")},
                     needs="create_template needs a name, a format and blocks, or copy_of a template's id.",
                     one_of=("name", "copy_of")),
    "project": project_answer,
    "tags": ("documents",),
}

UPDATE_TEMPLATE_SPEC: Dict[str, Any] = {
    "name": "update_template",
    "action": "platform_update_template",
    "reads_only": False,
    "description": (
        "Change one of this workspace's document templates: its template_id and only what changes. The block "
        "tree is checked as the studio checks it. A platform starter is never changed: copy it with "
        "create_template(copy_of=...) and update the copy. Nothing is deleted."
    ),
    "input_schema": _schema({"template_id": _string("The template's id, from list_templates."),
                             **TEMPLATE_BODY_SCHEMA}, ("template_id",)),
    "scope": forward({k: k for k in ("template_id", *TEMPLATE_BODY_FIELDS)}, ("template_id",),
                     "update_template needs the template_id, from list_templates, and what changes."),
    "project": project_answer,
    "tags": ("documents",),
}

LIST_PLAYBOOKS_SPEC: Dict[str, Any] = {
    "name": "list_playbooks",
    "action": "platform_list_playbooks",
    "description": (
        "List this workspace's playbooks (the owner's saved, repeatable workflows): name, trigger, "
        "status and steps. Use it before get_playbook or run_playbook to find the right one."
    ),
    "input_schema": _schema({"status": {"type": "string", "enum": list(PLAYBOOK_STATUSES),
                                        "description": "Only active or inactive playbooks (default all)."}}),
    "scope": forward({"status": "status_filter"}),
    "project": project_answer,
    "tags": ("playbooks",),
}

GET_PLAYBOOK_SPEC: Dict[str, Any] = {
    "name": "get_playbook",
    "action": "platform_get_playbook",
    "description": "Read one playbook in full: its steps, its trigger and its recent runs.",
    "input_schema": _schema(dict(PLAYBOOK_REF)),
    "scope": forward({"playbook_name": "playbook_name", "playbook_id": "playbook_id"},
                     needs=NAME_A_PLAYBOOK, one_of=("playbook_name", "playbook_id")),
    "project": project_answer,
    "tags": ("playbooks",),
}

RUN_PLAYBOOK_SPEC: Dict[str, Any] = {
    "name": "run_playbook",
    "action": "platform_execute_playbook",
    "reads_only": False,
    "description": (
        "Start a run of one of this workspace's playbooks. It runs on its own and appears on the "
        "board; this returns as soon as it has started. Only run one your ticket asks for."
    ),
    "input_schema": _schema({
        **PLAYBOOK_REF,
        "inputs": {"type": "object",
                   "description": "What the playbook's steps read, as key: value. One text goes in as {\"input\": \"…\"}."},
        "wait_for_me": {"type": "boolean",
                        "description": "true to hold the run's card in Review for the owner's check."},
    }),
    "scope": forward({"playbook_name": "playbook_name", "playbook_id": "playbook_id",
                      "inputs": "input_data", "wait_for_me": "wait_for_me"},
                     needs=NAME_A_PLAYBOOK, one_of=("playbook_name", "playbook_id")),
    "project": project_answer,
    "tags": ("playbooks",),
}

GET_LATEST_REPORT_SPEC: Dict[str, Any] = {
    "name": "get_latest_report",
    "action": "platform_get_latest_report",
    "description": (
        "Read another agent's most recent report in this workspace, e.g. the research a writer "
        "builds on. Name the agent; optionally the kind of report."
    ),
    "input_schema": _schema({
        "agent_name": _string("The agent whose report to read."),
        "agent_id": {"type": "integer", "description": "The agent's id (instead of its name)."},
        "report_type": {"type": "string", "enum": list(REPORT_TYPES), "description": "Only this kind of report."},
    }),
    "scope": forward({"agent_name": "agent_name", "agent_id": "agent_id", "report_type": "report_type"},
                     needs="get_latest_report needs the agent: agent_name, or agent_id.",
                     one_of=("agent_name", "agent_id")),
    "project": project_answer,
    "tags": ("reports",),
}

LIST_MISSIONS_SPEC: Dict[str, Any] = {
    "name": "list_missions",
    "action": "platform_list_missions",
    "description": "List this workspace's recent missions: goal, state and how many steps each has.",
    "input_schema": _schema({
        "state": {"type": "string", "enum": list(MISSION_STATES), "description": "Only missions in this state."},
        "limit": {"type": "integer", "description": "How many (default 10)."},
    }),
    "scope": forward({"state": "state", "limit": "limit"}),
    "project": project_answer,
    "tags": ("missions",),
}

GET_MISSION_SPEC: Dict[str, Any] = {
    "name": "get_mission",
    "action": "platform_get_mission",
    "description": "Read one mission in full: its goal, its steps, each step's result so far, and where it stands.",
    "input_schema": _schema({"mission_id": _string("The mission's id, or its card's number (#0188).")},
                            ("mission_id",)),
    "scope": forward({"mission_id": "mission_id"}, ("mission_id",),
                     "get_mission needs the mission: its id, or its card's number."),
    "project": project_answer,
    "tags": ("missions",),
}

SEARCH_MISSION_FINDINGS_SPEC: Dict[str, Any] = {
    "name": "search_mission_findings",
    "action": "platform_field_query",
    "description": (
        "Search what the other agents on your ticket's mission have found so far, best matches "
        "first. For a ticket outside a mission it searches what earlier missions in this workspace "
        "found. Check it before redoing work another step may have done."
    ),
    "input_schema": _schema({
        "query": _string("What you are looking for, in plain words."),
        "limit": {"type": "integer", "description": "How many findings (default 10)."},
    }, ("query",)),
    "scope": forward({"query": "query", "limit": "top_k"}, ("query",),
                     "search_mission_findings needs a query: what you are looking for."),
    "project": project_answer,
    "tags": ("missions",),
}

WORK_TOOL_SPECS: Tuple[Dict[str, Any], ...] = (
    GENERATE_DOCUMENT_SPEC, LIST_TEMPLATES_SPEC, GET_TEMPLATE_SCHEMA_SPEC, RENDER_PREVIEW_SPEC,
    CREATE_TEMPLATE_SPEC, UPDATE_TEMPLATE_SPEC,
    LIST_PLAYBOOKS_SPEC, GET_PLAYBOOK_SPEC, RUN_PLAYBOOK_SPEC,
    GET_LATEST_REPORT_SPEC,
    LIST_MISSIONS_SPEC, GET_MISSION_SPEC, SEARCH_MISSION_FINDINGS_SPEC,
)
# Every grouped tool, in the catalogue's order (data, graph, documents, playbooks,
# reports, missions): ``session_tools`` appends these after its core ten.
GROUP_TOOL_SPECS: Tuple[Dict[str, Any], ...] = DATA_TOOL_SPECS + WORK_TOOL_SPECS
