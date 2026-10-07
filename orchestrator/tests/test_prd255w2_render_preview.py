"""PRD-255 US-012: an agent renders a page into its session's folder and opens it.

``render_preview`` (``platform_render_preview``) draws one page of a document
template or of a Deliverable as a PNG, writes it into ``sessions/<ticket>/`` of the
calling session, and answers with the path. It runs only for a session's ticket
(server-side, strip-then-inject), only on the caller's workspace, never on a social
template, and a proposed kit is drawn without being saved.

Boundaries faked: the workspace worker, the template and Deliverable reads, and the
child-process render. The page renderer itself is exercised for real (WeasyPrint and
PDFium), as the F353 tests do.
"""
from __future__ import annotations

import asyncio
import copy
import io
import os
from types import SimpleNamespace as NS
from typing import Any, Dict, List, Optional
from uuid import UUID

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from PIL import Image  # noqa: E402

from modules.documents import template_page_render  # noqa: E402
from modules.documents.presets import LETTER  # noqa: E402
from modules.documents.thumbnails import job as thumbnail_job  # noqa: E402
from modules.documents.thumbnails import render as page_render  # noqa: E402
from modules.documents.thumbnails import sources as thumbnail_sources  # noqa: E402
from modules.documents.thumbnails.render import ThumbnailError, pdf_first_page_png  # noqa: E402
from modules.tools.discovery import handlers_render_preview as rp  # noqa: E402
from modules.tools.discovery.session_ticket import (  # noqa: E402
    SESSION_TICKET_PARAM,
    carries_the_session_ticket,
    session_ticket_params,
)
from modules.tools.execution import session_document_folder  # noqa: E402

TICKET = 2101
WS = UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e62")
OTHER_WS = UUID("9e8d7c6b-5a49-4382-a1b0-c9d8e7f6a5b4")
TEMPLATE_ID = UUID("1f2e3d4c-5b6a-4798-8a7b-6c5d4e3f2a1b")
OTHER_TEMPLATE_ID = UUID("2a3b4c5d-6e7f-4a8b-9c0d-1e2f3a4b5c6d")
SOCIAL_TEMPLATE_ID = UUID("3b4c5d6e-7f8a-4b9c-8d1e-2f3a4b5c6d7e")
DELIVERABLE_ID = UUID("4c5d6e7f-8a9b-4c0d-9e2f-3a4b5c6d7e8f")
OTHER_DELIVERABLE_ID = UUID("5d6e7f8a-9b0c-4d1e-8f3a-4b5c6d7e8f9a")
PNG = b"\x89PNG\r\n\x1a\n the drawn page"
STORED_KIT = {"brand_kit": {"name": "Hollow Leaf", "primary_color": "#1F3A2E"}}


class _Worker:
    """WorkspaceClient, faked: records what is written where."""

    written: Dict[Any, bytes] = {}

    def __init__(self, workspace_id: str) -> None:
        self.workspace_id = workspace_id

    async def write_binary(self, path: str, pieces: Any) -> Dict[str, Any]:
        self.written[(self.workspace_id, path)] = b"".join([piece async for piece in pieces])
        return {"success": True, "path": path}


class _Query:
    def __init__(self, row: Any) -> None:
        self.row = row

    def filter(self, *args: Any, **kwargs: Any) -> "_Query":
        return self

    def first(self) -> Any:
        return self.row


class _Db:
    """The session, faked: the caller's workspace row, and a count of commits (there must be none)."""

    def __init__(self) -> None:
        self.workspace = NS(id=WS, settings=copy.deepcopy(STORED_KIT))
        self.commits = 0

    def query(self, model: Any) -> _Query:
        return _Query(self.workspace if model.__name__ == "Workspace" else None)

    def commit(self) -> None:
        self.commits += 1


def _template(template_id: UUID, workspace_id: UUID, fmt: str = "pdf", blocks: Any = None) -> NS:
    return NS(id=template_id, workspace_id=workspace_id, name=f"Template {template_id.hex[:4]}", format=fmt,
              blocks=LETTER["blocks"] if blocks is None else blocks, sample_data=LETTER["sample_data"])


TEMPLATES = [
    _template(TEMPLATE_ID, WS),
    _template(OTHER_TEMPLATE_ID, OTHER_WS),
    _template(SOCIAL_TEMPLATE_ID, WS, fmt="social_image", blocks={"html": "<div></div>"}),
]


class _Templates:
    """DocumentTemplateService, faked: a template is found only in its own workspace."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def get_template(self, template_id: UUID, workspace_id: UUID) -> Optional[NS]:
        return next((t for t in TEMPLATES if t.id == template_id and t.workspace_id == workspace_id), None)


def _output(output_id: UUID, workspace_id: UUID, file_path: str, artifact_type: str = "document") -> Any:
    return thumbnail_job.Output(str(output_id), str(workspace_id), artifact_type, "agent_output", "generated",
                                file_path, 2048)


OUTPUTS = [
    _output(DELIVERABLE_ID, WS, "generated/report.pdf"),
    _output(OTHER_DELIVERABLE_ID, OTHER_WS, "generated/their-report.pdf"),
]


@pytest.fixture
def drawn(monkeypatch):
    """The worker, the reads and the child-process renders faked; what each render was given."""
    from modules.documents import template_service

    calls: Dict[str, List[Any]] = {"template": [], "deliverable": []}
    _Worker.written = {}
    monkeypatch.setattr(session_document_folder, "WorkspaceClient", _Worker)
    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)

    def render_template(template, kit, page_number, timeout_s=None):
        calls["template"].append({"template": template, "kit": kit, "page": page_number})
        return PNG

    def render_file(data, ext, timeout_s=None, page_number=1, whole_width_px=None):
        calls["deliverable"].append({"data": data, "ext": ext, "page": page_number, "width": whole_width_px})
        return PNG

    def load_output(db, workspace_id, output_id):
        return next((o for o in OUTPUTS if o.id == str(output_id) and o.workspace_id == str(workspace_id)), None)

    monkeypatch.setattr(template_page_render, "render_template_page_isolated", render_template)
    monkeypatch.setattr(page_render, "render_png_isolated", render_file)
    monkeypatch.setattr(thumbnail_job, "load_output", load_output)
    monkeypatch.setattr(thumbnail_sources, "read_source", lambda ws, storage, path: b"%PDF-1.7 " + path.encode())
    return calls


def _run(params: Dict[str, Any], db: Optional[_Db] = None, workspace_id: UUID = WS) -> Dict[str, Any]:
    return asyncio.run(rp.render_preview(db or _Db(), workspace_id, params))


def _in_session(**params: Any) -> Dict[str, Any]:
    """The params as the executor hands them over for a session's call (the ticket injected server-side)."""
    return session_ticket_params("platform_render_preview", params, {"session_task_id": TICKET})


# ── the page lands in the ticket's folder ──────────────────────────────────

def test_a_templates_page_is_written_into_the_tickets_folder_and_its_path_returned(drawn):
    answer = _run(_in_session(template_id=str(TEMPLATE_ID)))

    assert answer["success"] is True, answer
    path = answer["path"]
    assert path == f"sessions/{TICKET}/preview-template-{TEMPLATE_ID.hex}-p1.png"
    assert _Worker.written == {(str(WS), path): PNG}
    assert path in answer["note"] and answer["page"] == 1 and answer["template_id"] == str(TEMPLATE_ID)
    (call,) = drawn["template"]
    assert call["page"] == 1 and call["template"]["blocks"] == LETTER["blocks"]
    assert call["kit"]["primary_color"] == "#1F3A2E"             # the stored kit


def test_a_deliverables_page_is_drawn_from_its_own_file_at_the_page_asked(drawn):
    answer = _run(_in_session(deliverable_id=str(DELIVERABLE_ID), page=2))

    assert answer["success"] is True, answer
    assert answer["path"] == f"sessions/{TICKET}/preview-deliverable-{DELIVERABLE_ID.hex}-p2.png"
    (call,) = drawn["deliverable"]
    assert call == {"data": b"%PDF-1.7 generated/report.pdf", "ext": ".pdf", "page": 2,
                    "width": template_page_render.PREVIEW_WIDTH_PX}          # the whole page, readable


def test_the_same_page_drawn_again_replaces_its_picture(drawn):
    first = _run(_in_session(template_id=str(TEMPLATE_ID)))
    again = _run(_in_session(template_id=str(TEMPLATE_ID)))

    assert first["path"] == again["path"] and len(_Worker.written) == 1


def test_a_social_deliverables_image_is_copied_as_it_is(drawn):
    image_id = UUID("6e7f8a9b-0c1d-4e2f-9a4b-5c6d7e8f9a0b")
    OUTPUTS.append(_output(image_id, WS, "generated/post.png", artifact_type="image"))
    try:
        answer = _run(_in_session(deliverable_id=str(image_id)))
    finally:
        OUTPUTS.pop()

    assert answer["success"] is True, answer
    assert _Worker.written[(str(WS), answer["path"])] == b"%PDF-1.7 generated/post.png"   # the stored bytes
    assert drawn["deliverable"] == []                                                       # never re-rendered


# ── the caller's workspace only ────────────────────────────────────────────

def test_another_workspaces_template_is_not_found_and_nothing_is_drawn(drawn):
    answer = _run(_in_session(template_id=str(OTHER_TEMPLATE_ID)))

    assert answer == {"success": False, "error": f"No template {OTHER_TEMPLATE_ID} in this workspace."}
    assert drawn["template"] == [] and _Worker.written == {}


def test_another_workspaces_deliverable_is_not_found_and_nothing_is_drawn(drawn):
    answer = _run(_in_session(deliverable_id=str(OTHER_DELIVERABLE_ID)))

    assert answer == {"success": False, "error": f"No Deliverable {OTHER_DELIVERABLE_ID} in this workspace."}
    assert drawn["deliverable"] == [] and _Worker.written == {}


def test_the_page_is_written_into_the_callers_workspace_never_the_ids(drawn):
    answer = _run(_in_session(template_id=str(OTHER_TEMPLATE_ID)), workspace_id=OTHER_WS)

    assert answer["success"] is True, answer
    assert list(_Worker.written) == [(str(OTHER_WS), answer["path"])]


# ── a session only ─────────────────────────────────────────────────────────

def test_outside_a_session_it_refuses_and_writes_nothing(drawn):
    answer = _run({"template_id": str(TEMPLATE_ID)})

    assert answer == {"success": False, "error": rp.NEEDS_A_SESSION}
    assert drawn["template"] == [] and _Worker.written == {}


@pytest.mark.parametrize("context", [None, {}, {"board_task_id": 77}, {"conversation_id": "c-1"}])
def test_a_caller_supplied_ticket_is_stripped_so_a_chat_or_a_board_run_is_refused(drawn, context):
    params = session_ticket_params("platform_render_preview",
                                   {"template_id": str(TEMPLATE_ID), SESSION_TICKET_PARAM: TICKET}, context)

    assert SESSION_TICKET_PARAM not in params
    assert _run(params) == {"success": False, "error": rp.NEEDS_A_SESSION}
    assert _Worker.written == {}


def test_only_the_tools_that_write_there_are_given_the_ticket():
    params = session_ticket_params("platform_list_templates", {SESSION_TICKET_PARAM: 5}, {"session_task_id": TICKET})

    assert params == {}


def test_a_json_string_cannot_smuggle_a_ticket_past_the_strip():
    params = session_ticket_params("platform_render_preview", '{"_session_task_id": 9}', None)

    assert params == {}


def test_the_executor_carries_the_ticket_and_the_session_hands_it_over():
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS, PlatformActionExecutor
    from services import session_tools as st

    assert PLATFORM_HANDLERS["platform_render_preview"] is rp.render_preview
    layers, fn = [], PlatformActionExecutor.execute  # every decorator layer on execute (F381 adds one above)
    while fn is not None:
        layers, fn = [*layers, fn.__code__.co_filename], getattr(fn, "__wrapped__", None)
    assert any(name.endswith("session_ticket.py") for name in layers)
    tool = st.get_tool("render_preview")
    ctx = st.SessionContext(task_id=TICKET, agent_id=7, agent_name="Brand Designer", workspace_id=WS)
    assert tool.action == "platform_render_preview" and tool.reads_only
    assert st._caller_context(tool, ctx) == {"session_task_id": TICKET}


def test_the_real_executor_hands_the_handler_only_the_sessions_own_ticket(monkeypatch):
    from modules.tools.discovery import platform_executor as pe

    seen: List[Any] = []

    async def run_cleared(self, action_name, params, caller_context, cleared, handler):
        seen.append(params)
        return {"success": True}

    monkeypatch.setattr(pe.PlatformActionExecutor, "clear", lambda self, a, p, c: pe.Cleared(None, False, None, False))
    monkeypatch.setattr(pe.PlatformActionExecutor, "_run_cleared", run_cleared)
    executor = pe.PlatformActionExecutor(None, WS)
    spoof = {"template_id": "t", SESSION_TICKET_PARAM: 9}
    for action, context in [("platform_render_preview", None),
                            ("platform_render_preview", {"conversation_id": "c-1"}),
                            ("platform_render_preview", {"session_task_id": TICKET}),
                            ("platform_list_templates", {"session_task_id": TICKET})]:
        asyncio.run(executor.execute(action, spoof, caller_context=context))

    assert seen == [{"template_id": "t"}, {"template_id": "t"},
                    {"template_id": "t", SESSION_TICKET_PARAM: TICKET}, {"template_id": "t"}]


def test_the_decorator_hands_the_execute_the_injected_ticket():
    seen: List[Any] = []

    @carries_the_session_ticket
    async def execute(self, action_name, params, caller_context=None):
        seen.append(params)
        return {"success": True}

    asyncio.run(execute(None, "platform_render_preview", {"page": 1}, {"session_task_id": str(TICKET)}))

    assert seen == [{"page": 1, SESSION_TICKET_PARAM: TICKET}]


def test_the_session_tool_forwards_only_its_fields_and_needs_something_to_draw():
    from services import session_tools as st

    tool = st.get_tool("render_preview")
    ctx = st.SessionContext(task_id=TICKET, agent_id=7, agent_name="Brand Designer", workspace_id=WS)
    forwarded = st.resolve_parameters(tool, {"template_id": "t", "page": 2, SESSION_TICKET_PARAM: 1}, ctx)
    assert forwarded == {"template_id": "t", "page": 2}
    with pytest.raises(st.SessionToolRefused):
        st.resolve_parameters(tool, {"page": 2}, ctx)


# ── the proposal is drawn, never saved ─────────────────────────────────────

def test_a_proposed_kit_is_drawn_and_never_saved(drawn):
    db = _Db()
    answer = _run(_in_session(template_id=str(TEMPLATE_ID), brand_kit={"primary_color": "#7A2E3B"}), db=db)

    assert answer["success"] is True and answer["from_proposal"] is True, answer
    assert answer["path"].endswith("-p1-proposal.png") and "NOT saved" in answer["note"]
    assert drawn["template"][0]["kit"]["primary_color"] == "#7A2E3B"
    assert db.workspace.settings == STORED_KIT and db.commits == 0


def test_a_proposal_the_kit_would_refuse_draws_nothing(drawn):
    db = _Db()
    answer = _run(_in_session(template_id=str(TEMPLATE_ID), brand_kit={"primary_color": "green-ish"}), db=db)

    assert answer["success"] is False and "not valid" in answer["error"]
    assert drawn["template"] == [] and _Worker.written == {} and db.commits == 0


def test_a_proposal_cannot_point_the_kit_at_a_stored_file(drawn):
    answer = _run(_in_session(template_id=str(TEMPLATE_ID), brand_kit={"logo_path": "brand/other/logo.png"}))

    assert answer["success"] is False and "logo_path" in answer["error"]
    assert drawn["template"] == []


# ── what it refuses ────────────────────────────────────────────────────────

def test_a_social_template_is_refused(drawn):
    answer = _run(_in_session(template_id=str(SOCIAL_TEMPLATE_ID)))

    assert answer["success"] is False and "document templates (pdf, docx, xlsx) only" in answer["error"]
    assert drawn["template"] == [] and _Worker.written == {}


@pytest.mark.parametrize("params, error", [
    ({}, rp.NAME_ONE),
    ({"template_id": str(TEMPLATE_ID), "deliverable_id": str(DELIVERABLE_ID)}, rp.NAME_ONE),
    ({"template_id": str(TEMPLATE_ID), "page": 0}, rp.BAD_PAGE),
    ({"template_id": str(TEMPLATE_ID), "page": rp.MAX_PREVIEW_PAGE + 1}, rp.BAD_PAGE),
    ({"template_id": str(TEMPLATE_ID), "page": 1.5}, rp.BAD_PAGE),
    ({"deliverable_id": str(DELIVERABLE_ID), "brand_kit": {"primary_color": "#000000"}}, rp.PROPOSAL_ON_DELIVERABLE),
    ({"template_id": str(TEMPLATE_ID), "brand_kit": "navy"}, rp.PROPOSAL_NOT_AN_OBJECT),
])
def test_a_request_that_names_nothing_drawable_is_refused(drawn, params, error):
    assert _run(_in_session(**params)) == {"success": False, "error": error}
    assert _Worker.written == {}


def test_a_render_that_fails_says_why_and_writes_nothing(drawn, monkeypatch):
    def fails(*args, **kwargs):
        raise ThumbnailError("the render took longer than 60s")

    monkeypatch.setattr(template_page_render, "render_template_page_isolated", fails)
    answer = _run(_in_session(template_id=str(TEMPLATE_ID)))

    assert answer == {"success": False, "error": "The page could not be drawn: the render took longer than 60s"}
    assert _Worker.written == {}


# ── the renderer draws any page (F353 extended) ────────────────────────────

def _two_page_pdf() -> bytes:
    from weasyprint import HTML

    page = "<p style='font-size:40pt'>one</p><p style='page-break-before:always;font-size:40pt'>two</p>"
    return HTML(string=page).write_pdf()


def test_the_page_renderer_draws_page_two_and_refuses_a_page_that_is_not_there():
    pdf = _two_page_pdf()

    first, second = pdf_first_page_png(pdf), pdf_first_page_png(pdf, page_number=2)
    assert first != second and Image.open(io.BytesIO(second)).format == "PNG"
    with pytest.raises(ThumbnailError, match="there is no page 3: the document has 2 pages"):
        pdf_first_page_png(pdf, page_number=3)


def test_the_isolated_render_takes_a_page():
    pdf = _two_page_pdf()

    assert page_render.render_png_isolated(pdf, ".pdf", page_number=2) == pdf_first_page_png(pdf, page_number=2)
    with pytest.raises(ThumbnailError, match="no page 3"):
        page_render.render_png_isolated(pdf, ".pdf", page_number=3)


def test_a_deliverables_page_is_drawn_whole_and_readable_in_a_child_process():
    pdf = _two_page_pdf()

    png = page_render.render_png_isolated(pdf, ".pdf", page_number=2,
                                          whole_width_px=template_page_render.PREVIEW_WIDTH_PX)
    image = Image.open(io.BytesIO(png))
    assert image.width == template_page_render.PREVIEW_WIDTH_PX and image.height > image.width   # uncut
    assert png == page_render.render_png(pdf, ".pdf", 2, template_page_render.PREVIEW_WIDTH_PX)


def test_a_template_page_is_drawn_whole_from_the_kit_in_a_child_process():
    from modules.documents.brand_kit import get_brand_kit

    shape = {"name": LETTER["name"], "blocks": LETTER["blocks"], "sample_data": LETTER["sample_data"]}
    png = template_page_render.render_template_page_isolated(shape, get_brand_kit(STORED_KIT), 1)

    image = Image.open(io.BytesIO(png))
    assert image.format == "PNG" and image.width == template_page_render.PREVIEW_WIDTH_PX
    assert image.height > image.width                                   # the whole A4 page, uncut
