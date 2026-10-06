"""``platform_render_preview``: an agent draws a page and opens it (PRD-255 US-012).

The Brand designer must see the page it made to judge it and revise. The tool draws
one page of a document template or of a Deliverable as a PNG and writes it into the
calling session's own folder, ``sessions/<ticket>/`` (F349's folder rule,
``session_document_folder.write_to_session``), and answers with the path so the
session opens it.

* **A session only.** The ticket comes from the server (``session_ticket``'s
  strip-then-inject); with none, the tool refuses and writes nothing.
* **Read-only, apart from the PNG.** No template, Deliverable or kit is changed. A
  ``brand_kit`` proposal (the kit patch ``platform_update_brand_kit`` takes) is laid
  over the stored kit through the same validation, for this render only: it is
  never saved (FR-11; US-014's proposal card carries a page drawn from it).
* **The caller's workspace only.** A template or Deliverable of another workspace
  answers "not found", as an unknown id does.
* **Document templates only** (Decision Q3): a social template is refused. A social
  Deliverable's stored image is copied as it is.
* The CPU-bound print runs in a child process under a time limit
  (``template_page_render``, F353's ``render_png_isolated``), off the event loop.
  The action is ``read`` (it changes nothing the owner or the board sees), so the
  write-action rate limit does not apply: a draw is bounded by that time limit,
  and a session waits for each one. The same page drawn again replaces its picture.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.discovery.session_ticket import SESSION_TICKET_PARAM

logger = logging.getLogger(__name__)

MAX_PREVIEW_PAGE = 50
TEMPLATE, DELIVERABLE = "template", "deliverable"
REF_PARAMS = {TEMPLATE: "template_id", DELIVERABLE: "deliverable_id"}
PROPOSAL_PARAM = "brand_kit"
PNG_EXT = ".png"
# A Deliverable that is already a picture (a social image) is copied as it is.
COPIED_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp"})

NEEDS_A_SESSION = (
    "render_preview draws a page into your session's folder, so it runs only in a session working a "
    "ticket. Nothing was drawn."
)
NAME_ONE = "render_preview needs exactly one of template_id (from list_templates) or deliverable_id."
NOT_AN_ID = "{param} {ref!r} is not an id: list_templates (or the Deliverable) gives it."
BAD_PAGE = f"page is a whole number from 1 to {MAX_PREVIEW_PAGE}."
NOT_FOUND = "No {what} {ref} in this workspace."
SOCIAL_REFUSED = (
    "{name!r} is a {fmt} template: render_preview draws document templates (pdf, docx, xlsx) only. "
    "Nothing was drawn."
)
NO_BLOCKS = (
    "{name!r} has no block layout to draw from. Fill it with generate_document, then render_preview "
    "the Deliverable it makes."
)
NO_PAGE = "A {ext} Deliverable has no page to draw."
ONE_PAGE_IMAGE = "That Deliverable is a single image: only page 1 exists."
PROPOSAL_ON_DELIVERABLE = "brand_kit applies to a template: a Deliverable is drawn as it was made."
PROPOSAL_NOT_AN_OBJECT = "brand_kit is an object of kit fields, the ones platform_update_brand_kit takes."
PROPOSAL_REFUSED = "The proposed kit is not valid, so nothing was drawn: {errors}"
DRAW_FAILED = "The page could not be drawn: {reason}"
NOT_WRITTEN = "The page was drawn, but your session's folder could not be written. Try again."
OPEN_NOTE = "Open {path} to look at the page before you judge it."
PROPOSAL_NOTE = " It was drawn from your proposed kit, which is NOT saved: the workspace's kit is unchanged."


@dataclass(frozen=True)
class PreviewRequest:
    """What to draw: a template or a Deliverable, its page, and a kit proposal (templates only)."""

    kind: str
    ref: UUID
    page: int
    proposal: Optional[Dict[str, Any]] = None


def _failed(error: str) -> Dict[str, Any]:
    return {"success": False, "error": error}


def ticket_of(params: Dict[str, Any]) -> Optional[int]:
    """The session's ticket, as the server injected it; ``None`` outside a session."""
    ticket = params.get(SESSION_TICKET_PARAM)
    return ticket if isinstance(ticket, int) and not isinstance(ticket, bool) and ticket > 0 else None


def _page(raw: Any) -> Optional[int]:
    if raw is None:
        return 1
    try:
        page = int(raw)
    except (TypeError, ValueError):
        return None
    return page if 1 <= page <= MAX_PREVIEW_PAGE and str(raw).strip() == str(page) else None


def parse_request(params: Dict[str, Any]) -> Tuple[Optional[PreviewRequest], Optional[str]]:
    """The request ``params`` make, or why they make none."""
    named = {kind: str(params[p]).strip() for kind, p in REF_PARAMS.items() if str(params.get(p) or "").strip()}
    if len(named) != 1:
        return None, NAME_ONE
    kind, ref = next(iter(named.items()))
    try:
        ref_id = UUID(ref)
    except ValueError:
        return None, NOT_AN_ID.format(param=REF_PARAMS[kind], ref=ref)
    page = _page(params.get("page"))
    if page is None:
        return None, BAD_PAGE
    proposal = params.get(PROPOSAL_PARAM)
    if proposal is not None and not isinstance(proposal, dict):
        return None, PROPOSAL_NOT_AN_OBJECT
    if proposal and kind == DELIVERABLE:
        return None, PROPOSAL_ON_DELIVERABLE
    return PreviewRequest(kind, ref_id, page, proposal or None), None


def _render_kit(db: Session, workspace_id: UUID, proposal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The render-ready kit: the stored one, with ``proposal`` laid over it (validated, never saved).

    Raises ``pydantic.ValidationError`` on a proposal the kit's patch would refuse.
    """
    from core.models.workspaces import Workspace
    from modules.documents.brand_fonts import brand_kit_for_media_render
    from modules.documents.brand_kit import get_brand_kit, proposed_brand_kit

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    settings = getattr(workspace, "settings", None)
    kit = proposed_brand_kit(settings, proposal) if proposal else get_brand_kit(settings)
    return brand_kit_for_media_render(kit)


def _template_job(db: Session, workspace_id: UUID, request: PreviewRequest) -> Tuple[Optional[Callable[[], bytes]], str]:
    """A function that draws the template's page, or ``None`` and why not."""
    from pydantic import ValidationError

    from core.social_templates import is_social_format
    from modules.documents.brand_kit import PATCH_FIELDS, brand_kit_errors
    from modules.documents.template_page_render import render_template_page_isolated
    from modules.documents.template_service import DocumentTemplateService

    template = DocumentTemplateService(db).get_template(request.ref, workspace_id)
    if template is None:
        return None, NOT_FOUND.format(what="template", ref=request.ref)
    if is_social_format(template.format):
        return None, SOCIAL_REFUSED.format(name=template.name, fmt=template.format)
    if not template.blocks:
        return None, NO_BLOCKS.format(name=template.name)
    unknown = sorted(set(request.proposal or {}) - set(PATCH_FIELDS))
    if unknown:
        return None, PROPOSAL_REFUSED.format(errors=f"the kit has no field {', '.join(unknown)}")
    try:
        kit = _render_kit(db, workspace_id, request.proposal)
    except ValidationError as e:
        return None, PROPOSAL_REFUSED.format(errors=brand_kit_errors(e))
    shape = {"name": template.name, "blocks": template.blocks, "sample_data": template.sample_data}
    return (lambda: render_template_page_isolated(shape, kit, request.page)), PNG_EXT


def _deliverable_job(db: Session, workspace_id: UUID, request: PreviewRequest) -> Tuple[Optional[Callable[[], bytes]], str]:
    """A function that draws (or, for an image, reads) the Deliverable's page, or ``None`` and why not."""
    from modules.documents.template_page_render import PREVIEW_WIDTH_PX
    from modules.documents.thumbnails.job import load_output
    from modules.documents.thumbnails.render import SUPPORTED_EXTENSIONS, ThumbnailError, render_png_isolated
    from modules.documents.thumbnails.sources import MAX_SOURCE_BYTES, read_source

    output = load_output(db, workspace_id, request.ref)
    if output is None:
        return None, NOT_FOUND.format(what="Deliverable", ref=request.ref)
    image = output.ext in COPIED_IMAGE_EXTENSIONS
    if not image and output.ext not in SUPPORTED_EXTENSIONS:
        return None, NO_PAGE.format(ext=output.ext or "extensionless")
    if image and request.page != 1:
        return None, ONE_PAGE_IMAGE
    if (output.file_size_bytes or 0) > MAX_SOURCE_BYTES:
        return None, DRAW_FAILED.format(reason="the file is too big to draw")

    def draw() -> bytes:
        data = read_source(output.workspace_id, output.storage_type, output.file_path)
        if data is None:
            raise ThumbnailError("its file was not found")
        if image:
            return data
        return render_png_isolated(data, output.ext, page_number=request.page, whole_width_px=PREVIEW_WIDTH_PX)

    return draw, (output.ext if image else PNG_EXT)


def preview_name(request: PreviewRequest, ext: str) -> str:
    """The picture's bare file name: what was drawn, its page and whether from a proposal.

    The same page drawn again replaces its picture, so a revising session's folder
    keeps one file per page, not one per attempt.
    """
    proposal = "-proposal" if request.proposal else ""
    return f"preview-{request.kind}-{request.ref.hex}-p{request.page}{proposal}{ext}"


async def render_preview(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Draw one page of a template or Deliverable of this workspace into the session's folder; its path."""
    from modules.documents.thumbnails.render import ThumbnailError
    from modules.tools.execution.session_document_folder import write_to_session

    ticket = ticket_of(params)
    if ticket is None:
        return _failed(NEEDS_A_SESSION)
    request, problem = parse_request(params)
    if request is None:
        return _failed(problem or NAME_ONE)
    job_for = _template_job if request.kind == TEMPLATE else _deliverable_job
    draw, ext_or_problem = job_for(db, workspace_id, request)
    if draw is None:
        return _failed(ext_or_problem)
    try:
        data = await asyncio.to_thread(draw)
    except ThumbnailError as e:
        logger.warning("[render_preview] %s %s page %s not drawn: %s", request.kind, request.ref, request.page, e)
        return _failed(DRAW_FAILED.format(reason=e))
    path = await write_to_session(workspace_id, ticket, preview_name(request, ext_or_problem), data)
    if path is None:
        return _failed(NOT_WRITTEN)
    note = OPEN_NOTE.format(path=path) + (PROPOSAL_NOTE if request.proposal else "")
    return {"success": True, "path": path, REF_PARAMS[request.kind]: str(request.ref), "page": request.page,
            "from_proposal": bool(request.proposal), "note": note}


__all__ = ["MAX_PREVIEW_PAGE", "parse_request", "preview_name", "render_preview", "ticket_of"]
