"""generate_document, the agent tool (PRD-63): its arguments read at the boundary (F298).

F298 (night 8): a playbook step on #0249 failed with "Tool generate_document
failed: 'str' object has no attribute 'get'", and that Python text landed on the
owner's card. The model had sent ``data`` as a string (the checklist itself, or
the object written out as JSON), and the handler, then inline in
``AgentPlatformTools.execute_tool``, called ``data.get("sections")`` on it; that
function's blanket ``except`` turned the AttributeError into the tool's answer.

``data`` is now read here, before anything renders:

* an object is used as it is (copied: generation fills in its title);
* an object the model wrote out as JSON text is parsed;
* plain text, for a PDF with no template, is the document's body (``content``);
* a list, for a PDF with no template, is its sections;
* anything else is refused in plain words that say what to send instead.

A failure while the document is made answers in plain words too. The errors the
platform words itself (a missing column, an unfilled template field, the month's
render minutes) say what they say; anything else is logged with its stack, and
the agent is told the document was not made and what to do next.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlencode
from uuid import UUID

from sqlalchemy.orm import Session

from core.media_render_client import MediaRenderError
from core.media_render_quota import RenderQuotaExceeded
from core.social_templates import is_social_format
from modules.documents.models import UnresolvedDeliverableError
from modules.tools.formatting.result_formatter import ToolResultFormatter

logger = logging.getLogger(__name__)

TOOL_NAME = "generate_document"
DEFAULT_TITLE = "Document"
DEFAULT_FORMAT = "pdf"
# The query parameter frontend/components/deliverables/deliverable-deep-link.tsx opens.
DELIVERABLE_PARAM = "deliverable"
# The one format whose no-template render takes free text and a list of sections.
BODY_FORMAT = "pdf"

DATA_NOT_AN_OBJECT = (
    "The document was not made: 'data' must be an object, for example "
    "{{\"sections\": [{{\"title\": \"...\", \"content\": \"...\"}}]}}, not {kind}."
)
DATA_BAD_JSON = (
    "The document was not made: 'data' looks like JSON but is not a complete object. "
    "Send 'data' as an object, for example {\"sections\": [{\"title\": \"...\", \"content\": \"...\"}]}."
)
TEMPLATE_NEEDS_FIELDS = (
    "The document was not made: a template is filled from named fields, so 'data' must be an object "
    "with the fields platform_get_template_schema lists, not {kind}."
)
MISSING_CONTENT = (
    "Missing document content. Either pass a template_id/template_name, "
    "or include 'sections' (a list of {title, content} objects with "
    "substantial text) or 'content' (a string) in 'data'."
)
SOCIAL_NEEDS_TEMPLATE = (
    "A {fmt} is rendered from a {fmt} template: pass its template_id or "
    "template_name (platform_list_templates with format {fmt} lists them)."
)
NO_WORKSPACE = "The document was not made: this agent belongs to no workspace, so there is nowhere to save it."
BAD_TEMPLATE_ID = (
    "The document was not made: template_id {value!r} is not a template id. "
    "Use an id platform_list_templates gives, or pass template_name instead."
)
NOT_MADE = (
    "The document was not made because of a problem on our side (it has been logged). "
    "Try once more; if it fails again, put the content in your answer as text instead."
)
# Failures the platform words itself, so their message is safe to pass on.
AUTHORED_FAILURES = (ValueError, UnresolvedDeliverableError, RenderQuotaExceeded, MediaRenderError)


class DocumentArgsRefused(ValueError):
    """The tool's arguments cannot make a document; the message says what to send."""


@dataclass(frozen=True)
class DocumentRequest:
    """One generate_document call, read and checked."""

    title: str
    fmt: str
    data: Dict[str, Any]
    template_name: Optional[str]
    template_id: Optional[UUID]


def _kind(value: Any) -> str:
    return "text" if isinstance(value, str) else f"a {type(value).__name__}"


def _from_text(text: str, fmt: str, has_template: bool) -> Dict[str, Any]:
    """``data`` sent as a string: a JSON object written out, or the body itself."""
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise DocumentArgsRefused(DATA_BAD_JSON) from exc
        if isinstance(parsed, dict):
            return parsed
    if has_template:
        raise DocumentArgsRefused(TEMPLATE_NEEDS_FIELDS.format(kind=_kind(text)))
    if fmt != BODY_FORMAT:
        raise DocumentArgsRefused(DATA_NOT_AN_OBJECT.format(kind=_kind(text)))
    return {"content": text}


def document_data(raw: Any, fmt: str, has_template: bool) -> Dict[str, Any]:
    """The ``data`` object a call means, or :class:`DocumentArgsRefused`. Pure."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        return _from_text(raw, fmt, has_template)
    if isinstance(raw, list) and not has_template and fmt == BODY_FORMAT:
        return {"sections": list(raw)}
    if has_template:
        raise DocumentArgsRefused(TEMPLATE_NEEDS_FIELDS.format(kind=_kind(raw)))
    raise DocumentArgsRefused(DATA_NOT_AN_OBJECT.format(kind=_kind(raw)))


def _template_id(raw: Any) -> Optional[UUID]:
    if not raw:
        return None
    try:
        return UUID(str(raw))
    except (ValueError, TypeError) as exc:
        raise DocumentArgsRefused(BAD_TEMPLATE_ID.format(value=raw)) from exc


def _has_content(data: Dict[str, Any], fmt: str) -> bool:
    """A no-template document is built only from ``data``: it must carry something."""
    if not data:
        return False
    return fmt != BODY_FORMAT or bool(data.get("sections") or data.get("content"))


def document_request(parameters: Dict[str, Any]) -> DocumentRequest:
    """Read one call's arguments, or :class:`DocumentArgsRefused` in plain words. Pure."""
    fmt = str(parameters.get("format") or DEFAULT_FORMAT)
    template_name = parameters.get("template_name") or None
    template_id = _template_id(parameters.get("template_id"))
    has_template = bool(template_name or template_id)
    # PRD-251 US-117: a social image or video is its template, rendered.
    if is_social_format(fmt) and not has_template:
        raise DocumentArgsRefused(SOCIAL_NEEDS_TEMPLATE.format(fmt=fmt))
    data = document_data(parameters.get("data"), fmt, has_template)
    # PRD-167 S6: only the no-template path needs content in data; a template
    # defines its own structure and fills data.* fields.
    if not has_template and not _has_content(data, fmt):
        raise DocumentArgsRefused(MISSING_CONTENT)
    title = str(parameters.get("title") or DEFAULT_TITLE)
    return DocumentRequest(title, fmt, data, template_name, template_id)


def deliverable_open_url(deliverable_id: Optional[str]) -> str:
    """The owner's link to one Deliverable: the Deliverables page opened on it (F298).

    Short, unsigned and lasting, so an agent can copy it onto a card. The page
    fetches the file through the app's own document route as the signed-in
    owner (anonymously in the local edition). Without an id, the page itself.
    """
    from modules.documents.generation_service import deliverables_app_url

    page = deliverables_app_url()
    if not deliverable_id:
        return page
    joiner = "&" if "?" in page else "?"
    return f"{page}{joiner}{urlencode({DELIVERABLE_PARAM: str(deliverable_id)})}"


def failure(message: str) -> Dict[str, Any]:
    """The tool's answer when no document was made."""
    # "status" too: standardize_result reads success from it, and a bare False came back None
    return ToolResultFormatter.standardize_result({"success": False, "status": "error", "error": message}, TOOL_NAME)


def _agent_row(db: Session, agent_id: int) -> Any:
    from core.models import Agent

    return db.query(Agent).filter(Agent.id == agent_id).first()


async def run_generate_document(db: Session, parameters: Dict[str, Any], agent_id: int) -> Dict[str, Any]:
    """Make the document one generate_document call asks for, or say in plain words why not."""
    try:
        request = document_request(parameters if isinstance(parameters, dict) else {})
    except DocumentArgsRefused as refused:
        return failure(str(refused))
    agent = _agent_row(db, agent_id)
    workspace_id = getattr(agent, "workspace_id", None)
    if not workspace_id:
        return failure(NO_WORKSPACE)
    try:
        return await make_document(db, request, agent, workspace_id)
    except AUTHORED_FAILURES as exc:
        logger.warning("[generate_document] agent %s: %s", agent_id, exc)
        return failure(str(exc))
    except Exception:
        logger.exception("[generate_document] agent %s: the document could not be made", agent_id)
        return failure(NOT_MADE)


async def make_document(db: Session, request: DocumentRequest, agent: Any, workspace_id: UUID) -> Dict[str, Any]:
    """Render the document, register it as a Deliverable and answer with its links."""
    from modules.documents.generation_service import DocumentGenerationService

    logger.info("[generate_document] %s document: %r", request.fmt.upper(), request.title)
    service = DocumentGenerationService(db, workspace_id)
    result = await service.generate(
        title=request.title,
        format=request.fmt,
        data=request.data,
        workspace_id=workspace_id,
        template_name=request.template_name,
        template_id=request.template_id,
        user_id=getattr(agent, "user_id", None),
    )
    # PRD-167 S6: the rendered document is a Deliverable with source attribution.
    registration = service.register_as_deliverable(
        result,
        title=request.title,
        source_type="agent_output",
        agent_id=getattr(agent, "id", None),
        agent_name=getattr(agent, "name", None),
        template_id=request.template_id,
    ) or {}
    await _ingest(db, workspace_id, result, request.title, agent, registration)
    return ToolResultFormatter.standardize_result(
        {"success": True, "results": [_answer(service, result, registration)]}, TOOL_NAME
    )


def _answer(service: Any, result: Any, registration: Dict[str, Any]) -> Dict[str, Any]:
    """PRD-242 S4: the links an agent needs to deliver the document, and which template filled it."""
    from modules.documents.generation_service import deliverables_app_url

    return {
        "status": "success",
        "filename": result.filename,
        "format": result.format,
        "download_url": result.download_url,
        "size_kb": result.size // 1024,
        "content": result.content,
        "deliverable_id": registration.get("deliverable_id"),
        "app_url": deliverables_app_url(),
        "open_url": deliverable_open_url(registration.get("deliverable_id")),
        "share_url": service.share_link(result),
        "template_id": result.template_id,
        "template_name": result.template_name,
        # F331: the data keys the page does not show; an agent cannot see the page.
        "unused_data_keys": list(getattr(result, "unused_keys", None) or []),
    }


async def _ingest(db: Session, workspace_id: UUID, result: Any, title: str, agent: Any,
                  registration: Dict[str, Any]) -> None:
    """PRD-164 S3: the document's markdown becomes retrievable knowledge. Never fails the generation."""
    from services.knowledge_flywheel import ingest_agent_output

    agent_name = getattr(agent, "name", None)
    source_id = registration.get("deliverable_id") or result.filename
    try:
        await ingest_agent_output(
            db,
            workspace_id,
            content=result.content or "",
            filename=f"{(result.filename or title).rsplit('.', 1)[0]}.md",
            source="generated_document",
            source_id=str(source_id),
            title=title,
            description=f"Generated document: {title}"[:500],
            agent_name=agent_name,
            created_by=agent_name or "agent",
            extra_tags=[f"agent:{getattr(agent, 'id', None)}"],
        )
    except Exception:
        logger.exception("[generate_document] knowledge ingest failed for %r (the document stands)", title)
