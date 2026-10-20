"""A document an AI made carries the AI mark and a provenance row (PRD-264 EUAI-09).

Two decorators on ``DocumentGenerationService`` (``generation_service.py`` is over the
file-size limit, so the logic lives here and the service only names it):

* :func:`ai_documents_are_marked` wraps ``generate``. A call with no ``user_id`` is an
  agent's: the generate_document tool, a Playbook's document step and a Mission's
  report make their calls without one, because no person makes them (F344). A person's
  own request (the Studio's Generate, a template preview) names its user, and the
  document it fills in by hand is not AI-generated, so it gets no mark. A call whose
  person cannot be resolved (``resolve_user_pk`` is None: an API key) is marked: when
  in doubt, a document says AI made it.
* :func:`marked_before_it_is_kept` wraps ``_build_result``, the one step every PDF, DOCX
  and XLSX passes after its last write and before its persistence copy is uploaded. For
  an agent's document it writes the mark (``provenance_marks.mark_document``: the IPTC
  "composite with trained algorithmic media" term, AI text in the workspace's own
  template and letterhead, and a random instance id) and records the marked file's
  SHA-256 as a ``document`` row (``services.content_provenance.record_produced_file``),
  so the public check can say Automatos AI made it. A file that cannot be marked is deleted and the error raised:
  it is never delivered unmarked. Social images and videos are media-render's
  (EUAI-06a/06b) and pass through untouched.

``services`` is imported when a document is made, never when this module loads.
"""
from __future__ import annotations

import functools
import inspect
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .provenance_marks import COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA, MARKED_DOCUMENT_FORMATS, mark_document

logger = logging.getLogger(__name__)

Async = Callable[..., Awaitable[Any]]
# What a document's provenance row says made it: the toolkit, never a person or a model's prompt.
DOCUMENT_GENERATOR = "Automatos AI document generation"
DOCUMENT_KIND = "document"
DOCUMENT_SOURCE_TYPE = COMPOSITE_WITH_TRAINED_ALGORITHMIC_MEDIA


@dataclass(frozen=True)
class AiDocumentCall:
    """The agent's generate call in progress: the workspace its document's row belongs to."""

    workspace_id: Any


_AI_CALL: ContextVar[Optional[AiDocumentCall]] = ContextVar("prd264_ai_document_call", default=None)


def ai_documents_are_marked(generate: Async) -> Async:
    """Wrap ``DocumentGenerationService.generate``: an agent's call (no ``user_id``) is
    marked while it runs; a person's is not."""
    signature = inspect.signature(generate)

    @functools.wraps(generate)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        arguments = signature.bind_partial(self, *args, **kwargs).arguments
        workspace_id = arguments.get("workspace_id") or getattr(self, "workspace_id", None)
        call = AiDocumentCall(workspace_id) if arguments.get("user_id") is None else None
        token = _AI_CALL.set(call)
        try:
            return await generate(self, *args, **kwargs)
        finally:
            _AI_CALL.reset(token)
    return wrapped


def _marked(path: str, fmt: str) -> str:
    """Mark the file at ``path``; on failure delete it and raise, so it is never delivered unmarked."""
    try:
        return mark_document(path, fmt, DOCUMENT_SOURCE_TYPE)
    except Exception:
        logger.exception("[DocGen] the %s could not be marked as AI-generated; it is not delivered", fmt)
        Path(path).unlink(missing_ok=True)
        raise


def marked_before_it_is_kept(build_result: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``DocumentGenerationService._build_result(path, fmt, ...)``: an agent's PDF,
    DOCX or XLSX is marked and recorded before its result (and its upload) is built."""
    @functools.wraps(build_result)
    def wrapped(self: Any, path: str, fmt: str, *args: Any, **kwargs: Any) -> Any:
        call = _AI_CALL.get()
        if call is not None and fmt in MARKED_DOCUMENT_FORMATS:
            from services.content_provenance import record_produced_file

            sha256 = _marked(path, fmt)
            record_produced_file(call.workspace_id, sha256, DOCUMENT_KIND, DOCUMENT_GENERATOR)
        return build_result(self, path, fmt, *args, **kwargs)
    return wrapped


__all__ = [
    "DOCUMENT_GENERATOR",
    "DOCUMENT_KIND",
    "DOCUMENT_SOURCE_TYPE",
    "ai_documents_are_marked",
    "marked_before_it_is_kept",
]
