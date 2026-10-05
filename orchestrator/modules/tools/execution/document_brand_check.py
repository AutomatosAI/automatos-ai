"""A document made with words the brand kit bans says so (night 9b, prep for night 10).

Night 9b: PDF 49c0c2b1 said "Two Bags of Guji! ... our exquisite Guji coffee", and
the brand voice bans "exquisite"; nothing told the agent that made it. An agent's
generate_document (an API agent's, or a session's routed by name) runs through
``exec_document.execute_generate_document``; :func:`a_documents_banned_words_are_said`
wraps it. The words are not rewritten in the file: the result's first row carries
the note (``DOCUMENT_NOTE_KEY``), for the agent to make the document again without
them. The placeholder signature is filled before the render, by
``modules.documents.brand_signing``.

``services`` is imported when a document is made, never when this module loads.
"""
from __future__ import annotations

import asyncio
import functools
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional

from modules.tools.formatting.document_brand_note import DOCUMENT_NOTE_KEY

Async = Callable[..., Awaitable[Any]]


def _texts(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _texts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _texts(item)


def document_text(parameters: Any) -> str:
    """Every text in a generate_document call's title and data, one a line."""
    params = parameters if isinstance(parameters, dict) else {}
    return "\n".join(_texts([params.get("title"), params.get("data")]))


def with_document_note(result: Any, note: str) -> Any:
    """``result`` with ``note`` on its first row: a successful result only, as a new dict."""
    rows = result.get("results") if isinstance(result, dict) else None
    if not note or not rows or not result.get("success") or not isinstance(rows[0], dict):
        return result
    return {**result, "results": [{**rows[0], DOCUMENT_NOTE_KEY: note}, *rows[1:]]}


def _agent_workspace(db: Any, agent_id: Any) -> Any:
    from core.models import Agent
    from services.brand_rules import without_flushing

    if db is None or not agent_id:
        return None
    with without_flushing(db):
        agent = db.query(Agent).filter(Agent.id == agent_id).first()
    return getattr(agent, "workspace_id", None)


def a_documents_banned_words_are_said(execute: Async) -> Async:
    """Wrap ``exec_document.execute_generate_document``: the result names the banned words."""
    @functools.wraps(execute)
    async def wrapped(executor: Any, tool_name: str, parameters: Dict[str, Any], agent_id: int,
                      workspace_id: Any = None, trace_id: Optional[str] = None,
                      caller_context: Optional[Dict[str, Any]] = None) -> Any:
        from services import brand_rules as br

        # F341: the card the call works travels on to the render (only when there is a context).
        context = {"caller_context": caller_context} if caller_context is not None else {}
        result = await execute(executor, tool_name, parameters, agent_id, workspace_id=workspace_id, trace_id=trace_id,
                               **context)
        db = getattr(executor, "db", None)
        # Both reads off the event loop (F105, F330): a pool wait never stops it.
        workspace = workspace_id or await asyncio.to_thread(_agent_workspace, db, agent_id)
        kit = await br.kit_off_loop(db, workspace)
        phrases = ((kit or {}).get("voice") or {}).get("banned_phrases") or []
        return with_document_note(result, br.banned_note(br.banned_found(document_text(parameters), phrases)))
    return wrapped


__all__ = ["a_documents_banned_words_are_said", "document_text", "with_document_note"]
