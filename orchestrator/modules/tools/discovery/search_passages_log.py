"""PRD-256 P256-FIX-N17-11 (F443, night 16): what a document search handed back, in the log.

Ask 318 ("the November price letter", 22:02 on 14 Oct) read three platform_search_documents
answers and the October terms, then quoted October prices while the November sheet was in the
workspace. The night's rows record each call's arguments, not its passages, so whether the
sheet was among them, or cut after, could not be told. Each search now logs one INFO line: the
turn's conversation, the query, and each passage's file, score and date, in the order the
answer lists them (newest document first, ``services.contested_facts.dates_and_ranks``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List

from modules.tools.execution.turn_owner_words import held_conversation_id
from services.contested_facts import DATE_KEY

logger = logging.getLogger(__name__)

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

LOGGED = "[search-documents] conversation=%s query=%r passages=%d: %s"
QUERY_CHARS = 200   # the query as logged: the owner's words, cut
PASSAGE = "{file} score={score} date={date}"
NONE = "none"


def passages_logged(handler: Handler) -> Handler:
    """Wrap platform_search_documents' handler: log what it found, in the order it lists it."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        result = await handler(db, workspace_id, params)
        found = result.get("results") if isinstance(result, dict) and result.get("success") else None
        if isinstance(found, list):
            query = str(params.get("query") or "")[:QUERY_CHARS]
            logger.info(LOGGED, held_conversation_id(), query, len(found), passages_said(found))
        return result
    return wrapped


def passages_said(passages: List[Any]) -> str:
    """'harbourline-wholesale-terms-2026-11.md score=0.71 date=2026-11-01; …', or 'none'."""
    said = [PASSAGE.format(file=p.get("file_name"), score=p.get("score"),
                           date=(p.get("metadata") or {}).get(DATE_KEY))
            for p in passages if isinstance(p, dict)]
    return "; ".join(said) or NONE


__all__ = ["passages_logged", "passages_said"]
