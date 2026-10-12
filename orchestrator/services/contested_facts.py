"""P256-FIX-N13-3 (F410, night 13): a corrected fact wins over its superseded copies.

Night 13 (11–12 Oct): the owner told Auto at 22:05 that the club's dispatch day from November
is Wednesday (a stored memory, so a standing rule, FX-015) and wrote it into a ruling document.
Asked cold, Auto answered Monday on 4 of 17 asks (2 of 7 on identical words): five older
documents still said Monday, and their passages outnumbered the rule. Quay's 30-day terms, one
stale rival, were right 5 of 5. The count of stale passages decided, not the owner.

The ranking, here and nowhere else, with no list of facts:

- The owner's own stated rule outranks any document: the Standing Rules block says so
  (``RULES_OUTRANK_DOCUMENTS``), and a retrieval-first turn puts the owner's rules above its
  passages (consumers/chatbot/knowledge_prefetch.py).
- Among documents the newest outranks the older: a passage carries its document's date (the
  date its name gives, else the day it was uploaded: ``dated``), is listed newest first and
  says whether it is the newest or older (``ranked``). A passage dated before the owner's
  newest standing rule says so.
- Ties never fall to arrival order: score, then document, then text (``in_score_order``, which
  also decides which passages a prompt's top few hold), so ten identical asks read one context.
- Auto answers with the winning value and names the conflict in one line (``RANKING_RULE``).
"""
from __future__ import annotations

import functools
import re
from datetime import date, datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional
from uuid import UUID

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

# A passage's dating rides its ``metadata``: the result formatter keeps that dict whole.
DATE_KEY = "document_date"
STANDING_KEY = "standing"
RULES_STATED_KEY = "rules_stated"
NEWEST = "newest"
OLDER = "older"
OLDER_THAN_THE_RULES = "older than the owner's newest standing rule ({stated})"

RULES_OUTRANK_DOCUMENTS = "They outrank any document or passage that says otherwise, whatever its date."
RANKING_RULE = (
    "Where these give different values for one fact, the owner's standing rule outranks any document, and a "
    "newer document outranks an older one (each source gives its document's date; the newest is listed first). "
    "Answer with the winning value, and when they disagree say so in one line naming the older files that still "
    "say otherwise ('older files still say Monday: the packing checklist, …')."
)

_NAMED_DATE = re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)")


def document_date(name: Optional[str], uploaded: Any) -> Optional[str]:
    """A document's date (ISO): the first real date its name gives, else the day it was
    uploaded; None when it has neither."""
    for match in _NAMED_DATE.finditer(name or ""):
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
        except ValueError:
            continue
    if isinstance(uploaded, datetime):
        return uploaded.date().isoformat()
    return uploaded.isoformat() if isinstance(uploaded, date) else None


def _meta(passage: Dict[str, Any]) -> Dict[str, Any]:
    meta = passage.get("metadata")
    return meta if isinstance(meta, dict) else {}


def _as_id(value: Any) -> Optional[int]:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(value) if isinstance(value, str) and value.strip().isdigit() else None


def passage_date(passage: Dict[str, Any]) -> Optional[str]:
    """The date of the document a passage came from, when it was dated."""
    found = _meta(passage).get(DATE_KEY)
    return str(found) if found else None


def _with_meta(passage: Dict[str, Any], **values: Any) -> Dict[str, Any]:
    return {**passage, "metadata": {**_meta(passage), **values}}


def _score(passage: Dict[str, Any]) -> float:
    try:
        return float(passage.get("similarity") or passage.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _order(passage: Dict[str, Any]) -> tuple:
    """Newest document first (undated last), then score, then document, then text."""
    found = passage_date(passage)
    newest_first = -date.fromisoformat(found).toordinal() if found else 0
    return (found is None, newest_first, *_by_score(passage))


def _by_score(passage: Dict[str, Any]) -> tuple:
    return (-_score(passage), str(passage.get("document_id") or ""),
            str(passage.get("content") or passage.get("excerpt") or ""))


def in_score_order(passages: Iterable[Any]) -> List[Any]:
    """The passages by score, ties by document then text, never by arrival order: which of them
    a prompt's top few holds is the same on every identical ask."""
    items = list(passages)
    return sorted((p for p in items if isinstance(p, dict)), key=_by_score) + [p for p in items if not isinstance(p, dict)]


def _standing(found: Optional[str], newest: Optional[str], distinct: int, stated: Optional[str]) -> str:
    if found is None:
        return ""
    if stated and found < stated:
        return OLDER_THAN_THE_RULES.format(stated=stated)
    if distinct < 2:
        return ""
    return NEWEST if found == newest else OLDER


def ranked(passages: Iterable[Any]) -> List[Dict[str, Any]]:
    """The passages newest document first, each with its standing in its metadata: newest,
    older, or older than the owner's newest standing rule (``against_the_rules``)."""
    items = [p for p in passages if isinstance(p, dict)]
    dates = {d for d in (passage_date(p) for p in items) if d}
    newest = max(dates) if dates else None
    stated = max((str(_meta(p)[RULES_STATED_KEY]) for p in items if _meta(p).get(RULES_STATED_KEY)), default=None)
    return [_with_meta(p, **{STANDING_KEY: _standing(passage_date(p), newest, len(dates), stated)})
            for p in sorted(items, key=_order)]


def standing_note(passage: Dict[str, Any]) -> str:
    """What a source line says of its document's date: 'dated 2026-10-11, newest'; '' undated."""
    found = passage_date(passage)
    if not found:
        return ""
    standing = _meta(passage).get(STANDING_KEY)
    return f"dated {found}, {standing}" if standing else f"dated {found}"


def against_the_rules(passages: Iterable[Any], stated: Optional[str]) -> List[Any]:
    """The passages, each weighed against the owner's newest standing rule, stated on ``stated``."""
    if not stated:
        return list(passages)
    return [_with_meta(p, **{RULES_STATED_KEY: stated}) if isinstance(p, dict) else p for p in passages]


def _a_session(db: Any) -> bool:
    from sqlalchemy.orm import Session

    return isinstance(db, Session)


def document_dates(db: Any, workspace_id: Any, ids: Iterable[Any]) -> Dict[int, str]:
    """Each of these documents' date, in the workspace (``document_date``)."""
    from core.models.core import Document

    wanted = sorted({i for i in (_as_id(value) for value in ids) if i is not None})
    if not wanted:
        return {}
    rows = (db.query(Document.id, Document.filename, Document.original_filename, Document.upload_date)
            .filter(Document.id.in_(wanted), Document.workspace_id == UUID(str(workspace_id))).all())
    named = ((row.id, document_date(f"{row.original_filename or ''} {row.filename or ''}", row.upload_date))
             for row in rows)
    return {doc_id: found for doc_id, found in named if found}


def dated(db: Any, workspace_id: Any, result: Any, key: str = "results") -> Any:
    """``result`` with each passage under ``key`` carrying its document's date; as it was
    without a database session, a workspace, or a passage from a dated document."""
    found = result.get(key) if isinstance(result, dict) else None
    if not isinstance(found, list) or not found or workspace_id is None or not _a_session(db):
        return result
    dates = document_dates(db, workspace_id, [p.get("document_id") for p in found if isinstance(p, dict)])
    if not dates:
        return result
    return {**result, key: [_with_meta(p, **{DATE_KEY: dates[_as_id(p.get("document_id"))]})
                            if isinstance(p, dict) and _as_id(p.get("document_id")) in dates else p
                            for p in found]}


def dated_for_agent(db: Any, result: Any, agent_id: Any) -> Any:
    """A search_knowledge result dated in the searching agent's workspace, its passages in
    score order (``in_score_order``)."""
    from core.models import Agent

    found = result.get("results") if isinstance(result, dict) else None
    ordered = {**result, "results": in_score_order(found)} if isinstance(found, list) else result
    if not agent_id or not _a_session(db):
        return ordered
    workspace_id = db.query(Agent.workspace_id).filter(Agent.id == agent_id).scalar()
    return dated(db, workspace_id, ordered)


def dates_and_ranks(key: str) -> Callable[[Handler], Handler]:
    """Wrap a platform document search handler ``(db, workspace_id, params)``: its passages
    under ``key`` are dated and listed newest first, each with its standing."""
    def wrap(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            result = dated(db, workspace_id, await handler(db, workspace_id, params), key)
            found = result.get(key) if isinstance(result, dict) else None
            return {**result, key: ranked(found)} if isinstance(found, list) and found else result
        return wrapped
    return wrap


__all__ = ["RANKING_RULE", "RULES_OUTRANK_DOCUMENTS", "against_the_rules", "dated", "dated_for_agent",
           "dates_and_ranks", "document_date", "in_score_order", "ranked", "standing_note"]
