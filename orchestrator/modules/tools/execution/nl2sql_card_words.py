"""The owner's words on a board card reach the SQL writer (F301, night 9).

F077 (A) sends the person's own words for a chat turn to NL2SQL beside Auto's
restatement, because a restatement drops qualifiers. A board agent's restatement does
the same, and on a card nothing carried the owner's words:

* #1882 asked "Were any Harvest Club boxes late going out in September?". The Inventory
  Watchdog queried "all subscription orders that shipped late in September 2026"
  (audit row 289) and answered 9 — every plan. The Harvest Club had 1 late box of 68.
* #1859 ("Were any subscription club boxes late in September 2026?") went the same
  way: the Business Analyst's query dropped the plan (audit row 249), 9 again.

A card the owner wrote (``created_by_type == "user"``) is the owner's words for every
database question its run asks. Cards written by the platform (mission steps, Playbook
runs) are left out: their text is a planner's brief, not the person's words.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CARD_WORDS_CHARS = 2000       # the same bound as a chat turn's own words (OWNER_WORDS_CHARS)
OWNER_CARD = "user"


def _card_text(session: Any, task_id: int, workspace_id: str) -> Optional[str]:
    """The card's title and description when the owner wrote it, scoped to the workspace."""
    from core.models.core import BoardTask

    row = (
        session.query(BoardTask.title, BoardTask.description, BoardTask.created_by_type)
        .filter(BoardTask.id == task_id, BoardTask.workspace_id == str(workspace_id))
        .first()
    )
    if row is None or row[2] != OWNER_CARD:
        return None
    words = "\n".join(part.strip() for part in (row[0], row[1]) if part and part.strip())
    return words[:CARD_WORDS_CHARS] or None


def _read_in_own_session(task_id: int, workspace_id: str) -> Optional[str]:
    """Read the card in a short-lived session. Blocking: run it on a thread."""
    from core.database.database import SessionLocal

    session = SessionLocal()
    try:
        return _card_text(session, task_id, workspace_id)
    finally:
        session.close()


def _task_id(caller_context: Optional[Dict[str, Any]]) -> Optional[int]:
    """The board card this run works, from the server-built context (never a tool argument)."""
    raw = (caller_context or {}).get("board_task_id") if isinstance(caller_context, dict) else None
    try:
        return int(raw) if raw is not None and not isinstance(raw, bool) else None
    except (TypeError, ValueError):
        return None


async def card_words(
    caller_context: Optional[Dict[str, Any]], workspace_id: str, db_session: Optional[Any] = None
) -> Optional[str]:
    """The owner's words on the card this run works, or None (no card, a card the
    platform wrote, or the read failed — logged; the query then runs on the agent's
    words alone, as before).

    A borrowed request session is rolled back when the read fails: the turn's other
    tools share it (the F074 pattern)."""
    task_id = _task_id(caller_context)
    if task_id is None:
        return None
    try:
        if db_session is None:
            return await asyncio.to_thread(_read_in_own_session, task_id, workspace_id)
        return _card_text(db_session, task_id, workspace_id)
    except Exception:  # noqa: BLE001 — logged; the question still runs without the card
        logger.exception("F301: card %s words not read for the database question", task_id)
        if db_session is not None:
            db_session.rollback()
        return None
