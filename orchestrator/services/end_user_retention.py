"""An end user's retention: what is older than their ``retention_days`` is deleted (PRD-274 Wave 3, §5.5).

A product records, with an end user's consent, how many days their memories are kept
(``end_users.retention_days``; none: until erased). The sweep, for every end user with a number:

- deletes their facts made before the cutoff (``DurableMemoryStore.prune_older_than`` on
  ``mem:{ws}:end_user:{id}``), retired ones too;
- deletes their messages sent before the cutoff, in their own conversations (``chats.end_user_id``), and
  then each of their conversations left empty that was itself opened before the cutoff.

It runs once a day on the shared scheduler (``schedules_the_retention_sweep`` on ``UnifiedScheduler.start``),
at ``RETENTION_HOUR_UTC``; a run missed by up to ``RETENTION_MISFIRE_SECONDS`` (a deploy) still runs.

An end user whose erase has started is left to the erase. Running it twice deletes nothing more: each run
deletes only what is older than its own cutoff. One end user that fails is logged and counted, and the rest
are still swept. Their sessions (Redis) need nothing: a conversation untouched for a whole day has none
left at the default ``MEMORY_SESSION_TTL_SECONDS``.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

RETENTION_JOB_ID = "end_user_retention_sweep"
RETENTION_HOUR_UTC = 3
RETENTION_MINUTE_UTC = 40
RETENTION_MISFIRE_SECONDS = 6 * 3600

Kept = Tuple[str, str, int]  # (workspace_id, end_user_id, retention_days)


def _with_retention() -> List[Kept]:
    """Every end user with a retention, whose erase has not started."""
    from core.database.database import SessionLocal
    from core.models.end_users import EndUser

    db = SessionLocal()
    try:
        rows = (db.query(EndUser.workspace_id, EndUser.end_user_id, EndUser.retention_days)
                .filter(EndUser.retention_days.isnot(None), EndUser.erasing_at.is_(None)).all())
        return [(str(ws), end_user_id, int(days)) for ws, end_user_id, days in rows]
    finally:
        db.close()


def _prune_their_conversations(workspace_id: str, end_user_id: str, cutoff: datetime) -> Tuple[int, int]:
    """Deletes their messages sent before ``cutoff``, then their conversations left empty; (messages, chats)."""
    from sqlalchemy import exists, select

    from core.database.database import SessionLocal
    from core.models.core import Chat, Message

    before = cutoff.astimezone(timezone.utc).replace(tzinfo=None)  # the columns are naive UTC
    theirs = select(Chat.id).where(Chat.workspace_id == workspace_id, Chat.end_user_id == end_user_id)
    # Empty and itself older than the cutoff: a chat just opened has no message for a moment.
    empty = theirs.where(~exists().where(Message.chat_id == Chat.id), Chat.created_at < before)
    db = SessionLocal()
    try:
        messages = (db.query(Message).filter(Message.chat_id.in_(theirs), Message.created_at < before)
                    .delete(synchronize_session=False))
        chats = db.query(Chat).filter(Chat.id.in_(empty)).delete(synchronize_session=False)
        db.commit()
        return messages, chats
    finally:
        db.close()


async def sweep_end_user_retention(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Deletes, for each end user with a retention, what is older than it; the counts of the run."""
    from modules.memory.unified_memory_service import MemoryNamespace, get_unified_memory_service

    now = now or datetime.now(timezone.utc)
    kept = await asyncio.to_thread(_with_retention)
    durable = get_unified_memory_service()._durable
    totals = {"end_users": len(kept), "memories": 0, "messages": 0, "chats": 0, "failed": 0}
    for workspace_id, end_user_id, days in kept:
        cutoff = now - timedelta(days=days)
        try:
            totals["memories"] += await durable.prune_older_than(MemoryNamespace(workspace_id).end_user(end_user_id), cutoff)
            messages, chats = await asyncio.to_thread(_prune_their_conversations, workspace_id, end_user_id, cutoff)
        except Exception:
            logger.exception("[end_user_retention] end user %s of workspace %s not swept", end_user_id, workspace_id)
            totals["failed"] += 1
            continue
        totals["messages"] += messages
        totals["chats"] += chats
    logger.info("[end_user_retention] swept %s", totals)
    return totals


def schedules_the_retention_sweep(start: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``UnifiedScheduler.start``: once the shared scheduler runs, it sweeps end users' retention daily."""

    @functools.wraps(start)
    def started(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = start(self, *args, **kwargs)
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is not None:
            scheduler.add_job(sweep_end_user_retention, "cron", hour=RETENTION_HOUR_UTC, minute=RETENTION_MINUTE_UTC,
                              id=RETENTION_JOB_ID, replace_existing=True, max_instances=1, coalesce=True,
                              misfire_grace_time=RETENTION_MISFIRE_SECONDS)
        return result

    return started

