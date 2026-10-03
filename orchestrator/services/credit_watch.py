"""F244 (night 7): the AI credit warns before a night's work would run past it.

Night 7 started with $18.69 on the AI provider's account and spent $18.24. The credit
ran out at 06:53 and stopped the owner's morning at 07:14 with no warning: the
product never showed a balance.

In the local edition the owner pays for every call. There, work that is about to
start (``refuse_new_work``, the board's and the missions' gate) also has the AI
provider's balance looked at, at most every 15 minutes per workspace, off the
caller's path. When what is left is under what the last 24 hours of work cost (or
under LOW_CREDIT_FLOOR_USD), the bell says so, once a day.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

CHECK_EVERY_SECONDS = 15 * 60
LOW_CREDIT_FLOOR_USD = 2.0
LOW_EVENT = "credit_low"
LOW_TITLE = "AI credit is low"
LOW_NOTICE = ("Your AI credit has ${left:.2f} left on the AI provider account that pays for the agents' model calls. "
              "The last 24 hours of work cost ${day:.2f}, so the next night's work could run out partway. Top it up "
              "before then.")

_last_check: Dict[str, float] = {}
_noticed_on: Dict[str, str] = {}
_running: Set["asyncio.Task[Any]"] = set()


def watches_the_credit(guard: Callable[..., Optional[str]]) -> Callable[..., Optional[str]]:
    """Wrap ``refuse_new_work``: its answer is unchanged; the balance is looked at beside it."""
    @functools.wraps(guard)
    def wrapped(db: Any, workspace_id: Any, what: str) -> Optional[str]:
        reason = guard(db, workspace_id, what)
        _schedule_check(workspace_id)
        return reason
    return wrapped


def _schedule_check(workspace_id: Any) -> None:
    from config import config

    if not config.IS_LOCAL_EDITION or not workspace_id:
        return
    key, now = str(workspace_id), time.monotonic()
    if now - _last_check.get(key, float("-inf")) < CHECK_EVERY_SECONDS:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return      # no loop here (a sync caller): the next start that has one looks
    _last_check[key] = now
    task = loop.create_task(check_credit(key))
    _running.add(task)
    task.add_done_callback(_running.discard)


async def check_credit(workspace_id: str) -> Optional[float]:
    """What is left on the AI provider's account, after the bell has said it is low
    (once a day). None when it can't be read."""
    try:
        api_key, day = await asyncio.to_thread(_key_and_last_day, workspace_id)
        if not api_key:
            return None
        from core.llm.openrouter_analytics import OpenRouterAnalyticsService

        credits = await OpenRouterAnalyticsService().get_credits(api_key)
        if not credits:
            return None
        left = float(credits.get("total_credits") or 0) - float(credits.get("total_usage") or 0)
        if left < max(LOW_CREDIT_FLOOR_USD, day) and _first_notice_today(workspace_id):
            await _ring(workspace_id, left, day)
        return left
    except Exception:
        logger.exception("[credit-watch] could not read the AI credit for workspace %s", workspace_id)
        return None


def _key_and_last_day(workspace_id: str) -> Tuple[Optional[str], float]:
    """The key the calls use, and the last 24 hours' spend, from the database."""
    from sqlalchemy import text

    from core.database.database import get_db_session
    from core.llm.key_resolver import resolve_provider_key

    with get_db_session() as db:
        resolved = resolve_provider_key(db, "openrouter", workspace_id=workspace_id)
        day = db.execute(text("SELECT COALESCE(SUM(total_cost), 0) FROM llm_usage WHERE workspace_id = "
                              "CAST(:ws AS uuid) AND created_at >= :since"),
                         {"ws": workspace_id, "since": datetime.now(timezone.utc) - timedelta(days=1)}).scalar()
    return (resolved.api_key if resolved else None), float(day or 0)


def _first_notice_today(workspace_id: str) -> bool:
    today = datetime.now(timezone.utc).date().isoformat()
    if _noticed_on.get(workspace_id) == today:
        return False
    _noticed_on[workspace_id] = today
    return True


async def _ring(workspace_id: str, left: float, day: float) -> None:
    from core.database.database import SessionLocal
    from core.services.notification_dispatcher import NotificationDispatcher

    db = SessionLocal()
    try:
        await NotificationDispatcher(db, workspace_id).dispatch(
            event_type=LOW_EVENT, title=LOW_TITLE, message=LOW_NOTICE.format(left=max(left, 0.0), day=day),
            status="warn", severity="urgent")
        db.commit()
    finally:
        db.close()


__all__ = ["LOW_EVENT", "LOW_NOTICE", "check_credit", "watches_the_credit"]
