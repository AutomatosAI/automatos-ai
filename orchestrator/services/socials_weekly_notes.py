"""PRD-251C (C7, C9; US-C403): the weekly note goes out, once per plan and week.

The plan tick (``services/socials_plan_maker.run_tick``) calls :func:`send_due` on the leader.
For each active plan whose note is due (``weekly_note.note_key``) and not yet sent for that
week (``make.notified["weekly_note"]``), with Socials on in its workspace: the week's posts and
their numbers (``posted.posted``), the plan's health (``plan_health.health_for``) and Auto's
proposals (``proposals``) become the note (``weekly_note.compose``), recorded as sent, then sent
through the platform's notifications, linked to the plan's Posted view. A week with no post
out and nothing to fix sends nothing. One plan's failure is logged and never stops the rest.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, List, Optional, Tuple

from config import config
from core.models.workspaces import Workspace
from modules.socials import plan_health, plan_notify, plan_store, plans, posted, proposals, results, weekly_note
from modules.socials.settings import socials_off_reason

logger = logging.getLogger(__name__)


def posted_url(plan_id: Any) -> str:
    return f"{str(config.FRONTEND_URL).rstrip('/')}{weekly_note.POSTED_PATH.format(plan_id=plan_id)}"


def _aware(iso: Optional[str]) -> Optional[datetime]:
    """A stored time as UTC (SQLite gives it back without a zone)."""
    if not iso:
        return None
    moment = datetime.fromisoformat(iso)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def week_posts(db: Any, plan: Any, now: datetime) -> List[weekly_note.WeekPost]:
    """The plan's posts that went out in the seven days before ``now``, newest first."""
    since = now - timedelta(days=weekly_note.WEEK_DAYS)
    out = []
    for row in posted.posted(db, plan.workspace_id, plan_id=plan.id, limit=posted.MAX_POSTED):
        went = _aware(row["went_out_at"])
        if went is None or went < since:
            continue
        read = row["numbers"]
        clock = went.astimezone(plans.zone_of(plan)).strftime("%H:%M")
        out.append(weekly_note.WeekPost(row["title"], row["format"], clock, row["topic"],
                                        results.numbers_line(read["numbers"] if read else None), read["engagement"] if read else None))
    return out


def build(db: Any, plan: Any, workspace: Any, now: datetime) -> Optional[Tuple[str, str]]:
    """The plan's note for this week, or ``None`` when there is nothing to say."""
    posts = week_posts(db, plan, now)
    health = [item.title for item in plan_health.health_for(db, plan, workspace, now)]
    since = now - timedelta(days=proposals.LOOKBACK_DAYS)
    proposed = [item.title for item in proposals.proposals(plan, proposals.read_posts(db, plan, since))]
    if not posts and not health:
        return None
    return weekly_note.compose(plan.name, posts, health, proposed, posted_url(plan.id))


def _mark_sent(db: Any, plan: Any, key: str) -> None:
    make = dict(plan.make or {})
    notified = dict(make.get(plan_notify.NOTIFIED) or {})
    plan.make = {**make, plan_notify.NOTIFIED: {**notified, weekly_note.NOTE_KEY: key}}
    db.commit()


def _send_one(db: Any, plan: Any, now: datetime) -> bool:
    key = weekly_note.note_key(plan, now)
    if key is None or ((plan.make or {}).get(plan_notify.NOTIFIED) or {}).get(weekly_note.NOTE_KEY) == key:
        return False
    workspace = db.get(Workspace, plan.workspace_id)
    if workspace is None or socials_off_reason(workspace) is not None:
        return False
    note = build(db, plan, workspace, now)
    _mark_sent(db, plan, key)
    if note is None:
        return False
    plan_notify.notify_weekly_note(plan.workspace_id, plan.id, *note)
    return True


def send_due(session_factory: Callable[[], Any], now: datetime) -> int:
    """Every due weekly note sent (the module docstring); how many went out."""
    db = session_factory()
    sent = 0
    try:
        for plan in plan_store.active_plans(db):
            try:
                sent += int(_send_one(db, plan, now))
            except Exception:  # noqa: BLE001 — one plan's note never stops the rest; logged
                logger.exception("[Socials] the weekly note of plan %s failed", plan.id)
                db.rollback()
    finally:
        db.close()
    return sent
