"""PRD-251C (C7, US-C402): each published post's numbers, read 1 and 7 days after it went out.

The scheduler's leader runs :func:`run_reads` every ``SOCIALS_RESULTS_TICK_SECONDS``. A target
is due a reading (``result_reads.READINGS``) once that many days have passed since it went out,
for ``SOCIALS_RESULTS_READ_WINDOW_DAYS``, when nothing is kept for that reading yet; at most
``SOCIALS_RESULTS_MAX_READS_PER_TICK`` a tick, the oldest first. Only a published target with
its id on the platform is read, and only through its channel's read action
(``modules/socials/result_reads.py``), as the platform, in the post's own workspace:

* a workspace with Socials off, or a trial on the hosted edition (PRD-222: no background
  burn), is not read;
* a channel the registry says the workspace cannot read now (not connected, its action not
  synced, or deny-listed) is skipped, and tried again at the next tick while the window lasts;
* what the platform answers is kept in ``social_post_stats``, one row per target and
  reading: only the numbers it gave. A read it refuses is kept with none (logged), so a
  reading is never taken twice.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import anyio

from config import config
from core.models.socials import SocialPost, SocialPostStat, SocialPostTarget
from core.models.workspaces import Workspace
from modules.socials import result_reads
from modules.socials.capabilities import PUBLISHED_TARGET, runnable_actions
from modules.socials.recipes.toolkit import PLATFORM_AGENT_ID, error_of, output_of
from services.socials_plan_research import background_allowed

logger = logging.getLogger(__name__)

RESULTS_JOB_ID = "socials_results_tick"
NO_BACKGROUND_READS = "the workspace takes no background reads (Socials off, or a hosted trial)"


@dataclass(frozen=True)
class DueRead:
    """One reading of one target, captured off the session."""

    target_id: Any
    post_id: Any
    workspace_id: Any
    toolkit: str
    post_kind: str
    remote_id: str
    reading: int


def _session():
    from core.database.database import SessionLocal

    return SessionLocal()


def _utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def _readings_due(published_at: datetime, now: datetime, taken: set) -> List[int]:
    window = timedelta(days=int(config.SOCIALS_RESULTS_READ_WINDOW_DAYS))
    age = now - _utc(published_at)
    return [days for days in result_reads.READINGS if timedelta(days=days) <= age < timedelta(days=days) + window and days not in taken]


def due_reads(db: Any, now: datetime) -> List[DueRead]:
    """The readings due now (the module docstring), oldest first, at most a tick's worth."""
    window = timedelta(days=int(config.SOCIALS_RESULTS_READ_WINDOW_DAYS))
    rows = (
        db.query(SocialPostTarget, SocialPost.workspace_id)
        .join(SocialPost, SocialPost.id == SocialPostTarget.post_id)
        .filter(
            SocialPostTarget.status == PUBLISHED_TARGET,
            SocialPostTarget.remote_id.isnot(None),
            SocialPostTarget.toolkit.in_(list(result_reads.READS)),
            SocialPostTarget.published_at.isnot(None),
            SocialPostTarget.published_at >= now - timedelta(days=max(result_reads.READINGS)) - window,
            SocialPostTarget.published_at <= now - timedelta(days=min(result_reads.READINGS)),
        )
        .order_by(SocialPostTarget.published_at)
        .all()
    )
    taken: Dict[Any, set] = {}
    for target_id, reading in db.query(SocialPostStat.target_id, SocialPostStat.reading).filter(
        SocialPostStat.target_id.in_([target.id for target, _ in rows])
    ):
        taken.setdefault(target_id, set()).add(reading)
    due = [
        DueRead(target.id, target.post_id, workspace_id, target.toolkit, target.post_kind, target.remote_id, reading)
        for target, workspace_id in rows
        for reading in _readings_due(target.published_at, now, taken.get(target.id, set()))
    ]
    return due[: int(config.SOCIALS_RESULTS_MAX_READS_PER_TICK)]


async def _read(executor: Any, due: DueRead) -> Tuple[Dict[str, int], Optional[str]]:
    """The numbers one reading gives, or none and why the platform refused."""
    read = result_reads.READS[due.toolkit]
    result = await executor.execute_with_uploads(
        read.action, result_reads.params_for(read, due.remote_id, due.post_kind), agent_id=PLATFORM_AGENT_ID,
        workspace_id=due.workspace_id, app_name=due.toolkit.upper(), upload_params=(), way_through=None,
    )
    data = result.get("data") if isinstance(result.get("data"), dict) else {}
    if not result.get("success") or data.get("successful") is False:
        return {}, error_of({"error": result.get("error") or data.get("error")})
    return result_reads.numbers_from(read, output_of(result)), None


def _keep(due: DueRead, numbers: Dict[str, int], now: datetime) -> None:
    db = _session()
    try:
        db.add(SocialPostStat(
            workspace_id=due.workspace_id, post_id=due.post_id, target_id=due.target_id, reading=due.reading, read_at=now,
            numbers=numbers, source_action=result_reads.READS[due.toolkit].action,
        ))
        db.commit()
    finally:
        db.close()


def _readable(db: Any, workspace_id: Any) -> Dict[str, Optional[str]]:
    """Per toolkit, why the workspace cannot read it now (``None``: it can)."""
    actions = {toolkit: read.action for toolkit, read in result_reads.READS.items()}
    if not background_allowed(db.get(Workspace, workspace_id)):
        return {toolkit: NO_BACKGROUND_READS for toolkit in actions}
    return runnable_actions(db, workspace_id, actions)


def plan_reads(now: datetime) -> Tuple[List[DueRead], Dict[Any, Dict[str, Optional[str]]]]:
    """The readings due now, and per workspace of theirs what it can read (on its own session)."""
    db = _session()
    try:
        due = due_reads(db, now)
        return due, {workspace_id: _readable(db, workspace_id) for workspace_id in {item.workspace_id for item in due}}
    finally:
        db.close()


async def run_reads(now: Optional[datetime] = None, executor: Optional[Any] = None) -> Dict[str, int]:
    """One pass: every due reading taken and kept, or skipped; the counts by outcome. The
    database work runs in worker threads, each Composio call on a session of its own."""
    from modules.socials.publishing import SessionPerCall

    now = now or datetime.now(timezone.utc)
    counts = {"read": 0, "refused": 0, "skipped": 0}
    try:
        due, readable = await anyio.to_thread.run_sync(plan_reads, now)
        executor = executor or SessionPerCall(_session)
        for item in due:
            if readable.get(item.workspace_id, {}).get(item.toolkit, NO_BACKGROUND_READS) is not None:
                counts["skipped"] += 1
                continue
            numbers, refusal = await _read(executor, item)
            if refusal:
                logger.warning("[Socials] the %d-day read of target %s was refused: %s", item.reading, item.target_id, refusal)
            await anyio.to_thread.run_sync(_keep, item, numbers, now)
            counts["refused" if refusal else "read"] += 1
    except Exception:  # noqa: BLE001 — the next tick tries again; logged
        logger.exception("[Socials] the results tick failed")
    return counts


def register(scheduler: Any) -> bool:
    """The reads on the leader's scheduler; ``False`` when this worker hosts none."""
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler.add_job(
        run_reads, IntervalTrigger(seconds=int(config.SOCIALS_RESULTS_TICK_SECONDS)), id=RESULTS_JOB_ID,
        replace_existing=True, max_instances=1, coalesce=True,
    )
    logger.info("[Socials] posts' numbers are read every %ds", int(config.SOCIALS_RESULTS_TICK_SECONDS))
    return True
