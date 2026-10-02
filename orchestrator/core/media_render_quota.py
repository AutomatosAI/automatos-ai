"""The monthly render quota, and the render minutes it counts (PRD-251 S1.1c).

Every plan gets Socials; plans differ only in render minutes a month (owner,
2026-09-23): Basic 10, Pro 60, Business 240, as ``render_minutes_month`` on
each tier in ``config.PLAN_TIERS``. Enterprise and the local edition have no
quota until the owner sets one.

The quota belongs to the renderer, not to one feature. A Socials post's render
(``modules/socials/render.py``) and ``generate_document`` with a social format
(``modules/documents/generation_service.py``) both reserve their seconds before
they call media-render, and both book what they rendered. It lives in core,
beside the media-render client, because those two feature modules may not
import each other (``orchestrator/.importlinter``).

* **The quota.** The workspace's ``plan_limits['render_minutes_month']`` (plan
  assignment writes it, ``services/plan_tiers.plan_limits_for_tier``), or, for
  a workspace assigned before the key existed, its tier's value. An unknown or
  missing plan counts as the entry tier. ``0``, ``None`` or a negative number is
  no quota, as for ``max_agents``. The local edition never has one.
* **Minutes used.** The month's render units on the ``media`` lane (US-103):
  every finished render books its rendered seconds (:func:`book_render_seconds`),
  rounded up, as units at $0 under the ``media_render`` provider. The month is
  the UTC calendar month.
* **Minutes held (P251W1-RVW-3).** A render reserves its composition's declared
  duration (:func:`declared_seconds`) before anything reaches media-render: a
  ``reserved`` row on the same lane (:func:`reserve_render`). It counts until the
  render ends, when it is deleted (:func:`release_render`): done, once the
  render's own seconds are booked; failed, refused or timed out. One that nothing
  releases stops counting when its render's deadline has passed
  (:func:`reservation_lifetime`), and the boot reaper deletes it
  (:func:`release_expired_reservations`). The check and the reservation are one
  step under one lock per workspace, in this process and across every worker
  process (a Postgres advisory lock), so renders requested together each see
  the minutes the others hold.
* **The refusal.** A render starts only while the minutes used plus the minutes
  held are under the quota, and it is refused BEFORE anything reaches
  media-render (:class:`RenderQuotaExceeded`, HTTP 429). The render that crosses
  the line finishes and counts in full: one render, however many are requested
  at once. A still has no duration: it reserves and books no seconds, and is
  refused only once the month's minutes are gone.
"""
from __future__ import annotations

import asyncio
import logging
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from sqlalchemy import func, or_, text
from sqlalchemy.orm import Session

from config import config
from core.llm.providers import MEDIA_RENDER_PROVIDER
from core.llm.usage_context import LANE_MEDIA, usage_scope
from core.llm.usage_tracker import UsageTracker
from core.models.core import LLMUsage
from core.social_templates import SOCIAL_IMAGE, root_duration
from core.utils.timestamps import month_window_utc, utc_iso
from services.plan_tiers import get_tier

logger = logging.getLogger(__name__)

QUOTA_KEY = "render_minutes_month"
# A workspace whose plan names no tier counts as the entry tier (as exposure_for_plan does).
ENTRY_TIER = "basic"
SECONDS_PER_MINUTE = 60
# How the tab shows minutes: one decimal place.
MINUTES_DECIMALS = 1
# The booking's model id: the renderer's engine (core/llm/providers.py MEDIA_RENDER_PROVIDER).
RENDER_MODEL_ID = "hyperframes"
# A render in flight holds its seconds as an llm_usage row of this status; the
# tier is the one its booking gets (UsageTracker.track_media).
RESERVED_STATUS = "reserved"
RESERVATION_TIER = "direct"
# A reservation nothing released counts this long past SOCIALS_RENDER_MAX_WAIT_SECONDS:
# the render's own deadline starts a moment after it reserves, and it books after it.
RESERVATION_GRACE_SECONDS = 60
# The workspace's quota lock in the database, in a key space of its own ('rndq').
QUOTA_LOCK_NAMESPACE = 0x726E6471
_QUOTA_LOCK = text("SELECT pg_advisory_xact_lock(:namespace, hashtext(:key))")
# Within one process, a workspace's renders queue here before they take a connection.
_PROCESS_LOCKS: Dict[str, threading.Lock] = {}


class RenderQuotaExceeded(Exception):
    """The workspace has used its render minutes for this month; the message says so."""


def _quota_minutes(value: Any, source: str) -> Optional[float]:
    """A quota value → minutes, or ``None`` for no quota."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        logger.warning("[MediaRender] %s %s=%r is not a number of minutes; no render quota applies", source, QUOTA_KEY, value)
        return None
    return float(value) if value > 0 else None


def quota_minutes_for(workspace: Any) -> Optional[float]:
    """The workspace's monthly render quota in minutes, or ``None`` for none."""
    if (config.AUTH_EDITION or "").strip().lower() == "local":
        return None
    limits = getattr(workspace, "plan_limits", None) or {}
    if isinstance(limits, dict) and QUOTA_KEY in limits:
        return _quota_minutes(limits[QUOTA_KEY], "plan_limits")
    plan = getattr(workspace, "plan", None) or ENTRY_TIER
    tier = get_tier(plan) or get_tier(ENTRY_TIER) or {}
    return _quota_minutes(tier.get(QUOTA_KEY), f"plan tier {plan!r}")


def _naive_utc(moment: Optional[datetime]) -> datetime:
    """``moment`` (now by default) as llm_usage.created_at holds it: naive UTC."""
    moment = moment or datetime.now(timezone.utc)
    return moment.astimezone(timezone.utc).replace(tzinfo=None) if moment.tzinfo else moment


def reservation_lifetime() -> timedelta:
    """How long a reservation nothing released keeps counting: its render's deadline, and the grace."""
    return timedelta(seconds=config.SOCIALS_RENDER_MAX_WAIT_SECONDS + RESERVATION_GRACE_SECONDS)


def render_seconds_used(db: Any, workspace_id: Any, start: datetime, end: datetime) -> int:
    """Rendered seconds booked on the media lane in ``[start, end)`` (UTC); never a reservation."""
    # llm_usage.created_at is naive UTC (the column default is now()): compare naive.
    start_naive, end_naive = start.replace(tzinfo=None), end.replace(tzinfo=None)
    total = (
        db.query(func.coalesce(func.sum(LLMUsage.input_tokens), 0))
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.request_type == LANE_MEDIA,
            LLMUsage.provider == MEDIA_RENDER_PROVIDER,
            or_(LLMUsage.status.is_(None), LLMUsage.status != RESERVED_STATUS),
            LLMUsage.created_at >= start_naive,
            LLMUsage.created_at < end_naive,
        )
        .scalar()
    )
    return int(total or 0)


def render_seconds_reserved(db: Any, workspace_id: Any, now: Optional[datetime] = None) -> int:
    """Seconds the workspace's renders in progress hold: reservations still within their render's deadline."""
    since = _naive_utc(now) - reservation_lifetime()
    total = (
        db.query(func.coalesce(func.sum(LLMUsage.input_tokens), 0))
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.request_type == LANE_MEDIA,
            LLMUsage.provider == MEDIA_RENDER_PROVIDER,
            LLMUsage.status == RESERVED_STATUS,
            LLMUsage.created_at >= since,
        )
        .scalar()
    )
    return int(total or 0)


@dataclass(frozen=True)
class RenderQuota:
    used_seconds: int
    quota_minutes: Optional[float]
    period_start: datetime
    period_end: datetime
    plan_label: Optional[str] = None
    # Held by renders in progress (P251W1-RVW-3): counted, not yet booked.
    reserved_seconds: int = 0

    @property
    def used_minutes(self) -> float:
        return self.used_seconds / SECONDS_PER_MINUTE

    @property
    def reserved_minutes(self) -> float:
        return self.reserved_seconds / SECONDS_PER_MINUTE

    @property
    def exhausted(self) -> bool:
        """No render may start: the minutes used and the minutes held reach the quota."""
        held = self.used_seconds + self.reserved_seconds
        return self.quota_minutes is not None and held >= self.quota_minutes * SECONDS_PER_MINUTE

    def refusal(self) -> str:
        """What a refused render says: the plan, the minutes used and held, and when they come back."""
        plan = f" on the {self.plan_label} plan" if self.plan_label else ""
        resumes = f"{self.period_end.day} {self.period_end:%B}"
        used = (
            f"This workspace has used {self.used_minutes:.{MINUTES_DECIMALS}f} of its "
            f"{self.quota_minutes:g} render minutes this month{plan}"
        )
        if not self.reserved_seconds:
            return f"{used}. Rendering resumes on {resumes}."
        return (
            f"{used}, and renders in progress hold {self.reserved_minutes:.{MINUTES_DECIMALS}f} more until "
            f"they finish. Rendering resumes on {resumes}, or sooner if they use less."
        )

    def to_dict(self) -> Dict[str, Any]:
        remaining = None
        if self.quota_minutes is not None:
            remaining = max(0.0, self.quota_minutes - self.used_minutes - self.reserved_minutes)
        return {
            "used_minutes": round(self.used_minutes, MINUTES_DECIMALS),
            "used_seconds": self.used_seconds,
            "reserved_minutes": round(self.reserved_minutes, MINUTES_DECIMALS),
            "reserved_seconds": self.reserved_seconds,
            "quota_minutes": self.quota_minutes,
            "remaining_minutes": round(remaining, MINUTES_DECIMALS) if remaining is not None else None,
            "exhausted": self.exhausted,
            "period_start": utc_iso(self.period_start),
            "period_end": utc_iso(self.period_end),
        }


def _plan_label(workspace: Any) -> Optional[str]:
    if (config.AUTH_EDITION or "").strip().lower() == "local":
        return None
    tier = get_tier(getattr(workspace, "plan", None) or ENTRY_TIER) or {}
    return tier.get("display_name")


@dataclass(frozen=True)
class _Holder:
    """What the quota reads of a workspace, taken on the caller's thread."""

    workspace_id: Any
    quota_minutes: Optional[float]
    plan_label: Optional[str]


def _holder(workspace: Any) -> _Holder:
    return _Holder(workspace.id, quota_minutes_for(workspace), _plan_label(workspace))


def _reading(db: Any, holder: _Holder, now: Optional[datetime]) -> RenderQuota:
    start, end = month_window_utc(now)
    return RenderQuota(
        used_seconds=render_seconds_used(db, holder.workspace_id, start, end),
        quota_minutes=holder.quota_minutes,
        period_start=start,
        period_end=end,
        plan_label=holder.plan_label,
        reserved_seconds=render_seconds_reserved(db, holder.workspace_id, now),
    )


def render_quota(db: Any, workspace: Any, now: Optional[datetime] = None) -> RenderQuota:
    """This month's reading: seconds used, seconds held by renders in progress, and the quota."""
    return _reading(db, _holder(workspace), now)


# ── reserving a render's seconds (P251W1-RVW-3) ─────────────────────────────
def declared_seconds(blocks: Optional[Mapping[str, Any]], fmt: Optional[str]) -> int:
    """The seconds a render of ``blocks`` (a social template of format ``fmt``) reserves.

    A video: its composition's declared duration (the root's ``data-duration``),
    rounded up as its booking is, and at most the longest media-render renders
    (``SOCIALS_RENDER_MAX_DURATION_SECONDS``), which is also what a duration that
    cannot be read reserves. A still: none, as it books none (a still counting
    toward the plan's minutes is an open owner call).
    """
    if fmt == SOCIAL_IMAGE:
        return 0
    longest = int(config.SOCIALS_RENDER_MAX_DURATION_SECONDS)
    html = blocks.get("html") if isinstance(blocks, Mapping) else None
    duration = root_duration(html) if isinstance(html, str) else None
    if duration is None or duration <= 0:
        return longest
    return min(math.ceil(duration), longest)


@dataclass(frozen=True)
class RenderReservation:
    """The seconds a render in progress holds against its workspace's quota.

    ``row_id`` is its ``reserved`` row in llm_usage; ``None`` holds nothing (no
    quota applies, or a still, which books no seconds).
    """

    workspace_id: Any
    seconds: int = 0
    row_id: Optional[int] = None


def sessions_for(db: Any) -> Callable[[], Any]:
    """Sessions of their own on the database ``db`` is bound to, opened only when
    one is needed: a reservation commits alone, never inside the caller's
    transaction, and a render with nothing to hold never touches ``db``."""
    return lambda: Session(bind=db.get_bind())


def _process_lock(workspace_id: Any) -> threading.Lock:
    return _PROCESS_LOCKS.setdefault(str(workspace_id), threading.Lock())


def _lock_quota(db: Any, workspace_id: Any) -> None:
    """Hold the workspace's quota lock until this transaction ends, across every
    worker process. A database without advisory locks (SQLite, in the unit tests)
    has one process: the process lock is the lock there."""
    if db.get_bind().dialect.name == "postgresql":
        db.execute(_QUOTA_LOCK, {"namespace": QUOTA_LOCK_NAMESPACE, "key": str(workspace_id)})


def _reservation_row(workspace_id: Any, seconds: int, execution_id: str, now: Optional[datetime]) -> LLMUsage:
    return LLMUsage(
        workspace_id=workspace_id,
        model_id=RENDER_MODEL_ID,
        provider=MEDIA_RENDER_PROVIDER,
        tier=RESERVATION_TIER,
        execution_id=execution_id,
        request_type=LANE_MEDIA,
        input_tokens=seconds,
        output_tokens=0,
        total_tokens=seconds,
        cache_read_tokens=0,
        cache_write_tokens=0,
        input_cost=0.0,
        output_cost=0.0,
        total_cost=0.0,
        is_byok=False,
        status=RESERVED_STATUS,
        created_at=_naive_utc(now),
    )


def _reserve(
    session_factory: Callable[[], Any], holder: _Holder, seconds: int, execution_id: str, now: Optional[datetime]
) -> RenderReservation:
    with _process_lock(holder.workspace_id):
        db = session_factory()
        try:
            _lock_quota(db, holder.workspace_id)
            reading = _reading(db, holder, now)
            if reading.exhausted:
                logger.info(
                    "[MediaRender] render refused for workspace %s: %ss used and %ss held of %s min",
                    holder.workspace_id, reading.used_seconds, reading.reserved_seconds, reading.quota_minutes,
                )
                raise RenderQuotaExceeded(reading.refusal())
            if seconds <= 0:
                return RenderReservation(holder.workspace_id)
            row = _reservation_row(holder.workspace_id, seconds, execution_id, now)
            db.add(row)
            db.flush()
            row_id = row.id
            db.commit()
            return RenderReservation(holder.workspace_id, seconds, row_id)
        finally:
            db.close()  # a check that reserved nothing rolls back here, and the lock goes with it


async def reserve_render(
    session_factory: Callable[[], Any],
    workspace: Any,
    seconds: int,
    *,
    execution_id: str,
    now: Optional[datetime] = None,
) -> RenderReservation:
    """Check the month's quota and hold ``seconds`` for a render about to start, as one step.

    :class:`RenderQuotaExceeded` when the minutes used and the minutes held by
    renders in progress leave none. The reservation commits in a session of its
    own from ``session_factory``, off the event loop (the lock may wait); give it
    back with :func:`release_render` when the render ends. No quota: nothing held.
    """
    holder = _holder(workspace)
    if holder.quota_minutes is None:
        return RenderReservation(holder.workspace_id)
    return await asyncio.to_thread(_reserve, session_factory, holder, max(0, int(seconds)), execution_id, now)


def _release(session_factory: Callable[[], Any], reservation: RenderReservation) -> bool:
    db = session_factory()
    try:
        db.query(LLMUsage).filter(
            LLMUsage.id == reservation.row_id, LLMUsage.status == RESERVED_STATUS
        ).delete(synchronize_session=False)
        db.commit()
        return True
    except Exception:  # noqa: BLE001 — the render has ended: its hold expires with its deadline instead
        logger.exception(
            "[MediaRender] releasing reservation %s of workspace %s failed; it counts until its render's deadline",
            reservation.row_id, reservation.workspace_id,
        )
        return False
    finally:
        db.close()


async def release_render(session_factory: Callable[[], Any], reservation: Optional[RenderReservation]) -> bool:
    """Give back what a render held, once it has ended (and booked what it
    rendered); ``True`` when nothing is left held. A release that fails is
    logged, and the reservation stops counting at its render's deadline."""
    if reservation is None or reservation.row_id is None:
        return True
    return await asyncio.to_thread(_release, session_factory, reservation)


def release_expired_reservations(db: Any, now: Optional[datetime] = None) -> int:
    """Delete the reservations whose render's deadline has passed (the boot
    reaper): their render ended with a process that died. They count no longer;
    this keeps them out of llm_usage. The caller commits; the number deleted."""
    expired_before = _naive_utc(now) - reservation_lifetime()
    rows = (
        db.query(LLMUsage)
        .filter(
            LLMUsage.request_type == LANE_MEDIA,
            LLMUsage.provider == MEDIA_RENDER_PROVIDER,
            LLMUsage.status == RESERVED_STATUS,
            LLMUsage.created_at < expired_before,
        )
        .all()
    )
    for row in rows:
        db.delete(row)
    return len(rows)


def book_render_seconds(*, workspace_id: Any, execution_id: str, seconds: float, latency_ms: int) -> None:
    """Book a finished render's seconds on the media lane at $0: the units the quota counts.

    Call it through ``asyncio.to_thread`` from a coroutine: the tracker then
    writes inline, before the render returns, where on the event loop it would
    hand the row to the best-effort threads, which drop it when the pool is full
    (P251W1-RVW-3).
    """
    with usage_scope(request_type=LANE_MEDIA, execution_id=execution_id, workspace_id=workspace_id):
        UsageTracker.track_media(
            provider=MEDIA_RENDER_PROVIDER, model_id=RENDER_MODEL_ID, units=seconds, usd=0.0, latency_ms=latency_ms
        )
