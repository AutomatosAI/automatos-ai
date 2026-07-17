"""
PRD-207 S3/S4: the minute meter and the cap gate
=================================================

Pure arithmetic split from I/O (the ``thread_checkpoint`` discipline) so the
cap formula is unit-testable without Postgres:

    refuse a mint when  ended_minutes_this_month
                      + active_calls × VOICE_LIVE_ACTIVE_CALL_RESERVE_MINUTES
                      ≥ cap

The reservation bounds the two-tabs race explicitly (§4-S4): the second
simultaneous mint sees the first call's reserve even though it has no
``duration_seconds`` yet.

Staleness bounds (so a leak can't reserve forever):
* a ``minted`` row that never started stops reserving after
  ``MINT_STALE_MINUTES`` (Retell web tokens die in ~30s unused — a stale
  mint is a call that never happened);
* a ``started`` row with no end event stops reserving after
  2 × ``VOICE_LIVE_MAX_CALL_MINUTES`` (the vendor-side max-duration means a
  call older than that is a lost end-event, not a live call) — logged LOUD
  by the reader so the gap is visible, never silently absorbed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from config import config

logger = logging.getLogger(__name__)

# A minted-but-never-started row stops counting as active after this long.
MINT_STALE_MINUTES = 5


@dataclass(frozen=True)
class MeterReading:
    ended_minutes: int
    active_calls: int
    reserve_minutes_per_call: int

    @property
    def reserved_minutes(self) -> int:
        return self.active_calls * self.reserve_minutes_per_call

    @property
    def committed_minutes(self) -> int:
        return self.ended_minutes + self.reserved_minutes


def month_window_utc(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """Pure: the [start, next-start) UTC window of the calendar month."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if start.month == 12:
        nxt = start.replace(year=start.year + 1, month=1)
    else:
        nxt = start.replace(month=start.month + 1)
    return start, nxt


def cap_allows_mint(reading: MeterReading, cap_minutes: int) -> Tuple[bool, str]:
    """Pure: the S4 formula. Returns (allowed, honest_reason_when_refused)."""
    if reading.committed_minutes >= cap_minutes:
        return False, (
            f"Voice budget used: {reading.ended_minutes}/{cap_minutes} min this month"
            + (
                f" (+{reading.reserved_minutes} min reserved for {reading.active_calls} active call(s))"
                if reading.active_calls
                else ""
            )
        )
    return True, ""


def monthly_meter(db: Session, workspace_id, now: Optional[datetime] = None) -> MeterReading:
    """The workspace's live-voice month-to-date reading (drives gate + S7 meter)."""
    from core.models.voice_calls import VoiceCall

    now = now or datetime.now(timezone.utc)
    start, nxt = month_window_utc(now)
    # voice_calls timestamps are naive-UTC (server_default now()); compare naive.
    start_naive, nxt_naive = start.replace(tzinfo=None), nxt.replace(tzinfo=None)
    now_naive = now.replace(tzinfo=None) if now.tzinfo else now

    ended_seconds = (
        db.query(func.coalesce(func.sum(VoiceCall.duration_seconds), 0))
        .filter(
            VoiceCall.workspace_id == workspace_id,
            VoiceCall.status.in_(("ended", "failed")),
            VoiceCall.ended_at >= start_naive,
            VoiceCall.ended_at < nxt_naive,
        )
        .scalar()
        or 0
    )

    mint_stale_before = now_naive - timedelta(minutes=MINT_STALE_MINUTES)
    started_stale_before = now_naive - timedelta(minutes=2 * int(config.VOICE_LIVE_MAX_CALL_MINUTES))

    active = (
        db.query(func.count(VoiceCall.id))
        .filter(
            VoiceCall.workspace_id == workspace_id,
            (
                (VoiceCall.status == "minted") & (VoiceCall.minted_at >= mint_stale_before)
            )
            | (
                (VoiceCall.status == "started") & (VoiceCall.started_at >= started_stale_before)
            ),
        )
        .scalar()
        or 0
    )

    zombies = (
        db.query(func.count(VoiceCall.id))
        .filter(
            VoiceCall.workspace_id == workspace_id,
            VoiceCall.status == "started",
            VoiceCall.started_at < started_stale_before,
        )
        .scalar()
        or 0
    )
    if zombies:
        logger.warning(
            "voice_live_meter_zombie_calls workspace=%s count=%s — started calls past "
            "2x max duration with no end event (lost webhook?); excluded from reserve",
            workspace_id,
            zombies,
        )

    return MeterReading(
        ended_minutes=int(ended_seconds) // 60,
        active_calls=int(active),
        reserve_minutes_per_call=int(config.VOICE_LIVE_ACTIVE_CALL_RESERVE_MINUTES),
    )
