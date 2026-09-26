"""The monthly render quota, and the render minutes it counts (PRD-251 S1.1c).

Every plan gets Socials; plans differ only in render minutes a month (owner,
2026-09-23): Basic 10, Pro 60, Business 240, as ``render_minutes_month`` on
each tier in ``config.PLAN_TIERS``. Enterprise and the local edition have no
quota until the owner sets one.

The quota belongs to the renderer, not to one feature. A Socials post's render
(``modules/socials/render.py``) and ``generate_document`` with a social format
(``modules/documents/generation_service.py``) both check it before they call
media-render, and both book what they rendered. It lives in core, beside the
media-render client, because those two feature modules may not import each
other (``orchestrator/.importlinter``).

* **The quota.** The workspace's ``plan_limits['render_minutes_month']`` (plan
  assignment writes it, ``services/plan_tiers.plan_limits_for_tier``), or, for
  a workspace assigned before the key existed, its tier's value. An unknown or
  missing plan counts as the entry tier. ``0``, ``None`` or a negative number is
  no quota, as for ``max_agents``. The local edition never has one.
* **Minutes used.** The month's render units on the ``media`` lane (US-103):
  every finished render books its rendered seconds (:func:`book_render_seconds`),
  rounded up, as units at $0 under the ``media_render`` provider. The month is
  the UTC calendar month.
* **The refusal.** A render starts only while the month has minutes left, and
  it is refused BEFORE anything reaches media-render (:class:`RenderQuotaExceeded`,
  HTTP 429). The render that crosses the line finishes and counts in full.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import func

from config import config
from core.llm.providers import MEDIA_RENDER_PROVIDER
from core.llm.usage_context import LANE_MEDIA, usage_scope
from core.llm.usage_tracker import UsageTracker
from core.models.core import LLMUsage
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


def render_seconds_used(db: Any, workspace_id: Any, start: datetime, end: datetime) -> int:
    """Rendered seconds booked on the media lane in ``[start, end)`` (UTC)."""
    # llm_usage.created_at is naive UTC (the column default is now()): compare naive.
    start_naive, end_naive = start.replace(tzinfo=None), end.replace(tzinfo=None)
    total = (
        db.query(func.coalesce(func.sum(LLMUsage.input_tokens), 0))
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.request_type == LANE_MEDIA,
            LLMUsage.provider == MEDIA_RENDER_PROVIDER,
            LLMUsage.created_at >= start_naive,
            LLMUsage.created_at < end_naive,
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

    @property
    def used_minutes(self) -> float:
        return self.used_seconds / SECONDS_PER_MINUTE

    @property
    def exhausted(self) -> bool:
        return self.quota_minutes is not None and self.used_seconds >= self.quota_minutes * SECONDS_PER_MINUTE

    def refusal(self) -> str:
        """What a refused render says: the plan, the minutes and when they come back."""
        plan = f" on the {self.plan_label} plan" if self.plan_label else ""
        resumes = f"{self.period_end.day} {self.period_end:%B}"
        return (
            f"This workspace has used {self.used_minutes:.{MINUTES_DECIMALS}f} of its "
            f"{self.quota_minutes:g} render minutes this month{plan}. Rendering resumes on {resumes}."
        )

    def to_dict(self) -> Dict[str, Any]:
        remaining = None
        if self.quota_minutes is not None:
            remaining = max(0.0, self.quota_minutes - self.used_minutes)
        return {
            "used_minutes": round(self.used_minutes, MINUTES_DECIMALS),
            "used_seconds": self.used_seconds,
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


def render_quota(db: Any, workspace: Any, now: Optional[datetime] = None) -> RenderQuota:
    """This month's reading: seconds used and the quota."""
    start, end = month_window_utc(now)
    return RenderQuota(
        used_seconds=render_seconds_used(db, workspace.id, start, end),
        quota_minutes=quota_minutes_for(workspace),
        period_start=start,
        period_end=end,
        plan_label=_plan_label(workspace),
    )


def enforce_render_quota(db: Any, workspace: Any, now: Optional[datetime] = None) -> RenderQuota:
    """:class:`RenderQuotaExceeded` when the month's minutes are used up."""
    reading = render_quota(db, workspace, now)
    if reading.exhausted:
        logger.info(
            "[MediaRender] render refused for workspace %s: %ss used of %s min",
            workspace.id, reading.used_seconds, reading.quota_minutes,
        )
        raise RenderQuotaExceeded(reading.refusal())
    return reading


def book_render_seconds(*, workspace_id: Any, execution_id: str, seconds: float, latency_ms: int) -> None:
    """Book a finished render's seconds on the media lane at $0: the units the quota counts."""
    with usage_scope(request_type=LANE_MEDIA, execution_id=execution_id, workspace_id=workspace_id):
        UsageTracker.track_media(
            provider=MEDIA_RENDER_PROVIDER, model_id=RENDER_MODEL_ID, units=seconds, usd=0.0, latency_ms=latency_ms
        )
