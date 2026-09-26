"""PRD-251 D13: what a post and a workspace may spend on media, checked before any spend.

Footage and stills from the workspace's Composio generation toolkit (S1.8) are
priced before anything is submitted: the toolkit's own estimate where it prices
a call, else a configured ceiling per shot. The price must fit two caps:

* **the post's cap**, ``config.SOCIALS_MEDIA_POST_CAP_USD``: everything the
  post's renders have booked on the media lane (its footage, its premium voice)
  plus the price;
* **the workspace's monthly media cap**, the workspace setting
  ``socials.media_monthly_cap_usd`` (``modules/socials/settings.py``; the default
  is ``config.SOCIALS_MEDIA_MONTHLY_CAP_USD``): everything the workspace booked
  on the media lane this UTC calendar month, plus the price.

Over either cap nothing is submitted, and :class:`MediaCapExceeded` says which
cap, what was spent and what the work would cost. A monthly cap that cannot be
read spends nothing. The spend is what the media lane booked (``llm_usage``,
US-103): the same rows the workspace budget gate and the daily spend guard read.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func

from config import config
from core.llm.usage_context import LANE_MEDIA
from core.models.core import LLMUsage
from core.utils.timestamps import month_window_utc
from modules.socials.settings import media_monthly_cap_usd

# The post's execution id on the media lane: every booking a post's render makes.
POST_EXECUTION_PREFIX = "social_post:"
# Money is shown to the cent.
CENTS = 2


class MediaCapExceeded(Exception):
    """The work would take a post or the workspace over its media cap: nothing was submitted."""


def post_execution_id(post_id: Any) -> str:
    return f"{POST_EXECUTION_PREFIX}{post_id}"


def _naive(value: datetime) -> datetime:
    # llm_usage.created_at is naive UTC (the column default is now()): compare naive.
    return value.replace(tzinfo=None)


def post_spend_usd(db: Any, workspace_id: Any, post_id: Any) -> float:
    """Every dollar the post's renders booked on the media lane."""
    total = (
        db.query(func.coalesce(func.sum(LLMUsage.total_cost), 0.0))
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.request_type == LANE_MEDIA,
            LLMUsage.execution_id == post_execution_id(post_id),
        )
        .scalar()
    )
    return float(total or 0.0)


def month_spend_usd(db: Any, workspace_id: Any, start: datetime, end: datetime) -> float:
    """Every dollar the workspace booked on the media lane in ``[start, end)`` (UTC)."""
    total = (
        db.query(func.coalesce(func.sum(LLMUsage.total_cost), 0.0))
        .filter(
            LLMUsage.workspace_id == workspace_id,
            LLMUsage.request_type == LANE_MEDIA,
            LLMUsage.created_at >= _naive(start),
            LLMUsage.created_at < _naive(end),
        )
        .scalar()
    )
    return float(total or 0.0)


def _usd(amount: float) -> str:
    return f"${amount:.{CENTS}f}"


@dataclass(frozen=True)
class MediaSpend:
    """What a post and its workspace have spent on media, and their caps."""

    post_usd: float
    month_usd: float
    post_cap_usd: float
    monthly_cap_usd: float
    period_end: datetime
    # Why the monthly cap spends nothing: its stored value is not a number of dollars.
    problem: Optional[str] = None

    def refusal(self, price_usd: float, what: str) -> Optional[str]:
        """Why ``what``, priced at ``price_usd``, may not be submitted; ``None`` when it fits."""
        if self.problem:
            return f"{self.problem[:1].upper()}{self.problem[1:]}: nothing was submitted."
        if self.post_usd + price_usd > self.post_cap_usd:
            return (
                f"{what} would cost about {_usd(price_usd)}, and this post has spent {_usd(self.post_usd)} "
                f"of its {_usd(self.post_cap_usd)} media cap: nothing was submitted."
            )
        if self.month_usd + price_usd > self.monthly_cap_usd:
            return (
                f"{what} would cost about {_usd(price_usd)}, and this workspace has spent "
                f"{_usd(self.month_usd)} of its {_usd(self.monthly_cap_usd)} monthly media cap, which resets on "
                f"{self.period_end.day} {self.period_end:%B}: nothing was submitted."
            )
        return None

    def to_dict(self) -> dict:
        return {
            "month_usd": round(self.month_usd, CENTS),
            "monthly_cap_usd": self.monthly_cap_usd,
            "post_cap_usd": self.post_cap_usd,
            "period_end": self.period_end.isoformat(),
            "problem": self.problem,
        }


def media_spend(db: Any, workspace: Any, post_id: Any = None, now: Optional[datetime] = None) -> MediaSpend:
    """This month's reading for ``workspace``, and ``post_id``'s own spend when given."""
    start, end = month_window_utc(now)
    cap, problem = media_monthly_cap_usd(getattr(workspace, "settings", None))
    return MediaSpend(
        post_usd=post_spend_usd(db, workspace.id, post_id) if post_id is not None else 0.0,
        month_usd=month_spend_usd(db, workspace.id, start, end),
        post_cap_usd=float(config.SOCIALS_MEDIA_POST_CAP_USD),
        monthly_cap_usd=cap,
        period_end=end,
        problem=problem,
    )


def check_caps(spend: MediaSpend, price_usd: float, what: str) -> None:
    """:class:`MediaCapExceeded` unless ``what`` at ``price_usd`` fits both caps."""
    refusal = spend.refusal(price_usd, what)
    if refusal:
        raise MediaCapExceeded(refusal)
