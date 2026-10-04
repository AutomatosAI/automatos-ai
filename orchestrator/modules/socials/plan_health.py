"""PRD-251C (C9, US-C407): a plan's health, each item with one click to act.

The Plan page and the weekly note carry what needs the owner, checked now:

* **bank_low**: the content bank has fewer unused topics than the plan's next two batches
  need (the slots of the next two weeks for a weekly plan, two months for a monthly one, two
  days for a daily one). Act: Research again.
* **channel_missing**: a cadence row names a channel the workspace has not connected. Act:
  connect it.
* **no_video**: no video goes out in the next seven days. Act: add a video row (Cadence).
* **cap_close**: this month's render minutes, or AI media spend, are at
  ``CAP_CLOSE_SHARE`` of their cap or more. Act: the AI tools and caps.

Each item says what is wrong and what one click does; nothing changes by itself. Reads only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

from modules.socials import batches, plans

CAP_CLOSE_SHARE = 0.8
WEEK_DAYS = 7
BATCHES_AHEAD = 2
DAYS_AHEAD = {plans.DAILY: BATCHES_AHEAD, plans.WEEKLY: BATCHES_AHEAD * WEEK_DAYS, plans.MONTHLY: plans.MAX_WINDOW_DAYS}
RESEARCH, CADENCE, CONNECT, AI_TOOLS = "research", "cadence", "connect", "ai_tools"
ACTION_LABELS = {RESEARCH: "Research again", CADENCE: "Add a video row", CONNECT: "Connect it in Composio", AI_TOOLS: "AI tools and caps"}


@dataclass(frozen=True)
class HealthItem:
    id: str
    title: str
    detail: str
    action: str

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "detail": self.detail, "action": {"kind": self.action, "label": ACTION_LABELS[self.action]}}


@dataclass(frozen=True)
class Readings:
    """What the checks read, gathered by the caller (``health_for``)."""

    unused_topics: int
    connected: Sequence[str]
    render_share: Optional[float]  # this month's render minutes used over the quota; None: no quota
    spend_share: Optional[float]  # this month's AI media spend over the cap; None: no cap


def _upcoming(plan: Any, now: datetime, days: int) -> List[plans.Slot]:
    return plans.expand_slots(plan, now, now + timedelta(days=days))


def _bank_low(plan: Any, now: datetime, unused: int) -> Optional[HealthItem]:
    needed = len(_upcoming(plan, now, DAYS_AHEAD.get(batches.rhythm_of(plan), BATCHES_AHEAD)))
    if unused >= needed:
        return None
    return HealthItem("bank_low", "The content bank is running low",
                      f"{unused} unused topic{'' if unused == 1 else 's'} for the {needed} posts of the next two batches.", RESEARCH)


def _channel_missing(plan: Any, connected: Iterable[str]) -> Optional[HealthItem]:
    named = sorted({channel for row in plans.cadence_rows(plan) for channel in row.get("channels") or ()})
    missing = [channel for channel in named if channel not in set(connected)]
    if not missing:
        return None
    return HealthItem("channel_missing", "A channel of the cadence is not connected",
                      f"{', '.join(missing)}: its posts are not made until it is connected.", CONNECT)


def _no_video(plan: Any, now: datetime) -> Optional[HealthItem]:
    if any(slot.format == plans.VIDEO for slot in _upcoming(plan, now, WEEK_DAYS)):
        return None
    return HealthItem("no_video", "No video this week", "Short videos reach the most people on most channels: add a video row.", CADENCE)


def _cap_close(readings: Readings) -> Optional[HealthItem]:
    close = [(name, share) for name, share in (("render minutes", readings.render_share), ("AI media spend", readings.spend_share))
             if share is not None and share >= CAP_CLOSE_SHARE]
    if not close:
        return None
    detail = "; ".join(f"{name}: {round(share * 100)}% of this month's cap used" for name, share in close)
    return HealthItem("cap_close", "A cap is close", f"{detail}.", AI_TOOLS)


def health(plan: Any, now: datetime, readings: Readings) -> List[HealthItem]:
    """The plan's health items, most pressing first; empty when all is well."""
    found = [
        _channel_missing(plan, readings.connected),
        _bank_low(plan, now, readings.unused_topics),
        _cap_close(readings),
        _no_video(plan, now),
    ]
    return [item for item in found if item is not None]


def _share(used: float, cap: Optional[float]) -> Optional[float]:
    return used / cap if cap else None


def health_for(db: Any, plan: Any, workspace: Any, now: datetime) -> List[HealthItem]:
    """The plan's health, its readings gathered from the workspace (the channels, the bank, the caps)."""
    from core import media_render_quota as render_quota
    from modules.socials import media_caps, plan_store
    from modules.socials.capabilities import social_channels

    quota = render_quota.render_quota(db, workspace, now)
    spend = media_caps.media_spend(db, workspace, now=now)
    readings = Readings(
        unused_topics=plan_store.bank_counts(db, [plan.id]).get(plan.id, {}).get("unused", 0),
        connected=[channel.toolkit for channel in social_channels(db, workspace.id)],
        render_share=_share(quota.used_minutes, quota.quota_minutes),
        spend_share=_share(spend.month_usd, spend.monthly_cap_usd),
    )
    return health(plan, now, readings)
