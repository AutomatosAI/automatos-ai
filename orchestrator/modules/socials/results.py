"""PRD-251C (C7): a post's numbers, from its channels' latest readings.

The results job keeps each published target's numbers at 1 and 7 days
(``social_post_stats``, ``services/socials_results.py``). A post's numbers are each of its
channels' latest reading (the 7-day one once taken), and their sum across the channels for
the numbers they gave; its engagement is everyone who acted on it
(``result_reads.engagement``). Read by the Posted view (US-C408), the weekly note (US-C403),
the proposals (US-C404) and research (US-C405). Another workspace's numbers are never read.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import UUID

from core.models.socials import SocialPostStat, SocialPostTarget
from modules.socials import result_reads


@dataclass(frozen=True)
class PostNumbers:
    """A post's numbers: summed across its channels, and each channel's own."""

    numbers: Mapping[str, int]
    by_channel: Mapping[str, Mapping[str, int]]
    reading: int  # the latest reading taken, in days after it went out
    read_at: Optional[datetime]

    @property
    def engagement(self) -> int:
        return result_reads.engagement(self.numbers)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numbers": dict(self.numbers),
            "by_channel": {toolkit: dict(numbers) for toolkit, numbers in self.by_channel.items()},
            "engagement": self.engagement,
            "reading": self.reading,
            "read_at": self.read_at.isoformat() if self.read_at else None,
        }


def _summed(rows: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for numbers in rows:
        for key, value in numbers.items():
            if isinstance(value, int) and not isinstance(value, bool):
                total[key] = total.get(key, 0) + value
    return total


def post_numbers(db: Any, workspace_id: UUID, post_ids: Iterable[UUID]) -> Dict[UUID, PostNumbers]:
    """Each post's numbers (the module docstring); a post nothing was read for is absent."""
    ids = list(dict.fromkeys(post_ids))
    if not ids:
        return {}
    rows = (
        db.query(SocialPostStat, SocialPostTarget.toolkit)
        .join(SocialPostTarget, SocialPostTarget.id == SocialPostStat.target_id)
        .filter(SocialPostStat.workspace_id == workspace_id, SocialPostStat.post_id.in_(ids))
        .all()
    )
    latest: Dict[UUID, Any] = {}
    for stat, toolkit in rows:
        kept = latest.get(stat.target_id)
        if kept is None or stat.reading > kept[0].reading:
            latest[stat.target_id] = (stat, toolkit)
    by_post: Dict[UUID, list] = {}
    for stat, toolkit in latest.values():
        by_post.setdefault(stat.post_id, []).append((stat, toolkit))
    return {post_id: _numbers_of(items) for post_id, items in by_post.items()}


def _numbers_of(items: list) -> PostNumbers:
    channels: Dict[str, list] = {}
    for stat, toolkit in items:
        channels.setdefault(toolkit, []).append(dict(stat.numbers or {}))
    by_channel = {toolkit: _summed(rows) for toolkit, rows in channels.items()}
    stats = [stat for stat, _ in items]
    return PostNumbers(
        numbers=_summed(by_channel.values()), by_channel=by_channel,
        reading=max(stat.reading for stat in stats), read_at=max((stat.read_at for stat in stats if stat.read_at), default=None),
    )
