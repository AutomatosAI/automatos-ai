"""
DatetimeContextSection — Current UTC datetime for temporal awareness.

Priority 8 (low — nice-to-have, trimmed before most other sections).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext


class DatetimeContextSection(BaseSection):
    """Injects the current UTC timestamp into the system prompt.

    Allows agents to reason about time-sensitive queries (scheduling,
    deadlines, "what day is it", etc.) without relying on the LLM's
    training-data cutoff.
    """

    name: str = "datetime_context"
    priority: int = 8
    max_tokens: Optional[int] = 50

    async def render(self, ctx: SectionContext) -> str:
        """Return a single-line UTC timestamp, rounded to the hour.

        F025: this line used to carry SECONDS. It sits inside the prompt prefix
        the provider caches, so a value that changes every turn made the whole
        prefix unmatchable — a ~34k-token re-read on the first call of every
        turn, for a precision nothing downstream uses. The model needs to know
        the date and roughly the time of day; to the hour it is byte-stable
        across a whole hour of turns and the prefix survives.
        """
        try:
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            return f"Current UTC time: {now.strftime('%Y-%m-%dT%H:00Z')} (to the hour)"
        except Exception:
            return ""
