"""
DatetimeContextSection — today's date and the time, for temporal awareness.

Priority 2 (never dropped). F323 (night 9b): it was priority 8, the first section
the budget drops, and said only "Current UTC time: …Z"; the Business Analyst read
"this summer" as 2024 in October 2026 (#1984). It now leads with today's date in
words in the workspace's zone (``services.todays_date``), then the UTC time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class DatetimeContextSection(BaseSection):
    """Injects today's date (the workspace's zone) and the UTC time into the system prompt.

    Allows agents to reason about time-sensitive queries (scheduling,
    deadlines, "what day is it", "this summer") without relying on the LLM's
    training-data cutoff.
    """

    name: str = "datetime_context"
    priority: int = 2
    max_tokens: Optional[int] = 80

    async def render(self, ctx: SectionContext) -> str:
        """Return today's date in words and the UTC time, both to the hour.

        F025: this line used to carry SECONDS. It sits inside the prompt prefix
        the provider caches, so a value that changes every turn made the whole
        prefix unmatchable — a ~34k-token re-read on the first call of every
        turn, for a precision nothing downstream uses. The model needs to know
        the date and roughly the time of day; to the hour it is byte-stable
        across a whole hour of turns and the prefix survives.
        """
        from services.todays_date import today_line

        try:
            now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
            today = today_line(ctx.db_session, ctx.workspace_id, now)
            return f"{today} Current UTC time: {now.strftime('%Y-%m-%dT%H:00Z')} (to the hour)"
        except Exception:
            logger.exception("DatetimeContextSection.render failed; the prompt goes without the date")
            return ""
