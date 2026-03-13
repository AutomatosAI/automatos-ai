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
        """Return a single-line UTC timestamp."""
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            return f"Current UTC time: {now}"
        except Exception:
            return ""
