"""
CustomSection — Workspace-level custom prompt sections from the DB.

Priority 9 (low — trimmed early when token budget is tight).

Queries the ``system_prompts`` / ``system_prompt_versions`` tables for
active prompts in the "custom" category and concatenates their content
into the system prompt.  If no custom prompts exist the section is
silently skipped.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class CustomSection(BaseSection):
    """Workspace-level custom prompt additions from the system_prompts table.

    Looks up active prompts whose category is ``"custom"`` and injects
    their latest active version content into the system prompt.

    All failures are caught — a missing table, empty results, or DB
    errors will never crash the prompt build.
    """

    name: str = "custom"
    priority: int = 9
    max_tokens: Optional[int] = 500

    async def render(self, ctx: SectionContext) -> str:
        """Return concatenated custom prompt content, or empty string."""
        try:
            return await self._load_custom_prompts(ctx)
        except Exception:
            logger.warning(
                "CustomSection.render failed — skipping custom prompts",
                exc_info=True,
            )
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _load_custom_prompts(self, ctx: SectionContext) -> str:
        """Query system_prompts for active custom-category prompts."""
        db = ctx.db_session
        if db is None:
            return ""

        try:
            from sqlalchemy import and_, select

            from core.models.system_prompts import (
                SystemPrompt,
                SystemPromptVersion,
            )

            # Find active prompts in the "custom" category with an active version
            stmt = (
                select(SystemPromptVersion.content)
                .join(SystemPrompt, SystemPromptVersion.prompt_id == SystemPrompt.id)
                .where(
                    and_(
                        SystemPrompt.category == "custom",
                        SystemPrompt.is_active.is_(True),
                        SystemPromptVersion.status == "active",
                    )
                )
                .order_by(SystemPrompt.slug)
            )

            # Support both sync and async sessions
            if hasattr(db, "execute"):
                result = db.execute(stmt)
            else:
                return ""

            rows = result.scalars().all()
            if not rows:
                return ""

            content = "\n\n".join(row for row in rows if row)

            if self.max_tokens and content:
                content = self.truncate(content, self.max_tokens)

            return content

        except Exception:
            logger.warning(
                "CustomSection: DB query for custom prompts failed",
                exc_info=True,
            )
            return ""
