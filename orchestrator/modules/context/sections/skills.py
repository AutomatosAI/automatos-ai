"""
SkillsSection — SKILL.md content for the agent's assigned skill.

Priority 4. Replaces skill injection in agent_factory.py's
_build_agent_system_prompt() and ensures heartbeat/recipe paths
also get consistent skill content.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class SkillsSection(BaseSection):
    """SKILL.md content for the agent's assigned skill(s).

    Loads the ``prompt_template`` field from the agent's active skills
    (via the ``agent_skills`` many-to-many relationship on the Agent model).
    """

    name: str = "skills"
    priority: int = 4
    max_tokens: Optional[int] = 3000

    async def render(self, ctx: SectionContext) -> str:
        """Load and return skill content for the agent."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("SkillsSection.render failed — skipping skill content")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        agent = ctx.agent
        if agent is None:
            return ""

        # Agent.skills is a relationship loaded via agent_skills association table
        skills = getattr(agent, "skills", None)
        if not skills:
            return ""

        # Filter to active skills only
        active_skills = [s for s in skills if getattr(s, "is_active", True)]
        if not active_skills:
            return ""

        parts: list[str] = []
        for skill in active_skills:
            content = self._get_skill_content(skill, ctx)
            if content:
                parts.append(content)

        if not parts:
            return ""

        # Join multiple skills with a separator
        combined = "\n\n---\n\n".join(parts) if len(parts) > 1 else parts[0]

        if self.max_tokens:
            combined = self.truncate(combined, self.max_tokens)
        return combined

    @staticmethod
    def _get_skill_content(skill, ctx: SectionContext) -> str:
        """Extract skill content from the skill record.

        Tries ``prompt_template`` first (the SKILL.md body stored in DB).
        Falls back to loading via SkillLoader if available and prompt_template
        is empty.
        """
        # Primary: prompt_template field on the Skill model
        content = getattr(skill, "prompt_template", None)
        if content and str(content).strip():
            return str(content).strip()

        # Fallback: use SkillLoader singleton if available
        skill_name = getattr(skill, "name", None)
        if skill_name and ctx.db_session is not None:
            try:
                from modules.agents.services.skill_loader import get_skill_loader

                loader = get_skill_loader(ctx.db_session)
                loaded = loader.load_skill_core(skill_name, db=ctx.db_session)
                if loaded and str(loaded).strip():
                    return str(loaded).strip()
            except Exception:
                logger.warning(
                    "SkillLoader fallback failed for skill %s",
                    skill_name,
                    exc_info=True,
                )

        return ""
