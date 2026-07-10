"""
SkillsSection — SKILL.md content for the agent's assigned skill.

Priority 4. Renders the agent's assigned SKILL.md content for all
execution paths (chatbot, task, heartbeat, recipe).
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

    PRD-137 Fix #5: the primary skill (highest-priority active skill) is
    rendered uncapped. Auxiliary skills share an aux budget. This stops
    Auto's 11K-token platform-management SKILL.md being truncated to 3K.
    """

    name: str = "skills"
    priority: int = 4
    # Primary skill is uncapped. Auxiliary skills share this budget.
    aux_max_tokens: int = 5000
    # Class-level max_tokens kept None so legacy callers don't truncate.
    max_tokens: Optional[int] = None

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

        # PRD-191 S4 (closes F054): Agent.skills arrives ordered by the REAL
        # attachment-level priority (agent_skills.priority DESC — model
        # order_by). The old sort keyed on a phantom Skill.priority attribute
        # that no column backed, so the uncapped-primary slot was load order.
        # PRD-191 S2: never render the same body twice — dedup by id, then by
        # (name, content_hash), keeping the first (= highest-priority) instance.
        seen_ids = set()
        seen_bodies = set()
        deduped = []
        for s in active_skills:
            sid = getattr(s, "id", None)
            body_key = (getattr(s, "name", None), getattr(s, "content_hash", None))
            if sid is not None and sid in seen_ids:
                continue
            if body_key != (None, None) and body_key in seen_bodies:
                continue
            if sid is not None:
                seen_ids.add(sid)
            seen_bodies.add(body_key)
            deduped.append(s)
        active_skills = deduped

        primary_text = self._get_skill_content(active_skills[0], ctx)
        aux_texts = [
            txt for txt in (self._get_skill_content(s, ctx) for s in active_skills[1:])
            if txt
        ]

        if not primary_text and not aux_texts:
            return ""

        aux_combined = "\n\n---\n\n".join(aux_texts) if aux_texts else ""
        if aux_combined and self.aux_max_tokens:
            aux_combined = self.truncate(aux_combined, self.aux_max_tokens)

        if primary_text and aux_combined:
            combined = primary_text + "\n\n---\n\n" + aux_combined
        else:
            combined = primary_text or aux_combined

        # Skill tool usage instructions
        skill_tool_names = self._extract_skill_tool_names(active_skills)
        if skill_tool_names:
            combined += (
                "\n\n## Using Your Skill Tools\n"
                f"You have access to: {', '.join(skill_tool_names)}\n"
                "When your task requires capabilities provided by these tools, "
                "you MUST use them via function calling. "
                "Analyze your task, check if any tools match, and CALL them — "
                "do not just describe what you would do."
            )

        return combined

    @staticmethod
    def _extract_skill_tool_names(skills) -> list[str]:
        """Extract tool names from skill tools_schema fields."""
        names: list[str] = []
        for skill in skills:
            schema = getattr(skill, "tools_schema", None)
            if not schema or not isinstance(schema, dict):
                continue
            for tool_def in schema.get("tools", []):
                name = tool_def.get("name")
                if name:
                    names.append(name)
        return names

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
