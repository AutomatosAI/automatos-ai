"""
SkillsSection — trigger-based skill activation for the agent's prompt.

Priority 4. PRD-202 S2: the per-skill prompt-cost cut.

Each turn this renders, for the agent's attached skills:

  * **L1 metadata only** — ``name`` + ``description`` (~50-100 tokens/skill) —
    for every non-core skill, plus a one-line instruction that the model may
    pull a skill's full body when the task matches its description (via the
    ``load_skill`` tool). The body is NOT pre-paid.
  * the **full L2 body** ONLY for the small ``core`` always-on set
    (``config.SKILL_CORE_ALWAYS_ON`` — Auto's ``platform-management``), which is
    an agent's core operating manual, not an optional capability (Q4).

This replaces the old always-inject render (every attached skill's full body
every turn, uncapped-primary + a 5,000-token aux budget). That path — and the
aux budget — are **deleted**, not flagged: an attached-but-irrelevant skill no
longer taxes every turn with thousands of tokens.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

# Fail-safe default when config cannot be imported (e.g. isolated unit tests).
# The authoritative source is config.SKILL_CORE_ALWAYS_ON.
_DEFAULT_CORE_ALWAYS_ON = ("platform-management",)


class SkillsSection(BaseSection):
    """Trigger-based skill activation: L1 metadata always, L2 body on demand.

    Auto's ``platform-management`` (the ``core`` set) stays always-L2; every
    other attached skill contributes only its L1 metadata each turn and loads
    its body when the model calls ``load_skill`` (matched on the description).
    """

    name: str = "skills"
    priority: int = 4
    # No aux budget: non-core skills render L1 metadata only (tiny), so there is
    # nothing large to cap. Class-level max_tokens stays None (legacy callers).
    max_tokens: Optional[int] = None

    async def render(self, ctx: SectionContext) -> str:
        """Load and return skill content for the agent (never raises)."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("SkillsSection.render failed — skipping skill content")
            return ""

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        agent = ctx.agent
        if agent is None:
            return ""

        skills = getattr(agent, "skills", None)
        if not skills:
            return ""

        active_skills = [s for s in skills if getattr(s, "is_active", True)]
        if not active_skills:
            return ""

        active_skills = self._dedup(active_skills)
        core_names = self._core_always_on_names()

        core_bodies: list[str] = []
        l1_entries: list[str] = []
        activated_core: list[str] = []
        offered_l1: list[str] = []

        for s in active_skills:
            name = getattr(s, "name", None) or "skill"
            # Full L2 body renders ONLY for the core always-on set — never
            # unconditionally. Every other skill is L1 + load_skill trigger.
            if name in core_names:
                body = self._core_skill_body(s, ctx)
                if body:
                    core_bodies.append(body)
                    activated_core.append(name)
            else:
                l1_entries.append(self._l1_metadata_line(s))
                offered_l1.append(name)

        parts: list[str] = []
        if core_bodies:
            parts.append("\n\n---\n\n".join(core_bodies))
        if l1_entries:
            parts.append(self._render_l1_catalog(l1_entries))

        skill_tool_names = self._extract_skill_tool_names(active_skills)
        if skill_tool_names:
            parts.append(
                "## Using Your Skill Tools\n"
                f"You have access to: {', '.join(skill_tool_names)}\n"
                "When your task requires capabilities provided by these tools, "
                "you MUST use them via function calling. "
                "Analyze your task, check if any tools match, and CALL them — "
                "do not just describe what you would do."
            )

        if not parts:
            return ""

        # Skill-activation signal (S2 measurement): which skills stayed always-on
        # (core) vs were offered at L1 this turn. The cost delta / activation rate
        # (§7) is read from these.
        logger.info(
            "[skills] activation: core_always_on=%s l1_offered=%s (ws=%s)",
            activated_core, offered_l1, getattr(ctx, "workspace_id", None),
        )

        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup(active_skills: list) -> list:
        """PRD-191 S2 (held): never present the same skill twice.

        Dedup by id, then by (name, content_hash), keeping the first (=
        highest-priority — Agent.skills arrives ordered by the real
        agent_skills.priority) instance.
        """
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
        return deduped

    @staticmethod
    def _core_always_on_names() -> set:
        """The core always-L2 skill names (Q4). Config is authoritative."""
        try:
            from config import config

            names = getattr(config, "SKILL_CORE_ALWAYS_ON", None)
            if names:
                return set(names)
        except Exception:
            pass
        return set(_DEFAULT_CORE_ALWAYS_ON)

    @staticmethod
    def _l1_metadata_line(skill) -> str:
        """One L1 line: name + description (~50-100 tokens), NOT the body."""
        name = getattr(skill, "name", None) or "skill"
        desc = getattr(skill, "description", None)
        desc = desc.strip() if isinstance(desc, str) else ""
        return f"- **{name}**: {desc}" if desc else f"- **{name}**"

    @staticmethod
    def _render_l1_catalog(entries: list[str]) -> str:
        """The L1 catalog + the load_skill trigger instruction."""
        return (
            "## Available Skills\n"
            "These skills are attached to you. Only their names and descriptions "
            "are loaded now — NOT their full instructions. When your task matches "
            "a skill's description, call `load_skill(name=\"<skill-name>\")` to "
            "load that skill's full instructions for this turn.\n"
            + "\n".join(entries)
        )

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
    def _core_skill_body(skill, ctx: SectionContext) -> str:
        """Full L2 body for a CORE always-on skill only.

        Reads ``prompt_template`` (the SKILL.md body in the DB); falls back to
        the loader's ``load_skill_core`` only when the template is empty and a
        real session is available. Never called for non-core skills.
        """
        content = getattr(skill, "prompt_template", None)
        if content and str(content).strip():
            return str(content).strip()

        skill_name = getattr(skill, "name", None)
        if skill_name and getattr(ctx, "db_session", None) is not None:
            try:
                from modules.agents.services.skill_loader import get_skill_loader

                loader = get_skill_loader(ctx.db_session)
                loaded = loader.load_skill_core(skill_name, db=ctx.db_session)
                if loaded and str(loaded).strip():
                    return str(loaded).strip()
            except Exception:
                logger.warning(
                    "SkillLoader fallback failed for core skill %s",
                    skill_name,
                    exc_info=True,
                )
        return ""
