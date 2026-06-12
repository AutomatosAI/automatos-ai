"""
AgentRosterSection — Available agents with capabilities for the coordinator.

Priority 3 (high, but can be trimmed before mission_context).
Renders workspace agents: id, name, skills, assigned tools, configured model.

Source: PRD-82A Section 12 Phase 3, PRD-102 Section 7.2
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class AgentRosterSection(BaseSection):
    """Renders available agents for the coordinator's system prompt.

    Reads agents from ``ctx.kwargs["roster_agents"]`` (list of Agent ORM
    objects or dicts).  If not provided, renders a fallback note.
    """

    name: str = "agent_roster"
    priority: int = 3
    max_tokens: Optional[int] = 6000

    async def render(self, ctx: SectionContext) -> str:
        """Build the agent roster block for the coordinator prompt."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("AgentRosterSection.render failed")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        # Roster UNKNOWN (caller supplied nothing) → render nothing, so modes
        # whose caller already presents a roster don't get a false "no agents"
        # claim. An explicit empty list still renders the honest fallback.
        if "roster_agents" not in ctx.kwargs:
            return ""

        agents = ctx.kwargs.get("roster_agents")
        if not agents:
            return (
                "## Agent Roster\n\n"
                "No agents available in this workspace."
            )

        # PRD-164 S1: optional {agent_id: avg verification score} map rendered
        # as a per-agent performance line (sourced from the agent_matcher
        # history map — no parallel performance computation).
        performance = ctx.kwargs.get("agent_performance") or {}

        parts: list[str] = ["## Agent Roster", ""]

        for agent in agents:
            agent_id = _attr_or_key(agent, "id", "?")
            name = _attr_or_key(agent, "name", "Unknown")
            agent_type = _attr_or_key(agent, "agent_type", "assistant")
            status = _attr_or_key(agent, "status", "unknown")
            description = _attr_or_key(agent, "description", None)

            # Model info from model_config JSON
            model_config = _attr_or_key(agent, "model_config", None) or {}
            model_id = (
                model_config.get("model_id", "default")
                if isinstance(model_config, dict)
                else "default"
            )

            # Skills
            skills_text = _render_skills(agent)

            # Tags (lightweight capability tags)
            tags = _attr_or_key(agent, "tags", None) or []
            tags_text = ", ".join(str(t) for t in tags) if tags else None

            # Build agent entry
            line = f"### {name} (#{agent_id})"
            parts.append(line)

            details: list[str] = [f"- **Type:** {agent_type}"]
            details.append(f"- **Status:** {status}")
            details.append(f"- **Model:** {model_id}")

            if description:
                # Truncate long descriptions
                desc = str(description)
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                details.append(f"- **Description:** {desc}")

            if skills_text:
                details.append(f"- **Skills:** {skills_text}")

            if tags_text:
                details.append(f"- **Tags:** {tags_text}")

            score = _performance_for(performance, agent_id)
            if score is not None:
                details.append(
                    f"- **Recent performance:** {score:.2f} avg verification score (30d)"
                )

            parts.extend(details)
            parts.append("")  # blank line between agents

        content = "\n".join(parts)
        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content


def _performance_for(performance: dict, agent_id: Any) -> Any:
    """Look up an agent's performance score tolerating int/str key drift."""
    if not performance:
        return None
    if agent_id in performance:
        return performance[agent_id]
    try:
        return performance.get(int(agent_id))
    except (TypeError, ValueError):
        return None


def _attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    """Read a value from an ORM object (attribute) or dict (key)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _render_skills(agent: Any) -> str:
    """Extract skill names from an agent record."""
    skills = _attr_or_key(agent, "skills", None)
    if not skills:
        return ""

    # ORM relationship: list of Skill objects
    if isinstance(skills, list):
        names: list[str] = []
        for skill in skills:
            name = _attr_or_key(skill, "name", None)
            if name:
                names.append(str(name))
        return ", ".join(names) if names else ""

    return ""
