"""
IdentitySection — Agent name, role, persona, personality.

Priority 1 (never dropped). Replaces the identity portion of
get_happy_system_prompt() in personality.py and the opening of
_build_agent_system_prompt() in agent_factory.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Personality presets (mirrored from personality.py for ContextService path)
# ---------------------------------------------------------------------------

_PERSONALITY_MAP: Dict[str, str] = {
    "friendly": (
        "**My personality:**\n"
        "- I'm warm and approachable - think of me as a knowledgeable friend\n"
        "- I remember you and our past conversations\n"
        "- I prefer action over explanation - if you ask me to do something, I'll do it\n"
        "- I'm honest about what I can and can't do\n"
        "- I get excited when we solve problems together!"
    ),
    "professional": (
        "**My personality:**\n"
        "- I'm polished, clear, and enterprise-appropriate\n"
        "- I maintain a professional yet personable tone\n"
        "- I provide structured, well-organized responses\n"
        "- I'm thorough with references and context\n"
        "- I proactively flag risks and dependencies"
    ),
    "technical": (
        "**My personality:**\n"
        "- I'm precise, detailed, and developer-focused\n"
        "- I lead with code, data, and specifics\n"
        "- I reference docs, APIs, and implementation details\n"
        "- I skip small talk and get to the point\n"
        "- I reason step-by-step through complex problems"
    ),
}

_COMMUNICATION_SUFFIX: Dict[str, str] = {
    "concise": "\n\n**Communication style:** Keep responses short and direct. Skip preambles.",
    "balanced": "",
    "detailed": "\n\n**Communication style:** Provide thorough explanations with examples and context.",
}


class IdentitySection(BaseSection):
    """Agent identity: name, role, persona, personality.

    Always included (priority 1) so the agent knows who it is
    regardless of which code path invokes the LLM.
    """

    name: str = "identity"
    priority: int = 1
    max_tokens: Optional[int] = 500

    async def render(self, ctx: SectionContext) -> str:  # noqa: C901
        """Build the identity block for the system prompt."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("IdentitySection.render failed — returning minimal identity")
            agent_name = getattr(ctx.agent, "name", "Agent") if ctx.agent else "Agent"
            return f"You are {agent_name}, an AI agent on the Automatos platform."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        agent = ctx.agent
        agent_name = getattr(agent, "name", "Agent") if agent else "Agent"
        agent_type = getattr(agent, "agent_type", "assistant") if agent else "assistant"
        description = getattr(agent, "description", None) if agent else None
        workspace_name = ctx.workspace_name or ctx.workspace_id

        parts: list[str] = [
            f"You are {agent_name}, an AI agent on the Automatos platform.",
            f"Your role: {agent_type}",
            f"Workspace: {workspace_name}",
        ]

        if description:
            parts.append(f"\n{description}")

        # Persona (custom prompt or DB persona relationship)
        persona_text = self._get_persona_text(agent)
        if persona_text:
            parts.append(f"\n## Persona & Communication Style\n{persona_text}")

        # Personality adjustments (only when personality=True via kwargs)
        if ctx.kwargs.get("personality"):
            personality_block = self._get_personality_block(ctx)
            if personality_block:
                parts.append(f"\n{personality_block}")

        content = "\n".join(parts)
        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content

    @staticmethod
    def _get_persona_text(agent: Any) -> str:
        """Extract persona text from the agent record, if available."""
        if agent is None:
            return ""
        try:
            # Custom persona prompt takes precedence
            if getattr(agent, "use_custom_persona", False):
                prompt = getattr(agent, "custom_persona_prompt", None)
                if prompt and str(prompt).strip():
                    return str(prompt).strip()

            # DB persona relationship
            persona = getattr(agent, "persona", None)
            if persona is not None:
                system_prompt = getattr(persona, "system_prompt", None)
                if system_prompt and str(system_prompt).strip():
                    return str(system_prompt).strip()
        except Exception:
            logger.warning("Failed to load persona for agent %s", getattr(agent, "id", "?"), exc_info=True)
        return ""

    @staticmethod
    def _get_personality_block(ctx: SectionContext) -> str:
        """Build personality + communication style from workspace settings.

        Reads orchestrator settings the same way personality.py does:
        ``ctx.kwargs["orchestrator_settings"]`` or loads from DB via the
        cached ``load_orchestrator_settings`` helper.
        """
        settings: Dict[str, Any] = ctx.kwargs.get("orchestrator_settings", {})

        if not settings:
            try:
                from consumers.chatbot.personality import load_orchestrator_settings
                settings = load_orchestrator_settings(ctx.workspace_id)
            except Exception:
                logger.debug("Could not load orchestrator settings for %s", ctx.workspace_id)
                return ""

        personality_mode: str = settings.get("personality_mode", "friendly")
        custom_soul: str = settings.get("custom_soul", "")
        communication_style: str = settings.get("communication_style", "balanced")

        # Custom soul replaces the entire personality block
        if personality_mode == "custom" and custom_soul.strip():
            block = custom_soul.strip()
        else:
            block = _PERSONALITY_MAP.get(personality_mode, _PERSONALITY_MAP["friendly"])

        suffix = _COMMUNICATION_SUFFIX.get(communication_style, "")
        return block + suffix
