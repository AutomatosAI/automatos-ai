"""
IdentitySection — Agent name, role, persona, personality.

Priority 1 (never dropped). Renders agent name, role, persona,
personality, and response formatting guidance.
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

    When ``personality=True`` (CHATBOT mode), generates the full
    personality-aware prompt via ``AutomatosPersonality`` — including
    platform skill, tool guidance, action response style, and
    self-learning instructions.  Memory is NOT included here (handled
    by ``MemorySection``).
    """

    name: str = "identity"
    priority: int = 1
    max_tokens: Optional[int] = None

    async def render(self, ctx: SectionContext) -> str:  # noqa: C901
        """Build the identity block for the system prompt."""
        try:
            if ctx.kwargs.get("personality"):
                return self._build_chatbot_identity(ctx)
            return self._build(ctx)
        except Exception:
            logger.exception("IdentitySection.render failed — returning minimal identity")
            agent_name = getattr(ctx.agent, "name", "Agent") if ctx.agent else "Agent"
            return f"You are {agent_name}, an AI agent on the Automatos platform."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        """Basic identity for non-chatbot modes (task_execution, heartbeat, etc.)."""
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

        # Response formatting guidance
        parts.append(
            "\n## Response Formatting\n"
            "When you receive API/tool results:\n"
            "- Synthesize data into clear, human-friendly prose — do NOT dump raw JSON\n"
            "- Use bullet points or short paragraphs for a non-technical reader"
        )

        content = "\n".join(parts)
        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content

    def _build_chatbot_identity(self, ctx: SectionContext) -> str:
        """Full chatbot personality prompt via AutomatosPersonality.

        Replaces the call to ``get_happy_system_prompt()`` in the old
        chatbot path.  Includes base identity, platform skill, tool
        guidance, action response style, and self-learning — but NOT
        memory (``MemorySection`` handles that separately).

        PRD-137 Fix #3: also appends the agent's own description and
        persona, so non-Auto agents (e.g. Shopify widget) carry their
        identity in the prompt without needing a second injection in
        ``StreamingChatService``. This makes ``IdentitySection`` the
        single owner of identity content for all modes.
        """
        from consumers.chatbot.personality import (
            AutomatosPersonality,
            load_orchestrator_settings,
        )

        agent_name = ctx.kwargs.get("agent_name")
        if not agent_name:
            agent_name = getattr(ctx.agent, "name", "Agent") if ctx.agent else "Agent"

        user_name = ctx.kwargs.get("_user_name") or ctx.kwargs.get("user_name")
        msg_count = len(ctx.messages or [])

        orch_settings: Dict[str, Any] = ctx.kwargs.get("orchestrator_settings", {})
        if not orch_settings:
            try:
                orch_settings = load_orchestrator_settings(ctx.workspace_id)
            except Exception:
                logger.debug("Could not load orchestrator settings for %s", ctx.workspace_id)
                orch_settings = {}

        parts = [
            AutomatosPersonality.get_base_system_prompt(
                user_name=user_name,
                agent_name=agent_name,
                msg_count=msg_count,
                orchestrator_settings=orch_settings or None,
            ),
            AutomatosPersonality.get_platform_skill(),
            AutomatosPersonality.get_tool_guidance_prompt(has_tools=True),
            AutomatosPersonality.get_action_response_style(),
            AutomatosPersonality.get_anti_patterns(),
            AutomatosPersonality.get_self_learning_instruction(),
        ]

        # PRD-137 Fix #3: append agent's own description + persona text.
        agent = ctx.agent
        description = getattr(agent, "description", None) if agent else None
        if description and str(description).strip():
            parts.append(f"\n## Agent Description\n{str(description).strip()}")

        persona_text = self._get_persona_text(agent)
        if persona_text:
            parts.append(f"\n## Persona & Communication Style\n{persona_text}")

        # Execution policy — always on for chatbot mode
        parts.append(
            "\n## Execution Policy\n"
            "If the user requests multiple distinct tasks, you may call tools "
            "multiple times to complete ALL tasks before producing your final answer. "
            "Prefer data-gathering (read/list/fetch) steps before side-effect "
            "(send/post/create/update) steps. "
            "Only send/post after you have the final content to send."
        )

        # No max_tokens truncation — the full chatbot personality is essential
        return "\n".join(parts)

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
