"""
ConversationSection — Message history formatting and trimming.

Priority 3. Does NOT contribute text to the system prompt (render() returns
empty string). Instead, provides ``format_messages()`` which the ContextService
calls to populate ``ContextResult.messages``.

Handles:
- Stripping system messages (we build our own)
- Converting ``parts`` format to plain text
- Trimming oldest messages when exceeding the token budget
"""

from __future__ import annotations

import logging
from typing import Optional

from core.attachment_refs import render_unresolved_file_part
from core.context_guard import count_tokens
from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

# Overhead per message in tokens (role + delimiters).
_MESSAGE_OVERHEAD_TOKENS = 4


class ConversationSection(BaseSection):
    """Message history formatting and budget-aware trimming.

    Messages are NOT part of the system prompt text — they go into
    ``ContextResult.messages``.  The ``render()`` method returns an empty
    string; callers should invoke ``format_messages()`` separately.
    """

    name: str = "conversation"
    priority: int = 3
    max_tokens: Optional[int] = None  # Dynamic — uses remaining budget

    async def render(self, ctx: SectionContext) -> str:
        """Messages don't contribute to the system prompt text."""
        return ""

    # ------------------------------------------------------------------
    # Public API — called by ContextService, not by render()
    # ------------------------------------------------------------------

    def format_messages(
        self,
        messages: list[dict] | None,
        budget_tokens: int | None = None,
        resolved_attachment_ids: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Format, filter, and trim messages for the LLM call.

        Steps:
            1. Strip system messages (we build our own system prompt).
            2. Convert ``parts`` format to plain text.
            3. Trim oldest messages if *budget_tokens* is exceeded.

        Args:
            resolved_attachment_ids: PRD-223 S0.4 — attachment ids the
                ContextService will inject via AttachmentResolver after this
                runs. Their file parts are left for the resolver to render.

        Returns:
            A new list of ``{"role": ..., "content": ...}`` dicts.
        """
        if not messages:
            return []

        try:
            formatted = self._convert(messages, resolved_attachment_ids)
            if budget_tokens and budget_tokens > 0:
                formatted = self._trim(formatted, budget_tokens)
            return formatted
        except Exception:
            logger.exception(
                "ConversationSection.format_messages failed — returning raw messages"
            )
            # Best-effort fallback: return non-system messages as-is
            return [
                m for m in messages
                if isinstance(m, dict) and m.get("role") != "system"
            ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert(
        messages: list[dict],
        resolved_attachment_ids: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """Strip system messages and normalise content to plain text."""
        result: list[dict[str, str]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role", "")
            if role == "system":
                continue

            content = msg.get("content", "")

            # Handle "parts" format (list of typed content blocks)
            if isinstance(content, list):
                content = _parts_to_text(content, resolved_attachment_ids)
            elif not isinstance(content, str):
                content = str(content) if content else ""

            result.append({"role": role, "content": content})
        return result

    @staticmethod
    def _trim(
        messages: list[dict[str, str]],
        budget_tokens: int,
    ) -> list[dict[str, str]]:
        """Drop oldest messages until total fits within *budget_tokens*.

        The most recent messages are preserved (trimming from the front).
        """
        total = sum(
            count_tokens(m.get("content", "")) + _MESSAGE_OVERHEAD_TOKENS
            for m in messages
        )

        if total <= budget_tokens:
            return messages

        # Walk from oldest to newest, dropping until within budget.
        trimmed: list[dict[str, str]] = list(messages)
        while trimmed and total > budget_tokens:
            dropped = trimmed.pop(0)
            total -= (
                count_tokens(dropped.get("content", ""))
                + _MESSAGE_OVERHEAD_TOKENS
            )

        return trimmed


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _parts_to_text(
    parts: list,
    resolved_attachment_ids: list[str] | None = None,
) -> str:
    """Convert a list of content parts to plain text.

    Handles:
    - ``{"type": "text", "text": "..."}``  → extract text
    - ``{"type": "file", "name": "..."}``  → marker, unless the
      AttachmentResolver is rendering that attachment into this prompt
      (PRD-223 S0.4 — see ``core.attachment_refs``)
    - Unknown types                        → skip
    """
    texts: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            if isinstance(part, str):
                texts.append(part)
            continue

        part_type = part.get("type", "text")
        if part_type == "text":
            text = part.get("text", "")
            if text:
                texts.append(text)
        elif part_type == "file":
            marker = render_unresolved_file_part(part, resolved_attachment_ids)
            if marker:
                texts.append(marker)
        # Skip image_url and other non-text parts
    return "\n".join(texts)
