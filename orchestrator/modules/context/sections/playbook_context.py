"""
PlaybookContextSection — Playbook step name, instructions, previous results.

Priority 2 (never dropped). Provides the agent with its current playbook
step so it knows what to execute during multi-step playbook workflows.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class PlaybookContextSection(BaseSection):
    """Playbook step instructions for multi-step workflows.

    Included in RECIPE mode. Priority 2 ensures it's never dropped by
    the budget manager — the agent must know which step it's executing.
    """

    name: str = "playbook_context"
    priority: int = 2
    max_tokens: Optional[int] = 2000

    async def render(self, ctx: SectionContext) -> str:
        """Build the playbook context block for the system prompt."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("PlaybookContextSection.render failed")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        recipe_step = ctx.recipe_step
        if not recipe_step:
            return ""

        # Extract fields from the recipe_step dict.
        # Callers should pass: name, step_number, total_steps,
        # instructions, previous_output (all optional except instructions).
        name = recipe_step.get("name", "")
        step_number = recipe_step.get("step_number")
        total_steps = recipe_step.get("total_steps")
        instructions = recipe_step.get("instructions", "")
        previous_output = recipe_step.get("previous_output", "")

        if not instructions and not name:
            return ""

        # Header
        parts: list[str] = []
        if name:
            parts.append(f"## Playbook: {name}")
        else:
            parts.append("## Playbook Step")

        # Step progress line
        if step_number is not None and total_steps is not None:
            step_label = recipe_step.get("step_name", "")
            if step_label:
                parts.append(
                    f"\n### Current Step: {step_number}/{total_steps}"
                    f" — {step_label}"
                )
            else:
                parts.append(
                    f"\n### Current Step: {step_number}/{total_steps}"
                )
        elif step_number is not None:
            parts.append(f"\n### Step {step_number}")

        # Instructions body
        if instructions:
            parts.append(f"\n{instructions}")

        # Previous step results (truncated to fit budget)
        if previous_output:
            previous_section = f"\n### Previous Step Results:\n{previous_output}"
            # Reserve tokens for everything above the previous output
            header_content = "\n".join(parts)
            header_tokens = self.estimate_tokens(header_content)
            if self.max_tokens and header_tokens < self.max_tokens:
                remaining = self.max_tokens - header_tokens
                previous_section = self.truncate(previous_section, remaining)
            parts.append(previous_section)

        content = "\n".join(parts)
        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content
