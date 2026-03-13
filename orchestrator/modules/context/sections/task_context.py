"""
TaskContextSection — Task description, status, priority, board context.

Priority 2 (never dropped). Provides the agent with its current task
assignment so it knows what to work on during task execution mode.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class TaskContextSection(BaseSection):
    """Current task description and board context.

    Included in TASK_EXECUTION mode so the agent knows what it's
    working on. Priority 2 ensures it's never dropped by the budget
    manager.
    """

    name: str = "task_context"
    priority: int = 2
    max_tokens: Optional[int] = 1500

    async def render(self, ctx: SectionContext) -> str:
        """Build the task context block for the system prompt."""
        try:
            return self._build(ctx)
        except Exception:
            logger.exception("TaskContextSection.render failed")
            return ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build(self, ctx: SectionContext) -> str:
        task_description = ctx.task_description
        if not task_description:
            return ""

        parts: list[str] = [
            "## Current Task",
            "",
            str(task_description),
        ]

        # Optional structured metadata from kwargs
        task_status = ctx.kwargs.get("task_status")
        task_priority = ctx.kwargs.get("task_priority")
        board_name = ctx.kwargs.get("board_name")

        metadata_lines: list[str] = []
        if task_status:
            metadata_lines.append(f"Status: {task_status}")
        if task_priority:
            metadata_lines.append(f"Priority: {task_priority}")
        if board_name:
            metadata_lines.append(f"Board: {board_name}")

        if metadata_lines:
            parts.append("")
            parts.extend(metadata_lines)

        # Dependency context instructions
        parts.append("")
        parts.append("## Working with Context and Dependencies")
        parts.append("")
        parts.append("When you receive '## DEPENDENCY CONTEXT' at the beginning of your task:")
        parts.append("1. This contains outputs from previous tasks that you need to use")
        parts.append("2. Read and understand all the context provided")
        parts.append("3. For compilation/report tasks: Synthesize the information into a coherent document")
        parts.append("4. For document generation tasks: Transform the input into the requested format")
        parts.append("")
        parts.append("When your task involves writing/creating documents:")
        parts.append("- Use the write_file tool to save your output")
        parts.append("- The task description will specify the output filename")
        parts.append("- Actually WRITE the content, don't just describe what you would write")

        content = "\n".join(parts)
        if self.max_tokens:
            content = self.truncate(content, self.max_tokens)
        return content
