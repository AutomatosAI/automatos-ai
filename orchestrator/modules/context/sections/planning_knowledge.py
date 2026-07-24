"""
PlanningKnowledgeSection — RAG retrieval on the planning goal (PRD-164 S1).

Retrieves workspace knowledge relevant to the goal being planned THROUGH the
existing PRD-157 retrieval path: ``RAGService.retrieve`` derives its scope via
``build_retrieval_filters`` (the single fail-closed choke point) — this section
never queries the vector store or documents table directly, so there is no
parallel retrieval path to audit.

Output is the budgeter's numbered-citation context ``[1]..[n]`` so planners
can cite real documents in the plan.
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

# Bounded retrieval for planning: enough to ground a plan, small enough for
# every planner (chat classification included) to afford.
_MAX_CHUNKS = 6
_SECTION_TOKEN_CAP = 4000


class PlanningKnowledgeSection(BaseSection):
    """Workspace knowledge retrieved for the goal under plan."""

    name: str = "planning_knowledge"
    priority: int = 3
    max_tokens: Optional[int] = _SECTION_TOKEN_CAP

    async def render(self, ctx: SectionContext) -> str:
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception(
                "PlanningKnowledgeSection.render failed — planning continues without RAG"
            )
            return ""

    async def _build(self, ctx: SectionContext) -> str:
        goal = (ctx.task_description or "").strip()
        if not goal or not ctx.workspace_id:
            return ""

        from modules.rag.service import get_rag_service

        rag = get_rag_service()
        # PRD-157 path: retrieve() resolves scope via build_retrieval_filters
        # (fail-closed) and applies the token budgeter + numbered citations.
        result = await rag.retrieve(
            query=goal,
            max_chunks=_MAX_CHUNKS,
            max_tokens=self.max_tokens,
            context_type="planning",
            workspace_id=str(ctx.workspace_id),
            team=ctx.kwargs.get("team"),
        )

        if not result or not result.chunks:
            return ""

        return (
            "### Workspace knowledge (retrieved for this goal)\n\n"
            f"{result.formatted_context}"
        )
