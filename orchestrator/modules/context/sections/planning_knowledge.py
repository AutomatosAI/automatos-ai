"""
PlanningKnowledgeSection — RAG retrieval on the planning goal (PRD-164 S1).

Retrieves workspace knowledge relevant to the goal being planned THROUGH the
existing PRD-157 retrieval path: ``RAGService.retrieve`` derives its scope via
``build_retrieval_filters`` (the single fail-closed choke point) — this section
never queries the vector store or documents table directly, so there is no
parallel retrieval path to audit.

Output is the budgeter's numbered-citation context ``[1]..[n]`` so planners
can cite real documents in the plan.

F287 (night 8): the planner wrote "mention a last order date (Friday 27 November
from [2] is a good reference)" into mission #0352's plan against the owner's
"Thursday 10 December", and the email step used it; [2] was a document an agent had
written. Chunks from documents an agent wrote (``source_type`` agent_output) are
dropped before the plan sees them (services/draft_guides.owners_own_chunks).
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
        from services.draft_guides import owners_own_chunks

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
        # F287: the owner's documents only, never what an agent wrote.
        result = owners_own_chunks(ctx.db_session, result, ctx.workspace_id)

        if not result or not result.chunks:
            return ""

        return (
            "### Workspace knowledge (retrieved for this goal)\n\n"
            f"{result.formatted_context}"
        )
