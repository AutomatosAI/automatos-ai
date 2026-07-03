"""
FieldMemorySection — workspace-scoped field / durable-memory digest.

The read-half of F021 (PRD-179 S1). Heartbeat agents were memory-blind by
design and the planning pack consulted documents + the knowledge graph but never
the field, so patterns a mission accumulates never reached the next mission's
planning (the compounding claim the OS review §12.6 flags as roadmap).

This section reads the WORKSPACE-persistent field — the collection Wave 8's
field-to-durable promotion writes into — via
``VectorFieldSharedContext.query_workspace`` and renders it through the ONE
existing digest builder (``field_scoring.budget_results`` +
``field_scoring.format_digest``, the same pipeline ``_attach_field_digest`` uses
for dispatch). No second builder, no second ranking.

Included in HEARTBEAT_AGENT and PLANNING modes only. Priority 7 (just after the
user-memory section at 6): dropped before task-critical sections under budget
pressure, but ahead of the knowledge-graph excerpt.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from modules.context.factory import get_shared_context
from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)


class FieldMemorySection(BaseSection):
    """Workspace-scoped accumulated-knowledge digest for planning + heartbeat.

    Reads the workspace-persistent field and renders the top patterns as the
    same ``## Field memory`` block agents already see at dispatch, so planners
    and recurring agents inherit what earlier missions learned.
    """

    name: str = "field_memory"
    priority: int = 7
    max_tokens: Optional[int] = None

    def __init__(self) -> None:
        super().__init__()
        from config import config

        # Reuse the field digest budget — the same cap the dispatch digest uses.
        self.max_tokens = config.FIELD_QUERY_TOKEN_BUDGET

    async def render(self, ctx: SectionContext) -> str:
        """Return the workspace field digest, or '' on any failure/emptiness.

        Never raises — a memory read must not crash a prompt build.
        """
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception(
                "FieldMemorySection.render failed — skipping field digest"
            )
            return ""

    async def _build(self, ctx: SectionContext) -> str:
        query = self._extract_query(ctx)
        if not query:
            return ""

        rows = await self._query_workspace_field(str(ctx.workspace_id), query)
        if not rows:
            return ""

        # Render through the ONE shared digest builder (PRD-166 S2/S3): trim to
        # the field budget, then format the same block dispatch pins.
        from modules.context import field_scoring
        from config import config

        kept, truncated = field_scoring.budget_results(
            rows, config.FIELD_QUERY_TOKEN_BUDGET
        )
        if not kept:
            return ""
        return field_scoring.format_digest(kept, truncated=truncated)

    @staticmethod
    async def _query_workspace_field(
        workspace_id: str, query: str
    ) -> List[dict[str, Any]]:
        """Workspace-scoped field read via the shared-context backend.

        Reaches the ``query_workspace`` method on the inner adapter through the
        instrumentation wrapper (the ``getattr(field, '_inner', field)`` idiom
        the coordinator uses). Returns [] when the backend is unavailable so the
        section degrades to nothing rather than failing the build.
        """
        from config import config

        field = get_shared_context()
        if field is None:
            return []
        inner = getattr(field, "_inner", field)
        query_workspace = getattr(inner, "query_workspace", None)
        if query_workspace is None:
            return []
        results = await query_workspace(
            workspace_id=workspace_id,
            query=query,
            top_k=config.FIELD_QUERY_TOP_K,
        )
        return list(results or [])

    @staticmethod
    def _extract_query(ctx: SectionContext) -> str:
        """Query the field with the latest user message, else the goal / task.

        In PLANNING mode there are no messages — ``task_description`` carries the
        goal, which is exactly what a planner should retrieve accumulated
        knowledge against.
        """
        if ctx.messages:
            for msg in reversed(ctx.messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
        return ctx.task_description or ""
