"""
PlanningHistorySection — prior mission outcomes for the planner (PRD-164 S1).

Recalls top-k mission summaries + task failures (and retry recoveries) via the
PRD-159 memory recall path — ``UnifiedMemoryService.search_short_term`` for the
typed L2 records ``MissionMemoryService`` writes, plus ``search_long_term`` for
semantic L3 recall when Mem0 is configured. No direct table access, no parallel
recall path.

This section carries the learning demo: a seeded prior-mission failure must
reach the planning LLM so a new plan visibly avoids the failed approach.
Priority 2 — never dropped under budget pressure.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from modules.context.sections.base import BaseSection, SectionContext

logger = logging.getLogger(__name__)

# The typed L2 records MissionMemoryService writes (PRD-131d / PRD-159).
_MISSION_CONTENT_TYPES = ["mission_summary", "task_failure", "retry_recovery"]
_TOP_K = 5
_LOOKBACK_DAYS = 90
_SECTION_TOKEN_CAP = 3000
_ITEM_CHAR_CAP = 700


class PlanningHistorySection(BaseSection):
    """Mission summaries + task failures recalled for the planner."""

    name: str = "planning_history"
    priority: int = 2
    max_tokens: Optional[int] = _SECTION_TOKEN_CAP

    async def render(self, ctx: SectionContext) -> str:
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception(
                "PlanningHistorySection.render failed — planning continues without history"
            )
            return ""

    async def _build(self, ctx: SectionContext) -> str:
        goal = (ctx.task_description or "").strip()
        if not ctx.workspace_id:
            return ""

        from modules.memory.unified_memory_service import get_unified_memory_service

        unified = get_unified_memory_service()
        workspace_id = str(ctx.workspace_id)

        # --- L2 typed recall: recent mission summaries / task failures ---
        l2_items: List[dict] = []
        try:
            l2_items = await unified.search_short_term(
                workspace_id=workspace_id,
                query="",
                days=_LOOKBACK_DAYS,
                limit=_TOP_K * len(_MISSION_CONTENT_TYPES),
                content_types=_MISSION_CONTENT_TYPES,
            )
        except Exception:
            logger.warning(
                "PlanningHistorySection: L2 recall failed — continuing", exc_info=True
            )

        # --- L3 semantic recall on the goal (Mem0, when configured) ---
        l3_items: List[dict] = []
        if goal:
            try:
                raw = await unified.search_long_term(
                    workspace_id=workspace_id, query=goal, limit=_TOP_K
                )
                l3_items = [
                    m for m in raw or []
                    if _meta_type(m) in _MISSION_CONTENT_TYPES
                ]
            except Exception:
                logger.warning(
                    "PlanningHistorySection: L3 recall failed — continuing",
                    exc_info=True,
                )

        failures = [
            m for m in l2_items
            if m.get("content_type") in ("task_failure",)
            or _summary_outcome(m) == "failed"
        ]
        others = [m for m in l2_items if m not in failures]

        if not failures and not others and not l3_items:
            return ""

        parts: List[str] = ["### Learnings from prior missions"]

        if failures:
            parts.append("")
            parts.append(
                "Prior failures in this workspace — do NOT repeat these approaches:"
            )
            for m in failures[:_TOP_K]:
                parts.append(f"- {_excerpt(m.get('content'))}")

        if others:
            parts.append("")
            parts.append("Prior mission outcomes:")
            for m in others[:_TOP_K]:
                parts.append(f"- {_excerpt(m.get('content'))}")

        if l3_items:
            parts.append("")
            parts.append("Related long-term memories:")
            for m in l3_items[:_TOP_K]:
                text = m.get("memory") or m.get("content") or ""
                if text:
                    parts.append(f"- {_excerpt(text)}")

        return "\n".join(parts)


def _excerpt(text: Any, limit: int = _ITEM_CHAR_CAP) -> str:
    flat = " ".join(str(text or "").split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


def _meta_type(item: dict) -> Optional[str]:
    meta = item.get("metadata") if isinstance(item, dict) else None
    if isinstance(meta, dict):
        return meta.get("type")
    return None


def _summary_outcome(item: dict) -> Optional[str]:
    meta = item.get("metadata") if isinstance(item, dict) else None
    if isinstance(meta, dict):
        return meta.get("outcome")
    return None
