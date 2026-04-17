"""
Mission Memory Service — PRD-131d Phase 1
==========================================

Stores mission (OrchestrationRun) outcomes into memory so future missions can
learn from what worked and what failed.

Writes two places via UnifiedMemoryService:
- L3 long-term (Mem0) workspace namespace — conversational summary for fact
  extraction and semantic retrieval
- L2 short-term (Postgres) with content_type="mission_summary", importance=0.7
  (completed) or 0.8 (failed) so failures promote to L3 faster
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.orchestration import OrchestrationRun, OrchestrationTask, TaskState
from modules.memory.unified_memory_service import (
    MemoryNamespace,
    get_unified_memory_service,
)

logger = logging.getLogger(__name__)


class MissionMemoryService:
    """Persist mission outcomes to L3 + L2 memory."""

    def __init__(self, db: Session):
        if db is None:
            raise ValueError("MissionMemoryService requires an injected DB session")
        self.db = db
        self._unified = get_unified_memory_service()

    async def store_mission_summary(
        self,
        run_id: UUID,
        outcome: str,
        failure_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store a mission summary in L3 (Mem0) + L2 (Postgres short-term).

        Args:
            run_id: OrchestrationRun.id
            outcome: "completed" or "failed"
            failure_reason: required when outcome == "failed" — the reason text
                            that gets stored verbatim so future retrieval can
                            surface "we tried X, it failed because Y".

        Returns:
            Dict with stored flags and any error strings.
        """
        run = self.db.query(OrchestrationRun).get(run_id)
        if run is None:
            raise ValueError(f"Mission run not found: {run_id}")

        workspace_id = str(run.workspace_id)
        narrative = self._build_narrative(run, outcome, failure_reason)

        errors: List[str] = []
        l3_ok = False
        l2_id: Optional[str] = None

        # L3 — Mem0 long-term, workspace-wide namespace
        try:
            ns = MemoryNamespace(workspace_id=workspace_id)
            l3_result = await self._unified.store_long_term_messages(
                user_id=ns.workspace(),
                messages=[
                    {
                        "role": "user",
                        "content": f"Mission: {(run.goal or '')[:500]}",
                    },
                    {
                        "role": "assistant",
                        "content": narrative[:6000],
                    },
                ],
                metadata={
                    "type": "mission_summary",
                    "run_id": str(run.id),
                    "outcome": outcome,
                    "workspace_id": workspace_id,
                },
            )
            l3_ok = not l3_result.get("error") if isinstance(l3_result, dict) else True
        except Exception as e:
            errors.append(f"L3 store failed: {e}")
            logger.warning(
                "[MissionMemory] L3 store failed for run %s", run.id, exc_info=True
            )

        # L2 — Postgres short-term, importance 0.7 (completed) / 0.8 (failed)
        importance = 0.8 if outcome == "failed" else 0.7
        try:
            l2_id = await self._unified.store_short_term(
                workspace_id=workspace_id,
                content=narrative[:4000],
                content_type="mission_summary",
                importance=importance,
                metadata={
                    "run_id": str(run.id),
                    "outcome": outcome,
                    "goal": (run.goal or "")[:500],
                    "failure_reason": (failure_reason or "")[:500] if failure_reason else None,
                },
            )
        except Exception as e:
            errors.append(f"L2 store failed: {e}")
            logger.warning(
                "[MissionMemory] L2 store failed for run %s", run.id, exc_info=True
            )

        result = {
            "run_id": str(run.id),
            "outcome": outcome,
            "l3_stored": l3_ok,
            "l2_row_id": l2_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }
        if errors:
            logger.warning("[MissionMemory] Partial store for run %s: %s", run.id, errors)
        else:
            logger.info(
                "[MissionMemory] Stored run=%s outcome=%s l2=%s l3=%s",
                run.id, outcome, l2_id, l3_ok,
            )
        return result

    # ------------------------------------------------------------------
    # Narrative builder
    # ------------------------------------------------------------------

    def _build_narrative(
        self,
        run: OrchestrationRun,
        outcome: str,
        failure_reason: Optional[str],
    ) -> str:
        """Build a conversational narrative of the mission outcome."""
        tasks: List[OrchestrationTask] = (
            self.db.query(OrchestrationTask)
            .filter(OrchestrationTask.run_id == run.id)
            .order_by(OrchestrationTask.sequence_number)
            .all()
        )

        verified = [t for t in tasks if TaskState(t.state) == TaskState.VERIFIED]
        failed = [t for t in tasks if TaskState(t.state) == TaskState.FAILED]

        summary = run.output_summary or {}
        duration_s = summary.get("total_duration_seconds")
        if duration_s is None and run.started_at and run.completed_at:
            duration_s = int((run.completed_at - run.started_at).total_seconds())

        lines: List[str] = []
        lines.append(f"Mission {outcome}: {(run.goal or 'Untitled mission')[:300]}")

        stats: List[str] = []
        stats.append(f"{len(verified)} tasks verified")
        if failed:
            stats.append(f"{len(failed)} failed")
        if duration_s is not None:
            stats.append(f"{duration_s}s duration")
        if run.tokens_used:
            stats.append(f"{run.tokens_used} tokens used")
        lines.append("Stats: " + ", ".join(stats))

        if outcome == "failed" and failure_reason:
            lines.append(f"Failure reason: {failure_reason[:500]}")

        if verified:
            lines.append("What worked:")
            for t in verified[:6]:
                excerpt = (t.output or "")[:200].replace("\n", " ")
                lines.append(f"  - {t.title}: {excerpt}")

        if failed:
            lines.append("What failed:")
            for t in failed[:6]:
                reason = (t.failure_detail or t.failure_reason_code or "unknown")[:200]
                lines.append(f"  - {t.title}: {reason}")

        consistency = summary.get("consistency") if isinstance(summary, dict) else None
        if consistency:
            lines.append(
                f"Consistency: passed={consistency.get('passed')} score={consistency.get('score')}"
            )
            issues = consistency.get("issues") or []
            for issue in issues[:3]:
                lines.append(
                    f"  - [{issue.get('severity')}] {issue.get('description', '')[:180]}"
                )

        return "\n".join(lines)
