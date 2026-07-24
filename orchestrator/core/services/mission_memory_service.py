"""
Mission Memory Service — PRD-131d Phase 1
==========================================

Stores mission (OrchestrationRun) outcomes into memory so future missions can
learn from what worked and what failed.

Writes two places via UnifiedMemoryService:
- L3 long-term (durable store) workspace namespace — conversational summary for fact
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
        Store a mission summary in L3 (durable store) + L2 (Postgres short-term).

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

        # L3 — durable long-term, workspace-wide namespace
        try:
            ns = MemoryNamespace(workspace_id=workspace_id)
            l3_result = await self._unified.store_long_term_messages(
                user_id=ns.workspace(),
                workspace_id=workspace_id,
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
    # Task-level failure + recovery capture (PRD-131d Phase 2)
    # ------------------------------------------------------------------

    async def store_task_failure(
        self,
        task: OrchestrationTask,
    ) -> Dict[str, Any]:
        """
        Store a permanently-failed task into memory so future missions can
        look up "we tried this task, it failed because X, after N attempts".

        Called from reconciler when a task transitions to FAILED permanently
        (verification max retries, partial max retries, stall max retries).

        Writes L2 short-term (content_type="task_failure", importance=0.8) and
        L3 long-term (durable workspace namespace).
        """
        run = self.db.query(OrchestrationRun).get(task.run_id)
        if run is None:
            raise ValueError(f"OrchestrationRun not found for task.run_id={task.run_id}")

        workspace_id = str(run.workspace_id)
        narrative = self._build_task_failure_narrative(run, task)

        errors: List[str] = []
        l3_ok = False
        l2_id: Optional[str] = None

        # L3 — durable workspace namespace
        try:
            ns = MemoryNamespace(workspace_id=workspace_id)
            l3_result = await self._unified.store_long_term_messages(
                user_id=ns.workspace(),
                workspace_id=workspace_id,
                messages=[
                    {
                        "role": "user",
                        "content": f"Task failed: {(task.title or '')[:300]}",
                    },
                    {
                        "role": "assistant",
                        "content": narrative[:6000],
                    },
                ],
                metadata={
                    "type": "task_failure",
                    "task_id": str(task.id),
                    "run_id": str(run.id),
                    "workspace_id": workspace_id,
                    "failure_reason_code": task.failure_reason_code,
                    "attempts": task.attempt_number or 0,
                },
            )
            l3_ok = not l3_result.get("error") if isinstance(l3_result, dict) else True
        except Exception as e:
            errors.append(f"L3 store failed: {e}")
            logger.warning(
                "[MissionMemory] L3 task_failure store failed for task %s",
                task.id, exc_info=True,
            )

        # L2 — importance 0.8 so failures promote to L3 fast
        try:
            l2_id = await self._unified.store_short_term(
                workspace_id=workspace_id,
                content=narrative[:4000],
                content_type="task_failure",
                importance=0.8,
                metadata={
                    "task_id": str(task.id),
                    "run_id": str(run.id),
                    "task_title": (task.title or "")[:300],
                    "failure_reason_code": task.failure_reason_code,
                    "failure_detail": (task.failure_detail or "")[:500],
                    "attempts": task.attempt_number or 0,
                    "max_retries": task.max_retries,
                    "agent_id": str(task.assigned_agent_id) if task.assigned_agent_id else None,
                },
            )
        except Exception as e:
            errors.append(f"L2 store failed: {e}")
            logger.warning(
                "[MissionMemory] L2 task_failure store failed for task %s",
                task.id, exc_info=True,
            )

        result = {
            "task_id": str(task.id),
            "run_id": str(run.id),
            "l3_stored": l3_ok,
            "l2_row_id": l2_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }
        if errors:
            logger.warning("[MissionMemory] Partial task_failure store for task %s: %s", task.id, errors)
        else:
            logger.info(
                "[MissionMemory] Stored task_failure task=%s l2=%s l3=%s",
                task.id, l2_id, l3_ok,
            )
        return result

    async def store_retry_recovery(
        self,
        task: OrchestrationTask,
    ) -> Dict[str, Any]:
        """
        Store a successful recovery after retry: task passed verification but
        took >0 prior attempts. Captures "what eventually worked" so future
        missions can short-circuit the dead-end path next time.

        Writes L2 short-term (content_type="retry_recovery", importance=0.75).
        """
        attempts = task.attempt_number or 0
        if attempts <= 0:
            # First-attempt success — nothing novel to capture
            return {"skipped": True, "reason": "first_attempt_success"}

        run = self.db.query(OrchestrationRun).get(task.run_id)
        if run is None:
            raise ValueError(f"OrchestrationRun not found for task.run_id={task.run_id}")

        workspace_id = str(run.workspace_id)
        narrative = self._build_retry_recovery_narrative(run, task)

        errors: List[str] = []
        l2_id: Optional[str] = None

        try:
            l2_id = await self._unified.store_short_term(
                workspace_id=workspace_id,
                content=narrative[:4000],
                content_type="retry_recovery",
                importance=0.75,
                metadata={
                    "task_id": str(task.id),
                    "run_id": str(run.id),
                    "task_title": (task.title or "")[:300],
                    "attempts_taken": attempts,
                    "agent_id": str(task.assigned_agent_id) if task.assigned_agent_id else None,
                },
            )
        except Exception as e:
            errors.append(f"L2 store failed: {e}")
            logger.warning(
                "[MissionMemory] L2 retry_recovery store failed for task %s",
                task.id, exc_info=True,
            )

        result = {
            "task_id": str(task.id),
            "run_id": str(run.id),
            "attempts_taken": attempts,
            "l2_row_id": l2_id,
            "stored_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
        }
        if not errors:
            logger.info(
                "[MissionMemory] Stored retry_recovery task=%s attempts=%d l2=%s",
                task.id, attempts, l2_id,
            )
        return result

    # ------------------------------------------------------------------
    # Narrative builders
    # ------------------------------------------------------------------

    def _build_task_failure_narrative(
        self,
        run: OrchestrationRun,
        task: OrchestrationTask,
    ) -> str:
        """Build a narrative for a permanently-failed task."""
        lines: List[str] = []
        lines.append(f"Task failed: {(task.title or 'Untitled task')[:300]}")
        if task.description:
            lines.append(f"Description: {task.description[:300]}")
        lines.append(f"Mission goal: {(run.goal or '')[:300]}")

        attempts = task.attempt_number or 0
        max_retries = task.max_retries
        lines.append(
            f"Attempts: {attempts}"
            + (f" / max {max_retries}" if max_retries else "")
        )

        if task.failure_reason_code:
            lines.append(f"Failure code: {task.failure_reason_code}")
        if task.failure_detail:
            lines.append(f"Failure detail: {task.failure_detail[:500]}")

        # Verification feedback, if any (stored by reconciler on retries)
        context = task.input_context or {}
        vfb = context.get("verification_feedback") if isinstance(context, dict) else None
        if isinstance(vfb, dict):
            reasoning = vfb.get("reasoning")
            if reasoning:
                lines.append(f"Last verifier reasoning: {str(reasoning)[:400]}")
            failures = vfb.get("failures") or []
            if failures:
                lines.append("Deterministic failures:")
                for f in list(failures)[:3]:
                    lines.append(f"  - {str(f)[:200]}")

        # Last agent output excerpt — what the agent actually produced before failing
        if task.output:
            excerpt = task.output[:400].replace("\n", " ")
            lines.append(f"Last output excerpt: {excerpt}")

        # Review feedback from verifier (stored in output_metadata)
        om = task.output_metadata or {}
        review = om.get("review_feedback") if isinstance(om, dict) else None
        if isinstance(review, dict):
            suggestions = review.get("suggestions") or []
            if suggestions:
                lines.append("Verifier suggestions:")
                for s in list(suggestions)[:3]:
                    lines.append(f"  - {str(s)[:200]}")

        return "\n".join(lines)

    def _build_retry_recovery_narrative(
        self,
        run: OrchestrationRun,
        task: OrchestrationTask,
    ) -> str:
        """Build a narrative for a task that eventually passed after retries."""
        lines: List[str] = []
        attempts = task.attempt_number or 0
        lines.append(
            f"Task recovered after {attempts} retries: {(task.title or 'Untitled')[:300]}"
        )
        lines.append(f"Mission goal: {(run.goal or '')[:300]}")

        # Capture previous-output from input_context — the version that failed
        context = task.input_context or {}
        prev = context.get("previous_output") if isinstance(context, dict) else None
        if isinstance(prev, str) and prev:
            lines.append(f"Initial (failed) output excerpt: {prev[:300].replace(chr(10), ' ')}")

        vfb = context.get("verification_feedback") if isinstance(context, dict) else None
        if isinstance(vfb, dict):
            reasoning = vfb.get("reasoning")
            if reasoning:
                lines.append(f"Verifier feedback that guided recovery: {str(reasoning)[:300]}")

        if task.output:
            final = task.output[:400].replace("\n", " ")
            lines.append(f"Final (passing) output excerpt: {final}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Mission narrative builder
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
