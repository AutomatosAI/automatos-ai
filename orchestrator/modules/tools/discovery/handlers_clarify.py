"""PRD-229 — handler for ``ask_orchestrator``: answer inline, or guide (US-002).

The calling task/run is resolved from SERVER-injected caller context
(``_run_id`` / ``_task_id`` / ``_agent_id`` / ``_field_id`` — threaded by the
coordinator's field_context and re-injected by the executor), NEVER from a
tool parameter. The answer round (US-001 ``answer_clarification``) runs inside a
hard time-box (``Config.CLARIFICATION_ANSWER_TIMEOUT``) that fits within the
executing task's asyncio.wait_for envelope; a timeout takes the cannot-answer
path.

US-002 mapping (pre-escalation):
  * grounded answer  → ``{answer, sources}``
  * cannot_answer / escalate_directly / timeout → ``{proceed_with_assumption}``
US-003 REPLACES the second branch with the escalation ladder (create a human
ask, park the task with a draft, return ``{parked, ask_id}``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

_ASSUME_GUIDANCE = (
    "Auto could not answer this from the available context. Proceed with your "
    "best, explicitly-stated assumption, note it in your output, and keep going "
    "— do not fabricate facts you do not have."
)


async def ask_orchestrator(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Answer a mid-run clarification inline, or guide the agent to proceed."""
    from config import Config
    from services.orchestrator_answers import ClarificationSubject, answer_clarification

    question = (params.get("question") or "").strip()
    category = params.get("category")

    # Caller identity is server-injected (exec_platform / the executor), never a
    # tool parameter. A client-supplied run_id/task_id is ignored.
    run_id = params.get("_run_id")
    task_id = params.get("_task_id")
    agent_id = params.get("_agent_id")
    field_id = params.get("_field_id")

    if not question:
        return {"success": False, "error": "question is required"}

    # No run/task context (not a mission execution lane) — Auto has no run to
    # answer from; tell the agent to proceed with a stated assumption.
    if not run_id or not task_id:
        return {
            "success": True,
            "proceed_with_assumption": _ASSUME_GUIDANCE,
            "message": _ASSUME_GUIDANCE,
        }

    task = _load_task(db, task_id)
    subject = ClarificationSubject(
        run_id=run_id,
        workspace_id=workspace_id,
        task_id=task_id,
        task=task,
        field_id=field_id,
        agent_id=int(agent_id) if agent_id else None,
    )

    # Hard time-box: the answer round must fit inside the task's execution
    # envelope (Config docs the arithmetic). A slow round → cannot_answer.
    try:
        result = await asyncio.wait_for(
            answer_clarification(db, subject, question, category=category),
            timeout=Config.CLARIFICATION_ANSWER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("[ask_orchestrator] answer round timed out for task %s", task_id)
        result = {"cannot_answer": True, "reason": "timeout"}
    except Exception:  # noqa: BLE001 — a broken answer round must not crash the agent
        logger.warning("[ask_orchestrator] answer round failed for task %s", task_id, exc_info=True)
        result = {"cannot_answer": True, "reason": "error"}

    if "answer" in result:
        return {
            "success": True,
            "answer": result["answer"],
            "sources": result.get("sources", []),
        }

    # cannot_answer / escalate_directly / timeout — US-002 pre-escalation guidance.
    # (US-003 replaces this with: create a human ask, park the task, return parked.)
    return {
        "success": True,
        "proceed_with_assumption": _ASSUME_GUIDANCE,
        "message": _ASSUME_GUIDANCE,
        "detail": {k: v for k, v in result.items() if k != "answer"},
    }


def _load_task(db: Session, task_id: Any) -> Any:
    """Load the calling OrchestrationTask (upstream-digest source). Fail-soft —
    a missing/failed load leaves ``task=None`` and retrieval falls back."""
    try:
        from core.models.orchestration import OrchestrationTask

        return (
            db.query(OrchestrationTask)
            .filter(OrchestrationTask.id == task_id)
            .first()
        )
    except Exception:  # noqa: BLE001
        logger.warning("[ask_orchestrator] task load failed for %s", task_id, exc_info=True)
        return None
