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

    task = _load_task(db, run_id, workspace_id, task_id)
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

    # US-003 — the escalation ladder: cannot_answer / escalate_directly / timeout
    # becomes a human ask (via PRD-225's shared ask_human), the task parks with a
    # labelled draft, and the caller gets {parked, ask_id} so it stops cleanly.
    from services.clarification_ladder import escalate_clarification

    # escalate_clarification is exception-safe ONCE the ask is placed (P229-RVW-5:
    # it swallows every post-ask_human failure and still returns {parked, ask_id}).
    # It only RAISES when ask_human itself failed — i.e. NO human ask was placed —
    # so falling back to proceed-with-assumption here is safe: there is nothing to
    # orphan, and a retry re-attempts a fresh ask rather than double-asking a
    # placed one.
    try:
        escalation = await escalate_clarification(
            db, subject, question,
            category=category if isinstance(category, str) else result.get("category"),
            agent_name=params.get("_agent_name"),
        )
    except Exception:  # noqa: BLE001 — only reached when no ask was placed
        logger.warning(
            "[ask_orchestrator] escalation failed before an ask was placed for task %s",
            task_id, exc_info=True,
        )
        return {
            "success": True,
            "proceed_with_assumption": _ASSUME_GUIDANCE,
            "message": _ASSUME_GUIDANCE,
        }

    return {
        "success": True,
        "parked": True,
        "ask_id": escalation.get("ask_id"),
        "message": escalation.get("message"),
        "detail": {k: v for k, v in result.items() if k != "answer"},
    }


def _load_task(db: Session, run_id: Any, workspace_id: Any, task_id: Any) -> Any:
    """Load the calling OrchestrationTask (upstream-digest source), SCOPED to the
    server-resolved subject (P229-RVW-2).

    The task must belong to ``run_id`` AND that run must belong to
    ``workspace_id``. ``OrchestrationTask`` carries no ``workspace_id`` column —
    the tenant boundary is reached via its run — so a smuggled foreign ``task_id``
    that slipped a same-shaped id past the executor could still never load a
    cross-tenant row here. Fail-soft: a missing / failed / out-of-scope load
    leaves ``task=None`` and retrieval falls back."""
    if not run_id or not workspace_id or task_id is None:
        return None
    try:
        from core.models.orchestration import OrchestrationRun, OrchestrationTask

        task = (
            db.query(OrchestrationTask)
            .filter(
                OrchestrationTask.id == task_id,
                OrchestrationTask.run_id == run_id,
            )
            .first()
        )
        if task is None:
            return None
        run = (
            db.query(OrchestrationRun)
            .filter(
                OrchestrationRun.id == run_id,
                OrchestrationRun.workspace_id == workspace_id,
            )
            .first()
        )
        return task if run is not None else None
    except Exception:  # noqa: BLE001
        logger.warning("[ask_orchestrator] task load failed for %s", task_id, exc_info=True)
        return None
