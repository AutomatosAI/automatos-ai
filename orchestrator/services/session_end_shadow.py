"""PRD-248 S5 / tuning (6 Oct): the session_end judgement, beside a session's landed result.

Shadow only. Once ``cli_host_service.apply_result`` has landed a Claude Code session's
result (its deliverables registered, the files and refusals on the ticket), the
decision engine reads what the session made beside the ticket's brief and dates and
says whether the work is complete, whether nothing was done, and whether the owner is
needed. Lazy, off by default, fail-open: the result is never touched.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Sequence

from sqlalchemy.orm import Session

from core.models.cli_hosts import CliHost
from core.models.core import BoardTask

logger = logging.getLogger(__name__)


def shadow_session_end(
    task: Any, ref: Dict[str, Any], payload: Dict[str, Any], exec_result: Dict[str, Any],
    files: Any, denials: Any, deliverables: Sequence[Dict[str, Any]],
) -> None:
    """PRD-248 S5 (shadow only): the decision engine reads what the session made (its
    deliverables and their first lines, the files written, the commands refused) beside
    the ticket's brief and dates, and its final message, and says whether the work is
    complete, whether nothing was done, and whether the owner is needed. Logged beside
    the status the board applied. Lazy, off by default, fail-open; the result is never
    touched. ``denials`` are the ticket's denial summaries (``_denial_summary``)."""
    try:
        from core.llm.decisions import MODE_OFF, get_decision_engine, judgements
        from services import decision_evidence

        engine = get_decision_engine()
        if engine.dials().session_end_mode == MODE_OFF:
            return
        engine.shadow(
            decision_evidence.shadow_session_end(
                engine,
                deliverable_refs=[dict(d) for d in deliverables],
                workspace_id=getattr(task, "workspace_id", None),
                task_id=getattr(task, "id", None),
                attempt=payload.get("attempt", ref.get("attempt")),
                title=getattr(task, "title", "") or "",
                description=getattr(task, "description", "") or "",
                final_text=str(payload.get("result_text") or payload.get("error") or ""),
                exit_reason=str(payload.get("exit_reason") or exec_result.get("status") or ""),
                files=list(files) if isinstance(files, (list, tuple)) else [],
                denials=list(denials) if isinstance(denials, (list, tuple)) else [],
                asked_on=decision_evidence.day_label(getattr(task, "created_at", None)),
                finished_on=decision_evidence.today_label(),
                platform_status=str(exec_result.get("status") or ""),
            ),
            purpose=judgements.PURPOSE_SESSION_END,
        )
    except Exception:  # noqa: BLE001 — never into a result
        logger.debug("[decision] session-end shadow skipped", exc_info=True)


def session_end_judged() -> bool:
    """Whether the session_end dial is on (the engine caches its dials); never raises."""
    try:
        from core.llm.decisions import MODE_OFF, get_decision_engine

        return get_decision_engine().dials().session_end_mode != MODE_OFF
    except Exception:  # noqa: BLE001
        return False


# A result that hands the ticket back to the queue is not the end of its work.
_RELEASE_STATUSES = frozenset({"usage_limit", "host_stopped"})


def judged_after_landing(apply: Callable[..., Any]) -> Callable[..., Any]:
    """PRD-248 tuning (6 Oct): the session_end judgement runs once the result has landed,
    so it reads what the ticket now records: the deliverables registered from the
    session's files, the files written and the refusals. Before, it ran ahead of the
    deliverables and saw only the final message. A decorator, so ``apply_result``
    stays as it is; shadow only, the returned result is never touched."""

    @functools.wraps(apply)
    async def wrapper(db: Session, host: CliHost, task_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        result = await apply(db, host, task_id, payload)
        status = str(payload.get("status") or "success").lower()
        if result.get("applied") and status not in _RELEASE_STATUSES and session_end_judged():
            try:
                task = db.query(BoardTask).filter(BoardTask.id == task_id).first()
                ref = dict(getattr(task, "runtime_ref", None) or {})
                landed = {"status": "error" if status == "error" else ("cancelled" if status == "cancelled" else "success")}
                shadow_session_end(task, ref, payload, landed, ref.get("files_touched") or [],
                                   ref.get("permission_denials") or [], ref.get("deliverables") or [])
            except Exception:  # noqa: BLE001 — never into a result
                logger.debug("[decision] session-end shadow skipped after landing", exc_info=True)
        return result

    return wrapper
