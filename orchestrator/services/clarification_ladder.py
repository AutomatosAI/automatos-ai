"""PRD-229 US-003 — the escalation ladder: cannot_answer becomes a human ask.

When Auto cannot answer a mid-run clarification (or the question is a governance
decision), the ladder escalates VERTICALLY to a human — worker → orchestrator →
human, never lateral. It reuses PRD-225's SHARED ask internals (``ask_human`` —
the same function ``platform_ask_human`` dispatches to; no parallel ask
construction, no HTTP self-call), parks the task by recording a labelled DRAFT of
its partial output on the task's EXISTING result JSONB, and marks the task as
awaiting the answer. On the human answer (PRD-225's answer path, UNCHANGED), the
Q&A is bridged into the task's next-run context so a re-run reads it verbatim.

Escalations are NEVER budget-limited (baked decision, Gerard 2026-08-27) — they
are visible and cheap by design; only Auto's ANSWERS are budgeted (US-001).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from core.models.orchestration_enums import ActorType, EventType
from services.orchestration_state import emit_event

logger = logging.getLogger(__name__)

# The label a parked task's partial output carries on the card (visibly a draft).
DRAFT_LABEL = "draft — parked awaiting a human answer"

# Keys on the task's EXISTING JSONB — no schema change, rebuild-don't-mutate.
DRAFT_KEY = "clarification_draft"          # on output_metadata (the result JSONB)
PENDING_KEY = "clarification_pending"      # on input_context (awaiting-answer marker)
RESUME_KEY = "clarification_resume"        # on input_context (answered Q&A for re-run)


async def escalate_clarification(
    db: Any,
    subject: Any,
    question: str,
    *,
    category: Optional[str] = None,
    partial_output: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Escalate a clarification to a human and park the task with a draft.

    Reuses PRD-225 ``ask_human`` to create the question grant (subject_type
    ``tool_call`` — the parked clarification call — carrying the task id). Returns
    ``{parked: True, ask_id}`` so the calling agent stops cleanly.
    """
    from modules.tools.discovery.handlers_asks import ask_human

    # --- create the human ask via 225's SHARED internals (no parallel path) ---
    ask_params = {
        "subject_type": "tool_call",
        "subject_id": str(subject.task_id),
        "question": question,
        "_agent_id": subject.agent_id,
        "_agent_name": agent_name,
    }
    ask_result = await ask_human(db, subject.workspace_id, ask_params)
    ask_id = ask_result.get("ask_id") if isinstance(ask_result, dict) else None

    # --- park the task: labelled DRAFT + awaiting marker on EXISTING JSONB ------
    _park_task_with_draft(subject.task, ask_id=ask_id, question=question, partial_output=partial_output)

    # --- record the escalation on the run event trail --------------------------
    _record_escalation(db, subject, question, ask_id=ask_id, category=category)

    return {
        "parked": True,
        "ask_id": ask_id,
        "message": (
            f"Parked this task and asked a human (ask #{ask_id}). Stop cleanly — "
            "the answer will resume the work with your draft preserved."
        ),
    }


def _park_task_with_draft(
    task: Any,
    *,
    ask_id: Any,
    question: str,
    partial_output: Optional[str],
) -> None:
    """Record the parked draft on the task's result JSONB and the awaiting marker
    on its input context. Rebuild-don't-mutate; no schema change. Fail-soft — a
    missing task (run-level ask) leaves the grant standing without a draft."""
    if task is None:
        return
    now = datetime.now(timezone.utc).isoformat()
    draft = {
        "label": DRAFT_LABEL,
        "ask_id": ask_id,
        "question": question,
        "partial_output": (partial_output or (getattr(task, "output", None) or "")),
        "parked_at": now,
    }
    task.output_metadata = {**(getattr(task, "output_metadata", None) or {}), DRAFT_KEY: draft}
    task.input_context = {
        **(getattr(task, "input_context", None) or {}),
        PENDING_KEY: {"ask_id": ask_id, "question": question, "parked_at": now},
    }


def _record_escalation(
    db: Any,
    subject: Any,
    question: str,
    *,
    ask_id: Any,
    category: Optional[str],
) -> None:
    """Append the escalation to the run's event trail (best-effort)."""
    payload = {
        "outcome": "escalated",
        "question": (question or "")[:2000],
        "ask_id": ask_id,
        "parked": True,
    }
    if category:
        payload["category"] = category
    try:
        emit_event(
            db,
            run_id=subject.run_id,
            event_type=EventType.CLARIFICATION_ESCALATED,
            actor_type=ActorType.COORDINATOR,
            actor_id="auto",
            task_id=subject.task_id,
            payload=payload,
        )
    except Exception:  # noqa: BLE001 — the trail must never break the escalation
        logger.warning("[clarify] failed to record escalation on run trail", exc_info=True)


# ---------------------------------------------------------------------------
# Resume — bridge the human answer into the task's next-run context
# ---------------------------------------------------------------------------

def pending_ask_id(task: Any) -> Optional[Any]:
    """The id of the human ask this task is parked on, or None. Pure."""
    pending = (getattr(task, "input_context", None) or {}).get(PENDING_KEY)
    return pending.get("ask_id") if isinstance(pending, dict) else None


def apply_answered_clarification(db: Any, task: Any) -> bool:
    """If the task's clarification ask has been ANSWERED (PRD-225's answer path,
    unchanged), bridge the Q&A into the task's next-run context and clear the
    awaiting marker. Returns True iff the task was resumed. Rebuild-don't-mutate.

    Reads ``grant.answer_text``/``question_md`` directly — independent of 225's
    internal subject-storage — so the bridge holds for any answered question.
    """
    ask_id = pending_ask_id(task)
    if ask_id is None:
        return False
    grant = _load_answered_grant(db, ask_id)
    if grant is None:
        return False  # still pending / dismissed — keep waiting

    draft = (getattr(task, "output_metadata", None) or {}).get(DRAFT_KEY) or {}
    resume = {
        "ask_id": ask_id,
        "question": getattr(grant, "question_md", None) or draft.get("question"),
        "answer": getattr(grant, "answer_text", None),
        "draft": draft.get("partial_output"),
    }
    input_ctx = dict(getattr(task, "input_context", None) or {})
    input_ctx.pop(PENDING_KEY, None)          # clear the awaiting marker
    input_ctx[RESUME_KEY] = resume            # inject the answered Q&A for the re-run
    task.input_context = input_ctx
    return True


def _load_answered_grant(db: Any, ask_id: Any) -> Any:
    """Return the grant iff it exists and is GRANTED (answered); else None."""
    try:
        from core.models.approval_grants import ApprovalGrant, GrantStatus

        grant = db.query(ApprovalGrant).filter(ApprovalGrant.id == ask_id).first()
        if grant is None:
            return None
        return grant if getattr(grant, "status", None) == GrantStatus.GRANTED.value else None
    except Exception:  # noqa: BLE001
        logger.warning("[clarify] answered-grant load failed for ask %s", ask_id, exc_info=True)
        return None


def render_resume_block(task: Any) -> Optional[str]:
    """The prompt block a re-run reads: the human Q&A plus the preserved draft.
    Pure — returns None when the task carries no answered clarification."""
    resume = (getattr(task, "input_context", None) or {}).get(RESUME_KEY)
    if not isinstance(resume, dict) or not resume.get("answer"):
        return None
    parts = [
        "\n## Answer to your earlier question",
        f"You asked: {resume.get('question')}",
        f"The human answered: {resume.get('answer')}",
        "Use this answer to continue the task.",
    ]
    draft = resume.get("draft")
    if draft:
        parts.append(f"\n## Your preserved draft (continue from here)\n\n{draft}")
    return "\n".join(parts)
