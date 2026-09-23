"""PRD-229 US-003 — the escalation ladder: cannot_answer becomes a human ask.

When Auto cannot answer a mid-run clarification (or the question is a governance
decision), the ladder escalates VERTICALLY to a human — worker → orchestrator →
human, never lateral. It reuses PRD-225's SHARED ask internals (``stage_question``
— the function ``platform_ask_human`` itself reaches after validation; no parallel
ask construction, no HTTP self-call), parks the task by recording a labelled DRAFT of
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

# The grant subject a clarification park is filed under: the parked clarification
# CALL, carrying the OrchestrationTask id. Not a board_task — the ladder's subject
# is a mission sub-task, and the answer comes back through
# ``_resume_clarification_if_parked``, not the board re-queue.
SUBJECT_TOOL_CALL = "tool_call"


class ClarificationAskNotPlaced(RuntimeError):
    """No question row could be staged, so NOTHING was asked of a human.

    Raised before the task is parked. ``handlers_clarify`` catches it and falls
    back to proceed-with-assumption: there is no orphaned ask to worry about and a
    retry files a fresh one. The alternative — parking behind an ask that does not
    exist — strands the task with no question on any surface.
    """


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

    Reuses PRD-225's SHARED ask internals — ``stage_question``, the function
    ``platform_ask_human`` itself reaches after validation — to create the
    ``kind='question'`` grant (subject_type ``tool_call``, the parked
    clarification call, carrying the OrchestrationTask id). Returns
    ``{parked: True, ask_id}`` so the calling agent stops cleanly.

    ``park=None``: the ladder parks its OWN subject below (an OrchestrationTask
    draft + awaiting-answer marker, read back by ``apply_answered_clarification``).
    ``ask_human``'s park flips a BoardTask, which this subject is not.

    Raises when no question row could be staged — the caller's contract for "no
    ask was placed" (handlers_clarify falls back to proceed-with-assumption).
    Never park behind an ask that does not exist: the task would wait forever
    with no question on any surface.
    """
    from modules.tools.discovery.handlers_asks import stage_question

    # --- create the human ask via 225's SHARED internals (no parallel path) ---
    # stage_question durably COMMITS the grant — the human sees it in the
    # Questions tab (the list route filters on kind alone) and gets the Telegram
    # ping the instant this returns. If it RAISES, no ask was placed, so we let
    # it propagate (the handler falls back safely).
    #
    # It is reached DIRECTLY, not through ``ask_human``: that tool refuses every
    # non-``board_task`` subject up front (P225-RVW-11) because ITS answer path
    # — ``_resume_tool_call``'s stored-call re-dispatch — no-ops for them. A
    # clarification park has a different, working answer path:
    # ``_resume_clarification_if_parked`` recognises this grant by
    # ``pending_ask_id(task) == grant.id`` and bridges the answer into the task's
    # next run. Routing the ladder through the tool got a refusal dict, a
    # ``None`` ask id, and a task parked behind a question that was never filed.
    staged = await stage_question(
        db, subject.workspace_id,
        subject_type=SUBJECT_TOOL_CALL,
        subject_id=str(subject.task_id),
        question=question,
        asked_by_agent_id=int(subject.agent_id) if subject.agent_id else None,
        agent_name=agent_name,
        park=None,
    )
    ask_id = staged.get("ask_id") if isinstance(staged, dict) else None
    if ask_id is None:
        raise ClarificationAskNotPlaced(
            f"the ask internals staged no question row for task {subject.task_id}: {staged!r}"
        )

    # --- park the task + commit it as durably as the ask (P229-RVW-5) ----------
    # The session is SHARED across every task in this mission-run tick
    # (coordinator_service opens ONE SessionLocal at :1521 and runs the concurrent
    # agents' I/O via asyncio.gather at :1750). stage_question already committed
    # the grant; the park + draft + trail here are only FLUSHED, so a SIBLING task's
    # tool error firing db.rollback() on the shared session (platform_executor.py
    # :1245) would wipe this uncommitted draft while the committed ask survives —
    # an ORPHANED human ask no answer can bridge back to (Gerard's baked
    # draft-on-park decision silently lost). So we COMMIT the park immediately,
    # mirroring the ask internals' own durably-park-first pattern.
    #
    # Everything past the placed ask is best-effort AND non-raising: the human HAS
    # been asked, so a persistence failure must NOT surface as "escalation failed"
    # (a retrying agent would file a DUPLICATE ask). On failure we discard the
    # half-written park (the committed grant still stands) and still report parked.
    try:
        _park_task_with_draft(subject.task, ask_id=ask_id, question=question, partial_output=partial_output)
        _record_escalation(db, subject, question, ask_id=ask_id, category=category)
        db.commit()
    except Exception:  # noqa: BLE001 — the ask is already durable; never double-ask
        logger.error(
            "[clarify] park+draft persist failed after ask %s was placed; the ask "
            "stands but the draft could not be preserved", ask_id, exc_info=True,
        )
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass

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
