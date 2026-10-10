"""PRD-253 Wave P — a session's plan is the Plan card: a question for the operator.

Plan runs on every CLI. A plan turn ends with the plan — Claude Code presents it
with ``ExitPlanMode``, every other CLI as its final message — and the host sends
it as a ``PlanReady`` event in the turn's final flush, before the turn's result.
Here it becomes ONE PRD-225 question on the ticket (marker ``cli_plan``), so it
reaches the Questions tab, the bell and Telegram with nothing new, and one entry
in the ticket's plan ledger (``runtime_ref["session_plans"]``). The turn's end
parks the ticket on it through the path a session's own question takes
(``_park_for_answer``), keeping the session id, and the answer resumes that same
session:

* **Approve** — the next claim works as Edit automatically, its prompt carrying the plan;
* **Reject** — the ticket goes to review, the answer as the reason;
* **the operator's own words** — the next claim plans again, its prompt carrying them.

On the last round (``MAX_PLAN_ROUNDS``) only Approve carries on: any other answer
rejects, so a ticket never plans without end. A plan the operator approved INSIDE
the turn (Claude Code's card, answered in time) is recorded as approved, with no
card. Each claim reads the ledger for its mode (``plan_state``), so the host never
guesses whether a resumed session already presented its plan.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy.orm import Session

from core.models.approval_grants import SUBJECT_BOARD_TASK
from core.models.core import Agent, BoardTask
from services.ticket_numbers import ticket_label
from services.cli_host_service import (
    MAX_ASK_ANSWER_KEPT,
    PARKED_FOR_ANSWER_REASON,
    _iso,
    _kept,
    _notify_status,
    _now,
    resume_on_answer,
)

logger = logging.getLogger(__name__)

PLANS_KEY = "session_plans"      # ``runtime_ref[<key>] = [{kind, version, attempt, plan, grant_id, asked_at, answer?, …}]``
PLAN_KIND = "plan"
PLAN_MARKER = "cli_plan"         # ``ApprovalGrant.details[<marker>] = {task_id, version, attempt}``
# The host's event (services/cli-host … permission_modes.PLAN_EVENT), kept in step
# by test_cli_presets_parity.py.
PLAN_EVENT = "PlanReady"
PLAN_APPROVE = "Approve"
PLAN_REJECT = "Reject"
# The card's two buttons. The third answer is the operator's own words — a
# "Discuss" button would send the word and no feedback.
PLAN_OPTIONS = (PLAN_APPROVE, PLAN_REJECT)
MAX_PLAN_ROUNDS = 5              # O6: on the last round only Approve carries on
MAX_PLAN_KEPT = 16_000           # the plan on the ticket, for the prompt that carries it out
MAX_PLAN_ON_CARD = 8_000         # the card shows this much; plan.md beside the deliverables has all of it
PARKED_FOR_PLAN_REASON = "Waiting for your approval of its plan (ask #{grant_id})"
# PRD-256 FX-011: an agent's send on Auto's ticket parks on its approval card (agent_sends marks its ledger entry).
PARKED_FOR_SEND_REASON = "Waiting for your click on its send card (approval #{grant_id}): nothing has gone out"
SEND_KIND = "send"
PLAN_REJECTED_FEEDBACK = "Plan rejected: {answer}"
SUPERSEDED_ANSWER = "(superseded by a newer plan)"

STATE_PLANNING = "planning"      # no plan yet: the claim plans
STATE_AWAITING = "awaiting"      # the Plan card is open
STATE_APPROVED = "approved"      # the claim works as Edit automatically
STATE_DISCUSSING = "discussing"  # answered in the operator's own words: the claim plans again
STATE_REJECTED = "rejected"      # the ticket went to review


# ── the ledger ───────────────────────────────────────────────────────────────

def session_plans(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The plans this ticket's sessions presented, oldest first."""
    return [p for p in (ref.get(PLANS_KEY) or []) if isinstance(p, dict)]


def is_plan_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("kind") == PLAN_KIND


def plan_verdict(answer: Any, *, last_round: bool = False) -> str:
    """``approved``, ``rejected`` or ``discussing``. An answer that starts with
    Approve approves and one that starts with Reject rejects; any other answer is
    the operator's feedback — except on the last round, where it rejects."""
    said = str(answer or "").strip().lower()
    if said.startswith(PLAN_APPROVE.lower()):
        return STATE_APPROVED
    if said.startswith(PLAN_REJECT.lower()) or last_round:
        return STATE_REJECTED
    return STATE_DISCUSSING


def _last_round(entry: Dict[str, Any]) -> bool:
    return int(entry.get("version") or 0) >= MAX_PLAN_ROUNDS


def _verdict(entry: Dict[str, Any]) -> str:
    return plan_verdict(entry.get("answer"), last_round=_last_round(entry))


def plan_state(ref: Dict[str, Any]) -> str:
    """Where the ticket's plan stands: the latest plan decides."""
    plans = session_plans(ref)
    if not plans:
        return STATE_PLANNING
    if not plans[-1].get("answered_at"):
        return STATE_AWAITING
    return _verdict(plans[-1])


def plan_marker(grant: Any) -> Optional[Dict[str, Any]]:
    """``{task_id, version, attempt}`` when this question row is a Plan card, else None."""
    details = getattr(grant, "details", None)
    marker = details.get(PLAN_MARKER) if isinstance(details, dict) else None
    if not isinstance(marker, dict) or marker.get("task_id") is None:
        return None
    return marker


def park_reason(entry: Dict[str, Any]) -> str:
    """Why a parked ticket waits: its Plan card, or its session's own question."""
    template = PARKED_FOR_PLAN_REASON if is_plan_entry(entry) else PARKED_FOR_ANSWER_REASON
    if isinstance(entry, dict) and entry.get("kind") == SEND_KIND:
        template = PARKED_FOR_SEND_REASON
    return template.format(grant_id=entry.get("grant_id"))


def rejected_plan(ref: Dict[str, Any]) -> bool:
    """The latest plan was rejected and the ticket has not been sent to review for it
    yet — the operator answered while the turn still ran."""
    plans = session_plans(ref)
    latest = plans[-1] if plans else None
    return bool(latest and latest.get("answered_at") and not latest.get("folded_at")
                and _verdict(latest) == STATE_REJECTED)


# ── a plan arrives ───────────────────────────────────────────────────────────

def _plan_event(events: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The turn's plan: the last ``PlanReady`` event that carries one."""
    found = [ev for ev in events
             if (ev.get("event") or ev.get("hook_event_name")) == PLAN_EVENT and str(ev.get("text") or "").strip()]
    return found[-1] if found else None


async def raise_session_plan(
    db: Session, task: BoardTask, ref: Dict[str, Any], events: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """The plan a turn reported onto the ticket: an approval given inside the turn
    is recorded; any other plan becomes the Plan card, which the turn's end parks
    on. Once per claim — a re-flushed event files nothing twice.

    Returns the ``runtime_ref`` the caller continues with — the one it came in with
    when filing fails: the flush must still reach the host, and the plan is still
    the turn's final message and its plan.md."""
    event = _plan_event(events)
    if event is None or any(p.get("attempt") == ref.get("attempt") for p in session_plans(ref)):
        return ref
    try:
        return await _record_plan(db, task, ref, event)
    except Exception:  # noqa: BLE001 — never break the flush over the Plan card
        logger.error("[cli-host] the plan of ticket #%s was not filed this flush", task.id, exc_info=True)
        db.rollback()
        return ref


async def _record_plan(db: Session, task: BoardTask, ref: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    now = _iso(_now())
    entry = {"kind": PLAN_KIND, "version": len(session_plans(ref)) + 1, "attempt": ref.get("attempt"),
             "plan": _plan_kept(str(event["text"]).strip()), "asked_at": now}
    if event.get("approved_in_turn"):
        # Approved on its card inside the turn: the session is already at work on it.
        entry = {**entry, "answer": PLAN_APPROVE, "answered_at": now, "folded_at": now, "approved_in_turn": True}
    else:
        entry = {**entry, "grant_id": await _file_plan_card(db, task, entry)}
    return _append_plan(db, task, entry)


def _plan_kept(text: str) -> str:
    if len(text) <= MAX_PLAN_KEPT:
        return text
    return text[:MAX_PLAN_KEPT] + f"\n\n[The plan is cut here at {MAX_PLAN_KEPT:,} characters; plan.md holds all of it.]"


async def _file_plan_card(db: Session, task: BoardTask, entry: Dict[str, Any]) -> int:
    """ONE question row through PRD-225's shared internals — UNPARKED, like a
    session's own ask: the turn's result has yet to land (``apply_result`` writes
    only a run still ``in_progress``), and the turn's end parks it."""
    from modules.tools.discovery.handlers_asks import stage_question

    agent = db.query(Agent).filter(Agent.id == task.assigned_agent_id).first() if task.assigned_agent_id else None
    agent_name = getattr(agent, "name", None)
    staged = await stage_question(
        db, task.workspace_id,
        subject_type=SUBJECT_BOARD_TASK, subject_id=str(int(task.id)),
        question=plan_question(task, agent_name, entry),
        options=list(PLAN_OPTIONS),
        asked_by_agent_id=task.assigned_agent_id, agent_name=agent_name,
        park=None,
        details={PLAN_MARKER: {"task_id": int(task.id), "version": entry["version"], "attempt": entry["attempt"]}},
    )
    ask_id = staged.get("ask_id") if isinstance(staged, dict) else None
    if ask_id is None:
        raise RuntimeError(f"the Plan card of ticket #{task.id} was not filed")
    logger.info("[cli-host] ticket #%s presented plan %s (ask #%s)", task.id, entry["version"], ask_id)
    return int(ask_id)


def plan_question(task: Any, agent_name: Optional[str], entry: Dict[str, Any]) -> str:
    """The Plan card, wherever it reaches the operator. The answer words are spelled
    out: a Telegram reply sees no buttons."""
    title = f" — {task.title}" if getattr(task, "title", None) else ""
    plan = entry["plan"]
    if len(plan) > MAX_PLAN_ON_CARD:
        plan = plan[:MAX_PLAN_ON_CARD] + "\n\n… the whole plan is in plan.md, beside the ticket's deliverables."
    if _last_round(entry):
        how = ("This is the last round: reply **Approve** to start the work. Any other answer rejects the "
               "plan and sends the ticket to review, with your words as the reason.")
    else:
        how = ("Reply **Approve** to start the work, or **Reject** to send the ticket to review — or answer "
               "in your own words, and the agent revises the plan and presents it again.")
    return (f"**{agent_name or 'The agent'} has a plan for {ticket_label(task)}{title}** "
            f"(round {entry['version']} of {MAX_PLAN_ROUNDS}).\n\n{plan}\n\n{how}")


def _append_plan(db: Session, task: BoardTask, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Write ONLY the plan ledger, onto the row as it is now: filing the card spent
    seconds (bell, Telegram) and the host's flush or an answer may have committed
    meanwhile. An older plan still waiting is superseded — its card closes and
    nothing resumes on it."""
    db.refresh(task)
    fresh = dict(task.runtime_ref or {})
    older = session_plans(fresh)
    waiting = [p for p in older if not p.get("answered_at")]
    if waiting:
        _close_plan_cards(db, task, waiting, by=f"cli-host:{fresh.get('host_id') or 'unknown'}")
    now = _iso(_now())
    kept = [p if p.get("answered_at") else {**p, "answer": SUPERSEDED_ANSWER, "answered_at": now, "folded_at": now}
            for p in older]
    task.runtime_ref = {**fresh, PLANS_KEY: [*kept, entry]}
    db.commit()
    return task.runtime_ref


def _close_plan_cards(db: Session, task: BoardTask, plans: List[Dict[str, Any]], *, by: str) -> int:
    from core.services.approval_grants import expire_pending_grants

    ids = [p["grant_id"] for p in plans if p.get("grant_id") is not None]
    return expire_pending_grants(db, task.workspace_id, ids, revoked_by=by) if ids else 0


# ── the operator answers ─────────────────────────────────────────────────────

def answer_session_plan(db: Session, grant: Any) -> bool:
    """PRD-225's answer path reached a Plan card. Approve, or the operator's own
    words, resumes the session that made the plan — the path a session's own
    question takes (``resume_on_answer``); Reject sends the ticket to review.
    True when the work moves. A card that is not the ticket's open plan (an older
    version, already answered) changes nothing."""
    marker = plan_marker(grant)
    if marker is None:
        return False
    task = (
        db.query(BoardTask)
        .filter(BoardTask.id == int(marker["task_id"]), BoardTask.workspace_id == grant.workspace_id)
        .first()
    )
    if task is None:
        logger.warning("[cli-host] plan card #%s names ticket #%s, which is not in workspace %s",
                       grant.id, marker["task_id"], grant.workspace_id)
        return False
    ref = dict(task.runtime_ref or {})
    plans = session_plans(ref)
    latest = plans[-1] if plans else None
    if latest is None or latest.get("answered_at") or int(latest.get("grant_id") or 0) != int(grant.id):
        logger.info("[cli-host] plan card #%s is not ticket #%s's open plan — nothing changes", grant.id, task.id)
        return False
    answer = _kept(str(getattr(grant, "answer_text", "") or ""), MAX_ASK_ANSWER_KEPT,
                   what="The owner's answer is", grant_id=grant.id)
    ref = {**ref, PLANS_KEY: [*plans[:-1], {**latest, "answer": answer, "answered_at": _iso(_now())}]}
    if rejected_plan(ref) and task.status == "blocked":
        reject_plan(db, task, ref)
        return False
    return resume_on_answer(db, task, ref, what=f"plan card #{grant.id}")


def reject_plan(db: Session, task: BoardTask, ref: Dict[str, Any]) -> str:
    """Rejected: a person decides what happens to the ticket next. It goes to
    review with the answer as the reason, and the plan is marked folded so nothing
    resumes on it. Returns the status written."""
    plans = session_plans(ref)
    latest = plans[-1]
    task.runtime_ref = {**ref, PLANS_KEY: [*plans[:-1], {**latest, "folded_at": _iso(_now())}]}
    task.status = "review"
    task.lease_until = None
    task.blocked_at = None
    task.blocked_reason = None
    task.completed_at = _now()
    task.review_feedback = PLAN_REJECTED_FEEDBACK.format(answer=latest.get("answer") or PLAN_REJECT)
    _notify_status(db, task)  # F119: before the commit that carries it
    db.commit()
    logger.info("[cli-host] ticket #%s: plan %s rejected — sent to review", task.id, latest.get("version"))
    return task.status


# ── the session that picks the work up ───────────────────────────────────────

def plan_fold_in(ref: Dict[str, Any]) -> str:
    """The operator's answer to the latest plan, for the prompt of the claim that
    picks the work up — once: the claim stamps ``folded_at``, as for answers."""
    plans = session_plans(ref)
    latest = plans[-1] if plans else None
    if latest is None or not latest.get("answered_at") or latest.get("folded_at"):
        return ""
    answer = str(latest.get("answer") or "")
    if _verdict(latest) == STATE_APPROVED:
        lines = ["## Your plan was approved — implement it now",
                 "The operator approved the plan below. Carry it out, verify it, and end your turn with "
                 "what you changed and how you checked it."]
        note = _after_first_word(answer)
        lines += ["", f"**The operator added:** {note}"] if note else []
    else:
        lines = ["## Feedback on your plan — revise it and present it again",
                 "You are still planning: make no changes yet. Revise the plan with the operator's answer "
                 "and present it again.",
                 "", f"**The operator's answer:** {answer}"]
    return "\n".join([*lines, "", "**Your plan:**", str(latest.get("plan") or "")])


def _after_first_word(answer: str) -> str:
    """What the operator wrote after the verdict: "Approve — keep the old API" → "keep the old API"."""
    parts = answer.strip().split(None, 1)
    return parts[1].lstrip(" ,.:;-—–") if len(parts) > 1 else ""


__all__ = [
    "MAX_PLAN_ROUNDS", "PARKED_FOR_PLAN_REASON", "PLANS_KEY", "PLAN_APPROVE", "PLAN_EVENT", "PLAN_MARKER",
    "PLAN_OPTIONS", "PLAN_REJECT", "STATE_APPROVED", "STATE_AWAITING", "STATE_DISCUSSING", "STATE_PLANNING",
    "STATE_REJECTED", "answer_session_plan", "is_plan_entry", "park_reason", "plan_fold_in", "plan_marker",
    "plan_question", "plan_state", "plan_verdict", "raise_session_plan", "reject_plan", "rejected_plan",
    "session_plans",
]
