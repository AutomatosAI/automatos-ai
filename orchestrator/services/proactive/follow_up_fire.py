"""PRD-268 §6.3 'When it fires', FR-9, FR-14, K11 — a follow-up fires once, in the check-in lane.

``ScheduledTaskService.execute_task`` fires a follow-up row as it fires every row: the row
locked, the credit check, ``count_the_run`` completing the one-shot, the commit. Then
``_deliver`` hands it to :func:`fire_follow_up` on the same session, never to the chat path.

One ``proactive_runs`` row records the firing: kind ``follow_up``, Auto's agent id. Its start
refuses, with that row and no model call, when:

* the person no longer belongs to the workspace (``skipped_off``, ``no_addressee``);
* the workspace's mode is not live (``skipped_off``: ``switched_off``, or ``not_live`` in shadow);
* W1's ``gates.start_refusal`` refuses for the person: their pause, ``background_allowed``,
  the spend ceiling, the workspace's model-run cap (``follow_up=True``: no check-in interval).

Otherwise one turn of W1's loop runs in the CHECK_IN lane as Auto (``runner.AUTO``,
``runner.heartbeat_llm``, the ``check_in`` purpose: one ask, never paid from the trial). Its
prompt (``services.proactive.follow_up_prompts``) quotes, as data, the reason, the origin chat's
last ``PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES`` messages while that chat is still the person's, and
their open and recently hidden titles. Its calls carry ``check_in_user_id``, so a suggestion is
addressed to the person. The follow-up tools are chat-only, so the lane refuses them: no
follow-up schedules another (§12-Q9).

How it ends (K11):

* a suggestion recorded: ``suggested``; W2's ``deliver_run`` delivers it once the run row commits;
* ``FOLLOW_UP_SILENT``, or no text: ``silent``, nothing posted;
* any other text: ``posted``, ONE message through ``deliver_background_message``: in the origin
  chat while it is the person's (the messenger's owner check, given the person), else their Auto
  thread; source ``{origin 'follow_up', follow_up_id}``, announced as ``auto_message``. The model
  read untrusted data with nobody watching, so its words are posted as plain text that renders
  as itself (no link, image, code or HTML) and cut to a short message (``posted_text``). A post
  that does not land is an ``error`` (``not_posted``);
* any failure: ``error``, named by its class. The message is posted before the run row is
  written, so ``posted`` always means it landed; a failure writing the row after a post that
  landed is an ``error`` run beside that message.

Nothing re-registers or retries the row: ``count_the_run`` completed it before this ran. The
firing and its outcome each write a workspace audit row after their commit, never the reason.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.models.proactive import RUN_ERROR, RUN_POSTED, RUN_SILENT, RUN_SKIPPED_OFF, RUN_SUGGESTED
from services.proactive.delivery import deliver_run
from services.proactive.failures import failure_class
from services.proactive.follow_up_prompts import (
    FollowUpFacts,
    evidence_ref,
    follow_up_task,
    is_silent,
    posted_text,
    start_message,
)
from services.proactive.gates import NO_ADDRESSEE, SWITCHED_OFF, Refusal, start_refusal
from services.proactive.loop import LoopResult, LoopSpec, run_loop
from services.proactive.runner import (
    AUTO,
    CHECK_IN_PURPOSE,
    CheckInResult,
    heartbeat_llm,
    record_error,
    record_refusal,
)
from services.proactive.runs import KIND_FOLLOW_UP, RunRecord, known_titles, suggestions_made, write_run
from services.proactive.switches import MODE_LIVE, MODE_OFF, effective_mode

logger = logging.getLogger(__name__)

__all__ = ["FIRED_ACTION", "NOT_LIVE", "NOT_POSTED", "ORIGIN", "OUTCOME_ACTION", "fire_follow_up"]

# The message's provenance (chat_messenger's source; auto_message announces it).
ORIGIN = "follow_up"
# What refused or failed a follow-up, as its run's error_class.
NOT_LIVE = "not_live"
NOT_POSTED = "not_posted"
FIRED_ACTION = "proactive:follow_up_fired"
OUTCOME_ACTION = "proactive:follow_up_outcome"
AUDIT_RESOURCE = "follow_up"
TEXT_PART = "text"

# The origin chat's newest messages, while the chat is the person's own in this workspace.
_LAST_MESSAGES = text("""
    SELECT m.role, m.parts FROM messages m JOIN chats c ON c.id = m.chat_id
    WHERE m.chat_id = CAST(:chat AS uuid) AND c.workspace_id = CAST(:ws AS uuid) AND c.user_id = :user
      AND m.role IN ('user', 'assistant')
    ORDER BY m.created_at DESC, m.id DESC
    LIMIT :limit
""")


@dataclass(frozen=True)
class _FollowUp:
    """The fired row as plain values, read once: whose it is, where it came from and why."""

    id: int
    workspace_id: UUID
    user_id: Optional[int]
    agent_id: Optional[int]
    chat_id: Optional[str]
    reason: str
    scheduled_at: Optional[datetime]


@dataclass(frozen=True)
class _Plan:
    """A follow-up that goes to the model: its run, its task and start message (in the person's
    language), and its model."""

    record: RunRecord
    task: str
    start: str
    llm: Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _follow_up(task: Any, task_id: int) -> _FollowUp:
    user_id, agent_id = getattr(task, "for_user_id", None), getattr(task, "target_agent_id", None)
    chat_id = getattr(task, "origin_chat_id", None)
    return _FollowUp(id=int(task_id), workspace_id=UUID(str(task.workspace_id)),
                     user_id=int(user_id) if user_id is not None else None,
                     agent_id=int(agent_id) if agent_id is not None else None,
                     chat_id=str(chat_id) if chat_id else None, reason=str(getattr(task, "description", "") or ""),
                     scheduled_at=getattr(task, "created_at", None))


def _audit(db: Session, follow_up: _FollowUp, action: str, details: Dict[str, Any]) -> None:
    """A workspace audit row for the follow-up, after its commit (W1's K15); never its reason."""
    from core.workspaces.audit import AuditService

    AuditService(db).log(workspace_id=str(follow_up.workspace_id), user_id=follow_up.user_id, action=action,
                         actor_type="agent", resource_type=AUDIT_RESOURCE, resource_id=str(follow_up.id),
                         details={"agent_id": follow_up.agent_id, **details})


def _refusal(db: Session, follow_up: _FollowUp, now: datetime) -> Optional[Refusal]:
    """Why the follow-up may not reach the model, or None: the person, the mode, then W1's gates."""
    from services.push.recipients import AUTO_MESSAGE, recipients

    person = follow_up.user_id
    if person is None or not recipients(db, follow_up.workspace_id, AUTO_MESSAGE, person):
        return Refusal(RUN_SKIPPED_OFF, NO_ADDRESSEE)
    mode = effective_mode(db, follow_up.workspace_id)
    if mode != MODE_LIVE:
        return Refusal(RUN_SKIPPED_OFF, SWITCHED_OFF if mode == MODE_OFF else NOT_LIVE)
    return start_refusal(db, follow_up.workspace_id, person, now, f"follow-up {follow_up.id}", agent_id=None,
                         follow_up=True)


def _said(parts: Any) -> str:
    texts = [str(part.get("text") or "") for part in (parts or [])
             if isinstance(part, Mapping) and part.get("type") == TEXT_PART]
    return " ".join(" ".join(texts).split())


def _last_messages(db: Session, follow_up: _FollowUp) -> Tuple[Dict[str, str], ...]:
    """The origin chat's last messages, oldest first, as ``{role, text}``; none when the chat is
    gone or no longer the person's."""
    if follow_up.chat_id is None or follow_up.user_id is None:
        return ()
    rows = db.execute(_LAST_MESSAGES, {"chat": follow_up.chat_id, "ws": str(follow_up.workspace_id),
                                       "user": follow_up.user_id,
                                       "limit": config.PROACTIVE_FOLLOW_UP_CONTEXT_MESSAGES}).fetchall()
    said = ({"role": str(row.role), "text": _said(row.parts)} for row in reversed(rows))
    return tuple(message for message in said if message["text"])


def _language(db: Session, follow_up: _FollowUp, now: datetime) -> str:
    """The person's language, as their brief reads it (W2's K7)."""
    from services.brief.strings import brief_language
    from services.mobile.digest import person_today

    return brief_language(person_today(db, follow_up.workspace_id, follow_up.user_id, now).locale)


def _prepare(db: Session, follow_up: _FollowUp, record: RunRecord, now: datetime) -> Union[CheckInResult, _Plan]:
    """The follow-up up to its model call: refused, or a plan."""
    refusal = _refusal(db, follow_up, now)
    if refusal is not None:
        return record_refusal(db, record, refusal)
    open_titles, hidden_titles = known_titles(db, follow_up.workspace_id, follow_up.user_id, now)
    facts = FollowUpFacts(follow_up_id=follow_up.id, reason=follow_up.reason, scheduled_at=follow_up.scheduled_at,
                          messages=_last_messages(db, follow_up), open_titles=open_titles, hidden_titles=hidden_titles)
    language = _language(db, follow_up, now)
    return _Plan(record=replace(record, window_until=now), task=follow_up_task(language, facts, now),
                 start=start_message(language, now), llm=heartbeat_llm(follow_up.workspace_id, follow_up.agent_id))


async def _converse(db: Session, follow_up: _FollowUp, plan: _Plan) -> LoopResult:
    """Build the turn's context and run it, as one ask in the ``check_in`` lane, for the person."""
    from core.llm.usage_context import usage_scope
    from modules.context import ContextMode, ContextService
    from services.credits.asks import ask_scope

    record = plan.record
    with (
        usage_scope(request_type=CHECK_IN_PURPOSE, agent_id=record.agent_id, workspace_id=str(record.workspace_id),
                    inherit=False),
        ask_scope(CHECK_IN_PURPOSE, workspace_id=record.workspace_id, background=True, new=True),
    ):
        context = await ContextService(db).build_context(
            mode=ContextMode.CHECK_IN, agent=AUTO, workspace_id=str(record.workspace_id),
            task_description=plan.task, model_id=getattr(getattr(plan.llm, "config", None), "model", None),
        )
        spec = LoopSpec(run_id=record.run_id, workspace_id=record.workspace_id, agent_id=record.agent_id,
                        system_prompt=context.system_prompt, start_message=plan.start,
                        tools=tuple(context.tools or ()), reads=(evidence_ref(follow_up.id),),
                        user_id=follow_up.user_id)
        return await run_loop(db, plan.llm, spec)


def _post(db: Session, follow_up: _FollowUp, said: str) -> bool:
    """Post the one message: in the origin chat while it is the person's, else their Auto thread.
    The messenger is handed the person only when it finds this very person (``recorded_self``);
    otherwise none, and no chat to honour."""
    from services.chat_messenger import deliver_background_message
    from services.mobile.digest import recorded_self

    person = recorded_self(db, follow_up.user_id)
    message = deliver_background_message(
        db, workspace_id=follow_up.workspace_id, text=posted_text(said),
        source={"origin": ORIGIN, "follow_up_id": follow_up.id},
        # No person the messenger can find: no origin chat either, so its owner check is never skipped.
        chat_id=follow_up.chat_id if person else None, clerk_user_id=person,
    )
    return message is not None


def _ending(db: Session, follow_up: _FollowUp, record: RunRecord, result: LoopResult) -> Tuple[str, Optional[str], int]:
    """How the turn ended: its outcome, its error class and the suggestions it made. A message
    is posted here, before the run row, so ``posted`` is written only once it landed."""
    made = suggestions_made(db, record.workspace_id, record.run_id)
    if result.error_class:
        return RUN_ERROR, result.error_class, made
    if made:
        return RUN_SUGGESTED, None, made
    if is_silent(result.final_text):
        return RUN_SILENT, None, 0
    return (RUN_POSTED, None, 0) if _post(db, follow_up, str(result.final_text)) else (RUN_ERROR, NOT_POSTED, 0)


def _spent(record: RunRecord, result: LoopResult) -> RunRecord:
    """The run with what its turn spent, so an error after the model call still names its model."""
    return replace(record, model=result.model, tokens=result.tokens, cost_usd=result.cost_usd)


def _finish(db: Session, follow_up: _FollowUp, record: RunRecord, result: LoopResult) -> CheckInResult:
    """End the turn and write the run row (``record`` carries its spend): its suggestions, and how it ended."""
    outcome, error_class, made = _ending(db, follow_up, record, result)
    write_run(db, replace(record, outcome=outcome, suggestions=made, error_class=error_class))
    logger.info("[follow-up] %d fired as run %s in %s: %s, $%.4f", follow_up.id, record.run_id,
                record.workspace_id, outcome, result.cost_usd)
    return CheckInResult(record.run_id, outcome, made, error_class)


async def _run(db: Session, follow_up: _FollowUp, record: RunRecord, now: datetime) -> CheckInResult:
    """Prepare, converse and record the follow-up's one turn; any failure is its run's error,
    keeping what the turn spent once it ran (the model-run cap counts it)."""
    try:
        planned = await asyncio.to_thread(_prepare, db, follow_up, record, now)
        if isinstance(planned, CheckInResult):
            return planned
        record = planned.record
        result = await _converse(db, follow_up, planned)
        record = _spent(record, result)
        return await asyncio.to_thread(_finish, db, follow_up, record, result)
    except Exception as exc:  # noqa: BLE001 — recorded as the run's error, by its class; never retried
        logger.exception("[follow-up] %d (run %s in %s) failed", follow_up.id, record.run_id, record.workspace_id)
        return await asyncio.to_thread(record_error, db, record, failure_class(exc))


async def fire_follow_up(db: Session, task: Any, task_id: int) -> CheckInResult:
    """Fire one follow-up row (already completed by ``count_the_run``): one run, silent, one
    message or a suggestion, its firing and outcome audited. Never retried, never re-registered."""
    follow_up = _follow_up(task, task_id)
    record = RunRecord(run_id=uuid4(), workspace_id=follow_up.workspace_id, kind=KIND_FOLLOW_UP,
                       outcome=RUN_SILENT, agent_id=follow_up.agent_id)
    await asyncio.to_thread(_audit, db, follow_up, FIRED_ACTION, {"run_id": str(record.run_id)})
    result = await _run(db, follow_up, record, _now())
    if result.outcome == RUN_SUGGESTED:
        await deliver_run(db, result.run_id)
    await asyncio.to_thread(_audit, db, follow_up, OUTCOME_ACTION, {
        "run_id": str(result.run_id), "outcome": result.outcome, "error_class": result.error_class})
    return result
