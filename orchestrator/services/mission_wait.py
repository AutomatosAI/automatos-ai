"""F170 (night 5, B52/B62) — a mission waiting behind another's Claude Code step says so.

Mission 81e8f37a was approved at 17:34:19 and shown running, with its first
steps queued. Nothing was said until 18:08:02, seven seconds after mission
fb8e0ef4's second Claude Code step (#1046) ended. The coordinator tick awaits
each Claude Code step inline (F160, a held design), so while one runs no other
mission's step is dispatched: 418 ticks were skipped in that window.

Until F160 changes that, the wait is named: on the mission (a RUN_WAITING event
whose stop_detail the mission page shows), in the chat the mission came from,
and in Auto's reply to an approval. The tick records which step it awaits
(``awaiting_session_step``: bookkeeping only, cleared in a finally). With no
record (another process, or after a restart) the blocker is inferred from the
open Claude Code step cards and worded "probably". Another workspace's step is
only ever "another mission's step".
"""
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.core import BoardTask
from core.models.orchestration import OrchestrationEvent, OrchestrationRun, OrchestrationTask
from core.models.orchestration_enums import ActorType, EventType, RunState, TaskState

logger = logging.getLogger(__name__)

APPROVED_LEAD = "Approved. It {maybe}starts when"
WAITING_LEAD = "Waiting: this mission's next steps {maybe}start when"
WAIT_TAIL = ": while a Claude Code step runs, other missions' steps wait"
ANOTHER_MISSION = "another mission's step"
GOAL_SHOWN_CHARS = 80
# The run's own work in flight: then it is not waiting behind another mission.
OWN_WORK_STATES = (TaskState.ASSIGNED.value, TaskState.RUNNING.value, TaskState.VERIFYING.value,
                   TaskState.RETRYING.value)
SESSION_CARD_OPEN = ("assigned", "in_progress", "blocked")


# ── (3) the record: the Claude Code steps the tick is awaiting (in-process) ──

@dataclass(frozen=True)
class AwaitedStep:
    run_id: str
    task_id: str
    workspace_id: str
    title: str
    since: datetime
    ticket_id: Optional[int] = None


_lock = threading.Lock()
_awaited: Dict[str, AwaitedStep] = {}


@contextmanager
def awaiting_session_step(*, run_id: Any, task_id: Any, workspace_id: Any, title: Any) -> Iterator[Callable[[Any], None]]:
    """The tick awaits this step's Claude Code session. Recorded for the length
    of the wait and cleared in a finally; yields the lane's ``on_poll``, which
    notes the ticket. Bookkeeping only: nothing here changes the wait."""
    key = str(task_id)
    with _lock:
        _awaited[key] = AwaitedStep(str(run_id), key, str(workspace_id), str(title or ""),
                                    datetime.now(timezone.utc))
    try:
        yield lambda ticket: _saw_ticket(key, ticket)
    finally:
        with _lock:
            _awaited.pop(key, None)


def _saw_ticket(key: str, ticket: Any) -> None:
    ticket_id = getattr(ticket, "id", None)
    with _lock:
        step = _awaited.get(key)
        if step is not None and step.ticket_id != ticket_id:
            _awaited[key] = replace(step, ticket_id=ticket_id)


def awaited_steps() -> List[AwaitedStep]:
    with _lock:
        return list(_awaited.values())


# ── what a run waits behind, and how that is said ─────────────────────────────

@dataclass(frozen=True)
class Blocker:
    step_title: str
    goal: str
    ticket_id: Optional[int]
    since: Optional[datetime]
    same_workspace: bool
    certain: bool          # from the tick's record, not inferred


def blocker_for(db: Session, run: OrchestrationRun) -> Optional[Blocker]:
    """A Claude Code step of ANOTHER mission that ``run``'s steps wait behind."""
    others = sorted((s for s in awaited_steps() if s.run_id != str(run.id)), key=lambda s: s.since)
    if others:
        step = others[0]
        same = step.workspace_id == str(run.workspace_id)
        goal = (db.query(OrchestrationRun.goal).filter(OrchestrationRun.id == step.run_id).scalar() or "") if same else ""
        return Blocker(step.title, goal, step.ticket_id, step.since, same, certain=True)
    row = (
        db.query(BoardTask.id, BoardTask.title, BoardTask.started_at, OrchestrationRun.goal,
                 OrchestrationRun.workspace_id)
        .join(OrchestrationRun, OrchestrationRun.id == BoardTask.orchestration_run_id)
        .filter(
            BoardTask.source_id.like("mission:%"),        # a card the session lane runs (F094)
            BoardTask.status.in_(list(SESSION_CARD_OPEN)),
            OrchestrationRun.state == RunState.RUNNING.value,
            OrchestrationRun.id != run.id,
        )
        .order_by(BoardTask.started_at.asc().nullslast())
        .first()
    )
    if row is None:
        return None
    card_id, title, started_at, goal, workspace_id = row
    same = str(workspace_id) == str(run.workspace_id)
    return Blocker(str(title or ""), str(goal or "") if same else "", card_id, started_at, same, certain=False)


def _clip(text: str) -> str:
    line = " ".join(text.split())
    return line if len(line) <= GOAL_SHOWN_CHARS else line[:GOAL_SHOWN_CHARS - 1] + "…"


def wait_sentence(blocker: Blocker, *, at_approval: bool) -> str:
    """"Approved. It starts when 'X' of mission 'Y' finishes: …" in plain words."""
    maybe = "" if blocker.certain else "probably "
    lead = (APPROVED_LEAD if at_approval else WAITING_LEAD).format(maybe=maybe)
    if not blocker.same_workspace:
        return f"{lead} {ANOTHER_MISSION} finishes{WAIT_TAIL}."
    who = f"'{_clip(blocker.step_title)}' of mission '{_clip(blocker.goal)}'"
    detail = [f"ticket #{blocker.ticket_id}"] if blocker.ticket_id else []
    if blocker.since is not None:
        detail.append(f"since {blocker.since.astimezone(timezone.utc):%H:%M} UTC")
    return f"{lead} {who} finishes{WAIT_TAIL}" + (f" ({', '.join(detail)})." if detail else ".")


def _event(db: Session, run: OrchestrationRun, sentence: str, blocker: Blocker) -> None:
    from services.orchestration_state import emit_event

    emit_event(
        db=db, run_id=run.id, event_type=EventType.RUN_WAITING, actor_type=ActorType.COORDINATOR,
        actor_id="coordinator",
        payload={"stop_detail": sentence, "certain": blocker.certain,
                 "ticket_id": blocker.ticket_id if blocker.same_workspace else None},
    )


def _narrate(db: Session, run: OrchestrationRun, sentence: str) -> None:
    """Into the chat the mission came from; on its own session, fail-soft."""
    from services.coordinator_service import _narrate_mission

    _narrate_mission(db, run, sentence, level="run", event="run_waiting")


# ── (1) at approval ──────────────────────────────────────────────────────────

def note_wait_at_approval(db: Session, run: OrchestrationRun) -> Optional[str]:
    """Said once, as the owner approves, when a Claude Code step of another
    mission holds the tick. Never stands in the way of the approval: the
    lookup and the event run in a SAVEPOINT, so a statement that fails there
    rolls back to it and the approval's own transaction goes on."""
    try:
        with db.begin_nested():
            blocker = blocker_for(db, run)
            sentence = wait_sentence(blocker, at_approval=True) if blocker is not None else None
            if sentence is not None:
                _event(db, run, sentence, blocker)
    except Exception:  # noqa: BLE001 -- the mission is approved either way
        logger.warning("[F170] could not say what mission %s waits for", getattr(run, "id", None), exc_info=True)
        return None
    if sentence is not None:
        _narrate(db, run, sentence)
    return sentence


def wait_note_of(db: Session, run_id: Any) -> Optional[str]:
    """The wait sentence said for a run in THIS transaction (by its approval),
    for the reply to that approval; None when none was."""
    try:
        payload = (
            db.query(OrchestrationEvent.payload)
            .filter(OrchestrationEvent.run_id == run_id,
                    OrchestrationEvent.event_type == EventType.RUN_WAITING.value,
                    OrchestrationEvent.created_at >= func.now())
            .order_by(OrchestrationEvent.created_at.desc())
            .limit(1)
            .scalar()
        )
    except Exception:  # noqa: BLE001
        return None
    return (payload or {}).get("stop_detail") if isinstance(payload, dict) else None


# ── (2) a wait that begins later ─────────────────────────────────────────────

def _waiting_since_noted(db: Session, run: OrchestrationRun, cutoff: datetime) -> bool:
    """True when this run's current wait is already said, or began after ``cutoff``."""
    last_other = (
        db.query(func.max(OrchestrationEvent.created_at))
        .filter(OrchestrationEvent.run_id == run.id,
                OrchestrationEvent.event_type != EventType.RUN_WAITING.value)
        .scalar()
    )
    if last_other is None or last_other > cutoff:
        return True
    last_wait = (
        db.query(func.max(OrchestrationEvent.created_at))
        .filter(OrchestrationEvent.run_id == run.id,
                OrchestrationEvent.event_type == EventType.RUN_WAITING.value)
        .scalar()
    )
    return last_wait is not None and last_wait >= last_other


def note_waiting_missions(db: Session, now: datetime, wait_after_s: int) -> List[Tuple[OrchestrationRun, str]]:
    """Every running mission whose steps have all sat queued for ``wait_after_s``
    behind another mission's Claude Code step gets one RUN_WAITING event per
    wait. Returns each run with its sentence, for the caller to narrate once it
    has committed (``narrate_waits``)."""
    cutoff = now - timedelta(seconds=wait_after_s)
    noted: List[Tuple[OrchestrationRun, str]] = []
    for run in db.query(OrchestrationRun).filter(OrchestrationRun.state == RunState.RUNNING.value).all():
        states = [state for (state,) in db.query(OrchestrationTask.state).filter(OrchestrationTask.run_id == run.id)]
        if TaskState.QUEUED.value not in states or any(s in OWN_WORK_STATES for s in states):
            continue
        try:
            with db.begin_nested():          # one run's failure never costs the others their note
                if _waiting_since_noted(db, run, cutoff):
                    continue
                blocker = blocker_for(db, run)
                if blocker is None:
                    continue
                sentence = wait_sentence(blocker, at_approval=False)
                _event(db, run, sentence, blocker)
        except Exception:  # noqa: BLE001
            logger.warning("[F170] could not note the wait of mission %s", run.id, exc_info=True)
            continue
        noted.append((run, sentence))
    return noted


def narrate_waits(db: Session, noted: List[Tuple[OrchestrationRun, str]]) -> None:
    """Each noted wait, into the chat its mission came from (after the commit)."""
    for run, sentence in noted:
        _narrate(db, run, sentence)
