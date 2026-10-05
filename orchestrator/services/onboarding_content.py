"""F314 (night 9): Auto opened a workspace already in use with its onboarding quiz.

Night 9's workspace (a coffee roaster with 12 of the owner's documents and a shop
database) sat at onboarding stage ``not_started``. The owner's first question, "A
café wants 10 kg of coffee next week. What do we charge them for delivery?", got
"Before we dive into the coffee order, could you tell me a little about your
business? 1. What's your business? …" (chat 9928b259). Auto saved the answers the
owner then gave but never moved the stage, so all night every Auto turn carried the
quiz's guidance (OnboardingSection) and was pinned to AutoBrain's Tier 0 ("onboarding
active"), ahead of the routes that would have read the board for "What needs me?"
(F307).

Onboarding that has not started is not offered to a workspace already in use: one
with documents, or cards on its board. Once the owner starts it (any stage past
``not_started``) or asks for it ("set up my workspace", OnboardingSection's own
trigger), it runs as before. ``quiz_not_wanted`` answers that; the decorator
``not_for_a_workspace_in_use`` puts it into ``onboarding_state.is_onboarding_active``,
which AutoBrain's Tier 0 and the tool router's onboarding prior both read.

F337 (night 10): the next night's workspace had moved on to ``questions`` (answers
saved on night 9, before F314), so the not-started rule passed it by. Asked "Can you
make me a wholesale price list for cafés, as a spreadsheet?" (chat d2564ecf), Auto
called no tool and asked "What's your business? … How comfortable are you with AI?".
A quiz the owner is doing moves: each answer is saved within minutes. One left at
``questions`` for ``QUIZ_IDLE_HOURS`` in a workspace in use was never theirs, so it is
not offered either; "set up my workspace" still starts it again, and a quiz answered
in the last hours runs as before.
"""
from __future__ import annotations

import functools
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, object_session
from sqlalchemy.orm.exc import UnmappedInstanceError

logger = logging.getLogger(__name__)

# A workspace in use: the owner's documents (or any document) or a card on its board.
_IN_USE = text(
    "SELECT EXISTS (SELECT 1 FROM documents WHERE workspace_id = CAST(:ws AS uuid)) "
    "OR EXISTS (SELECT 1 FROM board_tasks WHERE workspace_id = CAST(:ws AS uuid))"
)
# F337: the quiz's stage, and how long it may sit unanswered in a workspace in use.
QUIZ_STAGE = "questions"
QUIZ_IDLE_HOURS = 2


def in_use(db: Session, workspace_id: Any) -> bool:
    """Whether the workspace already holds documents or cards. Only a real ``True``
    from the database counts: anything else reads as not in use."""
    return db.execute(_IN_USE, {"ws": str(workspace_id)}).scalar() is True


def _stamp(raw: Any) -> Optional[datetime]:
    """An ISO time from the onboarding document, in UTC; None when it isn't one."""
    try:
        when = datetime.fromisoformat(str(raw))
    except (TypeError, ValueError):
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def quiz_left_idle(workspace: Any, now: Optional[datetime] = None) -> bool:
    """True when the quiz stage has seen no answer or advance for ``QUIZ_IDLE_HOURS``
    (F337): its last write, or the time it reached ``questions``. No time on record
    reads as idle: such a row predates the stamps."""
    from services.onboarding_state import get_onboarding

    doc = get_onboarding(workspace)
    stamps = [_stamp(doc.get("updated_at")), _stamp((doc.get("stages") or {}).get(QUIZ_STAGE))]
    latest = max((s for s in stamps if s is not None), default=None)
    return latest is None or (now or datetime.now(timezone.utc)) - latest > timedelta(hours=QUIZ_IDLE_HOURS)


def quiz_not_wanted(db: Optional[Session], workspace: Any, workspace_id: Any) -> bool:
    """True when the workspace is already in use and onboarding has not started, or
    sits idle at the quiz (F337), so the quiz is not offered. A read that fails leaves
    onboarding as it was."""
    from services.onboarding_state import INITIAL_STAGE, current_stage

    if db is None or workspace is None or not workspace_id:
        return False
    stage = current_stage(workspace)
    if stage != INITIAL_STAGE and not (stage == QUIZ_STAGE and quiz_left_idle(workspace)):
        return False
    try:
        with db.begin_nested():
            return in_use(db, workspace_id)
    except SQLAlchemyError:
        logger.exception("[F314] could not tell whether workspace %s is in use; onboarding left as it was",
                         workspace_id)
        return False


def _session_of(workspace: Any) -> Optional[Session]:
    """The session a loaded Workspace row belongs to; None for anything else."""
    try:
        return object_session(workspace)
    except UnmappedInstanceError:
        return None


def not_for_a_workspace_in_use(is_active: Callable[[Any], bool]) -> Callable[[Any], bool]:
    """Wrap ``onboarding_state.is_onboarding_active``: onboarding that has not started
    is not active in a workspace already in use."""
    @functools.wraps(is_active)
    def wrapped(workspace: Any) -> bool:
        if not is_active(workspace):
            return False
        return not quiz_not_wanted(_session_of(workspace), workspace, getattr(workspace, "id", None))
    return wrapped


__all__ = ["in_use", "not_for_a_workspace_in_use", "quiz_not_wanted"]


def answers_alone_never_start_it(set_segment: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``onboarding_state.set_segment``: in a workspace already in use whose
    onboarding has not started, saved answers are kept but do not start it.

    Night 9 (chat 9928b259): the owner, asked about their business in a workspace with
    12 documents, answered two of the three questions and said "Now please just
    answer". Saving those answers moved the stage on by itself (``implied_stage``,
    added for a new workspace whose model forgot ``advance_to``), and from then on
    the quiz's guidance and AutoBrain's Tier 0 pin rode every turn of the night. In a
    workspace in use, onboarding starts only when it is started: the owner asks to set
    up, and Auto advances the stage itself.
    """
    @functools.wraps(set_segment)
    def wrapped(db: Any, workspace: Any, segment: dict, *, commit: bool = True) -> Any:
        if not quiz_not_wanted(db, workspace, getattr(workspace, "id", None)):
            return set_segment(db, workspace, segment, commit=commit)
        return _kept_without_starting(db, workspace, segment, commit)
    return wrapped


def _kept_without_starting(db: Any, workspace: Any, segment: dict, commit: bool) -> Any:
    """The answers merged into the onboarding document, the stage left as it is."""
    from services import onboarding_state as state

    cleaned = state._clean_segment(segment)
    if not cleaned:
        raise ValueError("set_segment requires at least one of business/goal/comfort")
    doc = state.get_onboarding(workspace)
    doc = {**doc, "segment": {**(doc.get("segment") or {}), **cleaned}, "updated_at": state._now_iso()}
    logger.info("[F314] onboarding answers kept for workspace %s; it is in use, so they don't start onboarding",
                getattr(workspace, "id", None))
    return state._persist(db, workspace, doc, commit=commit)
