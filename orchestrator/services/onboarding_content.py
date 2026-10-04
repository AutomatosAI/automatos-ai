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
"""
from __future__ import annotations

import functools
import logging
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


def in_use(db: Session, workspace_id: Any) -> bool:
    """Whether the workspace already holds documents or cards. Only a real ``True``
    from the database counts: anything else reads as not in use."""
    return db.execute(_IN_USE, {"ws": str(workspace_id)}).scalar() is True


def quiz_not_wanted(db: Optional[Session], workspace: Any, workspace_id: Any) -> bool:
    """True when onboarding has not started and the workspace is already in use, so
    the quiz is not offered. A read that fails leaves onboarding as it was."""
    from services.onboarding_state import INITIAL_STAGE, current_stage

    if db is None or workspace is None or not workspace_id or current_stage(workspace) != INITIAL_STAGE:
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
