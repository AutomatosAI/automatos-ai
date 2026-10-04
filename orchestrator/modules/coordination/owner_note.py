"""F291 (night 8): the owner's note on a mission's plan reaches every step.

#0214 at 00:52: the owner approved a mission's plan with "Plan is fine. Use our real
Thursday delivery day and keep the email short." The card's Approve was refused, and
the mission's own Approve took no note, so the note went nowhere and the steps ran
without it.

An Approve of the plan with a note, from the card or from the mission, now keeps the
note on the mission (``run.config[OWNER_NOTE_KEY]``), and every step's prompt carries
it after the mission's goal, in the owner's words: a plain step's
(``MissionDispatcher.build_task_prompt``) and the synthesis step's
(``CoordinatorService._build_synthesis_prompt``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, Optional

from sqlalchemy.orm import Session, object_session
from sqlalchemy.orm.exc import UnmappedInstanceError

logger = logging.getLogger(__name__)

OWNER_NOTE_KEY = "owner_note"
OWNER_NOTE_CHARS = 2000
OWNER_NOTE_HEADING = "## The owner's note on this mission"
OWNER_NOTE_RULE = "The owner wrote this when they approved the plan. Follow it in this step's work."


def note_text(note: Any) -> str:
    """The note as it is kept: its text, trimmed to ``OWNER_NOTE_CHARS``; '' for none."""
    return str(note).strip()[:OWNER_NOTE_CHARS] if isinstance(note, str) else ""


def keep_owner_note(run: Any, note: Any) -> bool:
    """Keep the owner's approval ``note`` on ``run``: its config, rebuilt, never changed
    in place. The caller commits it with the approval. False when there is no note."""
    text = note_text(note)
    if not text:
        return False
    run.config = {**(run.config or {}), OWNER_NOTE_KEY: text}
    return True


def owner_note_of(task: Any) -> str:
    """The note the owner approved ``task``'s mission with, or ''."""
    db = _session_of(task)
    run_id = getattr(task, "run_id", None)
    if db is None or run_id is None:
        return ""
    from core.models.orchestration import OrchestrationRun

    run = db.get(OrchestrationRun, run_id)
    config: Dict[str, Any] = (run.config or {}) if run is not None else {}
    return note_text(config.get(OWNER_NOTE_KEY))


def with_the_owners_note(prompt: str, task: Any) -> str:
    """``prompt``, followed by the owner's note on the step's mission when there is one."""
    note = owner_note_of(task)
    return f"{prompt}\n\n{OWNER_NOTE_HEADING}\n{OWNER_NOTE_RULE}\n\n{note}" if note else prompt


def a_steps_prompt_carries_the_owners_note(build: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``MissionDispatcher.build_task_prompt`` (the step first)."""
    @functools.wraps(build)
    def wrapped(task: Any, *args: Any, **kwargs: Any) -> str:
        return with_the_owners_note(build(task, *args, **kwargs), task)
    return wrapped


def _session_of(task: Any) -> Optional[Session]:
    try:
        db = object_session(task)
    except UnmappedInstanceError:      # a plain object standing in for a step
        return None
    return db if isinstance(db, Session) else None


__all__ = ["OWNER_NOTE_CHARS", "OWNER_NOTE_HEADING", "OWNER_NOTE_KEY", "a_steps_prompt_carries_the_owners_note",
           "keep_owner_note", "note_text", "owner_note_of", "with_the_owners_note"]
