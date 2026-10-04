"""PRD-252 R2 — what an approval leaves on the ticket it decides: its result and
the owner's note.

F195: the approval's result goes on the ticket only while the ticket is still that
approval's 'done', so a send-back that landed while the approval's action ran
keeps the ticket as it sent it back.

F038 (night 1): ``approve`` threw its body away, so a note written while approving
was lost. The note now goes into the ticket's notes (``runtime_ref.session_notes``,
the list the viewer shows beside the operator's and the session's own notes), in
the approval's own transaction and only when the approval stands. It is a record
for people, not agent memory: no prompt reads it.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.core import BoardTask

# How an approval's note reads among the ticket's notes.
APPROVAL_NOTE_PREFIX = "Approved: "
# The operator's own notes say "you" on the ticket, as an operator note sent with
# PATCH /api/v1/tasks/{id} does.
OPERATOR_NOTE_BY = "you"


def record_approval(db: Session, task_id: int, *, workspace_id: Any, decided_at: datetime,
                    action_result: Optional[Dict[str, Any]], note: str = "") -> bool:
    """Put the approval's result, and its note when it has one, on the ticket while
    it is still this approval's 'done' (F195); the caller commits. True when the
    ticket is still this approval's."""
    kept = (func.coalesce(func.nullif(BoardTask.result, ""), json.dumps(action_result))
            if action_result else BoardTask.result)
    still_approved = bool(
        db.query(BoardTask)
        .filter(BoardTask.id == task_id, BoardTask.status == "done", BoardTask.completed_at == decided_at)
        .update({BoardTask.result: kept}, synchronize_session=False)
    )
    if still_approved and note:
        _keep_note(db, task_id=task_id, workspace_id=workspace_id, note=note)
    return still_approved


def keep_approval_note(db: Session, *, task_id: int, workspace_id: Any, note: str) -> None:
    """An approval's note from outside the board's Approve (Auto approving in chat
    on the owner's word, F259): the same words and the same "you" on the ticket."""
    _keep_note(db, task_id=task_id, workspace_id=workspace_id, note=note)


def _keep_note(db: Session, *, task_id: int, workspace_id: Any, note: str) -> None:
    """One jsonb append in the caller's transaction (append_session_note), never a
    whole-document write over a CLI host's concurrent event flush. The note is cut
    to leave room for its prefix within the notes' own bound."""
    from services.cli_host_service import MAX_ASK_QUESTION_KEPT, append_session_note

    room = MAX_ASK_QUESTION_KEPT - len(APPROVAL_NOTE_PREFIX)
    append_session_note(db, task_id=task_id, workspace_id=workspace_id,
                        note=f"{APPROVAL_NOTE_PREFIX}{note[:room]}", by=OPERATOR_NOTE_BY)
