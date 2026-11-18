"""Ticket PR fields: a ``runtime_ref`` key written by its own statement survives every whole-document write.

``BoardTask.runtime_ref`` has many writers that rebuild the whole document from a copy read
earlier in their request and assign it back: the CLI host's event flush
(``cli_host_service._absorb_events``), its result (``apply_result``), a claim (``_claim_ref``
builds a fresh document and carries only its own record), the board's run stamps. A key that a
single ``jsonb_set`` wrote in the meantime is lost the moment one of them commits.

``pull_request`` (``services/ticket_pull_request.py``) has exactly one writer, a ``jsonb_set``
of that key run inside a row lock, so the row always holds its latest value. Just before any ORM
UPDATE that writes ``runtime_ref``, this listener takes the row lock that UPDATE is about to
take, reads the stored value, and puts it back into the document being written; when the row
holds none (the owner took it off: ``ticket_pull_request.clear_pull_request``), a stale copy's is
dropped, so nothing an ORM writer read earlier can put it back. Under the lock nothing can write
the key in between: a writer that came first is read here, and one that comes later waits for
this commit and writes on top of it. No ORM writer ever adds the key itself.

The lock is the UPDATE's own (``FOR NO KEY UPDATE``: an UPDATE that changes no key column takes
that one, which a child ticket's foreign-key check does not wait on), taken one statement earlier
in the same transaction, so it adds no new wait and no new lock order. Raw SQL never reaches the
listener, and no raw statement rewrites the whole document: each one sets, merges or removes a
single key (``jsonb_set``, ``|| jsonb_build_object``, ``- key``). Postgres only: another dialect (a unit test on
SQLite) is left alone. The session's notes are NOT kept this way: ORM writers append to them too
(``session_folder_reach.notes_with_unreadable_folder``), so the stored list cannot simply win.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import event, inspect, text

from core.models.core import BoardTask

PULL_REQUEST_KEY = "pull_request"
# The keys whose stored value wins over a whole-document write: each has one writer, a single statement.
KEPT_KEYS = (PULL_REQUEST_KEY,)
POSTGRES = "postgresql"

_STORED_UNDER_LOCK = text(
    "SELECT runtime_ref -> CAST(:key AS text) FROM board_tasks WHERE id = :task_id FOR NO KEY UPDATE"
)


def with_stored_keys(ref: Any, stored: dict) -> Any:
    """``ref`` with each kept key as the row holds it (a new dict): the stored value, or no key at
    all when the row holds none. ``ref`` itself when it already agrees. Pure."""
    current = ref if isinstance(ref, dict) else {}
    if all(current.get(key) == value for key, value in stored.items()):
        return ref
    kept = {key: value for key, value in current.items() if key not in stored or stored[key] is not None}
    return {**kept, **{key: value for key, value in stored.items() if value is not None}}


@event.listens_for(BoardTask, "before_update")
def _keep_stored_keys(mapper: Any, connection: Any, task: BoardTask) -> None:
    """Put each kept key's stored value back into a ``runtime_ref`` this flush rewrites."""
    if connection.dialect.name != POSTGRES or task.id is None:
        return
    if not inspect(task).attrs.runtime_ref.history.has_changes():
        return
    stored = {
        key: connection.execute(_STORED_UNDER_LOCK, {"key": key, "task_id": task.id}).scalar()
        for key in KEPT_KEYS
    }
    kept = with_stored_keys(task.runtime_ref, stored)
    if kept is not task.runtime_ref:
        task.runtime_ref = kept
