"""The provenance record: write a marked file's row, and answer the public check (PRD-264 EUAI-10).

* **Write.** ``record_marked_file`` adds one ``content_provenance`` row when Automatos
  produces a marked file: media-render outputs (EUAI-06b) and generated documents and
  images (EUAI-09) call it. Chat text is marked in its payload only and is not recorded.
  The row joins the caller's transaction; the caller commits. Writing the same file
  twice in a workspace keeps the first row.
* **Answer.** ``provenance_answer`` is what the public route returns for a hash:
  ``{"made_by_automatos_ai": true, "kind": ..., "date": "YYYY-MM-DD"}`` from the
  earliest row with that hash, or ``{"made_by_automatos_ai": false}``. It reads only
  the kind and the date, so it can never carry a workspace, a person or a file name.
* **Record as the file is made.** ``record_produced_file`` is the writer's best-effort
  form for code that has no session of its own to commit (EUAI-09's documents and
  images): its own session, off the event loop (F105), committed at once.
"""
from __future__ import annotations

import logging
import re
from datetime import timezone
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.best_effort import off_loop
from core.models.content_provenance import (
    GENERATOR_MAX_LENGTH,
    MANIFEST_ID_MAX_LENGTH,
    PROVENANCE_KINDS,
    UNIQUE_FILE_CONSTRAINT,
    ContentProvenance,
)

logger = logging.getLogger(__name__)

# A SHA-256 as the route and the table take it: 64 lower-case hex characters, nothing else.
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
ANSWER_KEY = "made_by_automatos_ai"


def is_sha256_hex(value: object) -> bool:
    """True only for a string of exactly 64 lower-case hex characters."""
    return isinstance(value, str) and _SHA256_HEX.fullmatch(value) is not None


def _checked_text(value: object, field: str, max_length: int) -> str:
    """``value`` stripped, or a ValueError naming ``field`` when it is empty or too long."""
    text_value = value.strip() if isinstance(value, str) else ""
    if not text_value or len(text_value) > max_length:
        raise ValueError(f"{field} must be 1 to {max_length} characters")
    return text_value


def record_marked_file(
    db: Session,
    workspace_id: UUID | str,
    sha256: str,
    kind: str,
    generator: str,
    manifest_id: Optional[str] = None,
) -> bool:
    """Record a marked file Automatos produced, in the caller's transaction.

    ``sha256`` is the final file's digest (after its last encode and its mark), as 64
    lower-case hex characters; ``kind`` is one of ``PROVENANCE_KINDS``; ``generator``
    names the toolkit or model, never a person. Raises ValueError on any other input,
    including a ``workspace_id`` that is not a UUID.
    Returns True when a row was added, False when the workspace already holds that file.
    """
    if not is_sha256_hex(sha256):
        raise ValueError("sha256 must be 64 lower-case hex characters")
    if kind not in PROVENANCE_KINDS:
        raise ValueError(f"kind must be one of {', '.join(PROVENANCE_KINDS)}")
    values = {
        "workspace_id": UUID(str(workspace_id)),
        "sha256": sha256,
        "kind": kind,
        "generator": _checked_text(generator, "generator", GENERATOR_MAX_LENGTH),
        "manifest_id": None if manifest_id is None
        else _checked_text(manifest_id, "manifest_id", MANIFEST_ID_MAX_LENGTH),
    }
    statement = insert(ContentProvenance).values(**values).on_conflict_do_nothing(
        constraint=UNIQUE_FILE_CONSTRAINT,
    )
    return db.execute(statement).rowcount == 1


@off_loop
def record_produced_file(
    workspace_id: UUID | str,
    sha256: str,
    kind: str,
    generator: str,
    manifest_id: Optional[str] = None,
) -> Optional[bool]:
    """Record a marked file in a session of its own, committed at once (EUAI-09).

    Called on an event loop, the write runs on the best-effort threads and this returns
    None at once (``core.best_effort.off_loop``); anywhere else it runs inline. A write
    that fails is rolled back, logged with its kind and workspace and returns False: the
    file it describes is already marked and delivered, and only its hash lookup misses.
    The hash itself is never logged (FR-17): a log line must not let anyone tie a file
    to the workspace that made it, so the failure is logged by its class (a database
    error's text carries the statement's parameters, the hash among them).
    Inline, True when a row was added and False when it was not (already there, or failed).
    """
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        added = record_marked_file(db, workspace_id, sha256, kind, generator, manifest_id=manifest_id)
        db.commit()
        return added
    except (SQLAlchemyError, ValueError) as e:
        db.rollback()
        logger.error("[provenance] a %s file of workspace %s was not recorded (%s)",
                     kind, workspace_id, type(e).__name__)
        return False
    finally:
        db.close()


def provenance_answer(db: Session, sha256: str) -> Dict[str, Any]:
    """The public answer for ``sha256``: yes with the kind and UTC date of the first mark, or no.

    The caller validates the hash (``is_sha256_hex``). Only ``kind`` and ``created_at``
    are read, so the answer holds nothing about the workspace that made the file.
    """
    first = (
        db.query(ContentProvenance.kind, ContentProvenance.created_at)
        .filter(ContentProvenance.sha256 == sha256)
        .order_by(ContentProvenance.created_at.asc(), ContentProvenance.id.asc())
        .first()
    )
    if first is None:
        return {ANSWER_KEY: False}
    made_on = first.created_at.astimezone(timezone.utc).date().isoformat()
    return {ANSWER_KEY: True, "kind": first.kind, "date": made_on}
