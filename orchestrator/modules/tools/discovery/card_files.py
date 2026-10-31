"""F460 (night 18): platform_get_task names the files on the card.

Auto could not open a file it had just listed: it read #3125's answer (the CLI session's report,
cut at 2,000 characters), which named ``harbour-lights-launch/1-club-email.md`` relative to the
session's folder, and asked the worker for that, then for ``deliverables/sessions/3131/x.md`` and
for the host's absolute path. The card records its files as the worker reads them
(``sessions/3125/harbour-lights-launch/1-club-email.md``: the Deliverables registered on it), and
the answer never said so. Now it lists them, newest first, with the call that reads one.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

FILES_KEY = "deliverable_files"
FILES_NOTE_KEY = "files_note"
FILES_NOTE = ("These are the card's Deliverable files as the workspace reads them: open one with "
              "workspace_read_file {path: <the path exactly as listed>}.")


def card_file_paths(db: Session, workspace_id: Any, card_id: int) -> List[str]:
    """The file of each Deliverable registered on the card, newest first, each once."""
    from services.step_files import NAMED_FILES_PER_CARD, SESSION_FILE_SOURCE_TYPE, listed_or_raise

    listed = listed_or_raise(db, workspace_id, source_type=SESSION_FILE_SOURCE_TYPE, source_id=str(card_id),
                             limit=NAMED_FILES_PER_CARD)
    paths = [str(row.get("file_path")) for row in listed.get("deliverables") or [] if row.get("file_path")]
    return list(dict.fromkeys(paths))


def with_its_files(db: Session, workspace_id: Any, card: Any, answer: Dict[str, Any]) -> Dict[str, Any]:
    """``answer`` (platform_get_task's) with the card's Deliverable files beside it, read in a
    savepoint; as it came when the card has none or they can't be listed (logged)."""
    try:
        with db.begin_nested():
            paths = card_file_paths(db, workspace_id, card.id)
    except Exception:
        logger.exception("[F460] the files on card %s could not be listed", getattr(card, "id", "?"))
        return answer
    if not paths:
        return answer
    return {**answer, "task": {**answer["task"], FILES_KEY: paths, FILES_NOTE_KEY: FILES_NOTE}}


__all__ = ["FILES_KEY", "FILES_NOTE", "card_file_paths", "with_its_files"]
