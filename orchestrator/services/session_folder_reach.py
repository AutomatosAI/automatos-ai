"""F333 (night 10) — a session that runs where Automatos cannot read says so when it starts.

A Claude Code session's files are reachable from the platform only when its folder
maps into a workspace: under the deliverables root (``AUTOMATOS_WORKSPACE_DIR``),
under the projects folder (``LOCAL_PROJECTS_DIR``) or inside the workspace volume.
A host paired with a root anywhere else ran its sessions there, and nothing said so
until every card failed later on. Now the session's start (or its result, whichever
first reports the folder) writes one log line and one note on the ticket naming the
folder and the ones the platform can read.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

NOTE_BY = "Automatos"
NOTE_UNSET = "not set"
UNREADABLE_FOLDER_NOTE = (
    "This session runs in {cwd}, a folder Automatos cannot read: it is not inside the "
    "deliverables folder ({workspace_dir}) or the projects folder ({projects_dir}) this "
    "stack was started with. Files it saves there will not open from Deliverables, the "
    "card cannot check them and later mission steps cannot read them. Start the CLI host "
    "with its --default-root inside one of those folders, or set LOCAL_PROJECTS_DIR to a "
    "folder that holds this one."
)


def unreadable_folder_note(cwd: str, workspace_dir: Optional[str], projects_dir: Optional[str]) -> str:
    """What the owner reads on the ticket when its session's folder is out of reach."""
    return UNREADABLE_FOLDER_NOTE.format(
        cwd=cwd, workspace_dir=workspace_dir or NOTE_UNSET, projects_dir=projects_dir or NOTE_UNSET,
    )


def notes_with_unreadable_folder(
    ref: Dict[str, Any], notes_key: str, *, task_id: Any, cwd: str,
    workspace_dir: Optional[str], projects_dir: Optional[str],
) -> Optional[List[Dict[str, Any]]]:
    """The ticket's notes with the out-of-reach note added (a new list), or ``None``
    when the folder is readable (``ref['explorer_root']`` set) or the note is
    already there — the start and the result both report the folder."""
    if ref.get("explorer_root"):
        return None
    note = unreadable_folder_note(cwd, workspace_dir, projects_dir)
    notes = ref.get(notes_key)
    notes = list(notes) if isinstance(notes, list) else []
    if any(isinstance(n, dict) and n.get("note") == note for n in notes):
        return None
    logger.warning("[cli-host] ticket %s: session folder %s is outside every folder the platform reads "
                   "(deliverables %s, projects %s)", task_id, cwd, workspace_dir, projects_dir)
    return [*notes, {"note": note, "at": datetime.now(timezone.utc).isoformat(), "by": NOTE_BY}]
