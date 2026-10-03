"""The files of a session folder: the host's own, and what the session wrote there.

Moved out of ``session.py`` unchanged (PRD-253), which was past the 800-line bound.
PRD-245 S0.7: what a session writes in its own folder is copied into the ticket's
deliverables folder — never one of the host's own files, by name or by content.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Sequence

from .terminal_log import FILENAME as TERMINAL_LOG_FILENAME

log = logging.getLogger("automatos.cli_host.session")


# What the host itself writes into a session folder — never a deliverable.
HOST_OWNED_SESSION_FILES = frozenset({"ticket.md", "settings.json", "system_prompt.md", TERMINAL_LOG_FILENAME, "mcp.json"})
# The subset that holds this ticket's own credential in PLAINTEXT. Removed the
# moment the turn ends — the row's copy is revoked there too, but a file is what
# gets read later, and every finished ticket used to leave one behind.
CREDENTIAL_SESSION_FILES = ("mcp.json",)
# A host-owned file is excluded by CONTENT as well as by name: a session can read
# one and write it back under another name, and from Wave 1 one of them carries
# the ticket's own credential. Only small files are compared (the terminal log is
# bounded but large, and no session hand-copies it).
MAX_HOST_FILE_COMPARE_BYTES = 256 * 1024


def host_owned_blobs(session_dir: Path) -> List[bytes]:
    """The bytes of the host's own files in this session folder, for the content
    check below. Unreadable or oversized files are simply not compared."""
    blobs: List[bytes] = []
    for name in sorted(HOST_OWNED_SESSION_FILES):
        path = session_dir / name
        try:
            if path.is_file() and path.stat().st_size <= MAX_HOST_FILE_COMPARE_BYTES:
                blobs = [*blobs, path.read_bytes()]
        except OSError:
            continue
    return blobs


def _is_host_copy(path: Path, blobs: Sequence[bytes]) -> bool:
    """True when this file is one of the host's own under another name."""
    try:
        size = path.stat().st_size
        if size > MAX_HOST_FILE_COMPARE_BYTES:
            return False
        candidates = [b for b in blobs if len(b) == size]
        return bool(candidates) and path.read_bytes() in candidates
    except OSError:
        return False


def session_deliverables(files_touched: Sequence[str], session_dir: Path, cwd: Path) -> List[Path]:
    """The files a session wrote inside its own folder, relative to it (PRD-245
    S0.7) — never the host's own files (by name or by content), each once."""
    root = session_dir.resolve()
    blobs = host_owned_blobs(root)
    found: List[Path] = []
    for raw in files_touched:
        path = Path(raw)
        try:
            resolved = (path if path.is_absolute() else cwd / path).resolve()
            rel = resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError):
            continue
        if not rel.parts or str(rel) in HOST_OWNED_SESSION_FILES or rel in found:
            continue
        if _is_host_copy(resolved, blobs):
            log.warning("deliverable %s is a copy of one of the host's own session files — not landed", resolved)
            continue
        found = [*found, rel]
    return found


def land_session_deliverables(relatives: Sequence[Path], session_dir: Path, dest: Path) -> List[str]:
    """Copy each file into the ticket's deliverables folder — created on demand
    (0o755), names kept, an earlier copy overwritten. One file failing is a
    warning, never a lost result. Returns the copies' paths."""
    landed: List[str] = []
    for rel in relatives:
        source, target = session_dir / rel, dest / rel
        try:
            if not source.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            shutil.copy2(source, target)
        except OSError as exc:
            log.warning("deliverable %s not copied to %s: %s", source, target, exc)
            continue
        landed = [*landed, str(target)]
    return landed


__all__ = [
    "CREDENTIAL_SESSION_FILES", "HOST_OWNED_SESSION_FILES", "MAX_HOST_FILE_COMPARE_BYTES", "host_owned_blobs",
    "land_session_deliverables", "session_deliverables",
]
