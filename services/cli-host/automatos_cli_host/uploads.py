"""#848: upload what a session left in the ticket's deliverables folder.

With ``docker compose`` the deliverables folder (``<root>/sessions/<ticket>``) is
bind-mounted into the backend, so its files are already where the worker serves
them. On a cluster nothing is shared: when the claim's ``upload`` is enabled, each
file of the session's result that sits in that folder is sent to the backend
(``PUT …/tasks/<ticket>/files``) before the result. The result names them by their
paths here, as it always has, and the backend registers the ones it holds.

Limits, the claim's: ``upload.max_file_bytes`` per file and ``upload.max_total_bytes``
for the run. Never a symlink, and never a file outside the folder. A file that
cannot be uploaded is a warning on the log, never a lost result.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .api import BackendError

log = logging.getLogger("automatos.cli_host.uploads")

# The backend answers a transient failure (restarting, 5xx) — try a file again before giving up on it.
UPLOAD_ATTEMPTS = 3
UPLOAD_RETRY_SECONDS = 2.0


def _relative_to_folder(path: Path, folder: Path, root: Path) -> Optional[Path]:
    """``path`` relative to the folder, as given (``folder``) or resolved (``root``,
    e.g. macOS's ``/var`` → ``/private/var``). Only a path that names no symlink
    inside the folder matches as written; anything else is resolved first."""
    for base in (folder.absolute(), root):
        try:
            return path.absolute().relative_to(base)
        except ValueError:
            continue
    try:
        return path.resolve().relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None


def _relative_file(raw: str, folder: Path, root: Path) -> Optional[Tuple[Path, str]]:
    """``(file, path in the folder)`` for a regular file inside the folder; ``None`` for
    anything else: a symlink (the file or a folder on its way), outside, or gone."""
    rel = _relative_to_folder(Path(raw), folder, root)
    if rel is None:
        return None
    node = root
    for part in rel.parts:
        node = node / part
        if part in ("", ".", "..") or node.is_symlink():
            return None
    return (node, rel.as_posix()) if node.is_file() else None


def upload_candidates(files: Sequence[str], folder: Path) -> List[Tuple[Path, str]]:
    """The files of a result that sit in the ticket's deliverables folder, each once."""
    try:
        root = folder.resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    found: Dict[str, Path] = {}
    for raw in files:
        hit = _relative_file(raw, folder, root)
        if hit is not None and hit[1] not in found:
            found = {**found, hit[1]: hit[0]}
    return [(path, rel) for rel, path in found.items()]


def _send(api: Any, host_id: str, task_id: str, rel: str, data: bytes) -> bool:
    """One file, retried on a transient failure; True when the backend took it."""
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        try:
            api.upload_file(host_id, int(task_id), rel, data)
            return True
        except BackendError as exc:
            transient = exc.status == 0 or exc.status >= 500
            if not transient or attempt == UPLOAD_ATTEMPTS:
                log.warning("deliverable %s of task %s not uploaded: %s", rel, task_id, exc)
                return False
            time.sleep(UPLOAD_RETRY_SECONDS)
    return False


def _read(path: Path, rel: str, task_id: str) -> Optional[bytes]:
    """The file's bytes, or ``None`` when it went away or became unreadable since it was found."""
    try:
        return path.read_bytes()
    except OSError as exc:
        log.warning("deliverable %s of task %s not uploaded: %s", rel, task_id, exc)
        return None


def upload_deliverables(api: Any, host_id: str, ticket: Dict[str, Any], folder: Optional[Path],
                        files: Sequence[str]) -> List[str]:
    """Upload the result's files in the deliverables folder when the claim asks for it;
    the paths (in the folder) the backend took."""
    upload = ticket.get("upload") if isinstance(ticket.get("upload"), dict) else {}
    if not upload.get("enabled") or folder is None:
        return []
    task_id = str(ticket.get("task_id"))
    max_file = int(upload.get("max_file_bytes") or 0)
    budget = int(upload.get("max_total_bytes") or 0)
    taken: List[str] = []
    for path, rel in upload_candidates(files, folder):
        try:
            size = path.stat().st_size
        except OSError as exc:
            log.warning("deliverable %s of task %s not uploaded: %s", rel, task_id, exc)
            continue
        if size > max_file or size > budget:
            log.warning("deliverable %s of task %s not uploaded: %d bytes is past the limit", rel, task_id, size)
            continue
        data = _read(path, rel, task_id)
        if data is not None and _send(api, host_id, task_id, rel, data):
            budget -= size
            taken = [*taken, rel]
    return taken


__all__ = ["UPLOAD_ATTEMPTS", "upload_candidates", "upload_deliverables"]
