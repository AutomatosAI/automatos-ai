"""F332 (night 10): the brand kit's logo, as files in the folder the session saves into.

A session runs here, on the operator's machine, where the platform's document store
is out of reach: night 10's agents had no logo file and guessed. The claim carries the
ticket's workspace's uploaded logo and logo mark (``brand_files``: name, mime, base64
data, from ``orchestrator/services/session_brand_files.py``). They are written under
``brand/`` in the ticket's deliverables folder (``<root>/sessions/<ticket>``, the folder
the ticket file tells the session to save into, which the gate grants), else, on a host
with no default root, in the ticket folder. The ticket file names them.

The retest of #961 put them in the host's own ticket folder first: the session could
read them there, but copying one into its deliverables folder was a Bash command the
gate held for the operator, so the logo never reached the work. Where the session saves,
it needs no copy: it links or embeds the file as it is.

Only the names the backend sends are accepted (``logo.png``, ``logo-mark.jpg`` …), so a
claim can never write outside that folder, and a file is at most the upload limit.
"""
from __future__ import annotations

import base64
import binascii
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger("automatos.cli_host.session")

BRAND_FOLDER = "brand"
BRAND_FILE_NAME = re.compile(r"^logo(?:-mark)?\.(?:png|jpg)$")
# The backend's own upload limit for a logo (modules/documents/brand_logo.py).
MAX_BRAND_FILE_BYTES = 2 * 1024 * 1024
# As the deliverables folder itself (session.py): the Deliverables explorer reads it.
BRAND_FOLDER_MODE = 0o755


def _decoded(entry: Any) -> Optional[bytes]:
    """The file's bytes, or None for a name this host does not accept, bad data or an oversize file."""
    name = str(entry.get("name") or "") if isinstance(entry, dict) else ""
    if not BRAND_FILE_NAME.match(name):
        log.warning("brand file %r refused: not a logo name this host writes", name[:80])
        return None
    try:
        data = base64.b64decode(str(entry.get("data") or ""), validate=True)
    except (binascii.Error, ValueError):
        log.warning("brand file %s refused: its data is not base64", name)
        return None
    return data if 0 < len(data) <= MAX_BRAND_FILE_BYTES else None


def _clear(folder: Path) -> None:
    """Drop an earlier claim's logos: a logo since replaced (or deleted) must not linger."""
    for old in folder.glob("logo*") if folder.is_dir() else ():
        if BRAND_FILE_NAME.match(old.name):
            old.unlink(missing_ok=True)


def write_brand_files(ticket: Dict[str, Any], root: Path) -> List[Path]:
    """Write the claim's brand files under ``<root>/brand/`` (the folder the session saves
    into); the paths written. A file that cannot be written is a warning, never a failed session."""
    folder = root / BRAND_FOLDER
    entries = ticket.get("brand_files")
    written: List[Path] = []
    try:
        _clear(folder)
        for entry in entries if isinstance(entries, list) else []:
            data = _decoded(entry)
            if data is not None:
                folder.mkdir(parents=True, exist_ok=True, mode=BRAND_FOLDER_MODE)
                (folder / entry["name"]).write_bytes(data)
                written = [*written, folder / entry["name"]]
    except OSError as exc:
        log.warning("brand files not written to %s: %s", folder, exc)
    return written


def brand_files_note(paths: Sequence[Path]) -> str:
    """What the ticket file says of them; empty when there are none."""
    if not paths:
        return ""
    listed = "\n".join(f"- {path}" for path in paths)
    return ("\nBrand files: the brand kit's own logo, already in the folder you save into. Use these "
            "files as they are (link or embed them in what you make); never draw, fetch or guess a logo.\n"
            f"{listed}\n")


__all__ = [
    "BRAND_FILE_NAME", "BRAND_FOLDER", "BRAND_FOLDER_MODE", "MAX_BRAND_FILE_BYTES", "brand_files_note",
    "write_brand_files",
]
