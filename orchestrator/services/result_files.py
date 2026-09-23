"""F014 (night 1, #153): a ticket may not close ``done`` pointing at nothing.

#153 "Onboarding pack for a new wholesale café" closed ``done`` with a result
that read perfectly — "I wrote the onboarding pack and saved it here:
``deliverables/cafe_onboarding_pack.md``" — and that file never existed. Of
night 1's false ``done``s it was the one nobody could catch by reading the
ticket. Before a run closes, every file its result names that maps into the
workspace is looked up; a name that is not there sends the ticket to
``review`` with the names on it, never ``done``.

Only what can be checked is checked: a path with a folder and a file
extension, in the workspace (or the projects folder the worker mounts) — never
a URL, a command, a bare file name with no folder, or a path elsewhere on the
owner's machine. A worker that cannot answer is not a verdict: the ticket
closes as it would have.
"""
from __future__ import annotations

import asyncio
import logging
import posixpath
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

MAX_NAMED_FILES = 10
LIST_TIMEOUT_SECONDS = 5.0
CHECK_TIMEOUT_SECONDS = 10.0
NOTE_NAMES_SHOWN = 3

# What a deliverable is saved as. A name with any other ending is not taken for a file.
FILE_EXTENSIONS = frozenset({
    ".md", ".markdown", ".txt", ".rst", ".pdf", ".doc", ".docx", ".odt", ".rtf", ".csv", ".tsv", ".xls",
    ".xlsx", ".ods", ".ppt", ".pptx", ".odp", ".key", ".json", ".yaml", ".yml", ".toml", ".xml", ".html",
    ".css", ".js", ".ts", ".tsx", ".jsx", ".py", ".sql", ".sh", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".webp", ".zip", ".mp3", ".mp4", ".wav",
})
_WORD = r"[\w.@+-]"
_PATH = rf"(?:/|~/)?(?:{_WORD}+/)+{_WORD}+\.[A-Za-z0-9]{{1,8}}"
_FENCE_RE = re.compile(r"```.*?```", re.S)
_BACKTICK_RE = re.compile(r"`([^`\n]{1,300})`")
_LINK_RE = re.compile(r"\]\(([^)\s]{1,300})\)")
_BARE_RE = re.compile(rf"(?<![\w/.~:@-])({_PATH})(?![\w/])")
_WHOLE_PATH_RE = re.compile(rf"^{_PATH}$")


def named_files(text: str) -> List[str]:
    """The file paths a result names — a backtick span or markdown link that
    is a path, or a bare path in the prose — each with a folder and a known
    extension. A path inside code (a command, a fenced block) is not a claim
    that a file was written."""
    found: List[str] = []
    for regex in (_BACKTICK_RE, _LINK_RE):
        found += [m.group(1).strip() for m in regex.finditer(text) if _WHOLE_PATH_RE.match(m.group(1).strip())]
    prose = _BACKTICK_RE.sub(" ", _FENCE_RE.sub(" ", text))
    found += [m.group(1) for m in _BARE_RE.finditer(prose)]
    out: List[str] = []
    for name in found:
        if "://" in name or name.startswith("~"):
            continue                                   # a URL, or a home the worker cannot see
        if posixpath.splitext(name)[1].lower() in FILE_EXTENSIONS and name not in out:
            out.append(name)
    return out[:MAX_NAMED_FILES]


def worker_paths(named: Sequence[str], *, workspace_id: str, runtime_ref: Optional[Dict[str, Any]],
                 projects_dir: Optional[str]) -> List[Tuple[str, str]]:
    """``(as named, as the worker sees it)`` for each name that maps into the
    workspace. An API run's names are workspace paths; a session's are relative
    to the folder it ran in, which must itself map into the workspace."""
    from services.cli_host_service import workspace_relative_path

    ref = runtime_ref or {}
    session = bool(ref.get("host_id"))
    cwd = ref.get("cwd")
    base = workspace_relative_path(str(cwd), workspace_id, projects_dir) if cwd else None
    out: List[Tuple[str, str]] = []
    for name in named:
        if name.startswith("/"):
            rel = workspace_relative_path(name, workspace_id, projects_dir)
        elif session:
            rel = posixpath.normpath(posixpath.join(base, name)) if base else None
        else:
            rel = posixpath.normpath(name)
        if rel and rel != "." and not rel.startswith(("../", "/")) and rel != "..":
            out.append((name, rel))
    return out


async def _names_in(client: Any, folder: str) -> Optional[Set[str]]:
    """The names in a workspace folder; ``None`` when the worker cannot say
    (unreachable, refused, or a listing cut short)."""
    try:
        listing = await asyncio.wait_for(client.list_dir(folder), timeout=LIST_TIMEOUT_SECONDS)
    except Exception:  # noqa: BLE001 — an unanswered lookup is not a verdict
        logger.warning("[result-files] could not list %s", folder, exc_info=True)
        return None
    if not isinstance(listing, dict) or listing.get("success") is False or listing.get("truncated"):
        return None
    return {str(entry.get("name")) for entry in listing.get("entries") or [] if isinstance(entry, dict)}


async def _missing(pairs: Sequence[Tuple[str, str]], client: Any) -> List[str]:
    listings: Dict[str, Optional[Set[str]]] = {}
    missing: List[str] = []
    for named, rel in pairs:
        folder, name = posixpath.split(rel)
        folder = folder or "."
        if folder not in listings:
            listings[folder] = await _names_in(client, folder)
        names = listings[folder]
        if names is not None and name not in names:
            missing.append(named)
    return missing


async def missing_files_note(task: Any, text: str, workspace_id: Any, *,
                             projects_dir: Optional[str] = None, client: Any = None) -> Optional[str]:
    """The line to put on a ticket whose result names files that are not in
    the workspace, or ``None`` when every checkable name is there."""
    pairs = worker_paths(named_files(text or ""), workspace_id=str(workspace_id),
                         runtime_ref=getattr(task, "runtime_ref", None), projects_dir=projects_dir)
    if not pairs:
        return None
    if client is None:
        from core.workspace_client import WorkspaceClient

        client = WorkspaceClient(str(workspace_id))
    try:
        missing = await asyncio.wait_for(_missing(pairs, client), timeout=CHECK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("[result-files] ticket %s: the file check ran out of time", getattr(task, "id", "?"))
        return None
    if not missing:
        return None
    shown = ", ".join(f"`{name}`" for name in missing[:NOTE_NAMES_SHOWN])
    more = f" and {len(missing) - NOTE_NAMES_SHOWN} more" if len(missing) > NOTE_NAMES_SHOWN else ""
    return (f"Not found when this ticket closed: {shown}{more} — the result names "
            f"{'it' if len(missing) == 1 else 'them'}, but the workspace has no such "
            f"{'file' if len(missing) == 1 else 'files'}. Sent to review instead of done.")
