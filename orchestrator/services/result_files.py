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
from dataclasses import dataclass
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


def _session_places(name: str, base: str) -> List[str]:
    """Where a session's named file can be. Relative to the folder it ran in,
    and, when the name is written from the deliverables root, from that root:
    night 5's #980 ran in ``sessions/980`` and named
    ``deliverables/sessions/980/…overview.md``, which the join alone looked for
    as ``sessions/980/deliverables/sessions/980/…`` (F161)."""
    from services.cli_host_service import configured_workspace_dir

    places = [posixpath.normpath(posixpath.join(base, name))]
    root = configured_workspace_dir()
    head, _, rest = name.partition("/")
    rooted = rest if root and rest and head == posixpath.basename(root) else name
    if rooted != name or rooted.startswith(base + "/"):
        places.append(posixpath.normpath(rooted))
    return list(dict.fromkeys(places))


def worker_paths(named: Sequence[str], *, workspace_id: str, runtime_ref: Optional[Dict[str, Any]],
                 projects_dir: Optional[str]) -> List[Tuple[str, str]]:
    """``(as named, as the worker sees it)`` for each place a name may map to in
    the workspace; a name can have more than one. An API run's names are
    workspace paths; a session's are relative to the folder it ran in, which must
    itself map into the workspace, or written from the deliverables root."""
    from services.cli_host_service import workspace_relative_path

    ref = runtime_ref or {}
    session = bool(ref.get("host_id"))
    cwd = ref.get("cwd")
    base = workspace_relative_path(str(cwd), workspace_id, projects_dir) if cwd else None
    out: List[Tuple[str, str]] = []
    for name in named:
        if name.startswith("/"):
            places = [workspace_relative_path(name, workspace_id, projects_dir)]
        elif session:
            places = _session_places(name, base) if base else []
        else:
            places = [posixpath.normpath(name)]
        for rel in places:
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
    """The names found in none of their places. A place the worker cannot list
    is not a verdict, so its name is not missing."""
    listings: Dict[str, Optional[Set[str]]] = {}
    unsettled: Dict[str, bool] = {}          # name -> still looked for (in order)
    for named, rel in pairs:
        unsettled.setdefault(named, True)
        folder, name = posixpath.split(rel)
        folder = folder or "."
        if folder not in listings:
            listings[folder] = await _names_in(client, folder)
        names = listings[folder]
        if names is None or name in names:
            unsettled[named] = False
    return [named for named, looking in unsettled.items() if looking]


@dataclass(frozen=True)
class FileCheck:
    note: str          # what to append to the result
    review: bool       # a named file is nowhere at all: the ticket goes to review


def _knowledge_base_matches(db: Any, workspace_id: Any, names: Sequence[str]) -> Set[str]:
    """The named files that exist as knowledge-base DOCUMENTS instead. Night 1's
    #153 saved its pack with platform_upload_document as a document called
    ``deliverables/cafe_onboarding_pack.md`` — real work, in the knowledge base,
    not a file on disk. Matched by name or by its last part."""
    if db is None or not names:
        return set()
    from core.models.core import Document
    from sqlalchemy import or_

    candidates = sorted({*names, *(posixpath.basename(n) for n in names)})
    try:
        rows = (
            db.query(Document.filename, Document.original_filename)
            .filter(Document.workspace_id == workspace_id,
                    or_(Document.filename.in_(candidates), Document.original_filename.in_(candidates)))
            .all()
        )
    except Exception:  # noqa: BLE001 — a lookup that fails leaves the names unexplained
        logger.warning("[result-files] knowledge-base lookup failed", exc_info=True)
        return set()
    stored = {name for row in rows for name in row if name}
    return {n for n in names if n in stored or posixpath.basename(n) in stored}


def _shown(names: Sequence[str]) -> str:
    more = f" and {len(names) - NOTE_NAMES_SHOWN} more" if len(names) > NOTE_NAMES_SHOWN else ""
    return ", ".join(f"`{name}`" for name in names[:NOTE_NAMES_SHOWN]) + more


async def check_named_files(task: Any, text: str, workspace_id: Any, *, db: Any = None,
                            projects_dir: Optional[str] = None, client: Any = None) -> Optional[FileCheck]:
    """What to say about the files a result names: nothing when every
    checkable name is in the workspace; where it went when it was saved to the
    knowledge base instead; a review when a name is nowhere at all."""
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
    in_knowledge_base = _knowledge_base_matches(db, workspace_id, missing)
    nowhere = [n for n in missing if n not in in_knowledge_base]
    saved_as_documents = [n for n in missing if n in in_knowledge_base]
    lines: List[str] = []
    if nowhere:
        lines.append(f"Not found when this ticket closed: {_shown(nowhere)} — the result names "
                     f"{'it' if len(nowhere) == 1 else 'them'}, but the workspace has no such "
                     f"{'file' if len(nowhere) == 1 else 'files'}. Sent to review instead of done.")
    if saved_as_documents:
        lines.append(f"Saved to the knowledge base, not as a file in the workspace: {_shown(saved_as_documents)}"
                     f" — find {'it' if len(saved_as_documents) == 1 else 'them'} under Documents.")
    return FileCheck(note="\n".join(lines), review=bool(nowhere))
