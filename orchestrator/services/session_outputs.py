"""Which of a Claude Code session's files become its Deliverables (F335, night 10).

A session asked for a document often writes the program that builds it as well, and
every file it wrote was registered. The card's Deliverable was then the program:
#2050 ``mkpdf.py`` ("Mkpdf"), #2068 ``make_invoice.py``, #2079 ``build_letter.py``,
#2082 ``make_xlsx.py``, #2096 ``mk.py``, while the PDF, sheet or page it made sat
beside it. #2077 registered the same PDF twice, under two names.

So when a session made a document, its scripts are how it got there, not the work:
a script is a Deliverable only when the session made no document, or when the brief
asked for code. And one set of bytes is one Deliverable, under the first name the
session gave it. Files left out stay where the session wrote them.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# What a person opens and reads: a document, a sheet, a deck, a page, a picture, a video.
DOCUMENT_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".odt", ".rtf", ".txt",
    ".xls", ".xlsx", ".ods", ".csv", ".tsv",
    ".ppt", ".pptx", ".odp", ".key",
    ".html", ".htm", ".md", ".markdown",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp4", ".mov", ".webm",
})
# What a session runs to build a document.
SCRIPT_EXTENSIONS = frozenset({
    ".py", ".sh", ".bash", ".zsh", ".js", ".mjs", ".cjs", ".ts",
    ".rb", ".pl", ".php", ".ps1", ".bat", ".r", ".lua",
})
# Words in a brief that ask for code, so the script a session wrote IS the work; a brief
# that names a script file (``fix utils.py``) asks for it too. Not a bare "code": the
# owner's briefs say "discount code" and "QR code", and the runtime is "Claude Code".
CODE_REQUEST_WORDS = (
    "codebase", "source code", "script", "scripts", "scripting", "programming", "python", "javascript",
    "typescript", "bash", "repo", "repository", "refactor", "debug",
)
_CODE_REQUEST = re.compile(
    r"\b(?:" + "|".join(re.escape(word) for word in CODE_REQUEST_WORDS) + r")\b"
    + r"|\w(?:" + "|".join(re.escape(ext) for ext in sorted(SCRIPT_EXTENSIONS)) + r")\b",
    re.IGNORECASE,
)
HASH_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class SessionOutput:
    """One file a session wrote, as the workspace sees it. ``size`` and ``full`` are
    ``None`` when this process holds no bytes for it (a projects file: the worker serves it)."""

    host_path: str
    rel: str
    artifact_type: str
    size: Optional[int] = None
    full: Optional[Path] = None


def visible_size(path: Path) -> Optional[int]:
    """The file's size when this process can see it, else ``None``."""
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _extension(output: SessionOutput) -> str:
    return os.path.splitext(output.rel.lower())[1]


def brief_asks_for_code(brief: str) -> bool:
    """True when the ticket's brief asks for code, or names a script file."""
    return bool(_CODE_REQUEST.search(brief or ""))


def without_build_scripts(outputs: Sequence[SessionOutput], brief: str) -> List[SessionOutput]:
    """The outputs less their scripts, when the session also made a document and the
    brief did not ask for code. Otherwise all of them."""
    made_a_document = any(_extension(o) in DOCUMENT_EXTENSIONS for o in outputs)
    if not made_a_document or brief_asks_for_code(brief):
        return list(outputs)
    return [o for o in outputs if _extension(o) not in SCRIPT_EXTENSIONS]


def _digest(path: Path) -> Optional[str]:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as exc:
        logger.warning("[F335] could not read %s to compare it: %s", path, exc)
        return None
    return digest.hexdigest()


def _content_key(output: SessionOutput, shared_sizes: frozenset) -> Optional[Tuple[int, str]]:
    """Size and digest, for a file this process can read whose size another output
    shares. ``None`` otherwise: nothing to compare it with, or no bytes here."""
    if output.full is None or output.size is None or output.size not in shared_sizes:
        return None
    digest = _digest(output.full)
    return (output.size, digest) if digest is not None else None


def without_duplicates(outputs: Sequence[SessionOutput]) -> List[SessionOutput]:
    """Each workspace path once, and each set of bytes once, under the first name."""
    sizes = [o.size for o in outputs if o.size is not None]
    shared_sizes = frozenset(s for s in sizes if sizes.count(s) > 1)
    kept: List[SessionOutput] = []
    seen_keys: frozenset = frozenset()
    for output in outputs:
        key = _content_key(output, shared_sizes)
        if any(k.rel == output.rel for k in kept) or (key is not None and key in seen_keys):
            logger.info("[F335] %s repeats an earlier file of the session: not registered again", output.rel)
            continue
        kept = [*kept, output]
        seen_keys = seen_keys | ({key} if key is not None else frozenset())
    return kept


def pick_deliverables(outputs: Sequence[SessionOutput], brief: str) -> List[SessionOutput]:
    """The session's files that are its Deliverables, in the order it wrote them."""
    return without_duplicates(without_build_scripts(outputs, brief))


__all__ = [
    "CODE_REQUEST_WORDS", "DOCUMENT_EXTENSIONS", "HASH_CHUNK_BYTES", "SCRIPT_EXTENSIONS", "SessionOutput",
    "brief_asks_for_code", "pick_deliverables", "visible_size", "without_build_scripts", "without_duplicates",
]
