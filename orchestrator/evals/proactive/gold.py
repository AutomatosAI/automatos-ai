"""PRD-268 §6.7 (K14) — the gold sets, read from a folder outside the repository.

The repository is public, so no gold case is ever committed. Each eval's cases are one JSON
object per line in ``<eval>.jsonl`` inside ``config.PROACTIVE_EVAL_GOLD_DIR``. The folder is
refused, with a message that says what to set, when the setting is empty, when it names no
folder, and when it is inside this repository (where one ``git add`` would publish it). The
command line turns a refusal into exit 2.

The repository is the nearest folder above this file that holds ``.git``; an image carries no
``.git``, so there it is the orchestrator folder the code runs from.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from config import config

__all__ = ["GOLD_EVALS", "GOLD_SUFFIX", "GoldSetUnavailable", "conversation", "gold_dir", "read_gold",
           "repository_root"]

# The evals that read a gold set; precision is live and one brief is driven, so neither has one.
GOLD_EVALS = ("triage", "silence", "reply_chips", "spoken")
GOLD_SUFFIX = ".jsonl"
SETTING = "PROACTIVE_EVAL_GOLD_DIR"
# Who speaks in a case's history before its ask.
ROLES = ("user", "assistant")

_HERE = Path(__file__).resolve().parent
# evals/proactive -> evals -> the orchestrator folder.
_ORCHESTRATOR = _HERE.parents[1]


class GoldSetUnavailable(Exception):
    """No gold set can be read; the message says why and what to fix."""


def repository_root() -> Path:
    """The checkout this code runs from, or the orchestrator folder where there is none."""
    for folder in (_HERE, *_HERE.parents):
        if (folder / ".git").exists():
            return folder
    return _ORCHESTRATOR


def _inside(folder: Path, root: Path) -> bool:
    """True when ``folder`` is ``root`` or under it, compared by the file system, so a path
    spelled in another case on a case-insensitive disk is still inside."""
    return any(candidate.samefile(root) for candidate in (folder, *folder.parents))


def gold_dir() -> Path:
    """The configured gold folder, resolved; refused when unset, not a folder, or in the repository."""
    value = str(config.PROACTIVE_EVAL_GOLD_DIR or "").strip()
    if not value:
        raise GoldSetUnavailable(
            f"{SETTING} is not set. Point it at a folder outside this repository that holds one "
            f"<eval>{GOLD_SUFFIX} per eval.")
    folder = Path(value).expanduser().resolve()
    if not folder.is_dir():
        raise GoldSetUnavailable(f"{SETTING} is {folder}, which is not a folder.")
    root = repository_root()
    if _inside(folder, root):
        raise GoldSetUnavailable(
            f"{SETTING} is {folder}, inside this repository ({root}). Gold sets stay outside it and "
            "are never committed: move them and point the setting there.")
    return folder


def _cases(path: Path) -> List[Dict[str, Any]]:
    """Each non-blank line of ``path`` as a JSON object; a line that is not one is refused by number."""
    cases: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as exc:
                raise GoldSetUnavailable(f"{path} line {number} is not JSON: {exc.msg}.") from exc
            if not isinstance(case, dict):
                raise GoldSetUnavailable(f"{path} line {number} is not a JSON object.")
            cases.append(case)
    return cases


def read_gold(name: str) -> List[Dict[str, Any]]:
    """The cases of eval ``name`` from ``<gold dir>/<name>.jsonl``; refused when the folder is
    refused (``gold_dir``) or the file is missing, unreadable, malformed or empty."""
    if name not in GOLD_EVALS:
        raise ValueError(f"{name!r} reads no gold set; the evals that do are {', '.join(GOLD_EVALS)}")
    path = gold_dir() / f"{name}{GOLD_SUFFIX}"
    if not path.is_file():
        raise GoldSetUnavailable(f"{path} is missing: the {name} eval reads its cases from it, one JSON "
                                 "object per line.")
    try:
        cases = _cases(path)
    except (OSError, UnicodeDecodeError) as exc:
        raise GoldSetUnavailable(f"{path} cannot be read: {exc}.") from exc
    if not cases:
        raise GoldSetUnavailable(f"{path} has no cases.")
    return cases


def conversation(name: str, case: Mapping[str, Any], number: int) -> Tuple[Mapping[str, str], ...]:
    """A case's turn as chat messages: its ``history`` (optional, ``[{role: user | assistant,
    content}]``), then its ``ask`` as the person's message; refused by number when either is not."""
    said = case.get("ask")
    if not isinstance(said, str) or not said.strip():
        raise GoldSetUnavailable(f"{name} case {number}: ask is the person's message, as text.")
    history = case.get("history") or []
    fits = isinstance(history, list) and all(
        isinstance(m, Mapping) and m.get("role") in ROLES and isinstance(m.get("content"), str) for m in history)
    if not fits:
        raise GoldSetUnavailable(f"{name} case {number}: history is a list of {{role: user | assistant, content}}.")
    return (*({"role": m["role"], "content": m["content"]} for m in history), {"role": "user", "content": said})
