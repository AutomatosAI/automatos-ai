"""F378 (night 11, 7 Oct): another take starts from the post as it is, and can be undone.

B16: the Social Media Director's post (``brief: null``, ``copy: {}``) was retaken and rebuilt
from old workspace notes, with tasting notes nobody gave, dropping the price and the delivery
line; it replaced the good version in place, with no undo. B11: a retake ignored "don't invent
tasting notes" and added more. A retake (``api/socials_retake.py``) now:

* gives the composer the post's current take (:func:`current_take`: its copy and its fields'
  values) as the starting point, to change only what the owner's guidance asks;
* is refused when there is nothing to start from (:func:`nothing_to_retake`: no brief, no
  copy, no field with a value) instead of composing from whatever the workspace holds;
* records the take it replaces in the post's history (:func:`record_previous`: a ``retake``
  entry carrying ``previous``, the copy, variables and sources), before the new one is
  written, so :func:`restoring` can bring it back: the newest take not yet restored, each
  restore logged as ``retake_undone`` naming the take it undid (``undid``).

Pure: the post is read and its history appended to (a new list, ``service``'s log), nothing
is committed here.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Set

from modules.socials import service

ACTION_RETAKE = "retake"
ACTION_RETAKE_UNDONE = "retake_undone"
RETAKE_LOGGED = "Auto made another take. The take before it is kept, so it can be restored."
UNDO_LOGGED = "The take before Auto's last one was restored."
NOTHING_TO_RETAKE = "Nothing to retake: this post has no brief, no copy and no fields yet. Add a brief, then ask again."
NOTHING_TO_UNDO = "There is no earlier take of this post to restore."
TAKE_FIELDS = ("copy", "variables", "sources")


def _words(text: Any) -> bool:
    return isinstance(text, str) and bool(text.strip())


def _has_copy(copy: Any) -> bool:
    copy = copy if isinstance(copy, Mapping) else {}
    channels = copy.get("channels") if isinstance(copy.get("channels"), Mapping) else {}
    return _words(copy.get("base")) or any(_words(text) for text in channels.values())


def _values(variables: Any) -> Dict[str, Any]:
    """The fields that hold a value: ``{name: value}``."""
    items = variables.items() if isinstance(variables, Mapping) else ()
    return {
        name: spec.get("value") for name, spec in items
        if isinstance(spec, Mapping) and spec.get("value") not in (None, "")
    }


def nothing_to_retake(post: Any) -> bool:
    """Whether the post gives a retake nothing to start from: no brief, no copy, no field's value."""
    return not (_words(getattr(post, "brief", None)) or _has_copy(post.copy) or _values(post.variables))


def current_take(post: Any) -> Dict[str, Any]:
    """The post as it is, for the composer to start from: its copy and its fields' values."""
    copy = post.copy if isinstance(post.copy, Mapping) else {}
    return {"copy": dict(copy), "variables": _values(post.variables)}


def _take(post: Any) -> Dict[str, Any]:
    return {name: dict(getattr(post, name) or {}) for name in TAKE_FIELDS}


def record_previous(post: Any, actor: str, guidance: Optional[str]) -> None:
    """Log the take a retake is about to replace (``previous``), with the guidance it was given."""
    service._log(post, actor, ACTION_RETAKE, RETAKE_LOGGED, previous=_take(post), guidance=(guidance or "").strip() or None)


def _undone(log: Any) -> Set[Any]:
    return {entry.get("undid") for entry in log if isinstance(entry, Mapping) and entry.get("action") == ACTION_RETAKE_UNDONE}


def restoring(post: Any) -> Optional[Mapping[str, Any]]:
    """The newest retake entry whose earlier take has not been restored yet, or ``None``."""
    log = [entry for entry in (post.review_log or []) if isinstance(entry, Mapping)]
    undone = _undone(log)
    for entry in reversed(log):
        if entry.get("action") == ACTION_RETAKE and isinstance(entry.get("previous"), Mapping) and entry.get("at") not in undone:
            return entry
    return None


def restore_changes(entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The edit that brings the entry's earlier take back: its copy, variables and sources."""
    previous = entry.get("previous") or {}
    return {name: dict(previous.get(name) or {}) for name in TAKE_FIELDS}


def record_undo(post: Any, actor: str, entry: Mapping[str, Any]) -> None:
    """Log that the take before ``entry`` was restored (``undid`` names the retake undone)."""
    service._log(post, actor, ACTION_RETAKE_UNDONE, UNDO_LOGGED, undid=entry.get("at"))
