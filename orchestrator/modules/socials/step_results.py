"""PRD-251 D8 (US-301): what a channel step returns, and when a status step is done.

The adapter data (``channel_adapters.py``) says how a post is published; these three
keys of a step say what the publisher reads back from each call:

* ``returns``: ``{name: path}``. ``$steps.<id>`` reads the step's ``id``; a
  ``publish`` step's ``id`` is the target's remote id (its receipt), and a
  ``permalink`` any step returns is the receipt's link.
* ``permalink``: a link with ``{id}`` in it, built from the step's returned ``id``
  when no step returns a permalink of its own.
* ``until`` (a ``status`` step only): the step is called every
  ``SOCIALS_PUBLISH_POLL_SECONDS`` until the value at ``path`` is one of ``done``
  (the step succeeded) or one of ``failed`` (the target fails with the value at
  ``error``, else the state). ``absent`` says what a missing value means: ``done``
  or, by default, ``pending`` (call again).

A path is dotted keys (``processing_info.state``); ``a|b`` takes the first that
holds a value. Composio wraps a tool's output in envelopes that differ between
actions, so a path is looked for at the top of the output first, then under any
nested object, the shallowest first.

Pure data and pure functions: the parsers raise ``ValueError`` naming the bad
entry, and ``capabilities.py`` runs them when it loads the data.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Tuple

ID = "id"  # the returned value $steps.<id> reads, and a publish step's remote id
PERMALINK = "permalink"
ID_PLACEHOLDER = "{id}"
DONE, PENDING = "done", "pending"
ABSENT_MEANS = (DONE, PENDING)
_UNTIL_KEYS = frozenset({"path", "done", "failed", "error", "absent"})
# How deep a path is looked for under the output's top level.
MAX_DEPTH = 6


# ---- the data, checked ------------------------------------------------------


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be non-blank text")
    return value


def _texts(value: Any, where: str) -> Tuple[str, ...]:
    items = [] if value is None else value
    if not isinstance(items, list) or not all(isinstance(item, str) and item for item in items):
        raise ValueError(f"{where} must be a list of texts")
    return tuple(items)


def parse_returns(raw: Any, where: str) -> Mapping[str, str]:
    """``{name: path}``, or empty when the step returns nothing the publisher reads."""
    if raw is None:
        return MappingProxyType({})
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{where}.returns must be an object of names and paths")
    return MappingProxyType({_text(name, f"{where}.returns"): _text(path, f"{where}.returns.{name}") for name, path in raw.items()})


def parse_until(raw: Any, where: str, step_class: str, status_class: str) -> Optional[Mapping[str, Any]]:
    """A status step's end condition, or ``None`` (one successful call is done)."""
    if raw is None:
        return None
    if step_class != status_class:
        raise ValueError(f"{where}: only a {status_class} step polls until done")
    if not isinstance(raw, Mapping) or set(raw) - _UNTIL_KEYS:
        raise ValueError(f"{where}.until must be an object of {sorted(_UNTIL_KEYS)}")
    absent = raw.get("absent", PENDING)
    if absent not in ABSENT_MEANS:
        raise ValueError(f"{where}.until.absent must be one of {ABSENT_MEANS}")
    done = _texts(raw.get("done"), f"{where}.until.done")
    if not done:
        raise ValueError(f"{where}.until.done must name at least one state")
    error = raw.get("error")
    return MappingProxyType({
        "path": _text(raw.get("path"), f"{where}.until.path"),
        "done": done,
        "failed": _texts(raw.get("failed"), f"{where}.until.failed"),
        "error": _text(error, f"{where}.until.error") if error is not None else None,
        "absent": absent,
    })


def parse_permalink(raw: Any, where: str, returns: Mapping[str, str]) -> Optional[str]:
    """A link template with ``{id}``, on a step that returns an ``id``."""
    if raw is None:
        return None
    if not isinstance(raw, str) or ID_PLACEHOLDER not in raw or not raw.startswith("https://"):
        raise ValueError(f"{where}.permalink must be an https link with {ID_PLACEHOLDER} in it")
    if ID not in returns:
        raise ValueError(f"{where}.permalink needs the step to return an {ID}")
    return raw


# ---- reading a tool's output ------------------------------------------------


def _at(value: Any, keys: List[str]) -> Any:
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _nested(value: Any, depth: int = 0) -> List[Mapping[str, Any]]:
    """Every object under ``value``, the shallowest first."""
    if depth >= MAX_DEPTH:
        return []
    children = list(value.values()) if isinstance(value, Mapping) else list(value) if isinstance(value, list) else []
    found = [child for child in children if isinstance(child, Mapping)]
    for child in children:
        found.extend(_nested(child, depth + 1))
    return found


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def lookup(output: Any, path: Optional[str]) -> Any:
    """The first value ``path``'s alternatives find in ``output``, else ``None``."""
    if not path:
        return None
    for alternative in path.split("|"):
        keys = [key for key in alternative.strip().split(".") if key]
        for scope in [output, *_nested(output)]:
            value = _at(scope, keys)
            if _present(value):
                return value
    return None


def returned(output: Any, returns: Mapping[str, str]) -> Mapping[str, Any]:
    """What a step returned, by ``returns``' names; a name found nowhere is left out."""
    found = {name: lookup(output, path) for name, path in returns.items()}
    return MappingProxyType({name: value for name, value in found.items() if _present(value)})


def poll_state(output: Any, until: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """``(done | failed | pending, why)`` for one status call's output."""
    state = lookup(output, until["path"])
    if not _present(state):
        return until["absent"], None
    text = str(state)
    if text in until["done"]:
        return DONE, None
    if text in until["failed"]:
        why = lookup(output, until.get("error"))
        return "failed", str(why) if _present(why) else text
    return PENDING, None
