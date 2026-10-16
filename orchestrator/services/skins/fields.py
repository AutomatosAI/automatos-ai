"""PRD-259 §3 — small shared helpers for the skin validator (mirror of ``fields.ts``)."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

# Messages never echo more than this many characters of the input back.
ECHO_MAX = 40

SkinError = Dict[str, str]


def skin_error(code: str, path: str, message: str) -> SkinError:
    """One validation error: a code from SKIN_ERROR_CODES, a dotted path and a sentence."""
    return {"code": code, "path": path, "message": message}


def is_record(value: Any) -> bool:
    """True for a JSON object."""
    return isinstance(value, dict)


def one_of(choices: Iterable[str], value: Any) -> bool:
    """True when ``value`` is a string in ``choices``."""
    return isinstance(value, str) and value in tuple(choices)


def echo(text: str) -> str:
    """``text`` cut to ECHO_MAX code points for a message."""
    return f"{text[:ECHO_MAX]}…" if len(text) > ECHO_MAX else text


def join_path(prefix: str, key: str) -> str:
    """A dotted path, or the key alone at the top level."""
    return f"{prefix}.{key}" if prefix else key


def is_whole_number(value: Any) -> bool:
    """A JSON number with no fraction (8 and 8.0 alike, as JavaScript reads them); never a bool."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return isinstance(value, int) or value.is_integer()


def unknown_fields(record: Dict[str, Any], known: Iterable[str], prefix: str) -> List[SkinError]:
    """UNKNOWN_FIELD for every key of ``record`` that isn't in ``known``."""
    allowed = tuple(known)
    return [
        skin_error("UNKNOWN_FIELD", join_path(prefix, key), f'"{echo(key)}" isn\'t part of a skin. Remove it.')
        for key in record
        if key not in allowed
    ]
