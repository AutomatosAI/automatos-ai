"""PRD-268 §6.11.2 (US-C19b): the structured parts a background post carries beside its text.

A producer that has a card's data (the brief, a suggestion) posts it after Auto's text part:
``parts = [{type: 'text', text}, *parts]`` (``services.chat_messenger.post_background_message``).
A client that can draw the card draws it; every other client, and the model's history
(``modules.context.sections.conversation._parts_to_text``), reads the text and skips the part.

Only the types named here are saved, each at most ``CHAT_PART_MAX_BYTES`` of JSON. Anything
else raises ValueError naming its type, before the messenger reads or writes anything.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

STRUCTURED_PART_TYPES = ("brief@1", "suggestion@1")
# A brief is a handful of sections of short rows and a suggestion is one card: 32 KiB holds
# either many times over, and keeps a runaway producer from bloating messages.parts.
CHAT_PART_MAX_BYTES = 32 * 1024
# A refused part's type is caller data: the error quotes at most this much of it.
TYPE_IN_ERROR_MAX_CHARS = 64
# An escaped NUL (``\u0000`` after an even run of backslashes): Postgres JSONB refuses it, and
# only once the save runs, after the Auto thread may already have been made.
_ESCAPED_NUL = re.compile(r"(?<!\\)(?:\\\\)*\\u0000")


def _type_of(part: Any) -> str:
    """The part's declared type, or its Python type when it is not a mapping, cut for an error."""
    kind = str(part.get("type")) if isinstance(part, Mapping) else type(part).__name__
    return kind[:TYPE_IN_ERROR_MAX_CHARS]


def _as_saved(part: Mapping[str, Any], kind: str) -> Tuple[Dict[str, Any], str, int]:
    """The part read back from the JSON it is saved as, that JSON and its size in UTF-8 bytes;
    ValueError when it is not JSON (a lone surrogate included)."""
    try:
        encoded = json.dumps(dict(part), separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return json.loads(encoded), encoded, len(encoded.encode("utf-8"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError(f"chat part {kind!r} is not JSON: {exc}") from exc


def _checked(part: Any) -> Dict[str, Any]:
    """One structured part as a new dict, read back from the JSON it is saved as."""
    kind = _type_of(part)
    if not isinstance(part, Mapping):
        raise ValueError(f"chat part type {kind!r} is not one of {STRUCTURED_PART_TYPES}")
    saved, encoded, size = _as_saved(part, kind)
    if saved.get("type") not in STRUCTURED_PART_TYPES:
        raise ValueError(f"chat part type {kind!r} is not one of {STRUCTURED_PART_TYPES}")
    if _ESCAPED_NUL.search(encoded):
        raise ValueError(f"chat part {kind!r} holds a NUL character, which Postgres refuses")
    if size > CHAT_PART_MAX_BYTES:
        raise ValueError(f"chat part {kind!r} is {size} bytes, over {CHAT_PART_MAX_BYTES}")
    return saved


def checked_parts(parts: Optional[Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    """The structured parts to save after the text part, each a new dict read back from the JSON
    it is saved as (a tuple comes back a list); [] for None.

    Raises ValueError, naming the type, for a part that is not a mapping, whose type is not in
    ``STRUCTURED_PART_TYPES``, that is not JSON or holds a NUL character, or whose JSON is over
    ``CHAT_PART_MAX_BYTES``.
    """
    if parts is None:
        return []
    if isinstance(parts, (str, bytes, Mapping)):
        raise ValueError(f"chat parts must be a list of parts, not {type(parts).__name__}")
    return [_checked(part) for part in parts]
