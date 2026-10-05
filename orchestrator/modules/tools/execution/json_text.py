"""JSON a model wrote out as text, read the way it meant it (F346, night 10b).

F346: generate_document's ``data`` arrived as JSON text, and 4 of 55 calls failed:
the model had written an apostrophe as ``\\'`` ("O\\'Brien", "the client\\'s"),
which JSON does not allow (its only quote escape is ``\\"``). The text was refused
as "not a complete object", which told the model nothing it could fix.

:func:`loads_lenient` reads the text as JSON, accepting literal newlines inside
strings and ``\\'`` as ``'``. Text that still is not JSON raises
:class:`JSONTextProblem`, whose message names the problem and where it is, so the
caller can say exactly what to fix and can never report the call a success.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ``\'`` not itself escaped (an even run of backslashes before it is escaped backslashes).
_ESCAPED_APOSTROPHE = re.compile(r"(?<!\\)((?:\\\\)*)\\'")
# How much of the text either side of the problem the message quotes.
NEAR_CHARS = 20


class JSONTextProblem(ValueError):
    """The text is not JSON; the message says what is wrong and where."""


def describe(exc: json.JSONDecodeError) -> str:
    """A decode error in words: what, at which line and column, and the text around it."""
    near = exc.doc[max(0, exc.pos - NEAR_CHARS): exc.pos + NEAR_CHARS]
    return f"{exc.msg} at line {exc.lineno}, column {exc.colno}, near {near!r}"


def _loads(text: str) -> Any:
    return json.loads(text, strict=False)


def loads_lenient(text: str) -> Any:
    """``text`` as the JSON value it holds, ``\\'`` read as ``'``; :class:`JSONTextProblem` if it holds none."""
    try:
        return _loads(text)
    except json.JSONDecodeError as exc:
        problem = exc
    repaired = _ESCAPED_APOSTROPHE.sub(r"\1'", text)
    if repaired == text:
        raise JSONTextProblem(describe(problem)) from problem
    try:
        return _loads(repaired)
    except json.JSONDecodeError as exc:
        raise JSONTextProblem(describe(exc)) from exc


__all__ = ["JSONTextProblem", "describe", "loads_lenient"]
