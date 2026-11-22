"""A sed script read the way sed reads it, for the commands that reach outside it.

``sed`` is on a ticket's allowlist and its script is not path-checked
(``shell_data``), so the commands that run something or open a file are found here
and refused: ``e`` (a command, or the ``e`` flag of ``s``), ``r``/``R`` (read a file
in), ``w``/``W`` (write one) and the ``w`` flag of ``s``. They are found only where
sed reads them: a command at the start of the script, after ``;``, a newline, ``{``
or ``}``, or after an address and an optional ``!``; a flag after the third
delimiter of ``s``, whatever the delimiter. The same letter in a regex, a
replacement, a label or a line of ``a`` text is text (``/warning/p``, ``s/x/root/``).

A script this reading cannot follow with certainty is refused too: one holding a
control character other than tab and newline (GNU ends a label at ``\\v``, ``\\f`` or
``\\r``, which a blank here is not), an unterminated regex, an unknown command, text
after a command, or a regex that GNU, BSD and BusyBox sed end at different places.
They read a ``[`` bracket expression three ways, so a delimiter inside one
(``/[[:alpha:]/]x/``) ends the regex in one sed and not in another; ``[^/]``,
``[0-9]`` and ``[[:space:]]`` end the same way in all three.

One pass per reading, linear in the script, and no regex runs over it: the script is
a session's own text, read on the thread every decision is answered from.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Tuple

_BLANKS = " \t"
_DIGITS = "0123456789"
_COMMAND_END = ";\n}#"                       # what may follow a command; ``}`` and ``#`` are read next
_SUBSTITUTE_FLAGS = frozenset("gpiImM" + _DIGITS + _BLANKS)
_REACHES_OUT = {"e": "runs a command", "r": "reads a file", "R": "reads a file",
                "w": "writes a file", "W": "writes a file"}
# C0 but tab and newline, DEL and C1: refused outright, wherever they sit.
_CONTROL = frozenset(chr(code) for code in (*range(0x00, 0x09), *range(0x0B, 0x20), *range(0x7F, 0xA0)))


class _Refused(Exception):
    """A command that reaches outside the script, or a script this reading cannot follow."""


def sed_escape(script: str) -> Optional[str]:
    """Why ``script`` is refused, or None when it reaches nothing outside itself."""
    try:
        _read_script(script)
    except _Refused as refused:
        return str(refused)
    return None


class SedReader:
    """``sed_escape`` behind the ``search`` interface of the escape patterns
    (``ScriptVerb.escape``): not a regex, and nothing in it backtracks."""

    @staticmethod
    def search(script: str) -> Optional[str]:
        return sed_escape(script)


def _read_script(text: str) -> None:
    if any(ch in _CONTROL for ch in text):
        raise _Refused("a control character, which the seds read differently")
    i = 0
    while i < len(text):
        if text[i] in _BLANKS + ";\n":
            i += 1
        else:
            i = _read_command(text, _command_at(text, i))


def _read_command(text: str, i: int) -> int:
    """The index after the command at ``i``, refusing one that reaches outside the script."""
    if i >= len(text):
        raise _Refused("an address with no command")
    ch = text[i]
    if ch in _REACHES_OUT:
        raise _Refused(f"the {ch!r} command {_REACHES_OUT[ch]}")
    if ch == "{":                                # a block: a command follows at once (``{p}``)
        return i + 1
    argument_end = _ARGUMENT_END.get(ch)
    if argument_end is None:
        raise _Refused(f"{ch!r} is not a sed command this reading knows")
    return _end_of_command(text, argument_end(text, i + 1))


# ── addresses ────────────────────────────────────────────────────────────────

def _command_at(text: str, i: int) -> int:
    """Where the command starts: past an address, a second one after ``,`` and a ``!``."""
    end = _address_end(text, i, second=False)
    comma = _skip(text, end, _BLANKS)
    if end > i and comma < len(text) and text[comma] == ",":
        start = _skip(text, comma + 1, _BLANKS)
        end = _address_end(text, start, second=True)
        if end == start:
            raise _Refused("a ',' with no second address")
    at = _skip(text, end, _BLANKS)
    if at < len(text) and text[at] == "!":
        at = _skip(text, at + 1, _BLANKS)
    return at


def _address_end(text: str, i: int, second: bool) -> int:
    """The index after the address at ``i`` (``i`` when none starts there): a number,
    ``first~step``, ``$``, ``/re/`` or ``\\cREc`` with ``I``/``M`` flags, and as a
    second address ``+N`` or ``~N``."""
    ch = text[i] if i < len(text) else ""
    if ch in ("/", "\\"):
        return _address_flags_end(text, _regex_end(text, i + 1 if ch == "\\" else i))
    if ch and ch in _DIGITS:
        end = _skip(text, i, _DIGITS)
        tilde = _skip(text, end, _BLANKS)
        return _number_end(text, tilde + 1) if tilde < len(text) and text[tilde] == "~" else end
    if ch == "$":
        return i + 1
    if second and ch and ch in "+~":
        return _number_end(text, i + 1)
    return i


def _address_flags_end(text: str, i: int) -> int:
    """Past the ``I`` and ``M`` flags of a regex address (GNU allows blanks between them).
    Lowercase ``i`` is the insert command, not a flag."""
    while True:
        at = _skip(text, i, _BLANKS)
        if at >= len(text) or text[at] not in "IM":
            return i
        i = at + 1


def _number_end(text: str, i: int) -> int:
    start = _skip(text, i, _BLANKS)
    end = _skip(text, start, _DIGITS)
    if end == start:
        raise _Refused("a step with no number")
    return end


# ── the text of a command: each reader starts just after the command's letter ──

def _substitute_end(text: str, i: int) -> int:
    """Past ``s/re/replacement/flags``, its first delimiter at ``i``."""
    replaced = _plain_end(text, _regex_end(text, i), _delimiter(text, i))
    return _substitute_flags_end(text, replaced)


def _transliterate_end(text: str, i: int) -> int:
    """Past ``y/abc/xyz/``, its first delimiter at ``i``: no regex, so no brackets."""
    delimiter = _delimiter(text, i)
    return _plain_end(text, _plain_end(text, i + 1, delimiter), delimiter)


def _number_argument_end(text: str, i: int) -> int:
    """Past the optional number of ``q``, ``Q``, ``l`` and ``L`` (``q 5``)."""
    return _skip(text, _skip(text, i, _BLANKS), _DIGITS)


def _no_argument(text: str, i: int) -> int:
    return i


def _comment_end(text: str, i: int) -> int:
    """Past a ``#`` comment: at the newline, which no backslash escapes here."""
    end = text.find("\n", i)
    return len(text) if end < 0 else end


def _substitute_flags_end(text: str, i: int) -> int:
    """Past the flags of ``s``, refusing ``w`` (a file) and ``e`` (a command)."""
    while i < len(text) and text[i] not in _COMMAND_END:
        if text[i] in "we":
            raise _Refused(f"the {text[i]!r} flag of s {_REACHES_OUT[text[i]]}")
        if text[i] not in _SUBSTITUTE_FLAGS:
            raise _Refused(f"{text[i]!r} is not a flag of s")
        i += 1
    return i


def _end_of_command(text: str, i: int) -> int:
    at = _skip(text, i, _BLANKS)
    if at < len(text) and text[at] not in _COMMAND_END:
        raise _Refused("text after a command")
    return at


def _label_end(text: str, i: int) -> int:
    """Past a label: it ends at a blank, ``;``, ``}``, ``#`` or a newline (GNU's
    ``read_label``). The shortest reading: what follows is read as commands."""
    at = _skip(text, i, _BLANKS)
    while at < len(text) and text[at] not in _BLANKS + _COMMAND_END:
        at += 1
    return at


def _text_end(text: str, i: int) -> int:
    """Past the text of ``a``/``i``/``c``: at the first newline no backslash escapes."""
    while i < len(text) and text[i] != "\n":
        i += 2 if text[i] == "\\" else 1
    return i


def _plain_end(text: str, i: int, delimiter: str) -> int:
    """Past a replacement or a ``y`` string: at its unescaped delimiter, on one line."""
    while i < len(text) and text[i] != "\n":
        if text[i] == delimiter:
            return i + 1
        i += 2 if text[i] == "\\" else 1
    raise _Refused("a delimiter that is never closed")


def _delimiter(text: str, at: int) -> str:
    if at >= len(text) or text[at] in "\n\\":
        raise _Refused("a regex with no delimiter")
    return text[at]


def _skip(text: str, i: int, chars: str) -> int:
    while i < len(text) and text[i] in chars:
        i += 1
    return i


# Every command but ``{`` and the ones that reach outside the script, by how its
# argument is read. ``a``/``i``/``c`` text and a comment run to the end of the line.
_ARGUMENT_END: Mapping[str, Callable[[str, int], int]] = {
    "#": _comment_end, "s": _substitute_end, "y": _transliterate_end,
    **dict.fromkeys("aic", _text_end), **dict.fromkeys(":btTv", _label_end),
    **dict.fromkeys("lLqQ", _number_argument_end), **dict.fromkeys("=dDgGhHnNpPxzF}", _no_argument),
}


# ── a regex: where GNU, BSD and BusyBox sed end it ───────────────────────────

def _class_start(text: str, i: int) -> int:
    """Past a leading ``^`` and a leading ``]``: neither closes a bracket expression."""
    i += 1 if text[i:i + 1] == "^" else 0
    return i + (1 if text[i:i + 1] == "]" else 0)


def _gnu_class_end(text: str, i: int) -> Optional[int]:
    """GNU sed's ``snarf_char_class``: past the ``]`` that closes the bracket expression
    opened just before ``i``. ``[:``/``[.``/``[=`` open a class its own ``:]`` closes."""
    i, state, opener = _class_start(text, i), 0, ""
    while i < len(text) and text[i] != "\n":
        ch = text[i]
        if ch == "]" and state < 2:
            return i + 1
        if ch == "[":
            state = state or 1
        elif ch in ".:=" and state == 1:
            state, opener = 2, ch
        elif ch in ".:=" and state == 2 and ch == opener:
            state = 3
        else:
            state = 0 if ch == "]" and state == 3 else state & ~1
        i += 1
    return None


def _bsd_class_end(text: str, i: int) -> Optional[int]:
    """BSD sed's ``compile_ccl``: a class from ``[:`` runs to the next ``:]``."""
    i = _class_start(text, i)
    while i < len(text) and text[i] != "\n":
        if text[i] == "]":
            return i + 1
        if text[i] == "[" and text[i + 1:i + 2] in (".", ":", "="):
            close = text.find(text[i + 1] + "]", i + 2)
            if close < 0 or "\n" in text[i + 2:close]:
                return None
            i = close + 2
        else:
            i += 1
    return None


def _busybox_class_end(text: str, i: int) -> Optional[int]:
    """BusyBox sed: the first ``]`` closes, classes or not."""
    i = _class_start(text, i)
    end = text.find("]", i)
    return None if end < 0 or "\n" in text[i:end] else end + 1


# (class end, a ``[`` opens one even when it is the delimiter): BusyBox checks it first.
_BRACKET_READINGS: Tuple[Tuple[Callable[[str, int], Optional[int]], bool], ...] = (
    (_gnu_class_end, False), (_bsd_class_end, False), (_busybox_class_end, True))


def _regex_end_by(text: str, i: int, delimiter: str,
                  class_end: Callable[[str, int], Optional[int]], bracket_first: bool) -> Optional[int]:
    found: Optional[int] = i
    while found is not None and found < len(text) and text[found] != "\n":
        ch = text[found]
        if ch == "[" and (bracket_first or delimiter != "["):
            found = class_end(text, found + 1)
        elif ch == delimiter:
            return found + 1
        else:
            found += 2 if ch == "\\" else 1
    return None


def _regex_end(text: str, at: int) -> int:
    """Past the regex whose delimiter is at ``at``. Refused unless every reading ends
    it at the same place."""
    delimiter = _delimiter(text, at)
    ends = {_regex_end_by(text, at + 1, delimiter, *reading) for reading in _BRACKET_READINGS}
    if None in ends or len(ends) > 1:
        raise _Refused("a regex the seds do not end at the same place")
    return ends.pop()


__all__ = ["SedReader", "sed_escape"]
