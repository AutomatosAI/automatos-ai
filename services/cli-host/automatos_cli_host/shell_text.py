"""The text of a command line before it is words: here-documents and command substitutions.

Pure and stdlib-only, moved out of ``policy.py`` unchanged (PRD-253 Wave P), which
was well past the 800-line bound. The gate cuts a here-document's body before it
tokenises a line — a body is data — and judges the substitutions an unquoted
delimiter makes the shell run; it judges every ``$(…)`` and backtick body as a
command line of its own, and reads the word around one with the body replaced by
``SUBSTITUTION_MARK``.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

_HEREDOC_RE = re.compile(r"<<-?\s*(?:'([^']*)'|\"([^\"]*)\"|\\([A-Za-z_][A-Za-z0-9_]*)|([A-Za-z_][A-Za-z0-9_]*))")
SUBSTITUTION_MARK = "$_"     # stands in for a ``$(…)`` body once that body is judged on its own


# ── here-documents ───────────────────────────────────────────────────────────

def heredoc_end(text: str, start: int, delimiter: str) -> Optional[int]:
    """The end of the line that terminates a here-document whose body starts at
    ``start`` (``<<-`` lets the terminator be tab-indented); None when there is none."""
    pos = start
    while pos <= len(text):
        newline = text.find("\n", pos)
        line = text[pos:] if newline < 0 else text[pos:newline]
        if line.lstrip("\t") == delimiter:
            return len(text) if newline < 0 else newline
        if newline < 0:
            return None
        pos = newline + 1
    return None


def heredoc_delimiter(match: Any) -> Tuple[str, bool]:
    """The here-document's terminator, and whether it was QUOTED. A quoted
    delimiter (``<<'EOF'``) makes the body inert data; an unquoted one
    (``<<EOF``) expands the substitutions inside it as the shell reads it."""
    if match.group(1) is not None:
        return match.group(1), True
    if match.group(2) is not None:
        return match.group(2), True
    if match.group(3) is not None:      # ``<<\EOF`` — bash treats it exactly like ``<<'EOF'``
        return match.group(3), True
    return match.group(4), False


def heredocs(command: str) -> Tuple[str, List[str]]:
    """The command without its here-document bodies, plus the bodies whose
    delimiter was UNQUOTED.

    A body is data, not part of the command line, so it is cut before tokenising
    (the ``<<`` itself stays, so the redirection beside it is still judged). But
    an unquoted delimiter makes the shell RUN the substitutions in that body, so
    those bodies come back for judging. A body without its terminator is left
    where it is."""
    out = command
    expanded: List[str] = []
    pos = 0
    while True:
        match = _HEREDOC_RE.search(out, pos)
        if match is None:
            return out, expanded
        line_end = out.find("\n", match.end())
        if line_end < 0:
            return out, expanded
        delimiter, quoted = heredoc_delimiter(match)
        body_end = heredoc_end(out, line_end + 1, delimiter)
        if body_end is None:
            return out, expanded
        if not quoted:
            expanded = [*expanded, out[line_end + 1:body_end]]
        out = out[:line_end] + out[body_end:]
        pos = match.end()


# ── command substitutions inside a word ──────────────────────────────────────

def matching_paren(text: str, start: int) -> int:
    """Index of the ')' closing the '(' at ``start``; the end of the text when unbalanced."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def substitution_spans(word: str) -> List[Tuple[int, int, str]]:
    """``(start, end, body)`` of every ``$(…)`` and backtick substitution in a
    word — the tokenizer keeps them whole when quoted. Arithmetic ``$((…))`` is
    skipped; a nested substitution is found when its body is judged."""
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(word):
        if word.startswith("$((", i):
            i = matching_paren(word, i + 1) + 1
        elif word.startswith("$(", i):
            end = matching_paren(word, i + 1)
            spans = [*spans, (i, end + 1, word[i + 2:end])]
            i = end + 1
        elif word[i] == "`":
            end = word.find("`", i + 1)
            end = len(word) if end < 0 else end
            spans = [*spans, (i, end + 1, word[i + 1:end])]
            i = end + 1
        else:
            i += 1
    return spans


def substitutions(word: str) -> List[str]:
    return [body for _, _, body in substitution_spans(word)]


def backtick_bodies(line: str) -> List[str]:
    """The bodies of the backtick substitutions only — the ones the tokenizer
    cannot keep whole when unquoted."""
    return [body for start, _, body in substitution_spans(line) if line[start] == "`"]


def without_substitutions(word: str) -> str:
    """The word with each substitution replaced by ``SUBSTITUTION_MARK`` — an
    unresolved reference wherever the body's output would land."""
    out = ""
    last = 0
    for start, end, _ in substitution_spans(word):
        out += word[last:start] + SUBSTITUTION_MARK
        last = end
    return out + word[last:]


__all__ = [
    "SUBSTITUTION_MARK", "backtick_bodies", "heredoc_delimiter", "heredoc_end", "heredocs", "matching_paren",
    "substitution_spans", "substitutions", "without_substitutions",
]
