"""PRD-268 §6.2.4 — a suggestion is said once: subject keys, normalised titles and the dedupe rule.

Every suggestion carries a ``subject_key`` saying what it is about:

* ``agent:<id>:error:<class>`` — one agent's failing streak of one error class (§6.2.2);
* ``ticket:<number>`` — one board ticket, by its board number (``#0846`` is ``ticket:0846``);
* ``title:<normalised title>`` — anything else, keyed by what it says.

A new suggestion is a duplicate when an ``open`` one has the same key or the same
normalised title, or a ``hidden`` one has either and was hidden within the last 14 days.
A started or expired suggestion never blocks a new one.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional

__all__ = [
    "HIDDEN_WINDOW",
    "SUBJECT_KEY_LENGTH",
    "TICKET_DIGITS",
    "Candidate",
    "agent_error_key",
    "agent_error_prefix",
    "board_number",
    "error_class_name",
    "find_duplicate",
    "is_duplicate",
    "normalise_title",
    "ticket_key",
    "title_key",
]

# A hidden suggestion blocks the same one for two weeks; after that it may come back.
HIDDEN_WINDOW = timedelta(days=14)
# The width of proactive_suggestions.subject_key (core.models.proactive SUBJECT_KEY_LENGTH).
SUBJECT_KEY_LENGTH = 200
# A board number's digits (services.ticket_numbers NUMBER_DIGITS): ticket #0846.
TICKET_DIGITS = 4

# "#0846", "846" or a mission step's "#0051.3".
_TICKET_NUMBER = re.compile(r"^#?(\d+)(?:\.(\d+))?$")


def _is_punctuation(char: str) -> bool:
    return unicodedata.category(char).startswith("P")


def normalise_title(title: Any) -> str:
    """``title`` as dedupe compares it: Unicode-normalised (NFKC), case folded, runs of
    spaces made one, and trailing punctuation off. "VECTOR's check-ins  FAIL!" is
    "vector's check-ins fail"."""
    text = " ".join(unicodedata.normalize("NFKC", str(title or "")).casefold().split())
    end = len(text)
    while end and (_is_punctuation(text[end - 1]) or text[end - 1].isspace()):
        end -= 1
    return text[:end]


def _key(prefix: str, value: str) -> str:
    return f"{prefix}:{value}"[:SUBJECT_KEY_LENGTH]


def error_class_name(error_class: Any) -> str:
    """An error class as a streak counts it and its key writes it: case folded, spaces as
    underscores. "Provider refused" is ``provider_refused``; nothing is ""."""
    return "_".join(str(error_class or "").casefold().split())


def agent_error_prefix(agent_id: Any) -> str:
    """What every failing-streak key of one agent starts with: ``agent:578:error:``."""
    return f"agent:{int(agent_id)}:error:"


def agent_error_key(agent_id: Any, error_class: Any) -> str:
    """The key of one agent's failing streak: ``agent:578:error:401``."""
    error = error_class_name(error_class)
    if not error:
        raise ValueError("a failing streak's key needs its error class")
    return f"{agent_error_prefix(agent_id)}{error}"[:SUBJECT_KEY_LENGTH]


def board_number(value: Any) -> Optional[str]:
    """A ticket's board number as a key writes it, or None when ``value`` is not one:
    "#0846", "846" and 846 are "0846"; a mission step's "#0051.3" is "0051.3". A float is
    not one: sent as a JSON number, a step's number may have lost a 0 (F241)."""
    if isinstance(value, (bool, float)):
        return None
    match = _TICKET_NUMBER.match(str(value).strip())
    if match is None:
        return None
    seq, step = match.groups()
    padded = f"{int(seq):0{TICKET_DIGITS}d}"
    return f"{padded}.{int(step)}" if step else padded


def ticket_key(number: Any) -> str:
    """The key of a suggestion about one ticket, by its board number: ``ticket:0846``."""
    canonical = board_number(number)
    if canonical is None:
        raise ValueError(f"{number!r} is not a ticket's board number")
    return _key("ticket", canonical)


def title_key(title: Any) -> str:
    """The key of a suggestion about anything else: ``title:`` and its normalised title.
    A title that normalises to nothing says nothing, and is refused."""
    normalised = normalise_title(title)
    if not normalised:
        raise ValueError("a suggestion's title says nothing once normalised")
    return _key("title", normalised)


@dataclass(frozen=True)
class Candidate:
    """A suggestion a check-in wants to make: its title and, when it names one, its key."""

    title: str
    subject_key: Optional[str] = None

    @property
    def key(self) -> str:
        """The key the suggestion is stored under: its own, else its title's."""
        return self.subject_key or title_key(self.title)


def _same_subject(candidate: Candidate, row: Any) -> bool:
    if row.subject_key == candidate.key:
        return True
    title = normalise_title(candidate.title)
    return bool(title) and normalise_title(row.title) == title


def _hidden_at(row: Any) -> Optional[datetime]:
    return row.decided_at or row.last_seen_at or row.created_at


def _hidden_recently(row: Any, now: datetime) -> bool:
    hidden_at = _hidden_at(row)
    return hidden_at is not None and now - hidden_at <= HIDDEN_WINDOW


def find_duplicate(candidate: Candidate, open_rows: Iterable[Any], hidden_rows: Iterable[Any],
                   now: datetime) -> Optional[Any]:
    """The suggestion ``candidate`` repeats, or None: an open one about the same subject,
    else a hidden one about it hidden within ``HIDDEN_WINDOW`` of ``now``.

    Rows carry ``subject_key`` and ``title``; hidden rows also ``decided_at`` (when they
    were hidden), falling back to ``last_seen_at`` and ``created_at``.
    """
    for row in open_rows:
        if _same_subject(candidate, row):
            return row
    for row in hidden_rows:
        if _same_subject(candidate, row) and _hidden_recently(row, now):
            return row
    return None


def is_duplicate(candidate: Candidate, open_rows: Iterable[Any], hidden_rows: Iterable[Any],
                 now: datetime) -> bool:
    """True when ``candidate`` repeats an open suggestion or one hidden in the last 14 days."""
    return find_duplicate(candidate, open_rows, hidden_rows, now) is not None
