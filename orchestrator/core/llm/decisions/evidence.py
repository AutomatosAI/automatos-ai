"""PRD-248 tuning (6 Oct) — the evidence a judgement reads beside an agent's own account (pure).

Night 4 joined the session_end and report_triage rows to graded runs. Jev read the
agent's final message and nothing else, and agents describe wrong work as
confidently as right work: P(complete) was 0.92 on wrong runs and 0.73 on right
ones. Ticket #673 was asked for 17 Sep and did 22 Sep; it was read as complete at
0.93. So the state now carries what the work is, not only what the agent says
about it: the brief's dates and the dates found in the work, the deliverables'
names and first lines, the files written and the commands refused.

Comparing dates is on TypeSafe's own list of weak spots, so the dates are pulled
out here and the ones the brief names that the work never mentions are listed. The
model reads the lists; it is never asked to do the date arithmetic. No I/O.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LIST_MAX = 10
FILES_MAX = 20
NAME_MAX_CHARS = 120
LINES_MAX_CHARS = 300
REFUSAL_MAX_CHARS = 200

_MONTHS: Dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTH = "(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_ORDINAL = r"(?:st|nd|rd|th)?"
_YEAR = r"(?:,?\s+(20\d{2}))?"
_ISO = re.compile(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?{_MONTH}{_YEAR}\b", re.IGNORECASE)
_MONTH_DAY = re.compile(rf"\b{_MONTH}\s+(\d{{1,2}}){_ORDINAL}{_YEAR}\b", re.IGNORECASE)

DateKey = Tuple[int, int, Optional[int]]  # (month, day, year or None)


def _month(token: str) -> int:
    """The month a matched token names; 0 for a lower-case "may", which is far more often
    the verb ("you may 2 ...") than the month."""
    return 0 if token.rstrip(".") == "may" else _MONTHS[token.rstrip(".").lower()]


def _key(month: int, day: int, year: Optional[str]) -> Optional[DateKey]:
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        return None
    return month, day, int(year) if year else None


def _found(text: str) -> List[Tuple[int, DateKey]]:
    hits: List[Tuple[int, DateKey]] = []
    for match in _ISO.finditer(text):
        key = _key(int(match.group(2)), int(match.group(3)), match.group(1))
        if key:
            hits.append((match.start(), key))
    for match in _DAY_MONTH.finditer(text):
        key = _key(_month(match.group(2)), int(match.group(1)), match.group(3))
        if key:
            hits.append((match.start(), key))
    for match in _MONTH_DAY.finditer(text):
        key = _key(_month(match.group(1)), int(match.group(2)), match.group(3))
        if key:
            hits.append((match.start(), key))
    return sorted(hits)


def _label(key: DateKey) -> str:
    month, day, year = key
    return f"{day} {_MONTH_NAMES[month - 1]}" + (f" {year}" if year else "")


def dates_in(text: Any) -> List[str]:
    """The dates written in ``text`` ("17 Sep", "Sep 17th", "2026-09-17"), as "17 Sep [2026]",
    each once, in the order they appear."""
    seen: List[str] = []
    for _, key in _found(str(text or "")):
        label = _label(key)
        if label not in seen:
            seen.append(label)
    return seen


def _day_month(label: str) -> str:
    return " ".join(label.split()[:2])


def dates_missing(asked: Sequence[str], found: Sequence[str]) -> List[str]:
    """The asked dates whose day and month appear nowhere in ``found`` (a year on one side
    only still matches)."""
    present = {_day_month(label) for label in found}
    return [label for label in asked if _day_month(label) not in present]


def date_evidence(brief: Any, work: Iterable[Any]) -> Dict[str, List[str]]:
    """The brief's dates, the work's dates, and the brief's dates the work never mentions.
    Empty when the brief names no date."""
    asked = dates_in(brief)
    if not asked:
        return {}
    found = dates_in("\n".join(str(piece or "") for piece in work))
    return {
        "brief_dates": asked,
        "dates_in_the_work": found,
        "brief_dates_missing_from_the_work": dates_missing(asked, found),
    }


def deliverable_lines(deliverables: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Each deliverable as its name, its type and its first lines (when it has text)."""
    out: List[Dict[str, str]] = []
    for item in list(deliverables)[:LIST_MAX]:
        entry = {"name": str(item.get("name") or "")[:NAME_MAX_CHARS], "type": str(item.get("type") or "")}
        lines = str(item.get("first_lines") or "").strip()
        if lines:
            entry["first_lines"] = lines[:LINES_MAX_CHARS]
        out.append(entry)
    return out


def file_names(files: Sequence[Any]) -> List[str]:
    """The files written, as their paths (the last two parts), at most ``FILES_MAX``."""
    names: List[str] = []
    for raw in list(files)[:FILES_MAX]:
        parts = [p for p in str(raw or "").replace("\\", "/").split("/") if p]
        if parts:
            names.append("/".join(parts[-2:])[:NAME_MAX_CHARS])
    return names


def refusals(denials: Sequence[Any]) -> List[str]:
    """Each refused command as one line: the tool, what it was about, and why."""
    lines: List[str] = []
    for denial in list(denials)[:LIST_MAX]:
        if not isinstance(denial, Mapping):
            lines.append(str(denial)[:REFUSAL_MAX_CHARS])
            continue
        about = denial.get("subject") or ""
        reason = denial.get("reason") or ""
        text = f"{denial.get('tool') or '?'}: {about}".rstrip(": ") + (f" ({reason})" if reason else "")
        lines.append(text[:REFUSAL_MAX_CHARS])
    return lines


__all__ = [
    "date_evidence", "dates_in", "dates_missing", "deliverable_lines", "file_names", "refusals",
]
