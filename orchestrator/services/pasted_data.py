"""F199 (night 6): sums over data pasted into a ticket are computed with code,
and no figure is called verified unless code checked it.

- #1123 "Tuesday's numbers (pasted in)", agent 329, 04:19: the numbers were
  listed right and added up wrong, twice, to two different wrong totals.
- #1149 "What sold last week (order lines pasted in)", agent 327, 05:28: every
  figure was wrong, and its "Verification Totals" added up its own wrong
  numbers.
Neither agent ran a single tool. F181 gave a spreadsheet document its
count-with-code rule; data pasted into the brief had none.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

DATA_LINES_MIN = 5
_NUMBER = re.compile(r"(?<![\w.])[£$€]?\d[\d,]*(?:\.\d+)?%?")
_DELIMITER = re.compile(r"[,\t|;]")

PASTED_DATA_RULE = (
    "## Pasted data\n"
    "The brief carries {lines} lines of pasted data. Count, sum, total or average them with code, never "
    "by hand: write the lines to a file with workspace_write_file and compute with python3 through "
    "workspace_exec. Call a figure verified or checked only when code computed it in this run."
)
# The actions that run code over data.
CODE_ACTIONS = ("workspace_exec", "run_skill_script", "query_data")
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
_VERIFIED = re.compile(
    r"\bverif(?:ied|y|ication)\b|\b(?:double|cross|re)[- ]?check(?:ed)?\b"
    r"|\bchecked (?:the |these |all |every |each )?(?:totals?|sums?|figures?|numbers?|maths?|calculations?)\b", re.I)
# Figures: a figures noun, or an amount of money ("I checked and Room 12 is free" is not one).
_FIGURES = re.compile(r"\b(?:totals?|sums?|figures?|numbers?|counts?|amounts?|revenue|units|bags|orders)\b"
                      r"|[£$€]\s?\d", re.I)
NOT_VERIFIED_NOTE = ("\n\nCheck before relying on these figures: no code computed them in this run, so they are "
                     "not verified.")


def pasted_data_lines(text: object) -> int:
    """How many lines of ``text`` look like pasted data rows: two numbers or
    more, or a number and a delimiter."""
    rows = 0
    for line in str(text or "").splitlines():
        numbers = len(_NUMBER.findall(line))
        if numbers >= 2 or (numbers == 1 and _DELIMITER.search(line)):
            rows += 1
    return rows


def pasted_data_rule(brief: object) -> Optional[str]:
    """The count-with-code rule for a brief that carries pasted data, else None."""
    lines = pasted_data_lines(brief)
    return PASTED_DATA_RULE.format(lines=lines) if lines >= DATA_LINES_MIN else None


def unverified_figures_note(result: object, succeeded: Iterable[str]) -> Optional[str]:
    """The note for a result that calls its figures verified or checked when no
    code ran in its run, else None."""
    if any(code in str(action) for action in succeeded or () for code in CODE_ACTIONS):
        return None
    for sentence in _SENTENCE.findall(str(result or "")):
        if _VERIFIED.search(sentence) and _FIGURES.search(sentence):
            return NOT_VERIFIED_NOTE
    return None
