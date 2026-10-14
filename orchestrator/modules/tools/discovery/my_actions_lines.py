"""P256-FIX-N15-4 (night 15, F425): the record ``platform_my_actions`` gives the model, one short line each.

Night 15, 12 of 12 "what have you done for me tonight?" asks stopped between 23:50 and 23:55: the
read found the whole night, but each item went to the model as an indented JSON object, oldest
first, and a chat turn keeps about 2,000 tokens of a tool's answer, so it was cut after the first
couple of dozen. Here each write and click is one line ("23:01 card made #1588"), a run of the same
change shares one ("23:01 card made #1588, 23:05 #1589"), each day has its heading ("13 Oct:"),
and the lines, one text (a JSON list of strings costs about four more tokens a line), are fitted,
oldest first, to the tokens left after the answer's own fields. What does not fit is counted, and
the answer says from when on, and the ``since`` that reads it.
"""
from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple

from core.context_guard import count_tokens

from .my_actions_reads import DONE, WAITING, Item

ENTRY_TOKENS = 1        # what one line costs in the record beyond its words: the newline before the next
REF_CHARS = 32          # of a subject named by its words, not a board number
CARD_CHARS = 90         # of a clicked card's words
OUTCOME_CHARS = 80
DAY = "{moment.day} {moment:%b}:"     # "13 Oct:"
CLOCK = "%H:%M"
WAITING_SAID = "waiting for your click: {what}"
DECIDED_SAID = "asked your click, {decision}: {what}"
DECISIONS = {"approved": "you approved", "declined": "you declined", "lapsed": "it lapsed"}
CLICK_SAID = "{clock} you {decision} the card: {card}{numbers}{outcome}"


class Listed(NamedTuple):
    """The lines that fit, how many items did not, and the time of the first that did not."""

    lines: List[str]
    left_out: int
    next_since: Optional[str]


def record_lines(items: Sequence[Item], budget: int) -> Listed:
    """``items`` (writes and clicks, oldest first) as lines, as many as ``budget`` tokens hold. A cut
    falls between two times, never inside one (a turn's writes share theirs): the ``since`` that
    reads on from it then repeats nothing."""
    fitted = 0
    for spent, _ in _built(items):
        if spent > budget:
            break
        fitted += 1
    if fitted == len(items):
        return Listed(_lines_of(items), 0, None)
    cut = _time_boundary(items, fitted)
    return Listed(_lines_of(items[:cut]), len(items) - cut, items[cut]["at"])


def _built(items: Sequence[Item]) -> Iterator[Tuple[int, List[str]]]:
    """After each item: the tokens spent so far, and the lines so far."""
    lines: List[str] = []
    spent, last_day, last_key = 0, None, None
    for item in items:
        moment = datetime.fromisoformat(item["at"])
        day, key = DAY.format(moment=moment), _run_key(item)
        joins = key is not None and key == last_key and day == last_day
        added = [_joined(item, moment)] if joins else [*([day] if day != last_day else []), _alone(item, moment)]
        spent += sum(_tokens(text) for text in added) - (ENTRY_TOKENS if joins else 0)
        lines = [*lines[:-1], lines[-1] + added[0]] if joins else [*lines, *added]
        last_day, last_key = day, key
        yield spent, lines


def _lines_of(items: Sequence[Item]) -> List[str]:
    """Every line of ``items``: the lines after the last one is built."""
    last = deque(_built(items), maxlen=1)
    return last[0][1] if last else []


def _time_boundary(items: Sequence[Item], fitted: int) -> int:
    """The first item at the time of the first that did not fit; ``fitted`` when every item before it
    shares that time (a cut inside it is then the only way on)."""
    cut = fitted
    while cut > 0 and items[cut - 1]["at"] == items[fitted]["at"]:
        cut -= 1
    return cut or fitted


def by_time(writes: Sequence[Item], clicks: Sequence[Item]) -> List[Item]:
    """The writes and the clicks in one list, oldest first."""
    return sorted([*writes, *clicks], key=lambda item: datetime.fromisoformat(item["at"]))


def _tokens(text: str) -> int:
    """What ``text`` costs in the record, as the formatter writes it (inside one JSON string)."""
    return count_tokens(json.dumps(text)[1:-1]) + ENTRY_TOKENS


def _run_key(item: Item) -> Optional[Tuple[Any, ...]]:
    """Writes of the same change share a line; a click is always its own."""
    if "card" in item:
        return None
    return (item.get("status"), item.get("decision"), item.get("what"))


def _alone(item: Item, moment: datetime) -> str:
    clock = moment.strftime(CLOCK)
    if "card" in item:
        return _click(item, clock)
    return " ".join(part for part in (clock, _said(item), _ref(item)) if part)


def _joined(item: Item, moment: datetime) -> str:
    return ", " + " ".join(part for part in (moment.strftime(CLOCK), _ref(item)) if part)


def _said(item: Item) -> str:
    what = str(item.get("what") or "")
    if item.get("status") == WAITING:
        return WAITING_SAID.format(what=what)
    if item.get("status") not in (DONE, WAITING):
        return DECIDED_SAID.format(decision=DECISIONS.get(item.get("decision"), item.get("decision")), what=what)
    return what


def _ref(item: Item) -> str:
    """The card's board number, else the subject's words (cut), unless the change already says them."""
    if item.get("number"):
        return str(item["number"])
    subject = str(item.get("subject") or "").strip()
    if not subject or subject.lower() in str(item.get("what") or "").lower():
        return ""
    return f"'{_cut(subject, REF_CHARS)}'"


def _click(item: Dict[str, Any], clock: str) -> str:
    card = _cut(str(item.get("card") or ""), CARD_CHARS)
    numbers = [str(number) for number in item.get("numbers") or [] if str(number) not in card]
    outcome = str(item.get("outcome") or "")
    return CLICK_SAID.format(clock=clock, decision=item.get("decision"), card=card,
                             numbers=f" ({', '.join(numbers)})" if numbers else "",
                             outcome=f". {_cut(outcome, OUTCOME_CHARS)}" if outcome else "")


def _cut(text: str, chars: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= chars else text[: chars - 1].rstrip() + "…"


def record_text(listed: Listed) -> str:
    """The record as the answer carries it: one line each, oldest first."""
    return "\n".join(listed.lines)


__all__ = ["Listed", "by_time", "record_lines", "record_text"]
