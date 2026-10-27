"""F317 (night 9b, build 15): the cards on the board that already answer the owner's question.

Auto answered as if the team had found nothing. "The club box count is on the board — the
Analyst and the Business Analyst both did cards on it. What number did they give?" got
"neither the Analyst nor the Shopify Business Analyst's completed tasks directly address the
total number of club boxes" (b8d81ba9) while #1932 and others said 63; it had listed the two
agents' done cards and read one about cancellations. "Check what the team has already found
on the board this week first" got a count of cards by status (147 tasks, 7 in review).
"Do I need to reorder any Kirinyaga?" came from the 1 September paper while the Watchdog's
card from ten minutes earlier sat on the board (378c1e41); Auto read it only when told to.

This is the read behind ``consumers/chatbot/team_findings.py``: the cards in review or done
whose title or answer share the words of the owner's question, best match first. It is F305's
past-work source (``services/past_work.py``: the board's answers, one workspace-scoped query,
no embedding) with two differences for Auto, who reports to the owner: a card in review comes
back too, marked as not approved yet, and each part of a message that asks several things is
matched on its own words, so a brief's club-box line finds the club-box cards.

F460 (night 18): "Did the Bell Yard invoice come back?" ranked an old card whose answer happened
to hold "income" and "feedback" above #1848, the card titled for the Bell Yard invoice: title and
answer were one string, matched part-word, and "come" and "back" counted as what it was about. Now
each word matches whole (a plural too), a title hit weighs ``TITLE_WEIGHT`` against an answer's
one, ties go newest first, and a card whose title holds the words comes back before it has an
answer. "come", "came" and "back" say how the owner asks; "count" says what about.
"""
from __future__ import annotations

import math
import re
from functools import reduce
from typing import Any, Dict, List, Tuple
from uuid import UUID

STATUSES = ("done", "review")
MAX_PARTS = 4
PER_PART = 2
MAX_CARDS = 4
MAX_TERMS = 8
MIN_TERMS = 2
MATCH_SHARE = 0.5
EXCERPT_CHARS = 700
TITLE_WEIGHT = 2
NO_ANSWER = "(no answer on the card yet)"
# A whole word, or its plural: Postgres regex word bounds, the term escaped and bound as a parameter.
WHOLE_WORD = r"\m{term}(?:e?s)?\M"
_WORD = re.compile(r"[^\W_][\w'-]{2,}", re.UNICODE)
_PART = re.compile(r"[?.!;:\n]+|,\s+|\s+[-–—]\s+")
_SPACE = re.compile(r"\s+")
# Words that say how the owner asks, not what about.
_STOP = frozenset("""
what what's which when where whom whose who why how many much does did do done have has had having the and
for are was were our ours you your yours their them they this that these those with from into onto about there
here please tell give need needs any all can could would should will just still also now today week weeks this
yesterday tomorrow number total counted counts check checked look looked find found already earlier first
team board boards card cards task tasks ticket tickets agent agents analyst business watchdog support ops
manager creator helper get got say said says anything something everything every each some more most less than
then it's i'm i've you're don't isn't not but out off over under only very really like make sure know let want
going goes way one two did they've she he her his its we're come came back
""".split())


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith(("xes", "ches", "shes", "sses")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def terms_of(text: str) -> List[str]:
    """The words that say what ``text`` is about, stemmed, in order, at most ``MAX_TERMS``."""
    seen: List[str] = []
    for word in _WORD.findall(str(text or "").lower().replace("’", "'")):
        stem = _stem(word.strip("'-"))
        if word not in _STOP and stem not in _STOP and len(stem) >= 3 and stem not in seen:
            seen.append(stem)
    return seen[:MAX_TERMS]


def parts_of(message: str) -> List[List[str]]:
    """The terms of each part of the message that says enough to match on, at most ``MAX_PARTS``."""
    parts = [terms_of(piece) for piece in _PART.split(str(message or ""))]
    useful = [terms for terms in parts if len(terms) >= MIN_TERMS]
    whole = terms_of(message)
    return useful[:MAX_PARTS] or ([whole] if len(whole) >= MIN_TERMS else [])


def _hits(column: Any, term: str) -> Any:
    """``column`` holds ``term`` as a whole word (F460: "come" never matches "income")."""
    from sqlalchemy import func

    return func.coalesce(column, "").regexp_match(WHOLE_WORD.format(term=re.escape(term)), flags="i")


def _scores(terms: List[str]) -> Tuple[Any, Any, Any]:
    """(the terms a card holds anywhere, those its title holds, its rank: a title hit weighs
    ``TITLE_WEIGHT``, an answer's one)."""
    from sqlalchemy import case, literal, or_

    from core.models.core import BoardTask

    def total(hits: List[Any]) -> Any:
        return reduce(lambda added, hit: added + hit, hits, literal(0))

    on_title = [_hits(BoardTask.title, term) for term in terms]
    on_answer = [_hits(BoardTask.result, term) for term in terms]
    held = total([case((or_(title, answer), 1), else_=0) for title, answer in zip(on_title, on_answer)])
    titled = total([case((title, 1), else_=0) for title in on_title])
    rank = total([case((title, TITLE_WEIGHT), (answer, 1), else_=0) for title, answer in zip(on_title, on_answer)])
    return held, titled, rank


def _cards_for(db: Any, workspace_id: UUID, terms: List[str], limit: int) -> List[Any]:
    from sqlalchemy import func, or_

    from core.models.core import BoardTask

    held, titled, rank = _scores(terms)
    needed = max(MIN_TERMS, math.ceil(len(terms) * MATCH_SHARE))
    newest = func.coalesce(BoardTask.completed_at, BoardTask.updated_at, BoardTask.created_at)
    return (db.query(BoardTask)
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.status.in_(STATUSES),
                    or_(BoardTask.result.isnot(None), titled >= needed), held >= needed)
            .order_by(rank.desc(), newest.desc().nullslast(), BoardTask.id.desc())
            .limit(limit).all())


def _agent_names(db: Any, cards: List[Any]) -> Dict[int, str]:
    from core.models import Agent

    ids = {c.assigned_agent_id for c in cards if c.assigned_agent_id}
    return dict(db.query(Agent.id, Agent.name).filter(Agent.id.in_(ids)).all()) if ids else {}


def _finding(card: Any, number: Any, agent: str) -> Dict[str, Any]:
    when = card.completed_at or card.updated_at
    return {"number": number or f"ticket {card.id}", "title": card.title, "agent": agent, "status": card.status,
            "day": f"{when.day} {when.strftime('%b')}" if when else "an unknown day",
            "excerpt": _SPACE.sub(" ", str(card.result or "")).strip()[:EXCERPT_CHARS] or NO_ANSWER}


def team_findings(db: Any, workspace_id: Any, message: str) -> List[Dict[str, Any]]:
    """The cards in review or done that answer the parts of ``message``, best match first,
    each with its number, title, agent, status, day and the start of its answer."""
    from services.ticket_numbers import ticket_numbers

    workspace = UUID(str(workspace_id))
    cards: List[Any] = []
    for terms in parts_of(message):
        cards.extend(c for c in _cards_for(db, workspace, terms, PER_PART) if c not in cards)
    cards = cards[:MAX_CARDS]
    if not cards:
        return []
    numbers = ticket_numbers(db, workspace, cards)
    names = _agent_names(db, cards)
    return [_finding(c, numbers.get(c.id), names.get(c.assigned_agent_id, "an agent")) for c in cards]


__all__ = ["EXCERPT_CHARS", "MAX_CARDS", "parts_of", "team_findings", "terms_of"]
