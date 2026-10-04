"""F316 / F324 (night 9b, build 15): two things Auto said in its own words that no call did.

F316: Auto's shop counts changed from chat to chat. "How many Harvest Club boxes go out on
Monday 5 October?" came back "the document doesn't specify the total number" in every fresh
chat (search_knowledge only, from the October club box note), then 5,957 (2fe5384a), 87
(e77f789a) and 63. "How many cancelled April to September?" came back 2, 118, then "12" with
a list adding to 11 (170ac093); asked where 2 came from, "I retrieved that information from our
previous conversation" (7c726e13), and "4 … came from my memory" (50aac6a9) with no call in the
turn. "Were any boxes late?" was answered from search_knowledge alone, sourced "my memory …
stored on October 4th" (11b9a74a); "£1455.00 … I seem to have misplaced the source" (cb5a62e6).
F302 put platform_query_data on every turn and F303 checks a disputed figure, but nothing
checked that a figure given for a shop question came from the shop in that turn.

F324: "I've stored the information … in my memory. This will ensure that all agents, including
the Support Agent, are aware of this change going forward" (5e404e17). Its one call was
platform_store_memory; a minute later Support #1962 and the Analyst #1963 said nothing was
changing. Auto's memory is its own; the agents read the owner's documents and their cards.

Now, in a chat turn (Auto's own words; what it quotes for a draft is left out):

- "counted from your shop system": on a turn the chat marked as asking a figure from the shop
  (``consumers/chatbot/shop_figures.py``), a sentence that gives a quantity, says the figure
  isn't there ("doesn't specify the total number"), or gives Auto's memory or an earlier chat
  as the source, needs a call to the shop's database this turn. The tool loop nudges it once,
  as every claim is (F108); a reply that still has none is saved with ``SHOP_LINE``.
- "told to the whole team": a sentence saying the agents or the team are aware, know, or have
  been told needs a call that put it where agents read it: a note in the owner's documents
  (platform_upload_document) or a card's or an agent's brief. platform_store_memory backs
  nothing here. A reply that still says so is saved with ``TEAM_LINE``.
"""
from __future__ import annotations

import contextvars
import functools
import re
from typing import Callable, List, Optional

SHOP_LABEL = "counted from your shop system"
TEAM_LABEL = "told to the whole team"
SHOP_LINE = ("Just to be clear: I didn't count this from your shop system in this reply, so a figure here may "
             "come from a document or an earlier chat, not today's count. Ask me to count it from the shop and "
             "I will.")
TEAM_LINE = ("Just to be clear: your agents haven't been told. I only keep this in my own memory, and they read "
             "your documents and their cards, not my memory. Ask me to write it into a note in your documents "
             "and every agent will read it.")
LINES = {SHOP_LABEL: SHOP_LINE, TEAM_LABEL: TEAM_LINE}

# The shop's database, by the names its doors are recorded under (platform_execute's inner action).
_SHOP_CALLS = ("query_data", "query_database", "smart_query", "nl2sql", "sql")
# What puts a correction where the agents read it: the owner's documents, a card, an agent's brief.
_TEAM_CALLS = ("upload_document", "add_to_knowledge", "update_task", "create_task", "assign_task", "update_agent")

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|"
          r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b")
# A figure: "63", "5,957", "41.0", "£1455.00". Not "#0101", "14-day", "250g", "5th", "08:00".
_NUMBER = re.compile(r"(?<![#\w.,:/£$€-])([£$€]?)(\d[\d,]*(?:\.\d+)?)(?![\w:/-]|\.\d)")
_BEFORE_DATE = re.compile(r"^\s+" + _MONTH, re.I)
_AFTER_MONTH = re.compile(r"\b" + _MONTH + r"\s*$", re.I)
_YEAR = re.compile(r"^(?:19|20)\d\d$")
_LIST_MARK = re.compile(r"^\s*(?:[-*\u2022]|\d{1,2}[.)])\s+")
# A sentence, not split at a decimal point.
_SENTENCE = re.compile(r"(?:[^.!?\n]|\.(?=\d))+[.!?]?")
_NOT_THERE = re.compile(
    r"\b(?:does(?:n't| not)|do(?:n't| not)|did(?:n't| not))\s+(?:specify|say|give|show|list|include|mention|"
    r"contain|hold|have)\b|\bnot (?:available|specified|listed|mentioned|included)\b"
    r"|\b(?:can't|cannot|can not|couldn't|could not|unable to)\s+(?:tell|find|give|count|see|get|provide|retrieve|"
    r"access)\b|\bno (?:information|data|records?|details)\b",
    re.I)
_REMEMBERED = re.compile(
    r"\b(?:from|in) (?:my|our) (?:memory|memories|(?:previous|earlier|last) (?:conversation|chat)s?)\b"
    r"|\bi (?:remember(?:ed)?|recall(?:ed)?)\b|\bmisplaced the source\b",
    re.I)
_GROUP = (r"(?:(?:all|every|each)(?: (?:of )?(?:the|your|my))? (?:agents?|helpers?|team members?)"
          r"|(?:the|your|my) (?:whole |entire )?team|(?:the|your|my) agents|everyone|everybody)")
_TOLD = re.compile(
    r"\b" + _GROUP + r"\b[^.!?\n]{0,60}?\b(?:(?:are|is|will be|'re)\s+(?:now\s+|all\s+|fully\s+)*"
    r"(?:aware|informed|briefed|updated|up to date|in the loop)|(?:now |will |all )?knows?|"
    r"(?:has|have) been (?:told|informed|briefed|updated|notified))\b"
    r"|\bi(?:'ve| have)(?: (?:just|now|already|also))* (?:told|informed|briefed|notified|let)\b[^.!?\n]{0,40}"
    r"\b(?:team|agents?|everyone|everybody)\b"
    r"|\b(?:make|made|making) sure (?:that )?" + _GROUP + r"\b[^.!?\n]{0,30}\b(?:knows?|aware|informed)\b"
    r"|\bensures? (?:that )?" + _GROUP + r"\b[^.!?\n]{0,60}?\b(?:knows?|aware|informed)\b",
    re.I)
# Not a claim: an offer or a question, the reply saying they don't know yet, the knowing made the
# condition of something Auto would do ("the team will know once I post the note"), or the purpose
# of a plan ("I'll write it into a note so every agent knows").
_OFFER = re.compile(r"\?|\b(?:if you|would you|do you want|shall i|should i|want me to|i can|i could)\b", re.I)
_NEGATED = re.compile(r"\b(?:not|never|no longer)\b|n't\b", re.I)
_CONDITION = re.compile(r"\b(?:once|if|when|after|as soon as|until)\s+(?:i|you|we)\b", re.I)
_PLAN = re.compile(r"\b(?:i'll|i will|i'm going to|i am going to|let me)\b", re.I)
_PURPOSE = re.compile(r"\bso(?: that)?\s+" + _GROUP + r"\b", re.I)

# The turn asks the owner's shop for a figure (set by the chat, per turn).
_SHOP_TURN: contextvars.ContextVar[bool] = contextvars.ContextVar("f316_shop_figure_turn", default=False)


def mark_shop_figure_turn(asks: bool) -> contextvars.Token:
    """Mark this turn as asking (or not) a figure from the owner's shop; returns the reset token."""
    return _SHOP_TURN.set(bool(asks))


def shop_figure_turn() -> bool:
    """Whether this turn asks a figure from the owner's shop."""
    return _SHOP_TURN.get()


def _plain(text: str) -> str:
    return (text or "").replace("\u2019", "'")


def _is_figure(sentence: str, match: "re.Match[str]") -> bool:
    currency, digits = match.group(1), match.group(2)
    if not currency and _YEAR.match(digits):
        return False
    after, before = sentence[match.end():], sentence[:match.start()]
    return not (_BEFORE_DATE.match(after) or _AFTER_MONTH.search(before))


def gives_a_figure(sentence: str) -> bool:
    """Whether ``sentence`` gives a quantity (not a date, a card's number, a year or a time)."""
    return any(_is_figure(sentence, m) for m in _NUMBER.finditer(sentence))


def _sentences(text: str) -> List[str]:
    lines = (_LIST_MARK.sub("", line) for line in _plain(text).splitlines())
    return [s for line in lines for s in _SENTENCE.findall(line) if s.strip()]


def _shop_claim(sentence: str) -> bool:
    if sentence.rstrip().endswith("?"):
        return False
    return gives_a_figure(sentence) or bool(_NOT_THERE.search(sentence) or _REMEMBERED.search(sentence))


def _team_claim(sentence: str) -> bool:
    if not _TOLD.search(sentence) or _OFFER.search(sentence) or _NEGATED.search(sentence):
        return False
    planned = bool(_PLAN.search(sentence) and _PURPOSE.search(sentence))
    return not (planned or _CONDITION.search(sentence))


def _ran(succeeded: List[str], stems: tuple) -> bool:
    return any(stem in action for action in succeeded for stem in stems)


def unbacked_shop_or_team_claim(text: str, succeeded: List[str]) -> Optional[str]:
    """``SHOP_LABEL`` or ``TEAM_LABEL`` for the first such claim in Auto's own words that no
    call this turn backs, else None. The shop claim counts only on a shop-figure turn."""
    shop_turn = shop_figure_turn() and not _ran(succeeded, _SHOP_CALLS)
    team_open = not _ran(succeeded, _TEAM_CALLS)
    for sentence in _sentences(text):
        if shop_turn and _shop_claim(sentence):
            return SHOP_LABEL
        if team_open and _team_claim(sentence):
            return TEAM_LABEL
    return None


Check = Callable[..., Optional[str]]


def also_checks_the_shop_and_the_team(check: Check) -> Check:
    """Wrap ``action_claims.claimed_action_not_done``: when it finds nothing, a chat turn's
    reply is checked for an unbacked shop figure or "the team knows" (see the module)."""
    @functools.wraps(check)
    def wrapped(text: str, done: Optional[set] = None, *, promises: Optional[bool] = None) -> Optional[str]:
        found = check(text, done, promises=promises)
        if found or not _autos_words(promises):
            return found
        from .action_claims import _own_words

        return unbacked_shop_or_team_claim(_own_words(text or ""), [a.lower() for a in (done or ())])
    return wrapped


def _autos_words(promises: Optional[bool]) -> bool:
    if promises is not None:
        return promises
    from .action_claims import _auto_speaks

    return _auto_speaks()


Line = Callable[[str], str]


def says_it_for_the_shop_and_the_team(not_done: Line) -> Line:
    """Wrap ``claim_check.not_done``: the owner's line for these two claims, in Auto's words."""
    @functools.wraps(not_done)
    def wrapped(claim: str) -> str:
        return LINES.get(claim) or not_done(claim)
    return wrapped


__all__ = ["SHOP_LABEL", "SHOP_LINE", "TEAM_LABEL", "TEAM_LINE", "also_checks_the_shop_and_the_team",
           "gives_a_figure", "mark_shop_figure_turn", "says_it_for_the_shop_and_the_team", "shop_figure_turn",
           "unbacked_shop_or_team_claim"]
