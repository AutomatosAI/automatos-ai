"""Where an agent's facts come from (F300, F304 and F313, night 9).

Night 9 ran on a coffee roaster's workspace with a live shop database
('harbourline_shop') and twelve dated documents.

- F300: agents asked the owner for table and column names (#1856 asked three times,
  asks #1458, #1459 and #1461; #1886 wanted "the exact plan_code"; #1891 "the
  subscription_orders schema") and spent their five queries guessing columns. The
  owner: "I don't know table names, I run a roastery."
- F304: answers named sources they never read. #1858's third run said the £1,455
  "comes from the harbourline_shop database's retail orders table, which I accessed in
  a previous query"; both earlier runs had failed, and the figure was Auto's chat
  answer (report L81).
- F313: "how much Guji right now" was answered from the 1 September list (140 kg; the
  shop system said 118), and #1864, #1865 and #1869 used September stock with the
  database in reach.

Night 9b (F327 and F323): agents asked the owner for an account id, a lot code, a date
range and parameter names (Friction 2, 15, 33, 46); Auto offered a web search for the
owner's own stock; the Business Analyst read "this summer" as 2024 (#1984). The block
now says to look those up, keep the web out of the business's own data, and read a
season or month against today's date (services/todays_date.py puts it in every prompt).

``FACTS_RULES`` is what every API run of an agent's work is told about its facts. It
rides in ``services.step_lessons.ON_THE_CARD``, so a plain card, a mission step and a
playbook step all get it where they get "Where your answer goes".

``an_answer_names_only_what_it_read`` is the cheap check behind F304 on a board card's
answer, beside F199's (``services.pasted_data.unverified_figures_note``): an answer
that cites an earlier run's query, or gives the database as its source when no
database query worked in its run, gets a line saying so. Only the names of the actions
that worked reach the check, never what they returned, so a document cited wrongly
after a search (L11, L108) is the prompt's rule alone.
"""
from __future__ import annotations

import functools
import re
from typing import Callable, Iterable, Optional

FACTS_HEADING = "## Where your facts come from"
FACTS_RULES = (
    f"{FACTS_HEADING}\n"
    "A question about now (\"now\", \"current\", \"today\", \"left\") is answered from the live system first: your "
    "database tool, when you have one. A dated document (a list \"as of 1 Sep\") is not today's figure: use it only "
    "when the live system can't answer, and give its date.\n"
    "With a database tool, learn its tables and columns through the tool before your first question (ask it which "
    "tables and columns there are), then use those names. Never ask the owner for table, column or schema names: "
    "they don't know them. If the tool can't tell you, say so in your answer.\n"
    "Cite only what a tool returned in this run, by the name the tool gave it (the document's file name, the "
    "database's name). Nothing from an earlier run reaches this one: never write \"a previous query\". A figure "
    "from the brief, the owner's notes or another card is cited as coming from there.\n"
    "Look up ids, codes and names yourself with your tools (an account from its name, a lot from the stock): never "
    "ask the owner for an id, a code, a parameter name or a date range they already gave. The business's own data "
    "(its stock, orders and customers) is in its database and documents: never search the web for it.\n"
    "Read \"this summer\", \"last month\" or \"this year\" against today's date in your instructions (\"Today "
    "is …\"), never a year from memory."
)

EARLIER_RUN_NOTE = ("\n\nCheck the source before relying on this answer: it cites \"{cited}\", but nothing from an "
                    "earlier run reaches this one, so no tool read that source for this answer.")
NO_QUERY_NOTE = ("\n\nCheck the source before relying on this answer: it gives the database as its source, but no "
                 "database query worked in this run.")

# The actions that read the database (query_database, smart_query_database,
# platform_query_data, search_tables).
DATABASE_ACTIONS = ("query_data", "search_tables")

_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_EARLIER = re.compile(
    r"\b(?:a|an|the|my|our|that) (?:previous|earlier|prior|last) "
    r"(?:query|queries|run|search|searches|attempt|session|lookup|call)\b"
    r"|\bpreviously (?:accessed|retrieved|queried|pulled|fetched|read|obtained)\b", re.I)
_READ = re.compile(r"\b(?:accessed|retrieved|queried|pulled|fetched|read|obtained|looked up|got|comes? from|"
                   r"came from|source|according to|based on)\b", re.I)
_DATABASE = re.compile(r"\bdatabase\b|\b[a-z]+_[a-z_]+ table\b", re.I)
# A sentence about a failure, a correction or the brief is not a claim to have read a source.
_NOT_A_CLAIM = re.compile(r"n't\b|\b(?:not|no|never|cannot|unable|fail(?:ed|s)?|error|wrong|incorrect|mistake|"
                          r"brief|owner|card)\b", re.I)


def _claims(result: object) -> Iterable[str]:
    """The sentences of ``result`` that say where something came from."""
    for sentence in _SENTENCES.split(str(result or "")):
        if _READ.search(sentence) and not _NOT_A_CLAIM.search(sentence):
            yield sentence


def unread_source_note(result: object, succeeded: Iterable[str]) -> Optional[str]:
    """The owner's line for an answer that names a source its run never read, else None."""
    claims = list(_claims(result))
    for sentence in claims:
        earlier = _EARLIER.search(sentence)
        if earlier:
            return EARLIER_RUN_NOTE.format(cited=earlier.group(0))
    queried = any(name in str(action) for action in succeeded or () for name in DATABASE_ACTIONS)
    if not queried and any(_DATABASE.search(sentence) for sentence in claims):
        return NO_QUERY_NOTE
    return None


def an_answer_names_only_what_it_read(
        check: Callable[[object, Iterable[str]], Optional[str]]) -> Callable[[object, Iterable[str]], Optional[str]]:
    """Wrap a board card's answer check that takes the answer and the names of the
    actions that worked in its run (F199's): its note, then F304's."""
    @functools.wraps(check)
    def wrapped(result: object, succeeded: Iterable[str]) -> Optional[str]:
        ran = list(succeeded or ())
        notes = [check(result, ran), unread_source_note(result, ran)]
        return "".join(note for note in notes if note) or None
    return wrapped


__all__ = ["FACTS_HEADING", "FACTS_RULES", "an_answer_names_only_what_it_read", "unread_source_note"]
