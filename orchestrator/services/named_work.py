"""P256-FIX-N15-5 (F426): the words a search for a named piece of work matches on.

Night 15: "show me the Guji blog draft" got "there isn't a Guji blog draft", and "did the
Westmoor invoice come back?" got "I can't find a Westmoor invoice"; given the ticket
number, Auto showed both at once. platform_list_tasks had no search at all, and the
Deliverables search matched the owner's whole phrase, so "the Guji blog draft" found
neither a card titled "Guji blog draft" nor its file ``guji-blog-draft.md``.

A search now matches word by word: every word that names the work must appear, in any
order, and words that name nothing (the, my, of…) are left out. "Draft" names a stage,
not the work (a card waiting in Review IS a draft, whatever its title says), so it is
left out too when other words remain. Each word is letters and digits only, so it never
carries a LIKE wildcard. Pure: callers bind the patterns.

P256-FIX-N17-10 (F449): what a piece of the owner's work is called. ``PAPER_KINDS``, the
customer paperwork the hand-off table sends to the team (F337(c), consumers/chatbot/handoffs.py),
and ``WORK_NOUNS``, those and the other made work the owner asks to change ("make the Tidewater
quote shorter", "tweak the newsletter intro"): modules/tools/discovery/described_work.py.
"""
from __future__ import annotations

import re
from typing import Any, List

# Words that say nothing about WHICH work: articles, possessives and linking words.
FILLER_WORDS = frozenset({
    "a", "an", "the", "my", "our", "your", "their", "his", "her", "its",
    "this", "that", "these", "those", "of", "for", "to", "on", "in", "and", "about",
})
# Words for a piece of work's stage, not its name: kept only when nothing else is given.
STAGE_WORDS = frozenset({"draft", "drafts"})
# The most words one search matches on: each is one bound pattern in the query.
MAX_SEARCH_WORDS = 8
_WORD = re.compile(r"[^\W_]+")
# Customer paperwork, by what it is called (a regex alternation; F337(c)).
PAPER_KINDS = (r"letters?|invoices?|quotes?|quotations?|estimates?|proposals?|agreements?|contracts?|"
               r"flyers?|leaflets?|posters?|brochures?|price ?lists?|welcome sheets?|receipts?|newsletters?|"
               r"one-pagers?|menus?")
# Made work the owner names to change it: the paperwork and the rest (a regex alternation; F449).
# F460 (night 18): counts, lists, routines and checklists are made work the owner asks after too.
WORK_NOUNS = (PAPER_KINDS + r"|drafts?|posts?|intros?|introductions?|notes?|reports?|write-?ups?"
              r"|counts?|checklists?|lists?|routines?")


def search_words(raw: Any) -> List[str]:
    """The words of ``raw`` that name the work, lower-cased, in order, each once. A search
    of filler words only keeps them all: a search never widens to everything."""
    if not isinstance(raw, str):
        return []
    words = list(dict.fromkeys(word.lower() for word in _WORD.findall(raw)))
    named = [word for word in words if word not in FILLER_WORDS and len(word) > 1]
    names = [word for word in named if word not in STAGE_WORDS] or named or words
    return names[:MAX_SEARCH_WORDS]


def search_patterns(raw: Any) -> List[str]:
    """One ILIKE pattern per word of ``search_words(raw)``: a row matches when it holds all."""
    return [f"%{word}%" for word in search_words(raw)]


__all__ = ["FILLER_WORDS", "MAX_SEARCH_WORDS", "PAPER_KINDS", "STAGE_WORDS", "WORK_NOUNS", "search_patterns",
           "search_words"]
