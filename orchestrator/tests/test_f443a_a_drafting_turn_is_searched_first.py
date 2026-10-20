"""F443 (a), P256-FIX-N17-11 (night 16, ask 318): a turn that writes from the documents is searched first.

"I need the November price letter to go out…" is an instruction, so the turn searched nothing
before the model's first call (F085 searches questions only). Auto's own platform_search_documents
calls read the October terms, and the letter quoted £23/kg, a 6 kg minimum and "I found no November
price change in the documents" while harbourline-wholesale-terms-2026-11.md said £26 and 8 kg.
Every turn that night that ran search_knowledge with November price words found the sheet.

A drafting turn (a customer draft, or prices, terms or rates near a letter, note, list or sheet;
a newsletter) is now searched first, the whole message, under the drafting header that carries
the absence rule. Other instructions stay unsearched, and what counts as a question is unchanged.
Driven through test_f085's harness: the real ``_retrieval_first`` with only the search faked.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from consumers.chatbot import knowledge_prefetch as kp
from tests.test_f085_retrieval_first import HIT, _chat, _turn

ASK_318 = ("I need the November price letter to go out to the cafes this week. Write it from the new wholesale "
           "terms: the price per kilo and the minimum order.")
NOVEMBER = {"filename": "harbourline-wholesale-terms-2026-11.md", "source": "harbourline-wholesale-terms-2026-11.md",
            "title": "harbourline-wholesale-terms-2026-11.md", "similarity": 0.71, "document_id": 731,
            "content": "From 1 November: £26/kg, minimum order 8 kg.", "excerpt": "From 1 November: £26/kg."}


def test_the_price_letter_is_searched_first_and_carries_the_absence_rule(monkeypatch):
    svc, searched = _chat(monkeypatch, results=(NOVEMBER,))
    frames, llm_messages, prefetched = _turn(svc, ASK_318)

    assert [call["tool_name"] for call in searched] == ["search_knowledge"]
    assert searched[0]["tool_args"]["query"] == ASK_318                     # the whole message, once
    assert '"tool-start"' in frames[0] and "searched automatically" in frames[1]
    assert prefetched == [("search_knowledge", searched[0]["tool_args"])]
    handed = llm_messages[-1]
    assert handed["role"] == "system" and handed["content"].startswith(kp.DRAFTING_HEADER)
    assert kp.ABSENCE_RULE in handed["content"] and "harbourline-wholesale-terms-2026-11.md" in handed["content"]


@pytest.mark.parametrize("message", [
    "Create an agent called OPS",
    "Please send the invoice to Declan",
    "Can you create an agent called OPS?",
])
def test_an_instruction_that_drafts_nothing_is_not_searched(monkeypatch, message):
    svc, searched = _chat(monkeypatch, results=(HIT,))
    frames, llm_messages, prefetched = _turn(svc, message)

    assert (frames, prefetched, searched, len(llm_messages)) == ([], [], [], 1)


@pytest.mark.parametrize("message, drafts", [
    ("Write the November price letter to the cafes, with the new prices.", True),     # night 16
    ("Write the November newsletter for the subscribers.", True),
    ("Now do the same for wholesale — a short line cafes would read.", True),
    (ASK_318, True),
    ("Update the rates on the wholesale sheet for Quay.", True),
    ("Cancel #1482", False),
    ("Create an agent called OPS", False),
    ("Please send the invoice to Declan", False),
    ("What are our wholesale terms?", False),                                         # a question: F085's own
    ("Move the deadline line on card #1490 to Friday.", False),
    ("Draft a reply to the customer on card #1490 about the new prices.", False),     # the board's (F263)
    ("", False),
])
def test_what_counts_as_a_drafting_turn(message, drafts):
    assert kp.drafts_from_the_documents(message) is drafts


def test_a_question_is_searched_as_before():
    assert kp.searched_first("What are our wholesale terms?") == {}
    assert kp.searched_first(ASK_318) == {"question_only": False, "drafting": True}
    assert kp.searched_first("Create an agent called OPS") is None


def test_with_a_database_the_drafting_header_sends_a_number_to_it(monkeypatch):
    from modules.context.sections import documents_inventory as inv

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: 12)
    monkeypatch.setattr(inv, "connected_databases", lambda db, ws: ["harbourline_shop"])

    async def _search(args):
        return {"raw_result": {"results": [NOVEMBER]}}

    got = asyncio.run(kp.prefetch(None, uuid4(), ASK_318, search=_search, enabled=True, limit=5, min_score=0.3,
                                  question_only=False, drafting=True))
    assert got.message["content"].startswith(f"{kp.DRAFTING_HEADER} {kp.PREFETCH_DATABASE_NOTE}\n\n")
