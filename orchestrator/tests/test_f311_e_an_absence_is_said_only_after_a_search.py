"""F311 (night 9, L1/L93) — Auto says a document lacks something only after searching for it.

"What do we charge for delivery on 10 kg, and is it ever free?" got "the document doesn't
specify any delivery charges" in two fresh chats, from the passages retrieval-first found for
the whole message; the wholesale terms say "carriage", and the answer came only after "are you
sure?". The passages handed to Auto now say: no absence on these passages alone, search first
in the thing's own words and synonyms. Driven through the real prefetch with only the search,
the document count and the database list faked, as test_f078 does.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

from consumers.chatbot import knowledge_prefetch as kp

DELIVERY = "A café wants 10 kg next week. What do we charge for delivery, and is it ever free?"
MINIMUM = {"filename": "wholesale-terms-2026.md", "similarity": 0.71, "content": "Minimum order: 6 kg per delivery."}


def _handed_over(monkeypatch, message):
    from modules.context.sections import documents_inventory as inv

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: 12)
    monkeypatch.setattr(inv, "connected_databases", lambda db, ws: [])

    async def _search(args):
        return {"raw_result": {"results": [MINIMUM]}}

    got = asyncio.run(kp.prefetch(None, uuid4(), message, search=_search, enabled=True, limit=5, min_score=0.3))
    return got.message["content"]


def test_the_passages_say_an_absence_needs_a_search_of_autos_own(monkeypatch):
    content = _handed_over(monkeypatch, DELIVERY)
    assert "Never say a document does not give something on these passages alone" in content
    assert "call search_knowledge yourself with the specific words for it" in content
    assert "carriage or postage for delivery charges" in content
    assert "wholesale-terms-2026.md" in content


def test_each_question_of_several_carries_the_rule_too(monkeypatch):
    content = _handed_over(monkeypatch, "What is the minimum order?\nWhat do we charge for delivery?")
    assert content.startswith(kp.MULTI_HEADER)
    assert kp.ABSENCE_RULE in kp.MULTI_HEADER and kp.ABSENCE_RULE in kp.PREFETCH_HEADER
