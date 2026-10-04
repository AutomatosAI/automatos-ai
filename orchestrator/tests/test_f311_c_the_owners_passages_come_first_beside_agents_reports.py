"""F311 (night 9) — beside agents' reports, the owner's passages are handed over, first.

"Are you sure? I'm fairly certain the wholesale terms say something about carriage."
(chat 77116484, 13:03): the five passages handed over were five agents' reports
(0.70 to 0.73) and none of wholesale-terms-2026.md. Reports are not the owner's facts
(F269): the night's build dropped them from Auto's passages and none was left, and the
first answer had said the terms name no delivery charge (ledger L93). Now, when a report
is among the candidates, up to three of the passages are the owner's, one per document,
and they come first: the search tool shows the model its first four.
"""
from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from config import config
from modules.rag.service import RAGResult, RAGService
from modules.search import ContextItem

QUESTION = "Are you sure? I'm fairly certain the wholesale terms say something about carriage."
NOW = datetime(2026, 10, 4)
CARRIAGE = ("## Delivery\nOur van does fixed days:\n- Bristol: Thursday and Friday\n- Bath: Monday and Wednesday\n"
            "- Taunton and Cheltenham: Monday\n- Exeter: Tuesday\n- Plymouth: Tuesday and Wednesday\n"
            "- Bournemouth: Wednesday (every other week)\n\nCarriage is charged per drop:\n"
            "- under 12 kg: **£8.50**\n- 12 kg and over: **£5.00**")
PRICES = ("## Prices\n- Standard wholesale price: **£22.00 per kg** (Harbour Blend and the core range).\n"
          "- Cafés on our single origins pay **£24.00 per kg**. Prices are per kg of roasted coffee.")
TEAM = ("# Team and the week\n- **Gerard**: owner. Everything else.\n- **Tom Reed**: packing and dispatch, "
        "Monday, Wednesday and Friday. Monday is subscription post day.")
REPORTS = {1529: 0.734, 1552: 0.716, 1530: 0.698, 1535: 0.69, 1551: 0.68}
OWNERS = {1526, 1525}


def _report(doc_id):
    words = " ".join(f"word{n}" for n in range(60))
    return f"# Task report {doc_id}\nAn agent's answer about delivery for a café order. {words}"


def _candidate(doc_id, score, content, chunk=0):
    return {"id": f"doc_{doc_id}_chunk_{chunk}", "content": content, "expanded_content": content,
            "source_file": "wholesale-terms-2026.md" if doc_id == 1526 else f"doc-{doc_id}.md",
            "document_id": str(doc_id), "similarity": score, "metadata": {"chunk_index": chunk}}


def _candidates():
    reports = [_candidate(doc_id, score, _report(doc_id)) for doc_id, score in REPORTS.items()]
    return reports + [_candidate(1526, 0.64, CARRIAGE, 4), _candidate(1526, 0.62, PRICES, 1),
                      _candidate(1525, 0.58, TEAM)]


def _rag():
    rag = RAGService.__new__(RAGService)
    rag._ContextItem = ContextItem
    return rag


def _select(candidates):
    return asyncio.run(_rag()._optimize_with_context_optimizer(QUESTION, candidates, 5, 2000, 0.3))


@pytest.fixture(autouse=True)
def facts(monkeypatch):
    monkeypatch.setattr(config, "RAG_OWNER_PASSAGES_RESERVED", 3, raising=False)
    known = {**{str(d): (True, NOW) for d in REPORTS}, **{str(d): (False, NOW) for d in OWNERS}}
    monkeypatch.setattr(RAGService, "_document_ranking_facts",
                        classmethod(lambda cls, ids: {i: known[i] for i in ids if i in known}))


def test_the_carriage_section_is_handed_over_first_beside_five_reports():
    result = _select(_candidates())

    assert isinstance(result, RAGResult) and len(result.chunks) == 5
    assert "Carriage is charged per drop" in result.chunks[0]["content"]
    assert [c["document_id"] for c in result.chunks[:2]] == ["1526", "1525"]
    assert {c["document_id"] for c in result.chunks[2:]} <= {str(d) for d in REPORTS}
    assert result.sources_map[0]["source_file"] == "wholesale-terms-2026.md"
    assert "[1] (source: wholesale-terms-2026.md)\n## Delivery" in result.formatted_context


def test_dropping_the_reports_still_leaves_the_carriage_charges():
    """What the night's build did to Auto's retrieval-first passages (F269)."""
    result = _select(_candidates())
    owners_only = [c["content"] for c in result.chunks if c["document_id"] not in {str(d) for d in REPORTS}]

    assert any("- under 12 kg: **£8.50**" in content for content in owners_only)


def test_one_passage_per_owners_document_is_reserved():
    result = _select(_candidates())

    assert sum(1 for c in result.chunks if c["document_id"] == "1526") == 1


def test_without_a_report_the_selection_is_not_touched():
    owners_only = [_candidate(1526, 0.64, CARRIAGE, 4), _candidate(1526, 0.62, PRICES, 1),
                   _candidate(1525, 0.58, TEAM)]
    result = _select(owners_only)

    assert [c["content"] for c in result.chunks][0] == CARRIAGE


def test_none_reserved_leaves_the_reports_as_they_were_chosen(monkeypatch):
    monkeypatch.setattr(config, "RAG_OWNER_PASSAGES_RESERVED", 0, raising=False)
    result = _select(_candidates())

    assert {c["document_id"] for c in result.chunks} == {str(d) for d in REPORTS}
