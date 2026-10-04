"""F312 (night 9) — "blocks", "causes", "triggers" need the document to say so.

TESTER's pre-night graph check found "Guji blocks swaps", from "4 members have
asked to swap the Guji for decaf". The document says nothing blocks anything:
the model picked the nearest relation and the graph kept it as a claim.

Now a claim of effect is kept only when the edge's own wording (the document's
phrase) carries it; otherwise the link stays as ``related_to`` and the refused
name is kept on the edge. A phrase that does make the claim keeps it.
"""
from __future__ import annotations

import pytest

from tests import helpers_graph_extraction as h

GUJI = h.node("guji_shakiso", "Guji Shakiso", "product")
SWAPS = h.node("swaps", "Swaps", "process")
LATE_BOX = h.node("late_box", "Late Box", "concept")
RESEND = h.node("resend_policy", "Resend Policy", "rule")


@pytest.mark.asyncio
async def test_guji_does_not_block_swaps(monkeypatch):
    graph = await h.extract(monkeypatch, [
        GUJI, SWAPS, h.edge("guji_shakiso", "swaps", "blocks", "swap the Guji"),
    ], "club-box-october-2026.md")

    kept = h.edge_between(graph, "guji_shakiso", "swaps")
    assert kept["relation"] == "related_to"
    assert kept["relation_refused"] == "blocks"


@pytest.mark.asyncio
async def test_a_trigger_the_document_states_is_kept(monkeypatch):
    graph = await h.extract(monkeypatch, [
        LATE_BOX, RESEND,
        h.edge("late_box", "resend_policy", "triggers",
               "If it hasn't arrived by Thursday we send another one free"),
    ], "subscriber-faq.md")

    kept = h.edge_between(graph, "late_box", "resend_policy")
    assert kept["relation"] == "triggers"
    assert "relation_refused" not in kept


@pytest.mark.asyncio
async def test_a_block_the_document_states_is_kept(monkeypatch):
    graph = await h.extract(monkeypatch, [
        LATE_BOX, SWAPS,
        h.edge("late_box", "swaps", "blocks", "no swaps once the box has posted"),
    ], "subscriber-faq.md")

    assert h.edge_between(graph, "late_box", "swaps")["relation"] == "blocks"
