"""F312 (night 9) — a person is never produced, and an assignment points from them.

TESTER's pre-night graph check found "October 2026 Box produces Priya", from
"Card copy for the box is with Priya". The relation vocabulary had no owner
relation and nothing checked an extracted edge against what its endpoints are,
so the model's nearest-wrong guess went straight into the graph.

Now an edge whose relation the endpoints' declared types rule out is refused:
the link stays as ``related_to`` (so traversal still reaches Priya from the box)
and the refused name is kept on the edge. An assignment phrased the other way
round ("is with Priya") points from the person.
"""
from __future__ import annotations

import pytest

from tests import helpers_graph_extraction as h

BOX = h.node("october_2026_box", "October 2026 Box", "product")
PRIYA = h.node("priya", "Priya", "person")
CARD = h.node("box_card", "Box Card", "product")


@pytest.mark.asyncio
async def test_the_box_does_not_produce_priya_but_stays_linked_to_her(monkeypatch):
    graph = await h.extract(monkeypatch, [
        BOX, PRIYA,
        h.edge("october_2026_box", "priya", "produces", "Card copy for the box is with Priya"),
    ], "club-box-october-2026.md")

    kept = h.edge_between(graph, "october_2026_box", "priya")
    assert kept["relation"] == "related_to"
    assert kept["relation_refused"] == "produces"
    assert kept["relation_label"] == "Card copy for the box is with Priya"


@pytest.mark.asyncio
async def test_a_thing_assigned_to_a_person_reads_from_the_person(monkeypatch):
    graph = await h.extract(monkeypatch, [
        CARD, PRIYA,
        h.edge("box_card", "priya", "assigned to", "Card copy for the box is with Priya"),
    ], "club-box-october-2026.md")

    kept = h.edge_between(graph, "box_card", "priya")
    assert (kept["source"], kept["relation"], kept["target"]) == ("priya", "responsible_for", "box_card")
    assert "relation_refused" not in kept


@pytest.mark.asyncio
async def test_a_person_who_produces_something_keeps_the_relation(monkeypatch):
    graph = await h.extract(monkeypatch, [
        PRIYA, CARD,
        h.edge("priya", "box_card", "produces", "Priya writes the box cards"),
    ], "team-and-week.md")

    kept = h.edge_between(graph, "priya", "box_card")
    assert (kept["source"], kept["relation"], kept["target"]) == ("priya", "produces", "box_card")
    assert "relation_refused" not in kept
