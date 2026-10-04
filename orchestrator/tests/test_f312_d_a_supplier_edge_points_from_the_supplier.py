"""F312 (night 9) — "Supplies:" is a supplier relation, read from the supplier.

The night-9 graph filed all eight "Supplies:" lines of the importers document as
``produces`` — the vocabulary had no supplier relation — and four of them pointed
from the coffee to the importer ("Guji Shakiso produces Tidewater Importers").

Now ``supplies`` is in the vocabulary, a "supplied by" phrasing is turned round,
an edge written from the coffee to its importer is turned round by the declared
types, and "a coffee produces its importer" is refused.
"""
from __future__ import annotations

import pytest

from tests import helpers_graph_extraction as h

GUJI = h.node("guji_shakiso", "Guji Shakiso", "product")
TIDEWATER = h.node("tidewater_importers", "Tidewater Importers", "organization")
DOC = "importers-and-green-buying.md"


def _triple(kept):
    return kept["source"], kept["relation"], kept["target"]


@pytest.mark.asyncio
async def test_supplies_is_a_relation_of_its_own(monkeypatch):
    graph = await h.extract(monkeypatch, [
        GUJI, TIDEWATER, h.edge("tidewater_importers", "guji_shakiso", "supplies", "Supplies: Guji Shakiso"),
    ], DOC)

    assert _triple(h.edge_between(graph, "guji_shakiso", "tidewater_importers")) == (
        "tidewater_importers", "supplies", "guji_shakiso",
    )


@pytest.mark.asyncio
async def test_a_coffee_written_as_supplying_its_importer_is_turned_round(monkeypatch):
    graph = await h.extract(monkeypatch, [
        GUJI, TIDEWATER, h.edge("guji_shakiso", "tidewater_importers", "supplies", "Supplies: Guji Shakiso"),
    ], DOC)

    kept = h.edge_between(graph, "guji_shakiso", "tidewater_importers")
    assert _triple(kept) == ("tidewater_importers", "supplies", "guji_shakiso")
    assert kept["direction_repaired"] is True


@pytest.mark.asyncio
async def test_supplied_by_reads_from_the_supplier(monkeypatch):
    graph = await h.extract(monkeypatch, [
        GUJI, TIDEWATER, h.edge("guji_shakiso", "tidewater_importers", "supplied by", "Supplies: Guji Shakiso"),
    ], DOC)

    assert _triple(h.edge_between(graph, "guji_shakiso", "tidewater_importers")) == (
        "tidewater_importers", "supplies", "guji_shakiso",
    )


@pytest.mark.asyncio
async def test_a_coffee_never_produces_its_importer(monkeypatch):
    graph = await h.extract(monkeypatch, [
        GUJI, TIDEWATER, h.edge("guji_shakiso", "tidewater_importers", "produces", "Supplies"),
    ], DOC)

    kept = h.edge_between(graph, "guji_shakiso", "tidewater_importers")
    assert kept["relation"] == "related_to"
    assert kept["relation_refused"] == "produces"
