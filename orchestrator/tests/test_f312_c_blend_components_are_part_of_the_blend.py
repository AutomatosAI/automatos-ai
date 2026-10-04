"""F312 (night 9) — both halves of the Harbour Blend are part of it, and reach its cafés.

TESTER's pre-night graph check found Brazil Cerrado and Sumatra Gayo linked only
to their importer, not to the Harbour Blend the importers document says they are
"both halves of", so night 9's "Brazil Cerrado is late: which cafés?" never
reached Kiln Bakehouse. The prompt now asks for one ``part_of`` edge per part of
a whole; this pins what the platform does with what the model writes:

* a whole-to-part phrasing ("contains", "includes") is turned round, so each
  coffee is ``part_of`` the blend, not the blend ``part_of`` the coffee;
* a customer edge written the wrong way round ("Harbour Blend buys Kiln
  Bakehouse") is turned round by the declared types;
* through the real merge and graph build, Brazil Cerrado reaches Kiln Bakehouse
  through the Harbour Blend.
"""
from __future__ import annotations

import networkx as nx
import pytest

from tests import helpers_graph_extraction as h

BLEND = h.node("harbour_blend", "Harbour Blend", "product")
BRAZIL = h.node("brazil_cerrado", "Brazil Cerrado", "product")
SUMATRA = h.node("sumatra_gayo", "Sumatra Gayo", "product")
MERIDIAN = h.node("meridian_bean_traders", "Meridian Bean Traders", "organization")
KILN = h.node("kiln_bakehouse", "Kiln Bakehouse", "organization")

IMPORTERS = [
    BLEND, BRAZIL, SUMATRA, MERIDIAN,
    h.edge("meridian_bean_traders", "brazil_cerrado", "supplies", "Supplies: Brazil Cerrado"),
    h.edge("meridian_bean_traders", "sumatra_gayo", "supplies", "Supplies: Sumatra Gayo"),
    h.edge("harbour_blend", "brazil_cerrado", "contains", "both halves of the Harbour Blend"),
    h.edge("harbour_blend", "sumatra_gayo", "includes", "both halves of the Harbour Blend"),
]
CAFE_NOTES = [
    BLEND, KILN,
    h.edge("harbour_blend", "kiln_bakehouse", "buys", "Kiln Bakehouse (Plymouth), Tuesday: Harbour Blend"),
]


@pytest.mark.asyncio
async def test_each_half_of_the_blend_is_part_of_the_blend(monkeypatch):
    graph = await h.extract(monkeypatch, IMPORTERS, "importers-and-green-buying.md")

    for coffee in ("brazil_cerrado", "sumatra_gayo"):
        kept = h.edge_between(graph, coffee, "harbour_blend")
        assert (kept["source"], kept["relation"], kept["target"]) == (coffee, "part_of", "harbour_blend")
        assert (kept["_src"], kept["_tgt"]) == (coffee, "harbour_blend")


@pytest.mark.asyncio
async def test_a_cafe_buys_the_blend_whichever_way_the_model_wrote_it(monkeypatch):
    graph = await h.extract(monkeypatch, CAFE_NOTES, "cafe-notes.md")

    kept = h.edge_between(graph, "kiln_bakehouse", "harbour_blend")
    assert (kept["source"], kept["relation"], kept["target"]) == ("kiln_bakehouse", "buys", "harbour_blend")


@pytest.mark.asyncio
async def test_brazil_cerrado_reaches_kiln_bakehouse_through_the_blend(monkeypatch):
    from graphify.build import build_from_json

    from modules.knowledge.graph_service import GraphifyService

    extractions = [
        await h.extract(monkeypatch, IMPORTERS, "importers-and-green-buying.md"),
        await h.extract(monkeypatch, CAFE_NOTES, "cafe-notes.md"),
    ]
    graph = build_from_json(GraphifyService._merge_extractions(extractions))

    assert nx.shortest_path(graph, "brazil_cerrado", "kiln_bakehouse") == [
        "brazil_cerrado", "harbour_blend", "kiln_bakehouse",
    ]
    blend_edge = graph.edges["brazil_cerrado", "harbour_blend"]
    assert (blend_edge["_src"], blend_edge["relation"], blend_edge["_tgt"]) == (
        "brazil_cerrado", "part_of", "harbour_blend",
    )
