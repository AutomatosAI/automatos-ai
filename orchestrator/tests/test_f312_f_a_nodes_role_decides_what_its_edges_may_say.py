"""F312 (build 14) — a node's role in the document decides what its edges may say.

The build-14 rebuild stored every node as "concept", and two wrong edges stood:
"Meridian Bean Traders causes Harbour Blend" (from "If Meridian are late the
blend is the problem") and "Lantern Kitchen has_property Harbour Blend" (from
"12 kg a week": a café buying the blend). A declared type can't be relied on, so
a node's kind is read from its structural edges: the source of ``supplies`` is a
supplier, the source of ``buys`` a customer, the ends of ``part_of`` and the
target of ``supplies``/``buys`` products. Every node here is declared "concept".

Nothing is dropped: a refused edge keeps the link as ``related_to``.
"""
from __future__ import annotations

import pytest

from tests import helpers_graph_extraction as h


def _concept(node_id: str, label: str) -> dict:
    return h.node(node_id, label, "concept")


IMPORTERS = [
    _concept("meridian_bean_traders", "Meridian Bean Traders"),
    _concept("brazil_cerrado", "Brazil Cerrado"),
    _concept("sumatra_gayo", "Sumatra Gayo"),
    _concept("harbour_blend", "Harbour Blend"),
    _concept("payment_terms_45_days", "Payment terms: 45 days"),
    h.edge("meridian_bean_traders", "brazil_cerrado", "supplies", "Supplies: Brazil Cerrado"),
    h.edge("meridian_bean_traders", "sumatra_gayo", "supplies", "Supplies: Sumatra Gayo"),
    h.edge("brazil_cerrado", "harbour_blend", "part_of", "both halves of the Harbour Blend"),
    h.edge("meridian_bean_traders", "harbour_blend", "causes", "If Meridian are late the blend is the problem"),
    h.edge("meridian_bean_traders", "payment_terms_45_days", "has_property", "Payment terms: 45 days"),
]
CAFE_NOTES = [
    _concept("crane_kitchen", "Crane Kitchen"),
    _concept("lantern_kitchen", "Lantern Kitchen"),
    _concept("harbour_blend", "Harbour Blend"),
    h.edge("crane_kitchen", "harbour_blend", "buys", "Thursday: Harbour Blend"),
    h.edge("lantern_kitchen", "harbour_blend", "has_property", "about 12 kg a week"),
]


@pytest.mark.asyncio
async def test_a_supplier_does_not_cause_a_product(monkeypatch):
    graph = await h.extract(monkeypatch, IMPORTERS, "importers-and-green-buying.md")

    kept = h.edge_between(graph, "meridian_bean_traders", "harbour_blend")
    assert kept["relation"] == "related_to"
    assert kept["relation_refused"] == "causes"
    assert len(graph["edges"]) == 5   # nothing dropped


@pytest.mark.asyncio
async def test_a_value_of_a_supplier_stays_a_property(monkeypatch):
    graph = await h.extract(monkeypatch, IMPORTERS, "importers-and-green-buying.md")

    kept = h.edge_between(graph, "meridian_bean_traders", "payment_terms_45_days")
    assert kept["relation"] == "has_property"
    assert "relation_refused" not in kept


@pytest.mark.asyncio
async def test_a_cafe_with_a_weekly_quantity_of_the_blend_buys_it(monkeypatch):
    graph = await h.extract(monkeypatch, CAFE_NOTES, "cafe-notes.md")

    kept = h.edge_between(graph, "lantern_kitchen", "harbour_blend")
    assert (kept["source"], kept["relation"], kept["target"]) == ("lantern_kitchen", "buys", "harbour_blend")
    assert kept["relation_refused"] == "has_property"


@pytest.mark.asyncio
async def test_a_product_is_never_a_property_without_a_purchase(monkeypatch):
    graph = await h.extract(monkeypatch, [
        *IMPORTERS, h.edge("harbour_blend", "sumatra_gayo", "has_property", "the other half"),
    ], "importers-and-green-buying.md")

    kept = h.edge_between(graph, "harbour_blend", "sumatra_gayo")
    assert kept["relation"] == "related_to"
    assert kept["relation_refused"] == "has_property"


@pytest.mark.asyncio
async def test_each_node_carries_the_kind_its_role_gives_it(monkeypatch):
    graph = await h.extract(monkeypatch, IMPORTERS, "importers-and-green-buying.md")

    kinds = {n["id"]: n.get("kind") for n in graph["nodes"]}
    assert kinds["meridian_bean_traders"] == "supplier"
    assert kinds["brazil_cerrado"] == kinds["harbour_blend"] == "product"
    assert kinds["payment_terms_45_days"] is None
