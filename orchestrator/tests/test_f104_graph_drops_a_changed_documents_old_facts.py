"""F104 (night 3) — a document that changes takes its old facts out of the graph.

After the corrected Christmas sheet replaced the old one (F087), the Knowledge
Graph still held the old price beside the new one — the incremental build only
ever added — and the graph excerpt in agent prompts could show either. Now each
extracted node and edge names its document, and a changed document's earlier
facts are removed before its new extraction is merged.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import networkx as nx
import pytest

from modules.knowledge.graph_provenance import SOURCE_DOC_ATTR, prune_document_facts, stamp_source_document


def _graph():
    g = nx.Graph()
    g.add_node("christmas_box", label="Christmas Box", source_file="christmas-box-2026.md", source_doc_id=716)
    g.add_node("price_45", label="£45", source_file="christmas-box-2026.md", source_doc_id=716)
    g.add_node("the_salt_loft", label="The Salt Loft", source_file="wholesale-accounts.md", source_doc_id=885)
    g.add_edge("christmas_box", "price_45", relation="priced_at", source_doc_id=716)
    g.add_edge("christmas_box", "the_salt_loft", relation="stocked_by", source_doc_id=885)
    return g


def test_an_extraction_names_its_document_and_the_original_is_untouched():
    extraction = {"nodes": [{"id": "a"}], "edges": [{"source": "a", "target": "b"}], "hyperedges": []}
    stamped = stamp_source_document(extraction, 716)
    assert stamped[SOURCE_DOC_ATTR] == 716
    assert stamped["nodes"] == [{"id": "a", SOURCE_DOC_ATTR: 716}]
    assert stamped["edges"] == [{"source": "a", "target": "b", SOURCE_DOC_ATTR: 716}]
    assert extraction == {"nodes": [{"id": "a"}], "edges": [{"source": "a", "target": "b"}], "hyperedges": []}


def test_the_documents_edges_go_then_the_nodes_no_other_document_holds():
    g = _graph()
    assert prune_document_facts(g, 716) == (1, 1)
    assert "price_45" not in g                                   # only document 716 held it
    assert "christmas_box" in g                                  # document 885's edge still does
    assert g.has_edge("christmas_box", "the_salt_loft")


def test_facts_from_before_the_stamp_go_only_when_the_name_is_the_documents_own():
    g = nx.Graph()
    g.add_node("christmas_box", label="Christmas Box", source_file="christmas-box-2026.md")
    g.add_node("price_45", label="£45", source_file="christmas-box-2026.md")
    g.add_edge("christmas_box", "price_45", relation="priced_at", source_file="christmas-box-2026.md")
    assert prune_document_facts(g, 716) == (0, 0)                # no id on them and no name given
    assert prune_document_facts(g, 716, legacy_source_file="christmas-box-2026.md") == (1, 2)
    assert g.number_of_nodes() == 0


@pytest.mark.asyncio
async def test_a_replaced_documents_new_facts_take_the_place_of_its_old_ones():
    from modules.knowledge.graph_service import GraphifyService

    svc = GraphifyService()
    existing = _graph()
    new_version = {
        "nodes": [{"id": "christmas_box", "label": "Christmas Box"}, {"id": "price_48", "label": "£48"}],
        "edges": [{"source": "christmas_box", "target": "price_48", "relation": "priced_at"}],
        "hyperedges": [],
    }
    source = {"type": "document", "id": 716, "path": "christmas-box-2026.md", "text": "Christmas Box: £48",
              "team_access": [], "replaced": True}
    with patch.object(svc, "_collect_sources", AsyncMock(return_value=[source])), patch(
        "modules.knowledge.graph_extraction.extract_from_document", AsyncMock(return_value=new_version)
    ), patch("core.llm.create_llm_manager", return_value=MagicMock()), patch.object(
        svc, "_export_graph", AsyncMock()
    ), patch.object(svc, "_write_json", AsyncMock()), patch.object(
        svc, "_snapshot_and_diff", AsyncMock(return_value=None)
    ), patch.object(svc, "_write_build_report", AsyncMock()), patch.object(svc, "_prune_history", AsyncMock()):
        await svc._incremental_build("ws-1", existing, [{"type": "document", "id": 716}])
    assert "price_45" not in existing
    assert existing.has_edge("christmas_box", "price_48")
    assert existing.nodes["price_48"][SOURCE_DOC_ATTR] == 716
    assert existing.has_edge("christmas_box", "the_salt_loft")   # the other document's fact stays
