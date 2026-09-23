"""F104 (night 3): a document that changes takes its old facts out of the graph.

The incremental build only ever ADDED. When a document was replaced (F087) or
re-ingested, what its previous text contributed stayed in the Knowledge Graph
beside the new facts, and the graph excerpt in agent prompts could still carry
the old price. Every node and edge extracted from a document now names that
document's id; before a changed document's new extraction is merged, the edges
it sourced come out, then its nodes left with no edge.

Provenance is best-effort. The graph keeps one edge per node pair and one set
of attributes per node, so where two documents extract the same edge or node,
the one merged last owns it. Elements from graphs built before the stamp carry
no document id; for a replaced document — whose name is its own — they are
matched on the file name the extraction echoed back (``source_file``).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import networkx as nx

SOURCE_DOC_ATTR = "source_doc_id"


def stamp_source_document(extraction: Dict[str, Any], doc_id: Any) -> Dict[str, Any]:
    """A copy of ``extraction`` whose nodes and edges — and the extraction
    itself — name the document they came from."""
    return {
        **extraction,
        SOURCE_DOC_ATTR: doc_id,
        "nodes": [{**node, SOURCE_DOC_ATTR: doc_id} for node in extraction.get("nodes", [])],
        "edges": [{**edge, SOURCE_DOC_ATTR: doc_id} for edge in extraction.get("edges", [])],
    }


def prune_document_facts(graph: nx.Graph, doc_id: Any, *,
                         legacy_source_file: Optional[str] = None) -> Tuple[int, int]:
    """Remove from ``graph`` — in place, as the incremental build merges into
    it — the edges document ``doc_id`` sourced, then its nodes left with no
    edge. Returns (edges removed, nodes removed)."""
    def from_document(attrs: Dict[str, Any]) -> bool:
        if attrs.get(SOURCE_DOC_ATTR) is not None:
            return str(attrs[SOURCE_DOC_ATTR]) == str(doc_id)
        return legacy_source_file is not None and attrs.get("source_file") == legacy_source_file

    edges = [(u, v) for u, v, attrs in graph.edges(data=True) if from_document(attrs)]
    graph.remove_edges_from(edges)
    nodes = [n for n, attrs in graph.nodes(data=True) if from_document(attrs) and graph.degree(n) == 0]
    graph.remove_nodes_from(nodes)
    return len(edges), len(nodes)
