"""
Knowledge Graph direction: who does what to whom, on an undirected graph
==========================================================================

F312 (build 14): the workspace graph is an undirected ``nx.Graph``. Its
``source``/``target`` (and the ``u, v`` a traversal yields) are only networkx's
storage or walk order; the direction extraction gave each edge is in ``_src`` /
``_tgt``. Readers that used storage order showed "Harbour Blend buys Crane
Kitchen" for "Crane Kitchen buys Harbour Blend", and the graph tools listed each
neighbour as a "target" with no direction at all.

Every reader that hands an edge to an agent or the UI now orients it here: from
``_src``/``_tgt`` when they name the edge's two ends, else (a deterministic
mapper's edge, or a graph built before the stamp) the order it was given in.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

DIRECTION_OUTGOING = "outgoing"
DIRECTION_INCOMING = "incoming"
_DEFAULT_RELATION = "related_to"
_DEFAULT_SCORE = 1.0


def edge_ends(u: Any, v: Any, data: Mapping[str, Any]) -> tuple[str, str]:
    """``(source, target)`` of the edge ``u``-``v`` as extraction gave it."""
    src, tgt = data.get("_src"), data.get("_tgt")
    if src is not None and tgt is not None and {str(src), str(tgt)} == {str(u), str(v)}:
        return str(src), str(tgt)
    return str(u), str(v)


def oriented_pairs(graph: Any, pairs: Iterable[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Traversal pairs ``(u, v)`` reordered to read source -> target.

    ``u``/``v`` keep their original node-id objects so lookups into ``graph``
    still work; a pair with no edge between them is left as it is.
    """
    oriented: list[tuple[Any, Any]] = []
    for u, v in pairs:
        if graph.has_edge(u, v) and edge_ends(u, v, graph.edges[u, v]) == (str(v), str(u)):
            oriented.append((v, u))
        else:
            oriented.append((u, v))
    return oriented


def _label(graph: Any, node_id: str) -> str:
    attrs = graph.nodes[node_id] if node_id in graph else {}
    return str(attrs.get("label", node_id))


def describe_edge(graph: Any, u: Any, v: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    """The edge as a statement an agent can read: source, relation, target."""
    src, tgt = edge_ends(u, v, data)
    relation = data.get("relation", _DEFAULT_RELATION)
    src_label, tgt_label = _label(graph, src), _label(graph, tgt)
    return {
        "source": src,
        "source_label": src_label,
        "relation": relation,
        "relation_label": data.get("relation_label") or relation,
        "target": tgt,
        "target_label": tgt_label,
        "statement": f"{src_label} {relation} {tgt_label}",
    }


def direction_from(node_id: Any, u: Any, v: Any, data: Mapping[str, Any]) -> str:
    """Whether the edge leaves ``node_id`` (it is the source) or arrives at it."""
    return DIRECTION_OUTGOING if edge_ends(u, v, data)[0] == str(node_id) else DIRECTION_INCOMING


def edge_payload(u: Any, v: Any, data: Mapping[str, Any]) -> dict[str, Any]:
    """A single link in the renderer's edge shape, pointing the way it reads."""
    score = data.get("confidence_score")
    if score is None:
        score = data.get("confidence", data.get("weight", _DEFAULT_SCORE))
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = _DEFAULT_SCORE
    src, tgt = edge_ends(u, v, data)
    relation = data.get("relation", _DEFAULT_RELATION)
    return {
        "source": src,
        "target": tgt,
        "relation": relation,
        "relation_label": data.get("relation_label") or relation,
        "confidence": data.get("confidence", score),
        "confidence_score": score,
    }
