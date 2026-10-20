"""The graphs behind PRD-262 US-027's reference partitions.

Until US-027, Railway's API clustered the Knowledge Graph with graphify 0.8.31's
``cluster``, whose ``_partition`` called ``graspologic.partition.leiden`` (graspologic
3.3.0, over graspologic-native 1.3.1) with a fixed seed, on the graph and again on any
community too large to keep. US-027 serves both calls from
``modules/knowledge/leiden_partition.py``, which calls ``graspologic_native.leiden``
itself. Each graph here has one partition that is clearly the best by modularity, so
its golden can be checked by hand, and between them they reach what the adapter must
get right:

- ``ring_of_cliques``: four 6-cliques in a ring, one edge between neighbours, the
  bridges carrying a ``relation`` but no weight. The cliques (modularity 0.6875) beat
  any merge or split.
- ``resolution_limit``: two 6-cliques joined by one edge, beside a 4-clique whose edges
  weigh 200. In the whole graph the heavy clique makes the pair one community (merging
  the two cliques gains 1/m - d*d/(2m*m) > 0, with m = 1,231 and d = 31): 12 of 16 nodes,
  more than the quarter graphify keeps, so ``cluster`` runs Leiden again on the pair
  alone (graphify/cluster.py:198), where it falls apart into its two cliques. The
  pair's edges carry no weight attribute, so graspologic's default of 1.0 counts.
- ``weighted_bridge``: two triangles joined by an edge of weight 20. Unweighted, the
  triangles would be the communities; weighted, the bridge's two ends pair up, and so do
  the other two corners of each triangle. Only a partition that reads ``weight`` finds
  that.
- ``triangles_and_isolates``: two triangles joined by an edge, and two nodes with no
  edge. graspologic leaves isolates out of a partition, and ``cluster`` gives each its
  own community.

Two readers build them:

- ``scripts/ci/reference/leiden_partitions.py`` records what graphify made of each with
  graspologic installed, in ``leiden_partitions.json`` here;
- ``tests/test_prd262_leiden_adapter.py`` checks that the adapter reproduces it.

networkx only: the reference script loads this file in a venv that has none of the app.
"""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence

import networkx as nx

CLIQUE_SIZE = 6
HEAVY_WEIGHT = 200.0  # resolution_limit's 4-clique
BRIDGE_WEIGHT = 20.0  # weighted_bridge's bridge


def _names(prefix: str, count: int = CLIQUE_SIZE) -> List[str]:
    return [f"{prefix}{index}" for index in range(count)]


def _clique(graph: nx.Graph, nodes: Sequence[str], weight: Optional[float] = None) -> None:
    """Every pair of ``nodes`` joined, with ``weight`` when one is given."""
    attrs = {} if weight is None else {"weight": weight}
    for source, target in combinations(nodes, 2):
        graph.add_edge(source, target, **attrs)


def ring_of_cliques() -> nx.Graph:
    graph = nx.Graph()
    cliques = [_names(prefix) for prefix in "abcd"]
    for nodes in cliques:
        _clique(graph, nodes)
    for left, right in zip(cliques, cliques[1:] + cliques[:1]):
        graph.add_edge(left[-1], right[0], relation="links_to")
    return graph


def resolution_limit() -> nx.Graph:
    graph = nx.Graph()
    _clique(graph, _names("h", 4), weight=HEAVY_WEIGHT)
    left, right = _names("p"), _names("q")
    _clique(graph, left)
    _clique(graph, right)
    graph.add_edge(left[-1], right[0])
    return graph


def weighted_bridge() -> nx.Graph:
    graph = nx.Graph()
    _clique(graph, ["alpha", "beta", "gamma"], weight=1.0)
    _clique(graph, ["delta", "epsilon", "zeta"], weight=1.0)
    graph.add_edge("gamma", "delta", weight=BRIDGE_WEIGHT)
    return graph


def triangles_and_isolates() -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(["orphan", "loner"])
    _clique(graph, ["ant", "bee", "cat"])
    _clique(graph, ["dog", "eel", "fox"])
    graph.add_edge("cat", "dog")
    return graph


GRAPHS: Dict[str, Callable[[], nx.Graph]] = {
    "ring_of_cliques": ring_of_cliques,
    "resolution_limit": resolution_limit,
    "weighted_bridge": weighted_bridge,
    "triangles_and_isolates": triangles_and_isolates,
}
