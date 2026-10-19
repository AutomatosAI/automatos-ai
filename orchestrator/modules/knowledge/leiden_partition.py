"""Leiden communities through graspologic-native, for graphify's clustering (PRD-262 US-027).

``graphify.cluster.cluster`` (graphifyy 0.8.31) splits the Knowledge Graph into
communities with two partition calls: one on the graph (cluster.py:133) and one on each
community it re-splits (cluster.py:198). Both go through graphify's ``_partition``,
which runs ``graspologic.partition.leiden`` when graspologic imports and networkx's
Louvain when it does not. graspologic imports all fifteen of its subpackages, so it
brought about 585 MB (numba, llvmlite, POT, gensim, statsmodels, matplotlib,
scikit-learn, pandas) into Railway's image for a function that forwards to
graspologic-native.

``partition`` is that function as ``_partition`` called it, with graspologic-native
called directly:

- graphify's stable graph: nodes sorted by ``str``, edges by their ends and attributes,
  so the edge list reaches Leiden in the same order;
- graspologic 3.3.0's ``leiden`` on a networkx graph: each edge's ``weight``, 1.0 where
  an edge has none; node ids passed as ``str``, two nodes with the same ``str`` refused;
  randomness 0.001, ``extra_forced_iterations`` 0 (so one iteration) and modularity;
- ``_partition``'s random_seed 42 and trials 1, and the caller's resolution.

Importing this module installs ``partition`` as ``graphify.cluster._partition``, which
``cluster`` looks up at each call, so both calls reach it while graphify's isolate
handling, splitting and numbering stay graphify's own. graphifyy is pinned exactly in
requirements.in, and the install refuses a graphify whose ``_partition`` takes other
parameters. tests/test_prd262_leiden_adapter.py holds the partitions to the ones
graspologic 3.3.0 recorded on fixture graphs (test.yml's reference-goldens job
re-records them).
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Dict, Hashable, Iterable, List, Tuple

import graspologic_native
import networkx as nx
from graphify import cluster as graphify_cluster

# graphify.cluster._partition passes these to graspologic.partition.leiden.
RANDOM_SEED = 42
TRIALS = 1
# graspologic 3.3.0's leiden() defaults, which _partition leaves alone.
RANDOMNESS = 0.001
ITERATIONS = 1  # extra_forced_iterations (0) + 1
USE_MODULARITY = True
WEIGHT_ATTRIBUTE = "weight"
WEIGHT_DEFAULT = 1.0
# How graphify's cluster() calls _partition: the graph, then resolution.
GRAPHIFY_PARAMETERS = ("G", "resolution")

NativeEdge = Tuple[str, str, float]


def _edge_key(row: Tuple[Any, Any, Dict[str, Any]]) -> Tuple[str, str, str]:
    source, target, attrs = row
    return str(source), str(target), json.dumps(attrs, sort_keys=True, ensure_ascii=False, default=str)


def stable_graph(graph: nx.Graph) -> nx.Graph:
    """``graph`` rebuilt as graphify's _partition rebuilds it: nodes by str, then edges in order."""
    stable = nx.Graph()
    stable.add_nodes_from(sorted(graph.nodes(), key=str))
    for source, target, attrs in sorted(graph.edges(data=True), key=_edge_key):
        stable.add_edge(source, target, **attrs)
    return stable


def native_edges(graph: nx.Graph) -> List[NativeEdge]:
    """graspologic's edge list for a networkx graph: str ends, float weight, 1.0 by default."""
    return [(str(source), str(target), float(weight))
            for source, target, weight in graph.edges(data=WEIGHT_ATTRIBUTE, default=WEIGHT_DEFAULT)]


def node_labels(graph: nx.Graph) -> Dict[str, Hashable]:
    """Each node with an edge, by its str; two such nodes with one str are refused, as graspologic does."""
    labels: Dict[str, Hashable] = {}
    for node in _edge_ends(graph):
        if labels.setdefault(str(node), node) != node:
            raise ValueError("str(value) results in a collision between distinct values")
    return labels


def _edge_ends(graph: nx.Graph) -> Iterable[Hashable]:
    for source, target in graph.edges():
        yield source
        yield target


def partition(graph: nx.Graph, resolution: float = 1.0) -> Dict[Hashable, int]:
    """Leiden communities of ``graph``, {node: community}; a node with no edge is left out.

    graphify's ``_partition`` with graspologic installed, over graspologic-native.
    """
    if resolution <= 0:
        raise ValueError("resolution must be a positive float")
    stable = stable_graph(graph)
    labels = node_labels(stable)
    _quality, communities = graspologic_native.leiden(
        edges=native_edges(stable),
        starting_communities=None,
        resolution=resolution,
        randomness=RANDOMNESS,
        iterations=ITERATIONS,
        use_modularity=USE_MODULARITY,
        seed=RANDOM_SEED,
        trials=TRIALS,
    )
    return {labels[label]: community for label, community in communities.items()}


def install() -> None:
    """Serve graphify.cluster's partition calls from ``partition``. Idempotent.

    Raises ImportError when graphify's ``_partition`` is missing or takes other
    parameters: a graphifyy bump that changed how cluster() calls it needs this adapter
    checked first.
    """
    current = getattr(graphify_cluster, "_partition", None)
    if current is partition:
        return
    found = tuple(inspect.signature(current).parameters) if callable(current) else ()
    if found != GRAPHIFY_PARAMETERS:
        raise ImportError(f"graphify.cluster._partition takes {found}, not {GRAPHIFY_PARAMETERS}: "
                          "check modules/knowledge/leiden_partition.py against this graphifyy")
    graphify_cluster._partition = partition


install()
# graphify's cluster(), its partition calls served by ``partition`` from here on.
cluster = graphify_cluster.cluster
