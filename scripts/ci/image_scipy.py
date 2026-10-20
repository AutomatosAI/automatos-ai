"""PRD-262 row #17: the API image keeps the scipy the code reaches, and it runs.

orchestrator/Dockerfile's pybuild deletes the scipy subpackages no code path loads
(``PRUNED``) and fails the build when one it keeps (``KEPT``) is missing. CI runs this
inside the built API image, with the image's own Python, as the image's user:

    docker run --rm -i --entrypoint python IMAGE - < scripts/ci/image_scipy.py

It checks:

- every ``PRUNED`` subpackage is gone and every ``KEPT`` one imports (none of them imports
  a pruned one when it loads);
- every path by which the code reaches scipy runs, through the importers' own entry
  points: datasketch (imported whole, as graphify's dedup imports it, and MinHashLSH,
  which integrates with scipy.integrate.quad), graphify's dedup of near-identical
  entities, networkx's pagerank (scipy.sparse) through the code graph's PageRankRanker,
  the Knowledge Graph's chain as modules/knowledge/graph_service.py runs it (build,
  Leiden through graspologic-native, cohesion, god nodes, surprising connections), and
  core/math/probability_theory.py's Student's t (scipy.special.stdtrit), whose value it
  compares with the tables';
- none of it loaded a pruned subpackage, or graphify.extract, the one graphify module
  that imports the tree-sitter grammars pybuild leaves out (PRD-262 row #11).

The app's modules are imported from /app with the throwaway database settings
scripts/ci/image_site_packages.py imports ``main`` with (nothing connects on import;
orchestrator/tests/test_prd262_scipy_prune.py holds the two equal). Standard library apart
from the image's own packages. Exit 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from typing import Callable, List, Tuple

# orchestrator/tests/test_prd262_scipy_prune.py holds both lists equal to the Dockerfile's.
PRUNED = ("cluster", "datasets", "differentiate", "fftpack", "interpolate", "io", "misc", "ndimage", "odr",
          "signal", "stats")
KEPT = ("_lib", "constants", "integrate", "linalg", "optimize", "sparse", "spatial", "special")
NEVER_LOADED = ("graphify.extract",)
APP_DIR = "/app"
APP_IMPORT_ENV = {
    "POSTGRES_USER": "probe",
    "POSTGRES_PASSWORD": "probe",
    "POSTGRES_HOST": "127.0.0.1",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "probe",
    "AUTH_EDITION": "local",
    "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000000c1",
}
# Student's t, 97.5 % quantile with 4 degrees of freedom: 2.776 in every table.
T_975_DF4 = 2.7764451051977987
SAMPLE = (10.0, 12.0, 9.0, 11.0, 13.0)
TOLERANCE = 1e-9
# Two cliques joined by one edge: two communities whatever the seed.
CLIQUE_A, CLIQUE_B = ("alpha", "bravo", "charlie", "delta"), ("echo", "foxtrot", "golf", "hotel")


def subpackage_problems() -> List[str]:
    """A pruned subpackage still installed, or a kept one that does not import."""
    problems = [f"scipy.{name} is still installed" for name in PRUNED if importlib.util.find_spec(f"scipy.{name}")]
    for name in KEPT:
        try:
            importlib.import_module(f"scipy.{name}")
        except ImportError as error:
            problems.append(f"scipy.{name} does not import: {error}")
    return problems


def datasketch_problems() -> List[str]:
    """graphify's dedup imports datasketch whole; MinHashLSH picks its bands with scipy.integrate.quad."""
    import datasketch

    near, far = datasketch.MinHash(num_perm=128), datasketch.MinHash(num_perm=128)
    for word in "the knowledge graph keeps one node per entity".split():
        near.update(word.encode())
    for word in "an unrelated sentence about the weather".split():
        far.update(word.encode())
    lsh = datasketch.MinHashLSH(threshold=0.5, num_perm=128)
    lsh.insert("near", near)
    lsh.insert("far", far)
    found = lsh.query(near)
    return [] if found == ["near"] else [f"MinHashLSH found {found}, not ['near']"]


def dedup_problems() -> List[str]:
    """graphify's entity dedup: its MinHash/LSH pass runs on labels long enough to pass its entropy gate."""
    from graphify.dedup import deduplicate_entities

    nodes = [{"id": "a", "label": "Harbourline Wholesale Distribution Ltd", "source_file": "a.md"},
             {"id": "b", "label": "Harbourline Wholesale Distribution Limited", "source_file": "b.md"},
             {"id": "c", "label": "Quarterly revenue forecast for the northern region", "source_file": "c.md"}]
    kept, _edges = deduplicate_entities(nodes, [], communities={"a": 0, "b": 0, "c": 1})
    return [] if 2 <= len(kept) <= 3 else [f"deduplicate_entities kept {len(kept)} of 3 nodes"]


def pagerank_problems() -> List[str]:
    """The code graph ranks symbols with networkx's pagerank, which builds a scipy.sparse matrix."""
    from modules.codegraph.ranking.pagerank_ranker import PageRankRanker

    symbols = [{"id": index, "name": f"f{index}", "signature": f"def f{index}()"} for index in range(4)]
    relationships = [{"from_symbol_id": source, "to_symbol_id": 0} for source in (1, 2, 3)]
    ranked = PageRankRanker().rank_symbols(symbols, relationships, token_budget=10_000)
    return [] if ranked and ranked[0]["id"] == 0 else [f"PageRankRanker ranked {[s['id'] for s in ranked]}"]


def knowledge_graph_problems() -> List[str]:
    """graph_service's chain: build_from_json, Leiden, score_all, god_nodes, surprising_connections."""
    from graphify.analyze import god_nodes, surprising_connections
    from graphify.build import build_from_json
    from graphify.cluster import score_all

    from modules.knowledge.leiden_partition import cluster

    names = CLIQUE_A + CLIQUE_B
    edges = [(a, b) for clique in (CLIQUE_A, CLIQUE_B) for a in clique for b in clique if a < b]
    extraction = {"nodes": [{"id": name, "label": name.title(), "file_type": "document", "source_file": "kg.md"}
                            for name in names],
                  "edges": [{"source": a, "target": b, "relation": "mentions", "confidence": "EXTRACTED",
                             "source_file": "kg.md"} for a, b in [*edges, (CLIQUE_A[0], CLIQUE_B[0])]]}
    graph = build_from_json(extraction)
    communities = cluster(graph)
    score_all(graph, communities)
    god_nodes(graph)
    surprising_connections(graph, communities)
    found = sorted(sorted(members) for members in communities.values())
    expected = sorted([sorted(CLIQUE_A), sorted(CLIQUE_B)])
    return [] if found == expected else [f"Leiden found {found}, not the two cliques {expected}"]


def statistics_problems() -> List[str]:
    """core.math's confidence interval for a small sample uses Student's t from scipy.special."""
    from core.math.probability_theory import ProbabilityTheory

    interval = ProbabilityTheory.calculate_confidence_interval(list(SAMPLE), confidence_level=0.95)
    margin = (interval.upper_bound - interval.lower_bound) / 2
    expected = T_975_DF4 * interval.standard_error
    if math.isclose(margin, expected, rel_tol=TOLERANCE):
        return []
    return [f"the 95 % interval of {SAMPLE} has margin {margin}, not t(0.975, 4) x SE = {expected}"]


def loaded_problems() -> List[str]:
    """Whatever ran, no pruned subpackage and no NEVER_LOADED module was imported."""
    loaded = [f"scipy.{name}" for name in PRUNED if f"scipy.{name}" in sys.modules]
    loaded += [name for name in NEVER_LOADED if name in sys.modules]
    return [f"{name} was imported" for name in loaded]


CHECKS: Tuple[Tuple[str, Callable[[], List[str]]], ...] = (
    ("subpackages", subpackage_problems), ("datasketch", datasketch_problems), ("graphify dedup", dedup_problems),
    ("code graph pagerank", pagerank_problems), ("Knowledge Graph", knowledge_graph_problems),
    ("Student's t", statistics_problems), ("never loaded", loaded_problems),
)


def run(name: str, check: Callable[[], List[str]]) -> List[str]:
    """One check's problems; an exception is the check's failure, reported on one line."""
    try:
        return check()
    except Exception as error:  # noqa: BLE001 — a crash in a check is that check's failure, reported by name
        return [f"{type(error).__name__}: {' '.join(str(error).split())}"]


def main() -> int:
    os.environ.update(APP_IMPORT_ENV)
    sys.path = [APP_DIR] + [entry for entry in sys.path if entry and entry != APP_DIR]
    failed = False
    for name, check in CHECKS:
        problems = run(name, check)
        for problem in problems:
            print(f"::error title=scipy (PRD-262 row #17, {name})::{problem}")
        if not problems:
            print(f"{name}: ok")
        failed = failed or bool(problems)
    loaded = sorted(module for module in sys.modules if module.count(".") == 1 and module.startswith("scipy."))
    print(f"scipy subpackages loaded: {', '.join(loaded)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
