#!/usr/bin/env python3
"""Record the Leiden fixture graphs' partitions with graspologic (PRD-262 US-027).

Until US-027 Railway's API clustered the Knowledge Graph with graphify 0.8.31's
``cluster``. Its ``_partition`` called ``graspologic.partition.leiden`` (graspologic
3.3.0, over graspologic-native 1.3.1) with random_seed=42 and trials=1: once on the
graph, and again on any community too large to keep. Every other image fell back to
networkx's Louvain. US-027 serves both calls from
``orchestrator/modules/knowledge/leiden_partition.py``, which calls graspologic-native
itself. For every graph ``orchestrator/tests/fixtures/leiden_graphs.py`` builds, this
script writes what graphify made of it with graspologic installed:

- ``partition``: ``_partition(graph)``, the communities of one Leiden call, largest
  first, ties by their node ids (a partition's community numbers carry no meaning);
- ``clusters``: ``cluster(graph)``'s communities in its own order, after graphify's
  isolate handling and re-split.

Each community is a sorted list of node ids. The committed copy is
``orchestrator/tests/fixtures/leiden_partitions.json``; the permanent test
(``orchestrator/tests/test_prd262_leiden_adapter.py``) reads only that file.

test.yml's ``reference-goldens`` job installs ``requirements-leiden-partitions.lock``
here into a throwaway Python 3.11 venv (graspologic 3.3.0 requires Python below 3.12),
runs this script and diffs its output against the committed copy, so the reference is
proven in the run that ships the swap. The script refuses to run against any other
graphifyy, graspologic, graspologic-native or networkx, and when graspologic does not
import, because graphify would then fall back to Louvain without a word.

By hand, from the repository root, with Python 3.11:

    python3 -m venv /tmp/leiden
    /tmp/leiden/bin/pip install --no-deps --require-hashes --only-binary :all: \\
        --no-binary graspologic -r scripts/ci/reference/requirements-leiden-partitions.lock
    /tmp/leiden/bin/python scripts/ci/reference/leiden_partitions.py \\
        --out orchestrator/tests/fixtures/leiden_partitions.json

Exit 0 after writing the partitions, 2 when the installed versions are not the pinned
ones or graspologic does not import.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from importlib import metadata
from pathlib import Path
from types import ModuleType
from typing import Dict, Hashable, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[3]
GRAPHS = ROOT / "orchestrator" / "tests" / "fixtures" / "leiden_graphs.py"
# The clustering Railway ran, at the versions its [leiden] lock installed.
PINNED = {"graphifyy": "0.8.31", "graspologic": "3.3.0", "graspologic-native": "1.3.1", "networkx": "3.1"}


def load_graphs() -> ModuleType:
    """orchestrator/tests/fixtures/leiden_graphs.py, loaded from its file (the venv has no app)."""
    spec = importlib.util.spec_from_file_location("leiden_graphs", GRAPHS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def version_errors() -> List[str]:
    """Each pinned distribution that is missing or at another version, or graspologic failing to import."""
    errors = []
    for name, wanted in PINNED.items():
        try:
            found = metadata.version(name)
        except metadata.PackageNotFoundError:
            found = "not installed"
        if found != wanted:
            errors.append(f"{name}: wanted {wanted}, found {found}")
    if not errors:
        try:
            importlib.import_module("graspologic.partition")
        except ImportError as exc:
            errors.append(f"graspologic.partition does not import ({exc}), so graphify would use Louvain")
    return errors


def communities(groups: Iterable[Iterable[Hashable]]) -> List[List[str]]:
    """Each community's node ids sorted, the communities largest first, ties by their ids."""
    lists = [sorted(map(str, group)) for group in groups]
    return sorted(lists, key=lambda ids: (-len(ids), ids))


def record() -> Dict[str, Dict[str, Dict[str, List[List[str]]]]]:
    """Every fixture graph's partition and clusters, as graphify with graspologic makes them."""
    graphify_cluster = importlib.import_module("graphify.cluster")  # only the reference venv has graspologic
    graphs = {}
    for name, build in load_graphs().GRAPHS.items():
        by_community: Dict[int, List[Hashable]] = {}
        for node, community in graphify_cluster._partition(build()).items():
            by_community.setdefault(community, []).append(node)
        clusters = graphify_cluster.cluster(build())
        graphs[name] = {
            "partition": communities(by_community.values()),
            "clusters": [[str(node) for node in clusters[cid]] for cid in sorted(clusters)],
        }
    return {"graphs": graphs}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the partitions here (default: standard output)")
    args = parser.parse_args(argv)
    errors = version_errors()
    if errors:
        print("::error::the reference needs the pinned graspologic: " + "; ".join(errors))
        return 2
    text = json.dumps(record(), indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
