"""PRD-262 US-027: Leiden through graspologic-native reproduces graspologic's partitions.

Railway's API clustered the Knowledge Graph with graphify 0.8.31, whose ``_partition``
called ``graspologic.partition.leiden`` (graspologic 3.3.0), until US-027 moved the call
onto graspologic-native through modules/knowledge/leiden_partition.py. This suite holds
the adapter to graspologic without graspologic installed:

- on every graph tests/fixtures/leiden_graphs.py builds, the adapter's partition, and
  graphify's ``cluster`` with the adapter installed, return the communities graspologic
  recorded (tests/fixtures/leiden_partitions.json). The bar is §9-Q8's, taken 16 Oct:
  identical partitions, or else modularity within 1% of the golden's (a warning says
  which). test.yml's reference-goldens job re-records the goldens with graspologic 3.3.0
  whenever they, the graphs or the reference change;
- both of graphify's partition calls reach graspologic-native: the one on the graph and
  the re-split of a community too large to keep;
- graspologic-native gets the call graspologic 3.3.0 made: graphify's stable edge list
  (str ids, ``weight`` or 1.0), the caller's resolution, randomness 0.001, one
  iteration, modularity, seed 42 and one trial; communities map back to the nodes as
  they were; a str collision and a resolution of 0 are refused, as graspologic refused
  them;
- importing the adapter installs it as graphify.cluster._partition, once, and refuses a
  graphify whose ``_partition`` takes other parameters; graphify's source still calls
  ``_partition`` from those two places only, with seed 42 and one trial;
  graph_service takes ``cluster`` from the adapter;
- the goldens cover every graph; the reference lock is its input pinned and hashed, at
  the API lock's graspologic-native and networkx; the reference script refuses to
  record without graspologic; CI re-records whenever any of it changes.

The adapter and graspologic-native run here; graspologic never does.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import graphify.cluster as graphify_cluster
import networkx as nx
import pytest
import yaml

from modules.knowledge import leiden_partition as adapter
from tests.fixtures.leiden_graphs import GRAPHS, resolution_limit
from tests.fixtures.reference_locks import pins, unhashed

ROOT = Path(__file__).resolve().parents[2]
ORCH = ROOT / "orchestrator"
GOLDEN = json.loads((ORCH / "tests" / "fixtures" / "leiden_partitions.json").read_text(encoding="utf-8"))["graphs"]
REFERENCE = ROOT / "scripts" / "ci" / "reference"
REFERENCE_IN = REFERENCE / "requirements-leiden-partitions.in"
REFERENCE_LOCK = REFERENCE / "requirements-leiden-partitions.lock"
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
MODULARITY_TOLERANCE = 0.01  # §9-Q8: within 1% of the golden partition's modularity


def _communities(groups) -> list[list[str]]:
    """Each community's ids sorted, largest first, ties by their ids (the goldens' order)."""
    lists = [sorted(map(str, group)) for group in groups]
    return sorted(lists, key=lambda ids: (-len(ids), ids))


def _by_community(partition: dict) -> list[list[str]]:
    groups: dict = {}
    for node, community in partition.items():
        groups.setdefault(community, []).append(node)
    return _communities(groups.values())


def _modularity(graph: nx.Graph, communities: list[list[str]]) -> float:
    """Weighted modularity of ``communities`` on the part of ``graph`` they cover."""
    covered = {node for community in communities for node in community}
    return nx.community.modularity(graph.subgraph(covered), [set(c) for c in communities], weight="weight")


def _assert_meets_the_bar(label: str, graph: nx.Graph, got: list, golden: list) -> None:
    """§9-Q8: identical communities, or else modularity within 1% of the golden's."""
    if _communities(got) == _communities(golden):
        return
    want, found = _modularity(graph, golden), _modularity(graph, got)
    assert abs(found - want) <= MODULARITY_TOLERANCE * abs(want), (
        f"{label}: {got} differ from graspologic's {golden}, and modularity {found:.6f} is not within "
        f"1% of {want:.6f} (PRD-262 §9-Q8: then the adapter does not ship)")
    warnings.warn(f"{label}: not graspologic's partition; modularity {found:.6f} within 1% of {want:.6f}",
                  stacklevel=2)


def _spy_on_native(monkeypatch) -> list:
    """Each graspologic-native call's keyword arguments, passed on to the real function."""
    calls: list = []
    real = adapter.graspologic_native.leiden

    def leiden(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(adapter, "graspologic_native", SimpleNamespace(leiden=leiden))
    return calls


def _fake_native(monkeypatch, communities: dict) -> dict:
    """graspologic-native replaced by a function that records its call and returns ``communities``."""
    seen: dict = {}

    def leiden(**kwargs):
        seen.update(kwargs)
        return 0.5, communities

    monkeypatch.setattr(adapter, "graspologic_native", SimpleNamespace(leiden=leiden))
    return seen


# --------------------------------------------------------------------------- parity with graspologic


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_the_adapters_partition_is_graspologics(name):
    graph = GRAPHS[name]()
    got = _by_community(adapter.partition(graph))
    _assert_meets_the_bar(f"{name} partition", graph, got, GOLDEN[name]["partition"])


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_graphifys_cluster_on_the_adapter_returns_graspologics_clusters(name):
    graph = GRAPHS[name]()
    clusters = adapter.cluster(graph)
    assert sorted(clusters) == list(range(len(clusters))), "graphify numbers communities from 0"
    got = [clusters[cid] for cid in sorted(clusters)]
    assert sorted(node for community in got for node in community) == sorted(map(str, graph.nodes()))
    _assert_meets_the_bar(f"{name} clusters", graph, got, GOLDEN[name]["clusters"])


def test_both_of_graphifys_partition_calls_reach_graspologic_native(monkeypatch):
    """cluster.py:133 on the graph, then :198 on the 12-node community, more than a quarter of 16."""
    calls = _spy_on_native(monkeypatch)
    graph = resolution_limit()
    adapter.cluster(graph)
    ends = [sorted({end for edge in call["edges"] for end in edge[:2]}) for call in calls]
    pair = sorted(f"{side}{index}" for side in "pq" for index in range(6))
    assert ends == [sorted(graph.nodes()), pair], ends


def test_the_goldens_cover_every_graph_and_every_node_once():
    assert set(GOLDEN) == set(GRAPHS)
    for name, build in GRAPHS.items():
        graph = build()
        partition = [node for community in GOLDEN[name]["partition"] for node in community]
        clusters = [node for community in GOLDEN[name]["clusters"] for node in community]
        assert sorted(partition) == sorted(node for node in graph.nodes() if graph.degree(node)), name
        assert sorted(clusters) == sorted(graph.nodes()), name


# --------------------------------------------------------------------------- the call graspologic made


def test_graspologic_native_gets_the_call_graspologic_made(monkeypatch):
    seen = _fake_native(monkeypatch, {"a": 0, "b": 0, "c": 1})
    graph = nx.Graph()
    graph.add_edge("c", "a", weight=2)
    graph.add_edge("b", "a")
    graph.add_edge("c", "b", weight=0.5, relation="mentions")
    assert adapter.partition(graph, resolution=0.5) == {"a": 0, "b": 0, "c": 1}
    assert seen == {
        # graphify's stable order: nodes sorted, edges sorted by their ends, then walked
        # node by node; weight read, 1.0 where an edge has none.
        "edges": [("a", "b", 1.0), ("a", "c", 2.0), ("b", "c", 0.5)],
        "starting_communities": None,
        "resolution": 0.5,
        "randomness": 0.001,
        "iterations": 1,
        "use_modularity": True,
        "seed": 42,
        "trials": 1,
    }
    assert all(isinstance(weight, float) for _, _, weight in seen["edges"])


def test_node_ids_go_as_str_in_str_order_and_come_back_as_they_were(monkeypatch):
    seen = _fake_native(monkeypatch, {"10": 0, "9": 0, "x": 1})
    graph = nx.Graph([(9, 10), (10, "x")])
    assert adapter.partition(graph) == {9: 0, 10: 0, "x": 1}
    assert seen["edges"] == [("10", "x", 1.0), ("10", "9", 1.0)], "str order: '10' before '9'"
    assert seen["resolution"] == 1.0, "graphify's re-split passes no resolution"


def test_two_nodes_with_one_str_are_refused_as_graspologic_refused_them(monkeypatch):
    seen = _fake_native(monkeypatch, {})
    with pytest.raises(ValueError, match="collision between distinct values"):
        adapter.partition(nx.Graph([(1, "1"), ("1", "b")]))
    assert not seen, "refused before graspologic-native is called"


def test_a_resolution_that_is_not_positive_is_refused(monkeypatch):
    seen = _fake_native(monkeypatch, {})
    with pytest.raises(ValueError, match="resolution must be a positive float"):
        adapter.partition(nx.Graph([("a", "b")]), resolution=0)
    assert not seen


def test_a_node_without_an_edge_is_left_out_as_graspologic_left_it_out():
    graph = nx.Graph([("a", "b"), ("b", "c"), ("c", "a")])
    graph.add_node("alone")
    assert set(adapter.partition(graph)) == {"a", "b", "c"}


# --------------------------------------------------------------------------- the install


def test_importing_the_adapter_installs_it_in_graphify_once():
    assert graphify_cluster._partition is adapter.partition
    adapter.install()
    assert graphify_cluster._partition is adapter.partition
    assert adapter.cluster is graphify_cluster.cluster


@pytest.mark.parametrize("changed", [
    lambda G, resolution=1.0, seed=None: {},  # a parameter graphify might add
    lambda graph: {},  # one it might rename or drop
])
def test_the_install_refuses_a_graphify_whose_partition_changed(monkeypatch, changed):
    monkeypatch.setattr(graphify_cluster, "_partition", changed)
    with pytest.raises(ImportError, match="leiden_partition.py"):
        adapter.install()


def test_the_install_refuses_a_graphify_without_a_partition(monkeypatch):
    monkeypatch.delattr(graphify_cluster, "_partition")
    with pytest.raises(ImportError, match="takes \\(\\)"):
        adapter.install()


def _graphify_source() -> ast.Module:
    return ast.parse(Path(inspect.getfile(graphify_cluster)).read_text(encoding="utf-8"))


def test_graphify_calls_partition_from_two_places_with_seed_42_and_one_trial():
    """What the adapter stands in for, read from the installed graphify (pinned in requirements.in)."""
    tree = _graphify_source()
    callers = sorted(
        function.name
        for function in ast.walk(tree) if isinstance(function, ast.FunctionDef)
        for call in ast.walk(function)
        if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "_partition"
    )
    assert callers == ["_split_community", "cluster"], callers
    partition = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_partition")
    assert tuple(arg.arg for arg in partition.args.args) == adapter.GRAPHIFY_PARAMETERS
    settings = {
        target.slice.value: node.value.value
        for node in ast.walk(partition) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
        for target in node.targets if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)
    }
    assert settings["random_seed"] == adapter.RANDOM_SEED and settings["trials"] == adapter.TRIALS, settings
    assert "from graspologic.partition import leiden" in ast.unparse(partition)


def test_graph_service_takes_cluster_from_the_adapter():
    tree = ast.parse((ORCH / "modules" / "knowledge" / "graph_service.py").read_text(encoding="utf-8"))
    imported = {node.module: {alias.name for alias in node.names}
                for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert imported["graphify.cluster"] == {"score_all"}, "cluster comes from the adapter, not graphify"
    assert imported["modules.knowledge.leiden_partition"] == {"cluster"}


# --------------------------------------------------------------------------- the reference and CI


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_reference_lock_is_its_input_pinned_and_hashed():
    wanted = pins(REFERENCE_IN)
    assert wanted and pins(REFERENCE_LOCK) == wanted, "every package the reference runs is pinned in the .in"
    text = REFERENCE_LOCK.read_text(encoding="utf-8")
    assert text.splitlines()[1] == (
        "#    uv pip compile scripts/ci/reference/requirements-leiden-partitions.in --universal "
        "--python-version 3.11 --only-binary :all: --no-binary graspologic --generate-hashes "
        "-o scripts/ci/reference/requirements-leiden-partitions.lock"
    )
    assert unhashed(text) == []


def test_the_reference_runs_the_api_locks_graspologic_native_and_networkx():
    """The adapter and the reference call the same Rust code on the same graphs."""
    api, reference = pins(ORCH / "requirements.lock"), pins(REFERENCE_IN)
    for name in ("graspologic-native", "networkx", "graphifyy"):
        assert reference[name] == api[name], name
    assert reference["graspologic"] == "3.3.0"


def test_the_reference_script_refuses_to_record_without_graspologic(capsys):
    script = _load("prd262_leiden_reference", REFERENCE / "leiden_partitions.py")
    wanted = pins(REFERENCE_IN)
    assert script.PINNED and all(wanted[name] == version for name, version in script.PINNED.items())
    assert set(script.load_graphs().GRAPHS) == set(GRAPHS)
    assert script.version_errors(), "this suite's environment has no graspologic"
    assert script.main([]) == 2, "it refuses to record with anything but the pinned versions"
    assert "::error::the reference needs the pinned graspologic" in capsys.readouterr().out
    assert script.communities([["b", "a"], ["c"], ["e", "d"]]) == [["a", "b"], ["d", "e"], ["c"]]


def test_ci_re_records_the_partitions_whenever_they_or_their_inputs_change():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    match = next(step for step in workflow["jobs"]["changes"]["steps"] if step.get("id") == "match")
    paths = re.compile(match["env"]["REFERENCE_PATHS"])
    for path in ("scripts/ci/reference/leiden_partitions.py", "scripts/ci/reference/requirements-leiden-partitions.lock",
                 "orchestrator/tests/fixtures/leiden_graphs.py", "orchestrator/tests/fixtures/leiden_partitions.json"):
        assert paths.search(path), path
    step = next(step for step in workflow["jobs"]["reference-goldens"]["steps"]
                if "leiden_partitions.py" in step.get("run", ""))
    assert step["if"] == "${{ !cancelled() }}", "it runs even when a reference before it failed"
    assert step["env"]["LOCK"] == "scripts/ci/reference/requirements-leiden-partitions.lock"
    assert step["env"]["PARTITIONS"] == "orchestrator/tests/fixtures/leiden_partitions.json"
    run = step["run"]
    assert 'python -m venv "$VENV"' in run
    assert ('"$VENV/bin/pip" install --no-deps --require-hashes --only-binary :all: --no-binary graspologic '
            '-r "$LOCK"') in run
    assert '"$VENV/bin/python" scripts/ci/reference/leiden_partitions.py --out' in run
    assert 'diff -u <(jq -S . "$PARTITIONS")' in run and "exit 1" in run
    setup = next(step for step in workflow["jobs"]["reference-goldens"]["steps"]
                 if "setup-python" in step.get("uses", ""))
    assert setup["with"]["python-version"] == "3.11", "graspologic 3.3.0 requires Python below 3.12"
