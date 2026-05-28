"""
PRD-141 US-017: GraphRouter penalizes negative affinities.
==========================================================

``_query_affinities`` now returns an explicit ``(positive_boosts,
negative_penalties)`` pair instead of a single netted dict:

    * ``succeeds_for_intent`` / ``agent_prefers`` -> positive_boosts[action] += weight*confidence
    * ``fails_for_intent``                        -> negative_penalties[action] += weight*confidence  (stored as a POSITIVE magnitude)

``_expand_with_graph`` then scores each edge-expanded chain as
``score = cosine * edge_confidence + boost - penalty`` so a tool that
historically fails for an intent ranks lower.

``modules.tools.__init__`` eagerly imports the DB-backed executor chain, so we
leaf-load ``graph_router.py`` under a synthetic package and inject a fake
``action_semantic_index`` (module-top import) plus a fake
``core.database.database`` (lazy import inside ``_expand_with_graph``) into
``sys.modules``. ``core.models.tool_routing`` imports cleanly (only needs the
declarative Base), so ``_query_affinities`` runs against the REAL model with a
fake DB session.
"""
import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

_discovery_dir = _orchestrator_root / "modules" / "tools" / "discovery"
_PKG = "_us017_graph"


def _load_graph_router():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_discovery_dir)]
        sys.modules[_PKG] = pkg

    # Fake the module-top `from .action_semantic_index import get_action_semantic_index`
    # so loading graph_router doesn't pull numpy + the registry/embedding stack.
    asi_name = f"{_PKG}.action_semantic_index"
    if asi_name not in sys.modules:
        fake_asi = types.ModuleType(asi_name)
        fake_asi.get_action_semantic_index = lambda: SimpleNamespace(
            rank_actions=lambda *a, **k: []
        )
        sys.modules[asi_name] = fake_asi

    full = f"{_PKG}.graph_router"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _discovery_dir / "graph_router.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


_graph_mod = _load_graph_router()
GraphRouter = _graph_mod.GraphRouter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeAffinityDB:
    """Minimal stand-in: query(...).filter(...).all() yields the canned rows."""

    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _FakeQuery(self._rows)


def _aff(action_name, affinity_type, weight, confidence):
    return SimpleNamespace(
        action_name=action_name,
        affinity_type=affinity_type,
        weight=weight,
        confidence=confidence,
    )


def _extract_edge_type(filter_clause):
    """Pull the `edge_type == X` literal out of the and_() expression that
    GraphRouter._query_edges passes to db.query(...).filter(...).

    Walks the BooleanClauseList for the binary expression whose left column is
    `edge_type` and returns its bound value, so the fake DB can honour the same
    filter the real query would apply.
    """
    for clause in getattr(filter_clause, "clauses", [filter_clause]):
        left = getattr(clause, "left", None)
        right = getattr(clause, "right", None)
        if getattr(left, "key", None) == "edge_type" and right is not None:
            return getattr(right, "value", None)
    return None


class _FakeEdgeQuery:
    """query(...).filter(...).order_by(...).limit(...).all() that respects the
    edge_type filter built by _query_edges (so failed_after rows are excluded
    exactly as the real SQL would exclude them)."""

    def __init__(self, rows):
        self._rows = rows
        self._edge_type = None

    def filter(self, *clauses):
        for c in clauses:
            et = _extract_edge_type(c)
            if et is not None:
                self._edge_type = et
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        if self._edge_type is None:
            return list(self._rows)
        return [r for r in self._rows if r.edge_type == self._edge_type]


class _FakeEdgeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *a, **k):
        return _FakeEdgeQuery(self._rows)


def _edge(from_action, to_action, edge_type, confidence=1.0, weight=1.0, agent_id=None):
    return SimpleNamespace(
        from_action=from_action,
        to_action=to_action,
        edge_type=edge_type,
        confidence=confidence,
        weight=weight,
        agent_id=agent_id,
    )


@pytest.fixture
def router(monkeypatch):
    """A GraphRouter with the lazy `core.database.database` import faked out."""
    fake_db_mod = types.ModuleType("core.database.database")

    @contextmanager
    def _fake_session():
        yield object()

    fake_db_mod.get_db_session = _fake_session
    monkeypatch.setitem(sys.modules, "core.database.database", fake_db_mod)
    return GraphRouter()


# ---------------------------------------------------------------------------
# _query_affinities now returns (positive_boosts, negative_penalties)
# ---------------------------------------------------------------------------

def test_query_affinities_splits_positive_and_negative():
    rows = [
        _aff("good", "succeeds_for_intent", weight=1.0, confidence=0.8),
        _aff("pref", "agent_prefers", weight=0.5, confidence=0.6),
        _aff("bad", "fails_for_intent", weight=1.0, confidence=0.5),
    ]
    positive, negative = GraphRouter._query_affinities(
        _FakeAffinityDB(rows), ["good", "pref", "bad"], None
    )

    assert positive["good"] == pytest.approx(0.8)
    assert positive["pref"] == pytest.approx(0.3)
    # fails_for_intent is recorded as a POSITIVE magnitude in the penalties dict
    assert negative["bad"] == pytest.approx(0.5)
    # no cross-contamination between the two dicts
    assert "bad" not in positive
    assert "good" not in negative and "pref" not in negative


def test_negative_affinity_penalizes_score(router, monkeypatch):
    """A fails_for_intent action subtracts its penalty from the chain score."""
    monkeypatch.setattr(
        router, "_query_edges",
        lambda db, names, conf, aid: [
            {"from_action": "bad_tool", "to_action": "next", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
        ],
    )
    monkeypatch.setattr(
        router, "_query_affinities",
        lambda db, names, aid: ({}, {"bad_tool": 0.5}),
    )

    chains = router._expand_with_graph([("bad_tool", 0.8)], agent_id=None)

    expanded = [c for c in chains if c[2] == ["bad_tool", "next"]]
    assert len(expanded) == 1
    # 0.8 (cosine) * 1.0 (edge_conf) + 0 (boost) - 0.5 (penalty) = 0.3
    assert expanded[0][1] == pytest.approx(0.3)
    # penalty demonstrably lowered the score below the un-penalized cosine*conf
    assert expanded[0][1] < 0.8


def test_negative_signals_reduce_ranking(router, monkeypatch):
    """Of two equal-cosine chains, the one whose action fails ranks lower."""
    monkeypatch.setattr(
        router, "_query_edges",
        lambda db, names, conf, aid: [
            {"from_action": "tool_a", "to_action": "x", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
            {"from_action": "tool_b", "to_action": "y", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
        ],
    )
    monkeypatch.setattr(
        router, "_query_affinities",
        lambda db, names, aid: ({}, {"tool_b": 0.5}),
    )

    chains = router._expand_with_graph([("tool_a", 0.9), ("tool_b", 0.9)], agent_id=None)

    score = {c[0]: c[1] for c in chains if len(c[2]) == 2}
    assert score["tool_a"] == pytest.approx(0.9)   # no penalty
    assert score["tool_b"] == pytest.approx(0.4)   # 0.9 - 0.5
    assert score["tool_a"] > score["tool_b"]


def test_positive_boost_still_applies(router, monkeypatch):
    """The refactor preserves positive boosts (regression guard)."""
    monkeypatch.setattr(
        router, "_query_edges",
        lambda db, names, conf, aid: [
            {"from_action": "tool_a", "to_action": "x", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
        ],
    )
    monkeypatch.setattr(
        router, "_query_affinities",
        lambda db, names, aid: ({"tool_a": 0.3}, {}),
    )

    chains = router._expand_with_graph([("tool_a", 0.5)], agent_id=None)

    expanded = [c for c in chains if c[2] == ["tool_a", "x"]]
    # 0.5 * 1.0 + 0.3 - 0 = 0.8
    assert expanded[0][1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# PRD-141 US-018: failed_after edges are never expanded into chains
# ---------------------------------------------------------------------------

def test_failed_after_edge_not_expanded():
    """_query_edges only ever requests edge_type == 'used_after', so a
    failed_after edge sitting in the same table is filtered out at the DB layer
    and can never become a recommended chain.
    """
    rows = [
        _edge("good", "next", "used_after", confidence=1.0),
        _edge("good", "bad", "failed_after", confidence=1.0),
    ]
    fake_db = _FakeEdgeDB(rows)

    edges = GraphRouter._query_edges(fake_db, ["good"], 0.6, None)

    to_actions = {e["to_action"] for e in edges}
    assert "next" in to_actions       # used_after IS followed
    assert "bad" not in to_actions    # failed_after is NOT followed
    assert len(edges) == 1
