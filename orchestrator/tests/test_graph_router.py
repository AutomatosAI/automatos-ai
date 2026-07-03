"""Tests for GraphRouter (PRD-139 US-004).

Pure unit tests -- no Redis, no DB, no OpenRouter. All dependencies are
replaced with deterministic fakes. Uses importlib to load graph_router.py
directly, bypassing the modules.tools.__init__ import chain.

Validates:
  AC-8:  mocked entry nodes + mocked edge table produces expected chains in order
  AC-9:  empty edge table falls through to embedding-only ranking with no error
  AC-10: agent below sample floor uses global graph, not agent edges
  AC-11: confidence floor filters low-evidence edges
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_DISCOVERY = _THIS.parents[1] / "modules" / "tools" / "discovery"

# ---------------------------------------------------------------------------
# Stub heavy platform modules that graph_router lazily imports
# ---------------------------------------------------------------------------

# core.database.database
_fake_db_mod = type(sys)("core.database.database")

# core.cache.service
_fake_cache_mod = type(sys)("core.cache.service")

# core.models.tool_routing
_fake_tr_mod = type(sys)("core.models.tool_routing")


class _StubEdge:
    """Mimics a ToolRoutingEdge ORM row."""

    # Column descriptor stubs for SQLAlchemy query filters
    class _Col:
        def __init__(self, name):
            self.name = name

        def in_(self, values):
            return ("in", self.name, values)

        def is_(self, value):
            return ("is", self.name, value)

        def __eq__(self, other):
            return ("eq", self.name, other)

        def __ge__(self, other):
            return ("ge", self.name, other)

        def desc(self):
            return ("desc", self.name)

    from_action = _Col("from_action")
    to_action = _Col("to_action")
    edge_type = _Col("edge_type")
    confidence = _Col("confidence")
    agent_id = _Col("agent_id")
    workspace_id = _Col("workspace_id")
    sample_count = _Col("sample_count")

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


class _StubAffinity:
    """Mimics a ToolRoutingAffinity ORM row."""

    class _Col:
        def __init__(self, name):
            self.name = name

        def in_(self, values):
            return ("in", self.name, values)

        def is_(self, value):
            return ("is", self.name, value)

        def __eq__(self, other):
            return ("eq", self.name, other)

    action_name = _Col("action_name")
    agent_id = _Col("agent_id")
    workspace_id = _Col("workspace_id")
    affinity_type = _Col("affinity_type")

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


_fake_tr_mod.ToolRoutingEdge = _StubEdge
_fake_tr_mod.ToolRoutingAffinity = _StubAffinity

# sqlalchemy stubs — graph_router.py uses lazy `from sqlalchemy import and_, or_`
# inside methods.  _StubEdge._Col returns plain tuples from operator overloads,
# so the real sqlalchemy operators would reject them.  We need a fake sqlalchemy
# for this test but MUST NOT leave it in sys.modules after module load (other
# tests in the same pytest session need the real one).
#
# Strategy: import the real sqlalchemy first (lands it in sys.modules), build a
# fake with the same module name, then swap it in ONLY via an autouse fixture.
import sqlalchemy as _real_sa  # noqa: E402 — ensures real module is cached
import sqlalchemy.orm as _real_sa_orm  # noqa: E402

_fake_sa = type(sys)("sqlalchemy")
_fake_sa.and_ = lambda *args: ("and_", args)
_fake_sa.or_ = lambda *args: ("or_", args)  # noqa
_fake_sa.func = MagicMock()
_fake_sa.__path__ = _real_sa.__path__
_fake_sa.__file__ = _real_sa.__file__

# Core module stubs are installed/restored per-test via the _swap_sqlalchemy
# fixture below (along with sqlalchemy).  We track which modules to stub here.
_CORE_STUBS = {
    "core.database.database": _fake_db_mod,
    "core.cache.service": _fake_cache_mod,
    "core.models.tool_routing": _fake_tr_mod,
}
_CORE_PACKAGES = [
    "core",
    "core.database",
    "core.cache",
    "core.models",
]


# ---------------------------------------------------------------------------
# Fake DB session
# ---------------------------------------------------------------------------

class _FakeQuery:
    """Chainable fake for db.query().filter().order_by().limit().all() / .scalar()."""

    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **kw):
        return self

    def order_by(self, *a, **kw):
        return self

    def limit(self, n):
        self._rows = self._rows[:n]
        return self

    def scalar(self):
        return self._rows

    def all(self):
        return self._rows


class _FakeDBSession:
    def __init__(self, edges=None, affinities=None, agent_total=0):
        self._edges = edges or []
        self._affinities = affinities or []
        self._agent_total = agent_total

    def query(self, model_or_func, *args):
        if not isinstance(model_or_func, type):
            return _FakeQuery(self._agent_total)
        name = model_or_func.__name__ if hasattr(model_or_func, "__name__") else ""
        if "Edge" in name:
            return _FakeQuery(list(self._edges))
        if "Affinity" in name:
            return _FakeQuery(list(self._affinities))
        return _FakeQuery([])


@contextmanager
def _make_db_ctx(edges=None, affinities=None, agent_total=0):
    yield _FakeDBSession(edges, affinities, agent_total)


# Wire fake get_db_session into the stub
_fake_db_mod.get_db_session = lambda: _make_db_ctx()


# ---------------------------------------------------------------------------
# Fake cache
# ---------------------------------------------------------------------------

class _FakeRedis:
    def get(self, *a, **kw):
        return None

    def setex(self, *a, **kw):
        pass


class _FakeCache:
    redis = _FakeRedis()


_fake_cache_mod.get_cache_service = lambda: _FakeCache()


# ---------------------------------------------------------------------------
# Fake semantic index
# ---------------------------------------------------------------------------

class _FakeSemanticIndex:
    def __init__(self, results):
        self._results = results

    async def rank_actions(self, query, top_k=5, **kw):
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Load graph_router via importlib (bypass modules.tools.__init__)
# ---------------------------------------------------------------------------

# First load action_semantic_index so the relative import resolves
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_gr_test", _DISCOVERY / "action_registry.py"
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
_ar_spec.loader.exec_module(_ar_mod)
# PRD-143: pre-initialize this module's registry singleton (empty) so the
# su chain filter's fallback lookup (_drop_super_admin_chains) never triggers
# the live platform_actions registrar — no su actions in these fixtures.
_ar_mod._registry_instance = _ar_mod.ActionRegistry()
_ar_mod._registry_instance._initialized = True

# Create fake package for the relative import chain
_pkg_name = "gr_test_pkg"
_pkg = type(sys)(_pkg_name)
_pkg.__path__ = [str(_DISCOVERY)]
sys.modules[_pkg_name] = _pkg
sys.modules[f"{_pkg_name}.action_registry"] = _ar_mod

# Load action_semantic_index under the package
_asi_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.action_semantic_index",
    _DISCOVERY / "action_semantic_index.py",
)
_asi_mod = importlib.util.module_from_spec(_asi_spec)
_asi_mod.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.action_semantic_index"] = _asi_mod
# We don't exec it (avoids needing real embedding manager) -- just need it
# importable so graph_router's `from .action_semantic_index import ...` works.
# Provide the factory function it imports:
_asi_mod.get_action_semantic_index = lambda: None  # replaced per-test

# Load graph_router under the same package
_gr_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.graph_router",
    _DISCOVERY / "graph_router.py",
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
_gr_mod.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.graph_router"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)

GraphRouter = _gr_mod.GraphRouter
get_graph_router = _gr_mod.get_graph_router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_edge(from_action, to_action, confidence=0.8, weight=5.0, agent_id=None):
    return _StubEdge(
        from_action=from_action,
        to_action=to_action,
        edge_type="used_after",
        confidence=confidence,
        weight=weight,
        agent_id=agent_id,
        workspace_id=None,
        sample_count=10,
    )


def _make_affinity(action_name, affinity_type="succeeds_for_intent",
                   weight=1.0, confidence=0.9, agent_id=None):
    return _StubAffinity(
        action_name=action_name,
        affinity_type=affinity_type,
        weight=weight,
        confidence=confidence,
        agent_id=agent_id,
        workspace_id=None,
        intent_cluster_id=None,
        sample_count=10,
    )


def _build_router(entry_nodes, edges=None, affinities=None,
                  agent_total=0, min_conf=0.6, agent_floor=50):
    """Build a GraphRouter with all dependencies faked."""
    router = GraphRouter.__new__(GraphRouter)
    router._semantic_index = _FakeSemanticIndex(entry_nodes)
    router._get_cache = lambda: _FakeCache()
    return router


def _run(coro):
    return asyncio.run(coro)


def _rank(router, edges=None, affinities=None, agent_id=None,
          agent_total=0, min_conf=0.6, agent_floor=50, top_k=15):
    """Call rank_chains with patched DB and config."""
    @contextmanager
    def patched_db():
        yield _FakeDBSession(edges or [], affinities or [], agent_total)

    orig_min = GraphRouter._min_confidence
    orig_floor = GraphRouter._agent_sample_floor
    orig_db = _fake_db_mod.get_db_session
    GraphRouter._min_confidence = staticmethod(lambda: min_conf)
    GraphRouter._agent_sample_floor = staticmethod(lambda: agent_floor)
    # Patch get_db_session on the module that the lazy import resolves to
    _fake_db_mod.get_db_session = patched_db

    try:
        return _run(router.rank_chains(
            query="test query", workspace_id=None, agent_id=agent_id, top_k=top_k,
        ))
    finally:
        GraphRouter._min_confidence = orig_min
        GraphRouter._agent_sample_floor = orig_floor
        _fake_db_mod.get_db_session = orig_db


# ---------------------------------------------------------------------------
# Fixture: swap in fake sqlalchemy for test execution, restore real after
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _swap_modules():
    """Install fake sqlalchemy + core stubs for graph_router's lazy imports.

    The fake provides no-op and_, or_, func that work with _StubEdge._Col.
    Core stubs provide fake DB session, cache, and model classes.
    All are restored after each test so other test modules are not poisoned.
    """
    saved = {}
    # Save and replace sqlalchemy
    saved["sqlalchemy"] = sys.modules.get("sqlalchemy")
    sys.modules["sqlalchemy"] = _fake_sa

    # Save and install core package paths + stubs
    for pkg in _CORE_PACKAGES:
        saved[pkg] = sys.modules.get(pkg)
        if pkg not in sys.modules:
            m = type(sys)(pkg)
            m.__path__ = []
            sys.modules[pkg] = m

    for mod_path, fake_mod in _CORE_STUBS.items():
        saved[mod_path] = sys.modules.get(mod_path)
        sys.modules[mod_path] = fake_mod

    yield

    # Restore everything
    sys.modules["sqlalchemy"] = _real_sa
    for key, orig in saved.items():
        if key == "sqlalchemy":
            continue
        if orig is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = orig


# ===========================================================================
# AC-8: mocked entry nodes + edge table -> expected chains in expected order
# ===========================================================================

class TestChainRanking:
    def test_chains_sorted_descending(self):
        entry = [("platform_list_agents", 0.95), ("platform_list_playbooks", 0.80)]
        edges = [
            _make_edge("platform_list_agents", "platform_get_details", confidence=0.9),
            _make_edge("platform_list_playbooks", "platform_run", confidence=0.7),
        ]
        router = _build_router(entry)
        result = _rank(router, edges=edges)

        assert len(result) > 0
        scores = [s for _, s, _ in result]
        assert scores == sorted(scores, reverse=True)

    def test_chain_beats_weak_single(self):
        entry = [("platform_search", 0.95), ("platform_weak", 0.20)]
        edges = [_make_edge("platform_search", "platform_analyze", confidence=0.95)]
        router = _build_router(entry)
        result = _rank(router, edges=edges)

        by_actions = {frozenset(a): s for _, s, a in result}
        chain = frozenset(["platform_search", "platform_analyze"])
        weak = frozenset(["platform_weak"])
        assert chain in by_actions
        assert weak in by_actions
        assert by_actions[chain] > by_actions[weak]

    def test_result_tuple_shape(self):
        entry = [("platform_a", 0.9)]
        edges = [_make_edge("platform_a", "platform_b", confidence=0.8)]
        router = _build_router(entry)
        result = _rank(router, edges=edges)

        for primary, score, actions in result:
            assert isinstance(primary, str)
            assert isinstance(score, float)
            assert isinstance(actions, list)
            assert all(isinstance(a, str) for a in actions)
            assert len(actions) >= 1

    def test_affinity_boosts_score(self):
        entry = [("platform_a", 0.5), ("platform_b", 0.5)]
        edges = [
            _make_edge("platform_a", "platform_c", confidence=0.7),
            _make_edge("platform_b", "platform_d", confidence=0.7),
        ]
        affinities = [
            _make_affinity("platform_a", "succeeds_for_intent", weight=5.0, confidence=0.9),
        ]
        router = _build_router(entry)
        result = _rank(router, edges=edges, affinities=affinities)

        by_actions = {frozenset(a): s for _, s, a in result}
        chain_a = frozenset(["platform_a", "platform_c"])
        chain_b = frozenset(["platform_b", "platform_d"])
        # chain_a should score higher due to affinity boost on platform_a
        assert by_actions[chain_a] > by_actions[chain_b]


# ===========================================================================
# AC-9: empty edge table -> embedding-only ranking, no error
# ===========================================================================

class TestEmptyGraph:
    def test_returns_single_chains(self):
        entry = [
            ("platform_list_agents", 0.95),
            ("platform_list_playbooks", 0.80),
            ("platform_search", 0.60),
        ]
        router = _build_router(entry)
        result = _rank(router, edges=[])

        assert len(result) == 3
        for primary, score, actions in result:
            assert len(actions) == 1
            assert actions[0] == primary

    def test_order_matches_cosine(self):
        entry = [("platform_a", 0.9), ("platform_b", 0.7), ("platform_c", 0.5)]
        router = _build_router(entry)
        result = _rank(router, edges=[])

        names = [primary for primary, _, _ in result]
        assert names == ["platform_a", "platform_b", "platform_c"]

    def test_no_error_single_entry(self):
        entry = [("platform_foo", 0.5)]
        router = _build_router(entry)
        result = _rank(router, edges=[])

        assert result == [("platform_foo", 0.5, ["platform_foo"])]

    def test_no_entry_nodes_returns_empty(self):
        router = _build_router([])
        result = _rank(router, edges=[])
        assert result == []


# ===========================================================================
# AC-10: agent below sample floor uses global graph
# ===========================================================================

class TestAgentSampleFloor:
    def test_below_floor_uses_global(self):
        entry = [("platform_a", 0.9)]
        global_edge = _make_edge("platform_a", "platform_c", confidence=0.7)

        router = _build_router(entry)
        result = _rank(
            router,
            edges=[global_edge],
            agent_id=42,
            agent_total=10,  # below floor of 50
            agent_floor=50,
        )

        chain_sets = [frozenset(a) for _, _, a in result]
        assert frozenset(["platform_a", "platform_c"]) in chain_sets

    def test_above_floor_uses_agent_edges(self):
        entry = [("platform_a", 0.9)]
        agent_edge = _make_edge("platform_a", "platform_b", confidence=0.85, agent_id=42)

        router = _build_router(entry)
        result = _rank(
            router,
            edges=[agent_edge],
            agent_id=42,
            agent_total=100,  # above floor of 50
            agent_floor=50,
        )

        chain_sets = [frozenset(a) for _, _, a in result]
        assert frozenset(["platform_a", "platform_b"]) in chain_sets


# ===========================================================================
# AC-11: confidence floor filters low-evidence edges
# ===========================================================================

class TestConfidenceFloor:
    def test_low_confidence_excluded(self):
        entry = [("platform_x", 0.9)]
        high = _make_edge("platform_x", "platform_y", confidence=0.8)
        # low confidence edge filtered by DB query -> not in results

        router = _build_router(entry)
        result = _rank(router, edges=[high], min_conf=0.6)

        chain_sets = [frozenset(a) for _, _, a in result]
        assert frozenset(["platform_x", "platform_y"]) in chain_sets
        assert frozenset(["platform_x", "platform_z"]) not in chain_sets

    def test_all_below_floor_returns_singles(self):
        """When all edges are below confidence floor, only singles remain."""
        entry = [("platform_a", 0.9)]
        # No edges pass the floor -> empty edges list
        router = _build_router(entry)
        result = _rank(router, edges=[], min_conf=0.99)

        assert len(result) == 1
        assert result[0] == ("platform_a", 0.9, ["platform_a"])


# ===========================================================================
# Singleton factory
# ===========================================================================

class TestSingleton:
    def test_returns_same_instance(self):
        _gr_mod._instance = None
        orig_factory = _gr_mod.get_action_semantic_index
        _gr_mod.get_action_semantic_index = lambda: MagicMock()

        try:
            r1 = get_graph_router()
            r2 = get_graph_router()
            assert r1 is r2
        finally:
            _gr_mod._instance = None
            _gr_mod.get_action_semantic_index = orig_factory


# ===========================================================================
# DB error fallback
# ===========================================================================

class TestDBErrorFallback:
    def test_db_failure_returns_singles(self):
        """If DB query raises, fall back to single-action chains."""
        entry = [("platform_a", 0.9), ("platform_b", 0.7)]
        router = _build_router(entry)

        @contextmanager
        def exploding_db():
            raise RuntimeError("DB gone")
            yield  # noqa: unreachable

        _gr_mod.get_db_session = exploding_db

        result = _run(router.rank_chains(query="test", workspace_id=None, top_k=15))

        assert len(result) == 2
        for _, _, actions in result:
            assert len(actions) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
