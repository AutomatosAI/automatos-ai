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
import asyncio
import importlib.util
import math
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
    """Pull the allowed edge_type literal(s) out of the and_() expression that
    GraphRouter._query_edges passes to db.query(...).filter(...).

    Handles BOTH the scalar `edge_type == X` form and the collection
    `edge_type.in_((X, Y))` form (PRD-143 added meta_sibling alongside
    used_after), returning a set of allowed edge_type values so the fake DB can
    honour the same filter the real query would apply. Returns None when no
    edge_type clause is present.
    """
    for clause in getattr(filter_clause, "clauses", [filter_clause]):
        left = getattr(clause, "left", None)
        if getattr(left, "key", None) != "edge_type":
            continue
        right = getattr(clause, "right", None)
        value = getattr(right, "value", None)
        if value is None:
            continue
        # `==` binds a scalar; `.in_((...))` binds a list/tuple of values.
        if isinstance(value, (list, tuple, set)):
            return set(value)
        return {value}
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
        # _edge_type is a set of allowed edge_types (used_after, meta_sibling).
        return [r for r in self._rows if r.edge_type in self._edge_type]


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
        _FakeAffinityDB(rows), ["good", "pref", "bad"], None, None
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
        lambda db, names, conf, aid, ws: [
            {"from_action": "bad_tool", "to_action": "next", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
        ],
    )
    monkeypatch.setattr(
        router, "_query_affinities",
        lambda db, names, aid, ws: ({}, {"bad_tool": 0.5}),
    )

    chains = router._expand_with_graph([("bad_tool", 0.8)], agent_id=None, workspace_id=None)

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
        lambda db, names, conf, aid, ws: [
            {"from_action": "tool_a", "to_action": "x", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
            {"from_action": "tool_b", "to_action": "y", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
        ],
    )
    monkeypatch.setattr(
        router, "_query_affinities",
        lambda db, names, aid, ws: ({}, {"tool_b": 0.5}),
    )

    chains = router._expand_with_graph([("tool_a", 0.9), ("tool_b", 0.9)], agent_id=None, workspace_id=None)

    score = {c[0]: c[1] for c in chains if len(c[2]) == 2}
    assert score["tool_a"] == pytest.approx(0.9)   # no penalty
    assert score["tool_b"] == pytest.approx(0.4)   # 0.9 - 0.5
    assert score["tool_a"] > score["tool_b"]


def test_positive_boost_still_applies(router, monkeypatch):
    """The refactor preserves positive boosts (regression guard)."""
    monkeypatch.setattr(
        router, "_query_edges",
        lambda db, names, conf, aid, ws: [
            {"from_action": "tool_a", "to_action": "x", "confidence": 1.0,
             "weight": 1.0, "agent_id": None},
        ],
    )
    monkeypatch.setattr(
        router, "_query_affinities",
        lambda db, names, aid, ws: ({"tool_a": 0.3}, {}),
    )

    chains = router._expand_with_graph([("tool_a", 0.5)], agent_id=None, workspace_id=None)

    expanded = [c for c in chains if c[2] == ["tool_a", "x"]]
    # 0.5 * 1.0 + 0.3 - 0 = 0.8
    assert expanded[0][1] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# PRD-141 US-018: failed_after edges are never expanded into chains
# ---------------------------------------------------------------------------

def test_failed_after_edge_not_expanded():
    """_query_edges requests edge_type IN ('used_after', 'meta_sibling') only,
    so a failed_after edge sitting in the same table is filtered out at the DB
    layer and can never become a recommended chain.
    """
    rows = [
        _edge("good", "next", "used_after", confidence=1.0),
        _edge("good", "bad", "failed_after", confidence=1.0),
    ]
    fake_db = _FakeEdgeDB(rows)

    edges = GraphRouter._query_edges(fake_db, ["good"], 0.6, None, None)

    to_actions = {e["to_action"] for e in edges}
    assert "next" in to_actions       # used_after IS followed
    assert "bad" not in to_actions    # failed_after is NOT followed
    assert len(edges) == 1


# ---------------------------------------------------------------------------
# PRD-141 US-019: batched incremental tool-execution signal recorder
# ---------------------------------------------------------------------------
#
# signal_recorder.py is stdlib-only at module top, so we leaf-load it under the
# same synthetic package. _flush()/_upsert_*() lazily import
# core.database.database (the session) and core.services.edge_builder
# (wilson_lower_bound); both are injected as fakes per-test via monkeypatch so
# no DB creds are needed and there is no cross-file sys.modules pollution.


def _load_signal_recorder():
    full = f"{_PKG}.signal_recorder"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(
        full, _discovery_dir / "signal_recorder.py"
    )
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


_signal_mod = _load_signal_recorder()
ToolSignalRecorder = _signal_mod.ToolSignalRecorder
ToolSignal = _signal_mod.ToolSignal


class _Result:
    def __init__(self, rowcount: int):
        self.rowcount = rowcount


class _CapturingDB:
    """Records (sql, params) and returns a configurable rowcount for UPDATEs."""

    def __init__(self, update_rowcount: int = 0):
        self.executed = []  # list of (sql_text, params)
        self._update_rowcount = update_rowcount

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.executed.append((sql, params or {}))
        rc = self._update_rowcount if sql.strip().upper().startswith("UPDATE") else 1
        return _Result(rc)

    def flush(self):
        pass


class _SessionFactory:
    """Counts how many sessions are opened (must be exactly 1 per flush)."""

    def __init__(self, update_rowcount: int = 0):
        self.db = _CapturingDB(update_rowcount=update_rowcount)
        self.enter_count = 0

    @contextmanager
    def session(self):
        self.enter_count += 1
        yield self.db


def _wilson_real(successes, total, z=1.96):
    if total == 0:
        return 0.0
    p = successes / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    return (centre - spread) / denom


def _recorder_with_fake_db(monkeypatch, update_rowcount: int = 0):
    """A ToolSignalRecorder whose lazy core.* imports are faked."""
    factory = _SessionFactory(update_rowcount=update_rowcount)

    fake_db_mod = types.ModuleType("core.database.database")
    fake_db_mod.get_db_session = factory.session
    monkeypatch.setitem(sys.modules, "core.database.database", fake_db_mod)

    fake_eb = types.ModuleType("core.services.edge_builder")
    fake_eb.wilson_lower_bound = _wilson_real
    monkeypatch.setitem(sys.modules, "core.services.edge_builder", fake_eb)

    return ToolSignalRecorder(), factory


def _edge_stmts(db):
    return [(s, p) for s, p in db.executed if "tool_routing_edges" in s]


def _aff_stmts(db):
    return [(s, p) for s, p in db.executed if "tool_routing_affinities" in s]


def test_incremental_edge_update_success(monkeypatch):
    """A success signal with a prior action -> used_after edge + agent_prefers
    affinity. Fresh keys (no existing row) fall through to INSERT."""
    recorder, factory = _recorder_with_fake_db(monkeypatch, update_rowcount=0)

    asyncio.run(
        recorder._flush(
            [ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a")]
        )
    )

    edge_inserts = [(s, p) for s, p in _edge_stmts(factory.db) if "INSERT" in s.upper()]
    assert len(edge_inserts) == 1
    _, ep = edge_inserts[0]
    assert ep["from_action"] == "a"
    assert ep["to_action"] == "b"
    assert ep["edge_type"] == "used_after"

    aff_inserts = [(s, p) for s, p in _aff_stmts(factory.db) if "INSERT" in s.upper()]
    assert len(aff_inserts) == 1
    _, ap = aff_inserts[0]
    assert ap["action_name"] == "b"
    assert ap["affinity_type"] == "agent_prefers"


def test_incremental_edge_update_failure(monkeypatch):
    """A failure signal -> failed_after edge + fails_for_intent affinity."""
    recorder, factory = _recorder_with_fake_db(monkeypatch, update_rowcount=0)

    asyncio.run(
        recorder._flush(
            [ToolSignal("b", False, agent_id=1, workspace_id="ws", prior_action="a")]
        )
    )

    edge_inserts = [(s, p) for s, p in _edge_stmts(factory.db) if "INSERT" in s.upper()]
    assert len(edge_inserts) == 1
    assert edge_inserts[0][1]["edge_type"] == "failed_after"

    aff_inserts = [(s, p) for s, p in _aff_stmts(factory.db) if "INSERT" in s.upper()]
    assert len(aff_inserts) == 1
    assert aff_inserts[0][1]["affinity_type"] == "fails_for_intent"


def test_edge_upsert_increments_sample_count(monkeypatch):
    """Repeated identical signals collapse to ONE upsert that INCREMENTS
    sample_count (no duplicate rows). When the row already exists
    (update_rowcount=1) only the UPDATE runs — never a second INSERT."""
    recorder, factory = _recorder_with_fake_db(monkeypatch, update_rowcount=1)

    asyncio.run(
        recorder._flush(
            [ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a")
             for _ in range(3)]
        )
    )

    edge_stmts = _edge_stmts(factory.db)
    updates = [(s, p) for s, p in edge_stmts if s.strip().upper().startswith("UPDATE")]
    inserts = [(s, p) for s, p in edge_stmts if "INSERT" in s.upper()]

    # 3 dupes collapse to 1 update; row exists so NO insert (no duplicate edge)
    assert len(updates) == 1
    assert len(inserts) == 0

    sql, params = updates[0]
    assert params["inc"] == 3  # aggregated count
    assert "sample_count = tool_routing_edges.sample_count + :inc" in sql  # increment
    assert "IS NOT DISTINCT FROM" in sql  # null-safe scope match


def test_flush_uses_single_session(monkeypatch):
    """A mixed batch (multiple edges + affinities) opens exactly ONE DB session."""
    recorder, factory = _recorder_with_fake_db(monkeypatch, update_rowcount=0)

    batch = [
        ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a"),
        ToolSignal("c", False, agent_id=1, workspace_id="ws", prior_action="b"),
        ToolSignal("d", True, agent_id=2, workspace_id="ws2", prior_action="c"),
    ]
    asyncio.run(recorder._flush(batch))

    assert factory.enter_count == 1


def test_aggregate_collapses_and_skips_priorless_and_self_edges():
    """_aggregate is pure: dupes sum, missing/equal prior_action yields no edge
    (affinity still emitted)."""
    batch = [
        ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a"),
        ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a"),
        ToolSignal("x", True, agent_id=1, workspace_id="ws", prior_action=None),
        ToolSignal("y", True, agent_id=1, workspace_id="ws", prior_action="y"),
    ]
    edges, affinities = ToolSignalRecorder._aggregate(batch)

    assert edges[("a", "b", "used_after", 1, "ws")] == 2  # dupes summed
    # no edge for the prior-less signal or the self-transition
    assert all(ek[1] != "x" for ek in edges)
    assert ("y", "y", "used_after", 1, "ws") not in edges
    # affinities are always produced (one per distinct action)
    assert affinities[("b", "agent_prefers", 1, "ws")] == 2
    assert affinities[("x", "agent_prefers", 1, "ws")] == 1
    assert affinities[("y", "agent_prefers", 1, "ws")] == 1


def test_record_is_noop_without_running_loop(monkeypatch):
    """record() from a sync context (no event loop) must not raise and must not
    create a queue/task — it silently drops."""
    monkeypatch.setitem(
        sys.modules, "config",
        types.ModuleType("config"),
    )
    sys.modules["config"].config = SimpleNamespace(TOOL_SIGNAL_RECORDER_ENABLED=True)

    recorder = ToolSignalRecorder()
    recorder.record(ToolSignal("b", True, agent_id=1, workspace_id="ws", prior_action="a"))

    assert recorder._queue is None
    assert recorder._drain_task is None


def test_record_disabled_is_noop(monkeypatch):
    """When the flag is off, record() does nothing even on an event loop."""
    monkeypatch.setitem(sys.modules, "config", types.ModuleType("config"))
    sys.modules["config"].config = SimpleNamespace(TOOL_SIGNAL_RECORDER_ENABLED=False)

    recorder = ToolSignalRecorder()

    async def _drive():
        recorder.record(ToolSignal("b", True, agent_id=1))
        return recorder._queue, recorder._drain_task

    q, t = asyncio.run(_drive())
    assert q is None and t is None
