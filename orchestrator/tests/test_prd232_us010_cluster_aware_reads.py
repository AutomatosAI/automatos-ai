"""PRD-232 US-010: Cluster-aware reads — the graph expresses what it learned.

The deep review (C4) found the learned structures lobotomized at read time:
  * intent clusters were write-only — created nightly, never queried live;
  * ``_query_affinities`` summed succeeds/fails across EVERY intent (cluster-blind),
    so an action that fails for intent X but succeeds for intent Y looked neutral;
  * ``failed_after`` edges were written and never read (a write-only path).

This suite proves US-010 wires the read side honest, end to end:

  (a) ``rank_chains`` embeds the query, matches the nearest ToolRoutingIntentCluster
      by centroid cosine over a config threshold, and its ``action_names_hot`` join
      the entry candidates (a miss = embedding floor only);
  (b) ``_query_affinities`` gains the ``intent_cluster_id`` predicate, so a matched
      cluster's succeeds/fails apply PER-INTENT at full weight while cluster-blind
      (IS NULL) rows apply only as a weak, discounted global prior;
  (c) ``failed_after`` edges are read as an expansion PENALTY — no write-only tables.

Pure unit test — no Redis, no DB, no OpenRouter. ``graph_router.py`` is leaf-loaded
via importlib (bypassing ``modules.tools.__init__``'s heavy chain), every dependency
is a deterministic fake, and the fake DB faithfully evaluates the and_/or_ predicate
trees the real queries emit — including the ``intent_cluster_id`` and ``edge_type``
leaves that are the whole point of the fix. Embeddings are hand-built 3-vectors so
cluster cosine is exact and inspectable; no embedding provider is involved.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from unittest.mock import MagicMock

import pytest

_THIS = Path(__file__).resolve()
_DISCOVERY = _THIS.parents[1] / "modules" / "tools" / "discovery"

_MODEL = "test:model:3"
# A concrete tenant for the integration tests. US-004 admits a NULL-workspace
# used_after/failed_after row only as a meta_sibling global, so a rank_chains
# integration test must read as a real workspace to see its own learned edges.
_WS = "11111111-1111-1111-1111-111111111111"


# ---------------------------------------------------------------------------
# Column + row stubs (faithful to the real ToolRouting* models)
# ---------------------------------------------------------------------------
class _Col:
    """SQLAlchemy column-descriptor stub. Operators return inspectable tuples."""

    def __init__(self, name):
        self.name = name

    def in_(self, values):
        return ("in", self.name, list(values))

    def is_(self, value):
        return ("is", self.name, value)

    def __eq__(self, other):
        return ("eq", self.name, other)

    def __ge__(self, other):
        return ("ge", self.name, other)

    def desc(self):
        return ("desc", self.name)


class _StubEdge:
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
    action_name = _Col("action_name")
    agent_id = _Col("agent_id")
    workspace_id = _Col("workspace_id")
    affinity_type = _Col("affinity_type")
    intent_cluster_id = _Col("intent_cluster_id")

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


class _StubCluster:
    embedding_model_key = _Col("embedding_model_key")

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


_fake_tr_mod = type(sys)("core.models.tool_routing")
_fake_tr_mod.ToolRoutingEdge = _StubEdge
_fake_tr_mod.ToolRoutingAffinity = _StubAffinity
_fake_tr_mod.ToolRoutingIntentCluster = _StubCluster


# ---------------------------------------------------------------------------
# Fake sqlalchemy capturing and_/or_ predicate trees
# ---------------------------------------------------------------------------
import sqlalchemy as _real_sa  # noqa: E402

_fake_sa = type(sys)("sqlalchemy")
_fake_sa.and_ = lambda *args: ("and_", list(args))
_fake_sa.or_ = lambda *args: ("or_", list(args))
_fake_sa.func = MagicMock()
_fake_sa.__path__ = _real_sa.__path__
_fake_sa.__file__ = _real_sa.__file__


# ---------------------------------------------------------------------------
# Predicate evaluation — faithful enough that the intent_cluster_id / edge_type
# isolation under test is genuinely exercised (not flattened away).
# ---------------------------------------------------------------------------
def _eval_pred(pred: Any, row: Any) -> bool:
    if isinstance(pred, tuple) and pred:
        tag = pred[0]
        if tag == "and_":
            return all(_eval_pred(s, row) for s in pred[1])
        if tag == "or_":
            return any(_eval_pred(s, row) for s in pred[1])
        op, name = pred[0], pred[1]
        row_val = getattr(row, name, None)
        val = pred[2] if len(pred) > 2 else None
        if op == "eq":
            return row_val == val
        if op == "is":
            return row_val is val or row_val == val
        if op == "in":
            return row_val in val
        if op == "ge":
            try:
                return row_val is not None and row_val >= val
            except TypeError:
                return False
    return True


class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: List[Any] = []

    def filter(self, *args, **kw):
        self._filters.extend(args)
        return self

    def order_by(self, *a, **kw):
        return self

    def limit(self, n):
        return self

    def scalar(self):
        return 0

    def all(self):
        return [r for r in self._rows if all(_eval_pred(f, r) for f in self._filters)]


class _FakeDBSession:
    def __init__(self, edges=None, affinities=None, clusters=None):
        self._edges = edges or []
        self._affinities = affinities or []
        self._clusters = clusters or []

    def query(self, model_or_func, *args):
        if not isinstance(model_or_func, type):
            return _FakeQuery([])
        name = getattr(model_or_func, "__name__", "")
        if "Edge" in name:
            return _FakeQuery(self._edges)
        if "Affinity" in name:
            return _FakeQuery(self._affinities)
        if "Cluster" in name:
            return _FakeQuery(self._clusters)
        return _FakeQuery([])


class _FakeRedis:
    def get(self, *a, **kw):
        return None

    def setex(self, *a, **kw):
        pass


class _FakeCache:
    redis = _FakeRedis()


_fake_cache_mod = type(sys)("core.cache.service")
_fake_cache_mod.get_cache_service = lambda: _FakeCache()

_fake_db_mod = type(sys)("core.database.database")


@contextmanager
def _default_db_ctx():
    yield _FakeDBSession()


_fake_db_mod.get_db_session = _default_db_ctx

_CORE_STUBS = {
    "core.database.database": _fake_db_mod,
    "core.cache.service": _fake_cache_mod,
    "core.models.tool_routing": _fake_tr_mod,
}
_CORE_PACKAGES = ["core", "core.database", "core.cache", "core.models"]


# ---------------------------------------------------------------------------
# Fake semantic index — exposes embed_query (the US-010 seam) + rank_actions
# ---------------------------------------------------------------------------
class _FakeSemanticIndex:
    def __init__(self, entry_nodes, query_vec, model_key=_MODEL):
        self._entry = entry_nodes
        self._vec = query_vec
        self._model_key = model_key
        # empty registry -> _drop_ineligible_chains finds no gated actions
        self._registry = SimpleNamespace(get_all=lambda: [])

    async def rank_actions(self, query, top_k=5, **kw):
        return self._entry[:top_k]

    async def embed_query(self, query, embed_timeout_s=None):
        return self._vec, self._model_key


# ---------------------------------------------------------------------------
# Load graph_router via importlib under a synthetic package
# ---------------------------------------------------------------------------
_pkg_name = "gr232_us010_pkg"
if _pkg_name not in sys.modules:
    _pkg = type(sys)(_pkg_name)
    _pkg.__path__ = [str(_DISCOVERY)]
    sys.modules[_pkg_name] = _pkg

_asi_name = f"{_pkg_name}.action_semantic_index"
if _asi_name not in sys.modules:
    _fake_asi = type(sys)(_asi_name)
    _fake_asi.get_action_semantic_index = lambda: None
    sys.modules[_asi_name] = _fake_asi

_gr_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.graph_router", _DISCOVERY / "graph_router.py"
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
_gr_mod.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.graph_router"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)

GraphRouter = _gr_mod.GraphRouter


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _swap_modules():
    saved = {"sqlalchemy": sys.modules.get("sqlalchemy")}
    sys.modules["sqlalchemy"] = _fake_sa
    for name, mod in _CORE_STUBS.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = mod
    for pkg in _CORE_PACKAGES:
        if pkg not in sys.modules:
            saved.setdefault(pkg, None)
            m = type(sys)(pkg)
            m.__path__ = []
            sys.modules[pkg] = m
    yield
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def _cluster(cid, centroid, hot, model=_MODEL):
    return _StubCluster(
        id=cid,
        centroid_embedding=centroid,
        embedding_model_key=model,
        action_names_hot=hot,
        sample_query="q",
        sample_count=10,
    )


def _aff(action, atype, cluster_id, weight=5.0, conf=0.9, ws=None, agent=None):
    return _StubAffinity(
        action_name=action,
        affinity_type=atype,
        weight=weight,
        confidence=conf,
        workspace_id=ws,
        agent_id=agent,
        intent_cluster_id=cluster_id,
        sample_count=10,
    )


def _edge(from_a, to_a, etype="used_after", conf=1.0, ws=None, agent=None):
    return _StubEdge(
        from_action=from_a,
        to_action=to_a,
        edge_type=etype,
        confidence=conf,
        weight=5.0,
        agent_id=agent,
        workspace_id=ws,
        sample_count=10,
    )


def _build_router(entry_nodes, query_vec, model_key=_MODEL):
    r = GraphRouter.__new__(GraphRouter)
    r._semantic_index = _FakeSemanticIndex(entry_nodes, query_vec, model_key)
    r._get_cache = lambda: _FakeCache()
    return r


def _rank(
    router,
    *,
    edges=None,
    affinities=None,
    clusters=None,
    workspace_id=None,
    min_conf=0.6,
    threshold=0.6,
    discount=0.5,
    failed_weight=1.0,
    top_k=15,
    exclude_admin=True,
    include_super_admin=False,
):
    @contextmanager
    def patched_db():
        yield _FakeDBSession(edges or [], affinities or [], clusters or [])

    orig = {
        "min": GraphRouter._min_confidence,
        "floor": GraphRouter._agent_sample_floor,
        "thr": GraphRouter._cluster_match_threshold,
        "disc": GraphRouter._global_affinity_discount,
        "fw": GraphRouter._failed_after_penalty_weight,
        "db": _fake_db_mod.get_db_session,
    }
    GraphRouter._min_confidence = staticmethod(lambda: min_conf)
    GraphRouter._agent_sample_floor = staticmethod(lambda: 50)
    GraphRouter._cluster_match_threshold = staticmethod(lambda: threshold)
    GraphRouter._global_affinity_discount = staticmethod(lambda: discount)
    GraphRouter._failed_after_penalty_weight = staticmethod(lambda: failed_weight)
    _fake_db_mod.get_db_session = patched_db
    try:
        return asyncio.run(
            router.rank_chains(
                query="q",
                workspace_id=workspace_id,
                agent_id=None,
                top_k=top_k,
                exclude_admin=exclude_admin,
                include_super_admin=include_super_admin,
            )
        )
    finally:
        GraphRouter._min_confidence = orig["min"]
        GraphRouter._agent_sample_floor = orig["floor"]
        GraphRouter._cluster_match_threshold = orig["thr"]
        GraphRouter._global_affinity_discount = orig["disc"]
        GraphRouter._failed_after_penalty_weight = orig["fw"]
        _fake_db_mod.get_db_session = orig["db"]


def _score_by_actions(result):
    return {frozenset(actions): score for _p, score, actions in result}


# ===========================================================================
# AC1 — per-cluster affinity: same action ranks DOWN for X, UP for Y
# ===========================================================================
def test_per_cluster_affinity_ranks_down_for_x_up_for_y():
    """The headline C4 fix. ``platform_update_task_status`` FAILS for intent X and
    SUCCEEDS for intent Y. A query that lands in cluster X must de-rank its chain;
    the same action for a cluster-Y query must be boosted. If affinities were still
    summed cluster-blind (the bug), both queries would net zero and neither
    assertion below could hold."""
    A = "platform_update_task_status"
    entry = [(A, 0.4), ("control_action", 0.4)]
    edges = [_edge(A, "platform_list_tasks", ws=_WS), _edge("control_action", "control_next", ws=_WS)]
    clusters = [
        _cluster(1, [1.0, 0.0, 0.0], [A]),  # intent X
        _cluster(2, [0.0, 1.0, 0.0], [A]),  # intent Y
    ]
    affinities = [
        _aff(A, "fails_for_intent", cluster_id=1, ws=_WS),      # penalty 5*0.9 = 4.5 for X
        _aff(A, "succeeds_for_intent", cluster_id=2, ws=_WS),   # boost   5*0.9 = 4.5 for Y
    ]

    sx = _score_by_actions(
        _rank(_build_router(entry, [1.0, 0.0, 0.0]), edges=edges, affinities=affinities,
              clusters=clusters, workspace_id=_WS)
    )
    sy = _score_by_actions(
        _rank(_build_router(entry, [0.0, 1.0, 0.0]), edges=edges, affinities=affinities,
              clusters=clusters, workspace_id=_WS)
    )

    chain = frozenset([A, "platform_list_tasks"])
    control = frozenset(["control_action", "control_next"])
    assert chain in sx and chain in sy
    # X-query: the action that fails for this intent ranks BELOW the neutral control.
    assert sx[chain] < sx[control]
    # Y-query: the same action, succeeding for this intent, ranks ABOVE the control.
    assert sy[chain] > sy[control]
    # Same action, opposite direction — driven purely by which cluster matched.
    assert sy[chain] > sx[chain]


def test_query_affinities_isolates_per_cluster_with_discounted_global(monkeypatch):
    """``_query_affinities`` reads the matched cluster's rows at full weight, the
    cluster-blind (IS NULL) row as a discounted global prior, and NEVER another
    cluster's rows."""
    monkeypatch.setattr(GraphRouter, "_global_affinity_discount", staticmethod(lambda: 0.5))
    affinities = [
        _aff("A", "fails_for_intent", cluster_id=1),                      # 5*0.9 = 4.5
        _aff("A", "succeeds_for_intent", cluster_id=2),                   # must be excluded
        _aff("A", "succeeds_for_intent", cluster_id=None, weight=2.0, conf=1.0),  # global prior
    ]
    db = _FakeDBSession(affinities=affinities)

    pos, neg = GraphRouter._query_affinities(db, ["A"], None, None, 1)

    assert neg.get("A") == pytest.approx(4.5)         # per-intent fail, full weight
    assert pos.get("A") == pytest.approx(1.0)         # global prior 2.0*1.0 * 0.5 discount
    # cluster 2's succeeds (2.0*... would have been 0) never leaks in — proven by
    # pos being exactly the discounted global, with no cluster-2 contribution.


def test_no_cluster_id_reads_only_cluster_blind_rows():
    """With no matched cluster, only intent_cluster_id IS NULL rows apply (exact
    legacy behaviour) — a cluster-scoped row must not leak into a cluster-blind read."""
    affinities = [
        _aff("A", "succeeds_for_intent", cluster_id=7, weight=5.0, conf=1.0),   # cluster-scoped
        _aff("A", "succeeds_for_intent", cluster_id=None, weight=1.0, conf=1.0),  # global
    ]
    db = _FakeDBSession(affinities=affinities)
    pos, _neg = GraphRouter._query_affinities(db, ["A"], None, None)  # no cluster id
    assert pos.get("A") == pytest.approx(1.0)  # only the IS NULL row, at full weight


# ===========================================================================
# AC2 — cluster-match threshold from config; miss falls back cleanly
# ===========================================================================
def test_cluster_miss_falls_back_to_embedding_floor():
    """A query far from every centroid matches NO cluster, so the cluster-scoped
    affinity does not apply and routing stays at the embedding floor."""
    A = "platform_update_task_status"
    entry = [(A, 0.4)]
    edges = [_edge(A, "platform_list_tasks", ws=_WS)]
    clusters = [_cluster(1, [1.0, 0.0, 0.0], [A])]
    affinities = [_aff(A, "fails_for_intent", cluster_id=1, ws=_WS)]

    s = _score_by_actions(
        _rank(
            _build_router(entry, [0.0, 0.0, 1.0]),  # orthogonal to the only centroid
            edges=edges,
            affinities=affinities,
            clusters=clusters,
            threshold=0.6,
            workspace_id=_WS,
        )
    )
    chain = frozenset([A, "platform_list_tasks"])
    assert s[chain] == pytest.approx(0.4)  # 0.4*1.0, unpenalized — no cluster matched


def test_cluster_match_threshold_is_read_from_config():
    """The SAME query matches above a low threshold and misses above a high one —
    proving the cutoff is the config value, not a hardcode."""
    A = "platform_update_task_status"
    entry = [(A, 0.4)]
    edges = [_edge(A, "platform_list_tasks", ws=_WS)]
    clusters = [_cluster(1, [1.0, 0.0, 0.0], [A])]
    affinities = [_aff(A, "succeeds_for_intent", cluster_id=1, weight=2.0, conf=1.0, ws=_WS)]  # boost 2.0
    vec = [1.0, 1.0, 0.0]  # cosine to [1,0,0] = 1/sqrt(2) ≈ 0.707
    chain = frozenset([A, "platform_list_tasks"])

    matched = _score_by_actions(
        _rank(_build_router(entry, vec), edges=edges, affinities=affinities,
              clusters=clusters, threshold=0.6, workspace_id=_WS)
    )
    missed = _score_by_actions(
        _rank(_build_router(entry, vec), edges=edges, affinities=affinities,
              clusters=clusters, threshold=0.8, workspace_id=_WS)
    )
    assert matched[chain] == pytest.approx(2.4)  # 0.4 + 2.0 boost (0.707 >= 0.6)
    assert missed[chain] == pytest.approx(0.4)   # no boost (0.707 < 0.8)


def test_match_intent_cluster_picks_nearest_over_threshold():
    """The pure matcher: nearest centroid over the threshold, None on a miss or a
    non-matching embedding model."""
    clusters = [
        _cluster(1, [1.0, 0.0, 0.0], ["a"]),
        _cluster(2, [0.0, 1.0, 0.0], ["b"]),
    ]
    db = _FakeDBSession(clusters=clusters)

    match = GraphRouter._match_intent_cluster(db, [0.1, 0.9, 0.0], _MODEL, 0.6)
    assert match is not None
    assert match[0] == 2 and match[1] == ["b"]  # nearest is cluster 2

    # Same vector, threshold above its best similarity -> clean miss.
    assert GraphRouter._match_intent_cluster(db, [0.1, 0.9, 0.0], _MODEL, 0.999) is None
    # Centroids under a different embedding model are not comparable -> no candidates.
    assert GraphRouter._match_intent_cluster(db, [0.0, 1.0, 0.0], "other:model:3", 0.6) is None


# ===========================================================================
# AC2(a) — cluster hot actions join the entry candidates
# ===========================================================================
def test_cluster_hot_actions_enter_the_surface():
    """The right action arrives via the matched cluster even when the cosine
    entry ranking missed it — action_names_hot join the entry candidates."""
    entry = [("control_action", 0.4)]
    clusters = [_cluster(1, [1.0, 0.0, 0.0], ["platform_update_task_status"])]

    result = _rank(_build_router(entry, [1.0, 0.0, 0.0]), clusters=clusters)
    names = set()
    for _p, _s, actions in result:
        names.update(actions)

    assert "platform_update_task_status" in names, "cluster hot action must join the surface"
    assert "control_action" in names


def test_cluster_hot_admin_action_dropped_for_operator_but_kept_for_admin():
    """P232-RVW-4 (fail-closed): a matched cluster's action_names_hot can carry an
    admin_only action — organic clusters are built from cross-tenant aggregate logs
    and the US-007 seed excludes su but NOT admin. _merge_cluster_hot_actions adds
    them with no role check, so the final _drop_ineligible_chains net must enforce
    the caller's exclude_admin; otherwise rank_chains(exclude_admin=True) leaks an
    admin tool the moment TOOL_ROUTING_GRAPH flips on. super_admin exclusion stays
    unchanged."""
    admin_hot = "platform_admin_only_hot"
    su_hot = "platform_su_only_hot"
    op_hot = "platform_operator_hot"
    # The registry the drop net resolves roles from (the fake index's default is
    # empty; give it a role-carrying one).
    registry = SimpleNamespace(get_all=lambda: [
        SimpleNamespace(name=admin_hot, admin_only=True, super_admin_only=False),
        SimpleNamespace(name=su_hot, admin_only=False, super_admin_only=True),
        SimpleNamespace(name=op_hot, admin_only=False, super_admin_only=False),
        SimpleNamespace(name="platform_entry", admin_only=False, super_admin_only=False),
    ])
    # Query vector hits the cluster centroid exactly (cosine 1.0 > threshold 0.6),
    # so all three hot actions merge into the entry candidates.
    query_vec = [1.0, 0.0, 0.0]
    cluster = _cluster(7, [1.0, 0.0, 0.0], [admin_hot, su_hot, op_hot])

    # Operator caller (exclude_admin=True, include_super_admin=False): the admin AND
    # su hot actions must both be dropped; the operator hot action still surfaces.
    router = _build_router([("platform_entry", 0.9)], query_vec)
    router._semantic_index._registry = registry
    surfaced = {a for _p, _s, actions in _rank(router, clusters=[cluster]) for a in actions}
    assert admin_hot not in surfaced, "admin_only cluster hot action leaked to an operator (exclude_admin=True)"
    assert su_hot not in surfaced, "super_admin_only cluster hot action leaked (fail-closed regression)"
    assert op_hot in surfaced, "operator hot action must still surface"

    # Admin caller (exclude_admin=False): the admin hot action IS allowed; su is
    # still dropped — proving exclude_admin gates it, not a blanket drop.
    router2 = _build_router([("platform_entry", 0.9)], query_vec)
    router2._semantic_index._registry = registry
    surfaced2 = {a for _p, _s, actions in _rank(router2, clusters=[cluster], exclude_admin=False) for a in actions}
    assert admin_hot in surfaced2, "admin caller (exclude_admin=False) must see the admin action"
    assert su_hot not in surfaced2, "su still dropped for a non-su admin caller"


# ===========================================================================
# AC3 — failed_after is CONSUMED as a penalty (no write-only paths)
# ===========================================================================
def test_query_failed_after_reads_only_failure_edges():
    """``_query_failed_after`` returns {(from, to): confidence} for failed_after
    edges and never treats a used_after row as a failure."""
    edges = [
        _edge("good", "next", etype="used_after", conf=1.0),
        _edge("good", "bad", etype="failed_after", conf=0.8),
    ]
    db = _FakeDBSession(edges=edges)

    penalties = GraphRouter._query_failed_after(db, ["good"], 0.6, None, None)
    assert penalties == {("good", "bad"): 0.8}
    assert ("good", "next") not in penalties


def test_failed_after_penalizes_chain_score():
    """A chain whose transition has a learned failed_after edge is de-ranked by
    confidence * penalty weight."""
    entry = [("good", 0.9)]
    edges = [
        _edge("good", "bad", etype="used_after", conf=1.0, ws=_WS),    # builds the [good, bad] chain
        _edge("good", "bad", etype="failed_after", conf=0.8, ws=_WS),  # de-ranks it
    ]
    s = _score_by_actions(
        _rank(_build_router(entry, [1.0, 0.0, 0.0]), edges=edges, clusters=[],
              failed_weight=1.0, workspace_id=_WS)
    )
    chain = frozenset(["good", "bad"])
    # 0.9 * 1.0 (used_after) - 0.8 * 1.0 (failed_after penalty) = 0.1
    assert s[chain] == pytest.approx(0.1)


def test_failed_after_penalty_weight_scales_the_penalty():
    """The penalty magnitude is the config weight — halving it halves the drop."""
    entry = [("good", 0.9)]
    edges = [
        _edge("good", "bad", etype="used_after", conf=1.0, ws=_WS),
        _edge("good", "bad", etype="failed_after", conf=0.8, ws=_WS),
    ]
    chain = frozenset(["good", "bad"])
    half = _score_by_actions(
        _rank(_build_router(entry, [1.0, 0.0, 0.0]), edges=edges, failed_weight=0.5, workspace_id=_WS)
    )
    # 0.9 - 0.8*0.5 = 0.5
    assert half[chain] == pytest.approx(0.5)


def test_failed_after_is_consumed_not_write_only():
    """Grep proof (US-010c): graph_router now has a failed_after READ path that the
    expansion applies as a penalty — the previously write-only edges are consumed."""
    src = (_DISCOVERY / "graph_router.py").read_text()
    assert "_query_failed_after" in src                # dedicated read method exists
    assert 'edge_type == "failed_after"' in src        # it queries the failure edges
    assert "failed_penalties" in src                   # expansion applies them
    assert "- failed_pen" in src                       # subtracted from the chain score


# ===========================================================================
# Regression — tuple shape + no-cluster path unchanged
# ===========================================================================
def test_result_shape_and_no_cluster_path_is_unchanged():
    """With no clusters, rank_chains behaves exactly as before: single + edge
    chains with the (primary, score, [actions]) shape and cosine-only scoring."""
    entry = [("platform_a", 0.9), ("platform_b", 0.5)]
    edges = [_edge("platform_a", "platform_c", conf=0.8, ws=_WS)]
    result = _rank(_build_router(entry, [1.0, 0.0, 0.0]), edges=edges, clusters=[], workspace_id=_WS)

    for primary, score, actions in result:
        assert isinstance(primary, str)
        assert isinstance(score, float)
        assert isinstance(actions, list) and actions
    s = _score_by_actions(result)
    assert s[frozenset(["platform_a", "platform_c"])] == pytest.approx(0.9 * 0.8)
