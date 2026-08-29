"""PRD-177 S5: Per-tenant workspace isolation on GraphRouter edge reads.

The learned operating graph is PER-TENANT (owner decision, locked 2026-07-03).
GraphRouter edge/affinity reads MUST filter by ``workspace_id`` so workspace A
never sees workspace B's learned edges, and there is NO unfiltered global-read
fallback that bleeds one tenant's edges into another's routing.

Pure unit test — no Redis, no DB, no OpenRouter. graph_router.py is loaded
directly via importlib (bypassing modules.tools.__init__), and every dependency
is a deterministic fake. The fake DB session is *workspace-aware*: unlike the
no-op filter in test_graph_router.py, it parses the workspace_id predicate that
_query_edges / _query_affinities emit and only returns rows for that workspace.

Validates:
  * rank_chains(..., workspace_id=A) never returns B's edges (and vice versa).
  * A required workspace_id keyword forces every caller to make a tenant choice.
  * workspace_id IS NULL (unscoped/global rows) never returns a tenant's rows.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

_THIS = Path(__file__).resolve()
_DISCOVERY = _THIS.parents[1] / "modules" / "tools" / "discovery"

# ---------------------------------------------------------------------------
# Stub heavy platform modules that graph_router lazily imports
# ---------------------------------------------------------------------------
_fake_db_mod = type(sys)("core.database.database")
_fake_cache_mod = type(sys)("core.cache.service")
_fake_tr_mod = type(sys)("core.models.tool_routing")


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

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


_fake_tr_mod.ToolRoutingEdge = _StubEdge
_fake_tr_mod.ToolRoutingAffinity = _StubAffinity

# ---------------------------------------------------------------------------
# Fake sqlalchemy that captures and_/or_ predicate trees so the workspace-aware
# fake DB can evaluate them.
# ---------------------------------------------------------------------------
import sqlalchemy as _real_sa  # noqa: E402
import sqlalchemy.orm as _real_sa_orm  # noqa: E402

_fake_sa = type(sys)("sqlalchemy")
_fake_sa.and_ = lambda *args: ("and_", list(args))
_fake_sa.or_ = lambda *args: ("or_", list(args))
_fake_sa.func = MagicMock()
_fake_sa.__path__ = _real_sa.__path__
_fake_sa.__file__ = _real_sa.__file__

_CORE_STUBS = {
    "core.database.database": _fake_db_mod,
    "core.cache.service": _fake_cache_mod,
    "core.models.tool_routing": _fake_tr_mod,
}
_CORE_PACKAGES = ["core", "core.database", "core.cache", "core.models"]


# ---------------------------------------------------------------------------
# Predicate evaluation — the crux of tenant isolation testing.
# ---------------------------------------------------------------------------

def _eval_pred(pred: Any, row: Any) -> bool:
    """Recursively evaluate an and_/or_ predicate TREE against a stub row.

    Enforces the two dimensions under test — ``workspace_id`` AND ``edge_type``
    (PRD-232 US-004 admits NULL-workspace rows ONLY for ``meta_sibling``) — and
    treats every other leaf (from_action.in_, confidence, agent_id) as satisfied
    so the test stays focused on the tenant + type reconciliation. Honouring real
    OR/AND semantics is the point: a flatten-to-AND pass (the earlier
    ``_row_matches``) would reject workspace A's own row against the US-004
    ``or_(ws == A, and_(ws IS NULL, meta_sibling))`` filter.
    """
    if isinstance(pred, tuple) and pred:
        if pred[0] == "and_":
            return all(_eval_pred(s, row) for s in pred[1])
        if pred[0] == "or_":
            return any(_eval_pred(s, row) for s in pred[1])
        op, name = pred[0], pred[1]
        if name not in ("workspace_id", "edge_type"):
            return True  # not under test — treat as satisfied
        row_val = getattr(row, name, None)
        if op == "eq":
            return row_val == pred[2]
        if op == "is":
            return row_val is pred[2] or row_val == pred[2]
        if op == "in":
            return row_val in pred[2]
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
    def __init__(self, edges=None, affinities=None):
        self._edges = edges or []
        self._affinities = affinities or []

    def query(self, model_or_func, *args):
        if not isinstance(model_or_func, type):
            return _FakeQuery([])
        name = getattr(model_or_func, "__name__", "")
        if "Edge" in name:
            return _FakeQuery(self._edges)
        if "Affinity" in name:
            return _FakeQuery(self._affinities)
        return _FakeQuery([])


class _FakeRedis:
    def get(self, *a, **kw):
        return None

    def setex(self, *a, **kw):
        pass


class _FakeCache:
    redis = _FakeRedis()


_fake_cache_mod.get_cache_service = lambda: _FakeCache()


@contextmanager
def _default_db_ctx():
    yield _FakeDBSession([], [])


# Baseline so lazy `from core.database.database import get_db_session` resolves
# even before a test swaps in its workspace-seeded session via _rank().
_fake_db_mod.get_db_session = _default_db_ctx


class _FakeSemanticIndex:
    def __init__(self, results):
        self._results = results

    async def rank_actions(self, query, top_k=5, **kw):
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Load graph_router via importlib
# ---------------------------------------------------------------------------
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_gr177_test", _DISCOVERY / "action_registry.py"
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
_ar_spec.loader.exec_module(_ar_mod)
_ar_mod._registry_instance = _ar_mod.ActionRegistry()
_ar_mod._registry_instance._initialized = True

_pkg_name = "gr177_test_pkg"
_pkg = type(sys)(_pkg_name)
_pkg.__path__ = [str(_DISCOVERY)]
sys.modules[_pkg_name] = _pkg
sys.modules[f"{_pkg_name}.action_registry"] = _ar_mod

_asi_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.action_semantic_index",
    _DISCOVERY / "action_semantic_index.py",
)
_asi_mod = importlib.util.module_from_spec(_asi_spec)
_asi_mod.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.action_semantic_index"] = _asi_mod
_asi_mod.get_action_semantic_index = lambda: None

_gr_spec = importlib.util.spec_from_file_location(
    f"{_pkg_name}.graph_router",
    _DISCOVERY / "graph_router.py",
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
_gr_mod.__package__ = _pkg_name
sys.modules[f"{_pkg_name}.graph_router"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)

GraphRouter = _gr_mod.GraphRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_WS_A = "11111111-1111-1111-1111-111111111111"
_WS_B = "22222222-2222-2222-2222-222222222222"


def _edge(from_action, to_action, workspace_id, confidence=0.8, edge_type="used_after"):
    return _StubEdge(
        from_action=from_action,
        to_action=to_action,
        edge_type=edge_type,
        confidence=confidence,
        weight=5.0,
        agent_id=None,
        workspace_id=workspace_id,
        sample_count=10,
    )


def _build_router(entry_nodes):
    router = GraphRouter.__new__(GraphRouter)
    router._semantic_index = _FakeSemanticIndex(entry_nodes)
    router._get_cache = lambda: _FakeCache()
    return router


def _rank(router, workspace_id, edges=None, top_k=15):
    @contextmanager
    def patched_db():
        yield _FakeDBSession(edges or [], [])

    orig_min = GraphRouter._min_confidence
    orig_floor = GraphRouter._agent_sample_floor
    orig_db = _fake_db_mod.get_db_session
    GraphRouter._min_confidence = staticmethod(lambda: 0.6)
    GraphRouter._agent_sample_floor = staticmethod(lambda: 50)
    _fake_db_mod.get_db_session = patched_db
    try:
        return asyncio.run(
            router.rank_chains(
                query="test query",
                workspace_id=workspace_id,
                agent_id=None,
                top_k=top_k,
            )
        )
    finally:
        GraphRouter._min_confidence = orig_min
        GraphRouter._agent_sample_floor = orig_floor
        _fake_db_mod.get_db_session = orig_db


@pytest.fixture(autouse=True)
def _swap_modules():
    saved = {}
    for name in ["sqlalchemy"]:
        saved[name] = sys.modules.get(name)
    sys.modules["sqlalchemy"] = _fake_sa
    saved_orm = sys.modules.get("sqlalchemy.orm")
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
    if saved_orm is not None:
        sys.modules["sqlalchemy.orm"] = saved_orm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _chain_actions(result: List[Tuple[str, float, List[str]]]) -> set:
    names = set()
    for _primary, _score, actions in result:
        names.update(actions)
    return names


def test_rank_chains_requires_workspace_id():
    """workspace_id is a required keyword — calling without it is an error.

    A default (e.g. workspace_id=None-by-omission) would silently reintroduce
    the global read. Forcing the keyword makes every caller declare its tenant.
    """
    router = _build_router([("platform_list_agents", 0.9)])
    with pytest.raises(TypeError):
        asyncio.run(router.rank_chains(query="x", agent_id=None, top_k=5))


def test_graph_router_tenant_isolation():
    """Edges seeded in workspace A and B: a query scoped to A never returns B's
    edge target, and a query scoped to B never returns A's."""
    entry = [("send_email", 0.9)]
    edges = [
        _edge("send_email", "A_ONLY_FOLLOWUP", _WS_A),
        _edge("send_email", "B_ONLY_FOLLOWUP", _WS_B),
    ]

    result_a = _rank(_build_router(entry), workspace_id=_WS_A, edges=edges)
    names_a = _chain_actions(result_a)
    assert "A_ONLY_FOLLOWUP" in names_a, "workspace A must see its own learned edge"
    assert "B_ONLY_FOLLOWUP" not in names_a, "cross-tenant leak: A saw B's edge"

    result_b = _rank(_build_router(entry), workspace_id=_WS_B, edges=edges)
    names_b = _chain_actions(result_b)
    assert "B_ONLY_FOLLOWUP" in names_b, "workspace B must see its own learned edge"
    assert "A_ONLY_FOLLOWUP" not in names_b, "cross-tenant leak: B saw A's edge"


def test_null_workspace_rows_never_leak_to_a_tenant():
    """A pre-tenant / unscoped (workspace_id IS NULL) ``used_after`` edge must not
    surface for a specific tenant's read — learned co-occurrence is per-tenant,
    not global (PRD-232 US-004 admits NULL rows only for meta_sibling)."""
    entry = [("send_email", 0.9)]
    edges = [_edge("send_email", "GLOBAL_FOLLOWUP", None)]  # used_after, unscoped

    result_a = _rank(_build_router(entry), workspace_id=_WS_A, edges=edges)
    names_a = _chain_actions(result_a)
    assert "GLOBAL_FOLLOWUP" not in names_a, (
        "unscoped global used_after edge leaked into a tenant read"
    )


# ---------------------------------------------------------------------------
# PRD-232 US-004 — reconcile global meta_sibling bootstrap seeds with the lock
# ---------------------------------------------------------------------------

def test_global_meta_sibling_seed_surfaces_for_every_tenant():
    """PRD-143's metadata_graph_seed writes GLOBAL (workspace_id IS NULL)
    meta_sibling cold-start edges. US-004 admits those for EVERY tenant read at
    the confidence floor, so a zero-telemetry workspace is still graph-reachable —
    while used_after globals stay excluded (previous test)."""
    entry = [("send_email", 0.9)]
    edges = [_edge("send_email", "META_COLD_START", None, edge_type="meta_sibling")]

    for ws in (_WS_A, _WS_B):
        names = _chain_actions(_rank(_build_router(entry), workspace_id=ws, edges=edges))
        assert "META_COLD_START" in names, (
            f"global meta_sibling seed did not surface for {ws} (cold-start unreachable)"
        )


def test_tenant_scoped_meta_sibling_stays_tenant_scoped():
    """A meta_sibling edge that DOES carry a workspace_id is still tenant-scoped —
    only the unscoped (NULL) meta_sibling globals cross tenants."""
    entry = [("send_email", 0.9)]
    edges = [_edge("send_email", "A_META_ONLY", _WS_A, edge_type="meta_sibling")]

    names_a = _chain_actions(_rank(_build_router(entry), workspace_id=_WS_A, edges=edges))
    names_b = _chain_actions(_rank(_build_router(entry), workspace_id=_WS_B, edges=edges))
    assert "A_META_ONLY" in names_a
    assert "A_META_ONLY" not in names_b, "tenant-scoped meta_sibling leaked to another tenant"
