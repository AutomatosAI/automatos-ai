"""PRD-177 S4 (F015): the gated prompt-catalog chain-hints path folds learned edges.

The default-live routing path (rank_chains under the default-true
SEMANTIC_TOOL_ROUTING flag) already reaches learned edges — that is NOT what
F015 is about and it is deliberately left alone. F015 is the narrow
prompt-catalog CHAIN-HINTS path gated behind TOOL_ROUTING_GRAPH (default false,
config.py). This test proves, end-to-end through the REAL GraphRouter (not a
stubbed rank_chains), that:

  1. a learned ``used_after`` edge is folded into a 2-action chain, and
  2. PlatformActionsSection renders that chain as a "Likely Platform Action
     Chains" hint when the flag is on, and
  3. with the flag off the graph path (and its hints) is not taken.

Pure — no DB, no Redis, no network. The GraphRouter is loaded via importlib with
a workspace-aware fake DB (same idiom as test_prd177_graph_router_tenant.py) so a
seeded used_after edge really flows through the ranking + chain-hint code.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

_THIS = Path(__file__).resolve()
_DISCOVERY = _THIS.parents[1] / "modules" / "tools" / "discovery"
_orchestrator_root = str(_THIS.parents[1])
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

# ---------------------------------------------------------------------------
# Fakes: workspace-aware edge/affinity DB + sqlalchemy stub (see tenant test)
# ---------------------------------------------------------------------------
_fake_db_mod = type(sys)("core.database.database")
_fake_cache_mod = type(sys)("core.cache.service")
_fake_tr_mod = type(sys)("core.models.tool_routing")


class _Col:
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
    intent_cluster_id = _Col("intent_cluster_id")  # PRD-232 US-010b

    def __init__(self, **kw):
        for k, v in kw.items():
            object.__setattr__(self, k, v)


_fake_tr_mod.ToolRoutingEdge = _StubEdge
_fake_tr_mod.ToolRoutingAffinity = _StubAffinity

import sqlalchemy as _real_sa  # noqa: E402

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


def _eval_pred(pred, row):
    """Recursively evaluate an and_/or_ predicate TREE against a stub row.

    Enforces workspace_id AND edge_type — the dimensions PRD-232 US-004 couples
    in ``or_(ws == A, and_(ws IS NULL, edge_type == meta_sibling))`` — with real
    OR/AND semantics. A flatten-to-AND pass would wrongly reject a tenant's own
    used_after edge against that OR filter. Other leaves are treated as satisfied.
    """
    if isinstance(pred, tuple) and pred:
        if pred[0] == "and_":
            return all(_eval_pred(s, row) for s in pred[1])
        if pred[0] == "or_":
            return any(_eval_pred(s, row) for s in pred[1])
        op, name = pred[0], pred[1]
        if name not in ("workspace_id", "edge_type"):
            return True
        rv = getattr(row, name, None)
        if op == "eq":
            return rv == pred[2]
        if op == "is":
            return rv is pred[2] or rv == pred[2]
        if op == "in":
            return rv in pred[2]
    return True


class _FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters = []

    def filter(self, *args, **kw):
        self._filters.extend(args)
        return self

    def order_by(self, *a, **k):
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


class _FakeCache:
    class _R:
        def get(self, *a, **k):
            return None

        def setex(self, *a, **k):
            pass

    redis = _R()


_fake_cache_mod.get_cache_service = lambda: _FakeCache()


@contextmanager
def _default_ctx():
    yield _FakeDBSession([], [])


_fake_db_mod.get_db_session = _default_ctx


class _FakeSemanticIndex:
    def __init__(self, results):
        self._results = results

    async def rank_actions(self, query, top_k=5, **kw):
        return self._results[:top_k]


# ---------------------------------------------------------------------------
# Load real GraphRouter
# ---------------------------------------------------------------------------
_ar_spec = importlib.util.spec_from_file_location(
    "action_registry_gr177_s4", _DISCOVERY / "action_registry.py"
)
_ar_mod = importlib.util.module_from_spec(_ar_spec)
_ar_spec.loader.exec_module(_ar_mod)
_ar_mod._registry_instance = _ar_mod.ActionRegistry()
_ar_mod._registry_instance._initialized = True

_pkg = "gr177_s4_pkg"
_p = type(sys)(_pkg)
_p.__path__ = [str(_DISCOVERY)]
sys.modules[_pkg] = _p
sys.modules[f"{_pkg}.action_registry"] = _ar_mod

_asi_spec = importlib.util.spec_from_file_location(
    f"{_pkg}.action_semantic_index", _DISCOVERY / "action_semantic_index.py"
)
_asi_mod = importlib.util.module_from_spec(_asi_spec)
_asi_mod.__package__ = _pkg
sys.modules[f"{_pkg}.action_semantic_index"] = _asi_mod
_asi_mod.get_action_semantic_index = lambda: None

_gr_spec = importlib.util.spec_from_file_location(
    f"{_pkg}.graph_router", _DISCOVERY / "graph_router.py"
)
_gr_mod = importlib.util.module_from_spec(_gr_spec)
_gr_mod.__package__ = _pkg
sys.modules[f"{_pkg}.graph_router"] = _gr_mod
_gr_spec.loader.exec_module(_gr_mod)
GraphRouter = _gr_mod.GraphRouter


def _learned_edge(from_a, to_a, ws=None, conf=0.85):
    return _StubEdge(
        from_action=from_a, to_action=to_a, edge_type="used_after",
        confidence=conf, weight=8.0, agent_id=None, workspace_id=ws, sample_count=12,
    )


def _router(entry_nodes):
    r = GraphRouter.__new__(GraphRouter)
    r._semantic_index = _FakeSemanticIndex(entry_nodes)
    r._get_cache = lambda: _FakeCache()
    return r


def _rank(router, edges, workspace_id=None):
    @contextmanager
    def _ctx():
        yield _FakeDBSession(edges, [])

    orig_min = GraphRouter._min_confidence
    orig_floor = GraphRouter._agent_sample_floor
    orig_db = _fake_db_mod.get_db_session
    GraphRouter._min_confidence = staticmethod(lambda: 0.6)
    GraphRouter._agent_sample_floor = staticmethod(lambda: 50)
    _fake_db_mod.get_db_session = _ctx
    try:
        return asyncio.run(
            router.rank_chains("submit a report", workspace_id=workspace_id, agent_id=None, top_k=10)
        )
    finally:
        GraphRouter._min_confidence = orig_min
        GraphRouter._agent_sample_floor = orig_floor
        _fake_db_mod.get_db_session = orig_db


@pytest.fixture(autouse=True)
def _swap():
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_chain_hints_use_learned_edges():
    """A learned used_after edge is folded into a 2-action chain by the REAL
    GraphRouter — the exact input the prompt-catalog chain-hints block renders."""
    entry = [("platform_get_latest_report", 0.9)]
    edges = [_learned_edge("platform_get_latest_report", "platform_submit_report", ws="ws-1")]

    chains = _rank(_router(entry), edges, workspace_id="ws-1")

    # The learned edge produced a multi-action chain (not just the bare entry).
    multi = [c for c in chains if len(c[2]) > 1]
    assert multi, "the learned used_after edge must produce a 2-action chain"
    assert ["platform_get_latest_report", "platform_submit_report"] in [c[2] for c in multi]


def test_chain_hints_render_block_from_learned_chain():
    """PlatformActionsSection._build_chain_hints turns that learned chain into the
    '## Likely Platform Action Chains' block the prompt catalog injects."""
    # Load the section's pure hint renderer without its heavy render() deps.
    from modules.context.sections.platform_actions import PlatformActionsSection

    section = PlatformActionsSection()
    chains = [
        ("platform_get_latest_report", 0.92,
         ["platform_get_latest_report", "platform_submit_report"]),
        ("platform_list_agents", 0.8, ["platform_list_agents"]),  # single — no hint
    ]
    hints = section._build_chain_hints(chains)
    assert "## Likely Platform Action Chains" in hints
    assert "`platform_get_latest_report` then `platform_submit_report`" in hints
    # single-action chains never become hints
    assert "platform_list_agents" not in hints


def test_flag_off_skips_graph_chain_hints():
    """With TOOL_ROUTING_GRAPH off, the section's graph gate is closed — the
    chain-hints path is not taken (default behavior unchanged)."""
    from modules.context.sections.platform_actions import PlatformActionsSection

    section = PlatformActionsSection()

    class _Cfg:
        SEMANTIC_TOOL_ROUTING = True
        TOOL_ROUTING_GRAPH = False

    import config as _config_mod
    orig = _config_mod.config
    _config_mod.config = _Cfg()
    try:
        assert section._graph_routing_enabled() is False
    finally:
        _config_mod.config = orig
