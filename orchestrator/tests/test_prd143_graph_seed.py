"""
PRD-143 S12: one-shot tool-routing-graph seed backfill (authored, never prod-run).
==================================================================================

scripts/seed_tool_routing_graph.py drives edge_builder's recompute (it must NOT
reimplement edge math) to populate tool_routing_edges / tool_routing_affinities
from historical tool_execution_logs. These tests run it against an in-memory
fake DB store that simulates the Postgres ON CONFLICT upsert semantics keyed on
the real unique constraints (uq_tre_full_key / uq_tra_full_key), so idempotency
is asserted on actual row convergence, not on SQL text.

The GraphRouter tests leaf-load graph_router.py under a synthetic package with
a fake action_semantic_index (the idiom from test_graph_router_negative.py) and
point its lazy ``core.database.database`` import at the SAME fake store the
seed wrote to — proving a seeded intent routes through graph edges while an
unseeded intent falls back to the pure semantic floor.

Never touches a real database: get_db_session and get_embedding_manager are
monkeypatched at the importing modules' namespaces.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import os
import sys
import types
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

# Dummy creds satisfy the lazy engine config at import time (the established
# idiom from test_harness_self_management.py); no connection is ever opened.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

import core.services.edge_builder as edge_builder
import scripts.seed_tool_routing_graph as seed_mod
from core.models.composio_cache import ToolExecutionLog
from core.models.tool_routing import (
    ToolRoutingAffinity,
    ToolRoutingEdge,
    ToolRoutingIntentCluster,
)

WS_A = uuid.UUID("00000000-0000-4000-8000-00000000014a")
WS_B = uuid.UUID("00000000-0000-4000-8000-00000000014b")

QUERY_AGENTS = "show me my agents and open the first one"
QUERY_FILES = "read the project files and update them"
QUERY_UNSEEDED = "compress the deliverable folder"


# ---------------------------------------------------------------------------
# SQLAlchemy clause introspection (filters applied honestly by the fake store)
# ---------------------------------------------------------------------------


def _extract_triples(clause, out):
    """Flatten and_()/BinaryExpression clauses into (key, op, value) triples."""
    sub = getattr(clause, "clauses", None)
    if sub is not None:
        for c in sub:
            _extract_triples(c, out)
        return
    left = getattr(clause, "left", None)
    key = getattr(left, "key", None)
    if key is None:
        return
    op_name = getattr(getattr(clause, "operator", None), "__name__", "")
    value = getattr(getattr(clause, "right", None), "value", None)
    if op_name in ("is_", "is_op"):
        out.append((key, "is_null", None))
    elif op_name == "in_op":
        out.append((key, "in", list(value) if value is not None else []))
    elif op_name == "eq":
        out.append((key, "eq", value))
    elif op_name == "ge":
        out.append((key, "ge", value))


def _match(row, triples) -> bool:
    for key, op, value in triples:
        actual = getattr(row, key, None)
        if op == "is_null":
            if actual is not None:
                return False
        elif op == "in":
            if str(actual) not in {str(v) for v in value}:
                return False
        elif op == "eq":
            if actual is None or str(actual) != str(value):
                return False
        elif op == "ge":
            if actual is None or not actual >= value:
                return False
    return True


# ---------------------------------------------------------------------------
# Fake store: one object acts as the SQLAlchemy session for every module
# ---------------------------------------------------------------------------

_EDGE_KEY = ("from_action", "to_action", "edge_type", "workspace_id", "agent_id")
_AFF_KEY = ("action_name", "affinity_type", "workspace_id", "agent_id", "intent_cluster_id")


class _FakeQuery:
    def __init__(self, store, model):
        self._store = store
        self._model = model
        self._triples = []
        self._limit = None

    def filter(self, *clauses):
        for c in clauses:
            _extract_triples(c, self._triples)
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _rows(self):
        if self._model is ToolExecutionLog:
            rows = list(self._store.logs)
        elif self._model is ToolRoutingIntentCluster:
            rows = list(self._store.clusters)
        elif self._model is ToolRoutingEdge:
            rows = [SimpleNamespace(**v) for v in self._store.edges.values()]
        elif self._model is ToolRoutingAffinity:
            rows = [SimpleNamespace(**v) for v in self._store.affinities.values()]
        else:
            rows = []
        return [r for r in rows if _match(r, self._triples)]

    def all(self):
        rows = self._rows()
        if self._limit is not None:
            rows = rows[: self._limit]
        return rows

    def delete(self, synchronize_session=None):
        doomed = self._rows()
        if self._model is ToolRoutingIntentCluster:
            self._store.clusters = [c for c in self._store.clusters if c not in doomed]
            return len(doomed)
        if self._model is ToolRoutingAffinity:
            doomed_keys = {
                k for k, v in self._store.affinities.items()
                if _match(SimpleNamespace(**v), self._triples)
            }
            for k in doomed_keys:
                del self._store.affinities[k]
            return len(doomed_keys)
        return 0


class _FakeStore:
    """Session + storage in one: logs in, edges/affinities/clusters out."""

    def __init__(self):
        self.logs = []
        self.edges = {}        # uq_tre_full_key tuple -> params dict
        self.affinities = {}   # uq_tra_full_key tuple -> params dict
        self.clusters = []     # ToolRoutingIntentCluster ORM instances
        self.statements = []   # every execute() call, for no-write assertions
        self._next_cluster_id = 1

    def query(self, model, *a, **k):
        return _FakeQuery(self, model)

    def execute(self, stmt, params=None):
        sql = str(stmt)
        params = dict(params or {})
        self.statements.append((sql, params))
        if "tool_routing_edges" in sql:
            self.edges[tuple(params[c] for c in _EDGE_KEY)] = params
        elif "tool_routing_affinities" in sql:
            self.affinities[tuple(params[c] for c in _AFF_KEY)] = params
        return SimpleNamespace(rowcount=1)

    def add(self, obj):
        obj.id = self._next_cluster_id
        self._next_cluster_id += 1
        self.clusters.append(obj)

    def flush(self):
        pass


class _FakeEmbeddingManager:
    def __init__(self):
        self.batch_calls = 0

    async def generate_embeddings_batch(self, queries):
        self.batch_calls += 1
        return [_vec(q) for q in queries]

    def get_provider_info(self):
        return {"provider": "fake", "model": "fake-embed", "dimension": 8}

    def get_dimension(self):
        return 8


def _vec(query: str):
    digest = hashlib.sha256(query.encode()).digest()
    return [b / 255.0 for b in digest[:8]]


@pytest.fixture
def fake_env(monkeypatch):
    store = _FakeStore()
    fake_em = _FakeEmbeddingManager()

    @contextmanager
    def _session():
        yield store

    monkeypatch.setattr(edge_builder, "get_db_session", _session)
    monkeypatch.setattr(seed_mod, "get_db_session", _session)
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: fake_em)
    return store, fake_em


# ---------------------------------------------------------------------------
# Log fixtures
# ---------------------------------------------------------------------------


def _paired_logs(from_action, to_action, n, workspace, query, base=None):
    """n turns; each turn executes from_action then to_action (both success)."""
    base = base or (datetime.utcnow() - timedelta(days=1))
    rows = []
    for i in range(n):
        turn_start = base + timedelta(minutes=10 * i)
        turn_id = f"{from_action}->{to_action}:{i}"
        for step, action in enumerate((from_action, to_action)):
            rows.append(SimpleNamespace(
                id=len(rows) + 1,
                agent_id=None,
                workspace_id=workspace,
                action_name=action,
                app_name="PLATFORM",
                status="success",
                user_query=query,
                executed_at=turn_start + timedelta(seconds=5 * step),
                router_decision={"turn_id": turn_id},
            ))
    return rows


def _seed_two_workspaces(store):
    store.logs.extend(
        _paired_logs("platform_list_agents", "platform_get_agent", 12, WS_A, QUERY_AGENTS)
    )
    store.logs.extend(
        _paired_logs("workspace_read_file", "workspace_write_file", 8, WS_B, QUERY_FILES)
    )


def _run_seed(**kwargs):
    return asyncio.run(seed_mod.seed_graph(**kwargs))


# ---------------------------------------------------------------------------
# Seed backfill
# ---------------------------------------------------------------------------


def test_seed_backfills_edges_from_logs(fake_env):
    """Historical logs become used_after edges; --workspace-id scopes the run."""
    store, _ = fake_env
    _seed_two_workspaces(store)

    summary = _run_seed(window_days=30, workspace_id=str(WS_A))

    key = ("platform_list_agents", "platform_get_agent", "used_after", str(WS_A), None)
    assert key in store.edges
    edge = store.edges[key]
    assert edge["sample_count"] == 12
    assert edge["confidence"] == pytest.approx(
        edge_builder.wilson_lower_bound(12, 12)
    )

    # Workspace-scoped: nothing from WS_B was loaded or written
    assert all(k[3] != str(WS_B) for k in store.edges)
    assert all(k[2] != str(WS_B) for k in store.affinities)
    assert summary.logs_processed == 24
    assert summary.edges == 1

    # Affinities were built from the same recompute (intent-scoped, WS_A only)
    assert summary.affinities > 0
    assert all(v["workspace_id"] == str(WS_A) for v in store.affinities.values())


def test_seed_is_idempotent(fake_env):
    """Re-running converges to the same rows — no duplicates, no count drift."""
    store, _ = fake_env
    _seed_two_workspaces(store)

    _run_seed(window_days=30)
    edges_1 = _edges_normalized(store)
    affs_1 = _affinities_normalized(store)
    clusters_1 = len(store.clusters)

    _run_seed(window_days=30)
    edges_2 = _edges_normalized(store)
    affs_2 = _affinities_normalized(store)

    assert edges_2 == edges_1
    assert affs_2 == affs_1
    assert len(store.clusters) == clusters_1


def _edges_normalized(store):
    return {
        k: (v["weight"], v["confidence"], v["sample_count"])
        for k, v in store.edges.items()
    }


def _affinities_normalized(store):
    """Multiset of affinity rows, cluster-id agnostic (cluster PKs regenerate)."""
    return sorted(
        (
            v["action_name"], v["affinity_type"], v["workspace_id"],
            v["agent_id"], v["weight"], v["confidence"], v["sample_count"],
        )
        for v in store.affinities.values()
    )


def test_dry_run_writes_nothing(fake_env):
    """--dry-run computes candidate counts but never writes nor embeds."""
    store, fake_em = fake_env
    _seed_two_workspaces(store)

    summary = _run_seed(window_days=30, dry_run=True)

    assert summary.dry_run is True
    assert summary.logs_processed == 40
    assert summary.edges == 2  # both pairs clear the sample floor
    assert store.edges == {}
    assert store.affinities == {}
    assert store.clusters == []
    assert store.statements == []
    assert fake_em.batch_calls == 0


def test_refuses_without_yes(fake_env, monkeypatch):
    """Human-applied like a migration: no --yes, no run — even with --dry-run."""
    monkeypatch.setattr(
        seed_mod, "seed_graph",
        lambda **kw: pytest.fail("seed_graph must not be invoked without --yes"),
    )
    assert seed_mod.main([]) == 2
    assert seed_mod.main(["--dry-run"]) == 2
    assert seed_mod.main(["--workspace-id", str(WS_A)]) == 2


# ---------------------------------------------------------------------------
# GraphRouter over the seeded graph (leaf-load idiom)
# ---------------------------------------------------------------------------

_discovery_dir = _orchestrator_root / "modules" / "tools" / "discovery"
_PKG = "_prd143_graph_seed"


def _load_graph_router():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_discovery_dir)]
        sys.modules[_PKG] = pkg

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


class _FakeSemanticIndex:
    """Embedding floor: canned (action, cosine) entries per query."""

    def __init__(self, mapping):
        self._mapping = mapping
        self._registry = SimpleNamespace(get_all=lambda: [])

    async def rank_actions(self, query, top_k=5, **kwargs):
        return list(self._mapping.get(query, []))[:top_k]


def _router_over_store(monkeypatch, store, mapping):
    module = _load_graph_router()
    router = module.GraphRouter()
    router._semantic_index = _FakeSemanticIndex(mapping)

    fake_db_mod = types.ModuleType("core.database.database")

    @contextmanager
    def _session():
        yield store

    fake_db_mod.get_db_session = _session
    monkeypatch.setitem(sys.modules, "core.database.database", fake_db_mod)
    monkeypatch.setattr(router, "_get_cache", lambda: None)
    return router


def test_seeded_graph_routes_common_intent(fake_env, monkeypatch):
    """After seeding, a common intent expands into a non-empty edge chain."""
    store, _ = fake_env
    _seed_two_workspaces(store)
    _run_seed(window_days=30)

    router = _router_over_store(
        monkeypatch, store, {QUERY_AGENTS: [("platform_list_agents", 0.8)]}
    )
    chains = asyncio.run(router.rank_chains(QUERY_AGENTS, agent_id=None, top_k=10))

    assert chains, "seeded graph must return ranked tools"
    chain_actions = [c[2] for c in chains]
    assert ["platform_list_agents", "platform_get_agent"] in chain_actions
    # The graph-expanded chain outranks the bare embedding floor (0.8)
    assert chains[0][2] == ["platform_list_agents", "platform_get_agent"]
    assert chains[0][1] > 0.8


def test_unseeded_intent_falls_back_to_semantic(fake_env, monkeypatch):
    """An intent with no seeded edges gets exactly the semantic-index floor."""
    store, _ = fake_env
    _seed_two_workspaces(store)
    _run_seed(window_days=30)

    router = _router_over_store(
        monkeypatch, store, {QUERY_UNSEEDED: [("workspace_zip_files", 0.7)]}
    )
    chains = asyncio.run(router.rank_chains(QUERY_UNSEEDED, agent_id=None, top_k=10))

    assert chains == [("workspace_zip_files", 0.7, ["workspace_zip_files"])]
