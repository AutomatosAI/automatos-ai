"""
PRD-232 US-007 — seed intent clusters from the corpus, survive the nightly rebuild.
===================================================================================

The synthetic-utterance seed (``scripts/seed_tool_routing_graph``) writes
``provenance='seeded'`` ToolRoutingIntentCluster rows so the tool-routing graph
routes DAY-ONE: a live query phrased like any seeded utterance matches the seeded
centroid (``GraphRouter._match_intent_cluster``) and the right action surfaces,
before any telemetry accrues. The trap (spec §7): the nightly ``edge_builder``
recompute deletes-and-reinserts its clusters — without a provenance marker it would
wipe the seeds at 03:00 UTC. US-007 makes the rebuild provenance-scoped (organic
only), so seeds persist.

These tests never touch a real database or an LLM:
  * The fake DB session is a faithful and_/or_ predicate evaluator over the REAL
    ToolRouting* models (so US-010's ``or_(cluster == X, cluster IS NULL)`` affinity
    filter is exercised honestly, not flattened to AND).
  * The embedding managers are hermetic local fakes — a hashed vector for the
    edge-builder rebuild tests, and a real (if simple) LEXICAL bag-of-words for the
    routing test, so lexical overlap proves an unseen phrasing lands on the right
    seeded cluster. NEITHER is ``DeterministicEmbeddingProvider`` (banned outside
    test fixtures by PRD-185 S3; it would poison every centroid).

The single-head + provenance-column checks (AC1) are pure file/model reads.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import math
import os
import re
import sys
import types
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

# Dummy creds satisfy the lazy engine config at import time (established idiom);
# no connection is ever opened.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

import core.services.edge_builder as edge_builder  # noqa: E402
import scripts.seed_tool_routing_graph as seed_mod  # noqa: E402
from core.models.composio_cache import ToolExecutionLog  # noqa: E402
from core.models.tool_routing import (  # noqa: E402
    ToolRoutingAffinity,
    ToolRoutingEdge,
    ToolRoutingIntentCluster,
)


# ===========================================================================
# Faithful and_/or_ predicate evaluator over REAL SQLAlchemy clauses
# ===========================================================================


def _eval_clause(clause, row) -> bool:
    """Recursively evaluate a real SQLAlchemy clause against a row.

    Honours and_/or_ semantics (a flatten-to-AND pass would wrongly reject a row
    against US-010's ``or_(cluster == X, cluster IS NULL)`` affinity filter). Any
    leaf whose column the row lacks is treated as satisfied.
    """
    sub = getattr(clause, "clauses", None)
    if sub is not None:
        op = getattr(getattr(clause, "operator", None), "__name__", "")
        results = [_eval_clause(c, row) for c in sub]
        if op == "or_":
            return any(results)
        return all(results)  # and_ / plain ClauseList

    left = getattr(clause, "left", None)
    key = getattr(left, "key", None)
    if key is None:
        return True
    op_name = getattr(getattr(clause, "operator", None), "__name__", "")
    value = getattr(getattr(clause, "right", None), "value", None)
    actual = getattr(row, key, None)
    if op_name in ("is_", "is_op"):
        return actual is None
    if op_name == "in_op":
        return str(actual) in {str(v) for v in (value or [])}
    if op_name == "eq":
        return actual is not None and str(actual) == str(value)
    if op_name in ("ge", "ge_op"):
        return actual is not None and actual >= value
    return True


_EDGE_KEY = ("from_action", "to_action", "edge_type", "workspace_id", "agent_id")
_AFF_KEY = ("action_name", "affinity_type", "workspace_id", "agent_id", "intent_cluster_id")


class _FakeQuery:
    def __init__(self, store, model):
        self._store = store
        self._model = model
        self._clauses = []
        self._limit = None

    def filter(self, *clauses):
        self._clauses.extend(clauses)
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def _all_rows(self):
        if self._model is ToolExecutionLog:
            return list(self._store.logs)
        if self._model is ToolRoutingIntentCluster:
            return list(self._store.clusters)
        if self._model is ToolRoutingEdge:
            return [SimpleNamespace(**v) for v in self._store.edges.values()]
        if self._model is ToolRoutingAffinity:
            return [SimpleNamespace(**v) for v in self._store.affinities.values()]
        return []

    def _rows(self):
        return [r for r in self._all_rows() if all(_eval_clause(c, r) for c in self._clauses)]

    def all(self):
        rows = self._rows()
        return rows[: self._limit] if self._limit is not None else rows

    def delete(self, synchronize_session=None):
        doomed = self._rows()
        if self._model is ToolRoutingIntentCluster:
            self._store.clusters = [c for c in self._store.clusters if c not in doomed]
            return len(doomed)
        if self._model is ToolRoutingAffinity:
            doomed_keys = {
                k for k, v in self._store.affinities.items()
                if all(_eval_clause(c, SimpleNamespace(**v)) for c in self._clauses)
            }
            for k in doomed_keys:
                del self._store.affinities[k]
            return len(doomed_keys)
        return 0


class _FakeStore:
    """Session + storage in one, simulating the Postgres ON CONFLICT upsert."""

    def __init__(self):
        self.logs = []
        self.edges = {}
        self.affinities = {}
        self.clusters = []
        self.statements = []
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


# ===========================================================================
# Hermetic embedding managers (NOT DeterministicEmbeddingProvider)
# ===========================================================================


class _HashEM:
    """8-dim hashed vector — enough for the edge-builder rebuild tests, which
    care about cluster ROW lifecycle, not semantic proximity."""

    def get_provider_info(self):
        return {"provider": "fake", "model": "fake-embed", "dimension": 8}

    def get_dimension(self):
        return 8

    async def generate_embeddings_batch(self, queries, max_concurrent=5):
        return [[b / 255.0 for b in hashlib.sha256(q.encode()).digest()[:8]] for q in queries]


_LEX_DIM = 512
_STOP = {"the", "a", "an", "to", "of", "for", "and", "or", "my", "me", "i", "is",
         "are", "in", "on", "at", "with", "this", "that", "it", "all", "from"}


def _lex_tokens(text: str):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


def _lex_vec(text: str):
    v = [0.0] * _LEX_DIM
    for tok in _lex_tokens(text):
        v[int(hashlib.sha1(tok.encode()).hexdigest(), 16) % _LEX_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


class _LexEM:
    """Real (if simple) LEXICAL bag-of-words embedding. Deterministic, hermetic —
    lexical overlap = semantic proximity, so an unseen phrasing that shares words
    with an action's seeded utterances lands near that action's centroid."""

    def get_provider_info(self):
        return {"provider": "lex", "model": "lex", "dimension": _LEX_DIM}

    def get_dimension(self):
        return _LEX_DIM

    async def generate_embeddings_batch(self, texts, max_concurrent=5):
        return [_lex_vec(t) for t in texts]


# ===========================================================================
# Leaf-load graph_router without modules.tools.__init__'s heavy chain
# ===========================================================================

_DISCOVERY = _ORCH_ROOT / "modules" / "tools" / "discovery"
_PKG = "_prd232_us007"


def _load_graph_router():
    if _PKG not in sys.modules:
        pkg = types.ModuleType(_PKG)
        pkg.__path__ = [str(_DISCOVERY)]
        sys.modules[_PKG] = pkg
    asi_name = f"{_PKG}.action_semantic_index"
    if asi_name not in sys.modules:
        fake_asi = types.ModuleType(asi_name)
        fake_asi.get_action_semantic_index = lambda: SimpleNamespace(rank_actions=lambda *a, **k: [])
        sys.modules[asi_name] = fake_asi
    full = f"{_PKG}.graph_router"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, _DISCOVERY / "graph_router.py")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = _PKG
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def seed_env(monkeypatch):
    """A fake store + patched get_db_session for the seed module."""
    store = _FakeStore()

    @contextmanager
    def _session():
        yield store

    monkeypatch.setattr(seed_mod, "get_db_session", _session)
    monkeypatch.setattr(edge_builder, "get_db_session", _session)
    return store


def _seed_corpus(monkeypatch, store, corpus, cats, em):
    monkeypatch.setattr(seed_mod, "get_embedding_manager", lambda: em)
    return asyncio.run(
        seed_mod.seed_intent_clusters_from_corpus(corpus=corpus, action_categories=cats)
    )


# ---------------------------------------------------------------------------
# Fixtures: corpora
# ---------------------------------------------------------------------------

CORPUS = {
    "platform_update_task_status": [
        "close the ticket",
        "close the blocked tickets",
        "finish this task",
        "mark the card as done",
        "clear the blocked items on the board",
        "close out the open tickets",
        "wrap up the blocked tasks",
        "set the ticket status to done",
    ],
    "platform_send_email": [
        "send an email to the team",
        "email the customer back",
        "compose a message and send it",
        "shoot an email over to support",
        "draft an email reply",
        "send a message to the mailing list",
        "email everyone the weekly update",
        "write an email to the vendor",
    ],
}
CATS = {
    "platform_update_task_status": "tasks",
    "platform_send_email": "email",
    # a same-category sibling with NO utterances → appears only in the family
    "platform_list_tasks": "tasks",
}


# ===========================================================================
# AC1 — exactly one new revision; nullable provenance server_default 'organic'
# ===========================================================================


def test_single_alembic_head_at_our_revision():
    """Exactly one head after the migration, and it is prd232_cluster_provenance
    chained onto the prior single head. Uses alembic's own ScriptDirectory (the
    same machinery ``alembic heads`` uses) so merges/tuples are handled correctly.
    No DB — ScriptDirectory only reads the version files."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config()
    cfg.set_main_option("script_location", str(_ORCH_ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)

    heads = list(script.get_heads())
    assert heads == ["prd232_cluster_provenance"], f"expected single head, got {heads}"
    rev = script.get_revision("prd232_cluster_provenance")
    assert rev.down_revision == "prd225_s1_asks_on_grants"


def test_migration_adds_nullable_provenance_default_organic():
    mig = (_ORCH_ROOT / "alembic" / "versions" / "prd232_cluster_provenance.py").read_text()
    assert "tool_routing_intent_clusters" in mig
    assert "ADD COLUMN IF NOT EXISTS provenance" in mig
    assert "DEFAULT 'organic'" in mig
    assert "NOT NULL" not in mig.split("upgrade")[1].split("downgrade")[0]  # nullable
    assert "DROP COLUMN IF EXISTS provenance" in mig  # reversible


def test_model_provenance_column_is_nullable_organic_default():
    col = ToolRoutingIntentCluster.__table__.columns["provenance"]
    assert col.nullable is True
    assert col.default is not None and col.default.arg == "organic"
    assert col.server_default is not None
    assert "organic" in str(col.server_default.arg)


# ===========================================================================
# AC2 — an unseen phrasing lands on the right seeded cluster
# ===========================================================================


def test_seeded_cluster_routes_unseen_phrasing(seed_env, monkeypatch):
    store = seed_env
    em = _LexEM()
    n = _seed_corpus(monkeypatch, store, CORPUS, CATS, em)
    assert n == 2  # both actions with utterances seeded; list_tasks (no utts) is family only

    seeded = [c for c in store.clusters if c.provenance == "seeded"]
    assert len(seeded) == 2
    model_key = edge_builder.derive_embedding_model_key(em)
    assert all(c.embedding_model_key == model_key for c in seeded)

    gr = _load_graph_router()
    # A phrasing NOT in the corpus, but lexically an update-task-status intent.
    held_out = _lex_vec("close all the blocked tickets from vector")
    match = gr.GraphRouter._match_intent_cluster(store, held_out, model_key, 0.05)
    assert match is not None, "held-out phrasing matched no seeded cluster"
    _cluster_id, hot, _sim = match
    assert hot[0] == "platform_update_task_status"
    # its category family rode along (US-007: action + family)
    assert "platform_list_tasks" in hot
    # and it is decisively NOT the e-mail cluster
    assert "platform_send_email" not in hot


def test_unrelated_phrasing_prefers_its_own_cluster(seed_env, monkeypatch):
    """Sanity: an e-mail phrasing lands on the e-mail cluster, not tasks — the
    match is discriminative, not a single-attractor artefact."""
    store = seed_env
    em = _LexEM()
    _seed_corpus(monkeypatch, store, CORPUS, CATS, em)
    model_key = edge_builder.derive_embedding_model_key(em)

    gr = _load_graph_router()
    held_out = _lex_vec("shoot the customer a quick email reply")
    match = gr.GraphRouter._match_intent_cluster(store, held_out, model_key, 0.05)
    assert match is not None
    assert match[1][0] == "platform_send_email"


# ===========================================================================
# AC3 — seeded clusters survive the nightly rebuild; FK integrity preserved
# ===========================================================================


def _org_log(action, query, turn, offset):
    return SimpleNamespace(
        id=offset,
        agent_id=None,
        workspace_id=None,
        action_name=action,
        app_name="PLATFORM",
        status="success",
        user_query=query,
        executed_at=datetime(2026, 1, 1) + timedelta(seconds=offset),
        router_decision={"turn_id": turn},
    )


def test_seeded_clusters_survive_nightly(seed_env, monkeypatch):
    store = seed_env
    # Seed AND nightly share ONE embedding model key ('fake:fake-embed:8'), so the
    # nightly's delete (scoped to that key) targets the seeds too — and ONLY the
    # provenance='organic' filter spares them. This is the real test of the trap.
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: _HashEM())
    _seed_corpus(monkeypatch, store, CORPUS, CATS, _HashEM())
    seeded_before = [c for c in store.clusters if c.provenance == "seeded"]
    seeded_ids = {c.id for c in seeded_before}
    assert len(seeded_before) == 2
    assert all(c.embedding_model_key == "fake:fake-embed:8" for c in seeded_before)
    seeded_aff_keys_before = {
        k for k, v in store.affinities.items() if v["intent_cluster_id"] in seeded_ids
    }
    assert seeded_aff_keys_before, "seed must write per-cluster affinities"

    # 2) Nightly recompute over real telemetry (same model key). It must rebuild ITS
    #    organic clusters and leave every seeded row untouched.
    for i in range(6):
        store.logs.append(_org_log("platform_list_agents", "show me the agents", f"t{i}", i * 10))
        store.logs.append(_org_log("platform_get_agent", "open that agent", f"t{i}", i * 10 + 2))
    asyncio.run(edge_builder.build_edges(window=timedelta(days=3650)))

    # organic clusters were built under the SAME key the seeds use...
    organic = [c for c in store.clusters if c.provenance == "organic"]
    assert organic, "nightly must build organic clusters"
    assert all(c.embedding_model_key == "fake:fake-embed:8" for c in organic)
    # ...and every seeded cluster + its affinities SURVIVED the same-key delete
    seeded_after = [c for c in store.clusters if c.provenance == "seeded"]
    assert {c.id for c in seeded_after} == seeded_ids
    assert all(k in store.affinities for k in seeded_aff_keys_before)

    # FK integrity: no affinity points at a cluster id that no longer exists.
    live_cluster_ids = {c.id for c in store.clusters}
    for v in store.affinities.values():
        cid = v["intent_cluster_id"]
        assert cid is None or cid in live_cluster_ids, "orphaned affinity FK after rebuild"


def test_second_nightly_still_spares_seeds(seed_env, monkeypatch):
    """Two nightly runs in a row (same model key): organic churns, seeds stable."""
    store = seed_env
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: _HashEM())
    _seed_corpus(monkeypatch, store, CORPUS, CATS, _HashEM())
    seeded_ids = {c.id for c in store.clusters if c.provenance == "seeded"}

    for i in range(6):
        store.logs.append(_org_log("platform_list_agents", "list agents please", f"t{i}", i * 10))
        store.logs.append(_org_log("platform_get_agent", "open the agent", f"t{i}", i * 10 + 2))
    asyncio.run(edge_builder.build_edges(window=timedelta(days=3650)))
    asyncio.run(edge_builder.build_edges(window=timedelta(days=3650)))

    assert {c.id for c in store.clusters if c.provenance == "seeded"} == seeded_ids


# ===========================================================================
# AC4 — seeds never outrank organic rows of higher Wilson confidence
# ===========================================================================


def test_seeds_never_outrank_organic_higher_wilson(seed_env, monkeypatch):
    store = seed_env
    lex = _LexEM()
    # Seed just one action so its cluster id is deterministic.
    _seed_corpus(
        monkeypatch, store,
        {"platform_update_task_status": CORPUS["platform_update_task_status"]},
        {"platform_update_task_status": "tasks", "platform_list_tasks": "tasks"},
        lex,
    )
    seeded = [c for c in store.clusters if c.provenance == "seeded"]
    assert len(seeded) == 1
    cluster_id = seeded[0].id

    # The seeded affinity sits at the config floor (0.6), weight 1 → boost 0.6.
    seed_aff = next(
        v for v in store.affinities.values()
        if v["action_name"] == "platform_update_task_status" and v["intent_cluster_id"] == cluster_id
    )
    from config import config
    assert seed_aff["confidence"] == pytest.approx(config.TOOL_ROUTING_SEED_CLUSTER_CONFIDENCE)
    assert seed_aff["weight"] == 1.0

    # Organic evidence accrues for a DIFFERENT action under the SAME intent: 40/40
    # successes → high Wilson confidence, weight 40.
    organic_conf = edge_builder.wilson_lower_bound(40, 40)
    assert organic_conf > seed_aff["confidence"]  # higher Wilson than the seed floor
    edge_builder._upsert_affinities(store, [{
        "action_name": "platform_list_tasks",
        "affinity_type": "succeeds_for_intent",
        "workspace_id": None,
        "agent_id": None,
        "intent_cluster_id": cluster_id,
        "weight": 40.0,
        "confidence": organic_conf,
        "sample_count": 40,
    }])

    # Read them the way the router does (per-intent scope), and assert the organic
    # row's boost strictly dominates the seed's — the seed can NEVER outrank it.
    gr = _load_graph_router()
    positive, _negative = gr.GraphRouter._query_affinities(
        store, ["platform_update_task_status", "platform_list_tasks"], None, None, cluster_id
    )
    assert positive["platform_list_tasks"] > positive["platform_update_task_status"]
    assert positive["platform_update_task_status"] == pytest.approx(
        seed_aff["weight"] * seed_aff["confidence"]
    )


# ===========================================================================
# AC5 — idempotent re-run; dry run writes nothing; --yes gate intact
# ===========================================================================


def test_seed_is_idempotent(seed_env, monkeypatch):
    store = seed_env
    em = _LexEM()
    n1 = _seed_corpus(monkeypatch, store, CORPUS, CATS, em)
    clusters_1 = len([c for c in store.clusters if c.provenance == "seeded"])
    affs_1 = _seed_affinity_signature(store)

    n2 = _seed_corpus(monkeypatch, store, CORPUS, CATS, em)
    clusters_2 = len([c for c in store.clusters if c.provenance == "seeded"])
    affs_2 = _seed_affinity_signature(store)

    assert n1 == n2 == 2
    assert clusters_1 == clusters_2 == 2  # no duplication on re-run
    assert affs_1 == affs_2               # converges (cluster-id agnostic)


def _seed_affinity_signature(store):
    """Multiset of seeded affinity content, cluster-id agnostic (ids regenerate)."""
    return sorted(
        (v["action_name"], v["affinity_type"], v["weight"], round(v["confidence"], 6))
        for v in store.affinities.values()
    )


def test_dry_run_seeds_nothing(seed_env, monkeypatch):
    store = seed_env
    monkeypatch.setattr(seed_mod, "get_embedding_manager", lambda: pytest.fail("dry run must not embed"))
    n = asyncio.run(
        seed_mod.seed_intent_clusters_from_corpus(dry_run=True, corpus=CORPUS, action_categories=CATS)
    )
    assert n == 2                # counts what WOULD be seeded
    assert store.clusters == []  # but wrote nothing
    assert store.affinities == {}
    assert store.statements == []


def test_yes_gate_still_intact(monkeypatch):
    """The whole seed (edges + meta + clusters) refuses to run without --yes."""
    monkeypatch.setattr(
        seed_mod, "seed_intent_clusters_from_corpus",
        lambda *a, **k: pytest.fail("must not seed clusters without --yes"),
    )
    monkeypatch.setattr(
        seed_mod, "seed_graph",
        lambda **kw: pytest.fail("must not seed graph without --yes"),
    )
    assert seed_mod.main([]) == 2
    assert seed_mod.main(["--dry-run"]) == 2


def test_su_only_action_never_seeded(seed_env, monkeypatch):
    """Defence-in-depth: a corpus name absent from action_categories (e.g. an su
    action the registry filtered out) is never seeded."""
    store = seed_env
    corpus = dict(CORPUS)
    corpus["platform_query_loki_logs"] = ["tail the loki logs", "show error logs"]  # su → filtered
    _seed_corpus(monkeypatch, store, corpus, CATS, _LexEM())  # CATS has no loki entry

    seeded_actions = {c.action_names_hot[0] for c in store.clusters if c.provenance == "seeded"}
    assert "platform_query_loki_logs" not in seeded_actions
    assert seeded_actions == {"platform_update_task_status", "platform_send_email"}
