"""
PRD-232 P232-RVW-2 — the TWO-LAYER graph (§6.5, Gerard's ruling).
=================================================================

RVW-2 amends PRD-177's per-tenant lock (and US-004's meta_sibling-only NULL
admission): the learned graph reads as TWO layers —

  * a tenant's OWN rows (workspace_id == X) at full weight, and
  * a TEXT-FREE GLOBAL prior (workspace_id IS NULL, aggregated across tenants) at
    reduced weight (config.TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR),

so a zero-telemetry tenant still routes (rides the global prior) while a tenant's
own signal always dominates it, and the moat holds for tenant-SPECIFIC rows.

These tests reuse the faithful and_/or_ fake store + leaf-load idiom from
test_prd143_graph_seed.py (upgraded for §6.5's OR filters). No DB, no network, no
LLM — the write path is driven through edge_builder with a fake session, the read
path through the real GraphRouter over the same fake store.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import Text

import core.services.edge_builder as edge_builder
import scripts.seed_tool_routing_graph as seed_mod
from core.models.tool_routing import ToolRoutingAffinity, ToolRoutingEdge

from tests.test_prd143_graph_seed import (
    WS_A,
    WS_B,
    _FakeEmbeddingManager,
    _FakeStore,
    _load_graph_router,
    _paired_logs,
    _router_over_store,
)

_EDGE_KEY = ("from_action", "to_action", "edge_type", "workspace_id", "agent_id")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _put_edge(store, from_action, to_action, workspace_id, confidence,
              edge_type="used_after", agent_id=None, sample_count=10):
    """Insert one edge into the fake store's ON-CONFLICT map (keyed like the DB)."""
    params = {
        "from_action": from_action,
        "to_action": to_action,
        "edge_type": edge_type,
        "workspace_id": workspace_id,
        "agent_id": agent_id,
        "confidence": confidence,
        "weight": float(sample_count),
        "sample_count": sample_count,
    }
    store.edges[tuple(params[c] for c in _EDGE_KEY)] = params


def _pin_prior_factor(monkeypatch, value=0.5):
    module = _load_graph_router()
    monkeypatch.setattr(
        module.GraphRouter, "_global_prior_factor", staticmethod(lambda: value)
    )
    return module


def _score_of(chains, actions):
    for _primary, score, chain_actions in chains:
        if chain_actions == actions:
            return score
    return None


# ===========================================================================
# AC3 — a tenant edge outranks the IDENTICAL global prior edge for that tenant;
#       a different tenant sees ONLY the global prior (its own rows never leak).
# ===========================================================================


def test_tenant_edge_outranks_identical_global_prior(monkeypatch):
    _pin_prior_factor(monkeypatch, 0.5)
    store = _FakeStore()
    # Same (send_email -> SHARED) pair exists BOTH as this tenant's own row AND as
    # the global prior, at identical confidence. Plus a WS_A-only private edge.
    _put_edge(store, "send_email", "SHARED", str(WS_A), 0.9)   # tenant row
    _put_edge(store, "send_email", "SHARED", None, 0.9)        # identical global prior
    _put_edge(store, "send_email", "A_PRIVATE", str(WS_A), 0.9)  # WS_A-only

    router = _router_over_store(monkeypatch, store, {"q": [("send_email", 0.9)]})
    chains = asyncio.run(
        router.rank_chains("q", workspace_id=str(WS_A), agent_id=None, top_k=10)
    )

    # The tenant's own row (0.9 * 0.9 = 0.81) wins the dedup over the discounted
    # global prior (0.9 * 0.9 * 0.5 = 0.405) — proving the tenant layer dominates.
    shared = _score_of(chains, ["send_email", "SHARED"])
    assert shared == pytest.approx(0.81), (
        f"tenant edge must win at full weight (0.81), got {shared} — global prior leaked in"
    )
    assert _score_of(chains, ["send_email", "A_PRIVATE"]) == pytest.approx(0.81)


def test_different_tenant_sees_only_the_global_prior(monkeypatch):
    _pin_prior_factor(monkeypatch, 0.5)
    store = _FakeStore()
    _put_edge(store, "send_email", "SHARED", str(WS_A), 0.9)   # WS_A tenant row
    _put_edge(store, "send_email", "SHARED", None, 0.9)        # global prior
    _put_edge(store, "send_email", "A_PRIVATE", str(WS_A), 0.9)  # WS_A-only

    router = _router_over_store(monkeypatch, store, {"q": [("send_email", 0.9)]})
    chains = asyncio.run(
        router.rank_chains("q", workspace_id=str(WS_B), agent_id=None, top_k=10)
    )
    names = {a for _p, _s, acts in chains for a in acts}

    # WS_B has no rows of its own: it rides the discounted global prior for SHARED
    # (0.9 * 0.9 * 0.5 = 0.405) and NEVER sees WS_A's private tenant edge.
    assert _score_of(chains, ["send_email", "SHARED"]) == pytest.approx(0.405)
    assert "A_PRIVATE" not in names, "cross-tenant leak: WS_B saw WS_A's tenant-specific edge"


def test_none_read_sees_global_layer_at_full_weight(monkeypatch):
    """A system/eval read (workspace_id=None) is NOT discounted — global IS the data."""
    _pin_prior_factor(monkeypatch, 0.5)
    store = _FakeStore()
    _put_edge(store, "send_email", "SHARED", None, 0.9)  # global row

    router = _router_over_store(monkeypatch, store, {"q": [("send_email", 0.9)]})
    chains = asyncio.run(
        router.rank_chains("q", workspace_id=None, agent_id=None, top_k=10)
    )
    # Full weight on the system read: 0.9 * 0.9 = 0.81 (no prior_factor applied).
    assert _score_of(chains, ["send_email", "SHARED"]) == pytest.approx(0.81)


# ===========================================================================
# AC4 — privacy hard rule: no raw user text in the GLOBAL layer.
# ===========================================================================


def test_organic_clusters_carry_no_raw_user_text(monkeypatch):
    """Organic intent clusters are GLOBAL (no workspace_id column) so they must not
    store a raw user query in sample_query — it is redacted to an action-name label.
    Seeded rows (a synthetic authored utterance) are the only non-redacted text."""
    store = _FakeStore()
    fake_em = _FakeEmbeddingManager()

    @contextmanager
    def _session():
        yield store

    monkeypatch.setattr(edge_builder, "get_db_session", _session)
    monkeypatch.setattr(seed_mod, "get_db_session", _session)
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: fake_em)

    raw_queries = {"show me my agents and open the first one", "read the project files and update them"}
    store.logs.extend(_paired_logs("platform_list_agents", "platform_get_agent", 12, WS_A, "show me my agents and open the first one"))
    store.logs.extend(_paired_logs("workspace_read_file", "workspace_write_file", 8, WS_B, "read the project files and update them"))
    asyncio.run(seed_mod.seed_graph(window_days=30))

    assert store.clusters, "seed must have produced organic clusters"
    for c in store.clusters:
        assert getattr(c, "provenance", "organic") == "organic"
        # No raw user query text survives into the global cluster layer.
        assert c.sample_query not in raw_queries, (
            f"global organic cluster leaked raw user text: {c.sample_query!r}"
        )
        # It is the redacted action-name label instead.
        assert c.sample_query.startswith("(organic")


def test_global_edge_and_affinity_tables_have_no_free_text_column():
    """The global edge/affinity rows carry only action NAMES + numbers — the tables
    have no free-text (Text) column that could hold a user query."""
    for model in (ToolRoutingEdge, ToolRoutingAffinity):
        text_cols = [c.name for c in model.__table__.columns if isinstance(c.type, Text)]
        assert text_cols == [], (
            f"{model.__name__} has free-text column(s) {text_cols} — the global layer must be text-free"
        )


# ===========================================================================
# AC6 — the WRITE path: a full recompute writes per-tenant AND global edges;
#       a scoped run writes tenant rows only (no global-layer pollution).
# ===========================================================================


def _pair_logs_ws(from_action, to_action, n, workspace, query):
    """n turns of (from_action -> to_action), with a WORKSPACE-SCOPED turn_id so the
    SAME action pair in two workspaces does NOT collide on the session-grouping key
    (unlike test_prd143_graph_seed._paired_logs, whose turn_id is pair-only)."""
    base = datetime.utcnow() - timedelta(days=1)
    rows = []
    for i in range(n):
        turn_start = base + timedelta(minutes=10 * i)
        turn_id = f"{workspace}:{from_action}->{to_action}:{i}"
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


def _run_build(monkeypatch, store, **kwargs):
    fake_em = _FakeEmbeddingManager()

    @contextmanager
    def _session():
        yield store

    monkeypatch.setattr(edge_builder, "get_db_session", _session)
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: fake_em)
    return asyncio.run(edge_builder.build_edges(**kwargs))


def test_full_recompute_writes_two_tenant_rows_plus_one_global(monkeypatch):
    store = _FakeStore()
    # The SAME pair occurs in two workspaces (5 turns in A, 4 in B).
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 5, WS_A, "q-a"))
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 4, WS_B, "q-b"))

    summary = _run_build(monkeypatch, store)

    pair = ("platform_list_agents", "platform_get_agent", "used_after")
    a_key = (*pair, str(WS_A), None)
    b_key = (*pair, str(WS_B), None)
    global_key = (*pair, None, None)

    # 2 tenant rows (each workspace's own count) + 1 global row (summed).
    assert a_key in store.edges and store.edges[a_key]["sample_count"] == 5
    assert b_key in store.edges and store.edges[b_key]["sample_count"] == 4
    assert global_key in store.edges, "full recompute must write the global-prior edge"
    assert store.edges[global_key]["sample_count"] == 9  # 5 + 4 summed across tenants
    assert store.edges[global_key]["workspace_id"] is None
    assert summary.global_edges_built == 1


def test_scoped_run_writes_no_global_row(monkeypatch):
    """A single --workspace-id run writes ONLY that tenant's rows — a one-tenant
    'aggregate' would just be that tenant's data relabeled global (a leak)."""
    store = _FakeStore()
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 5, WS_A, "q-a"))
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 4, WS_B, "q-b"))

    summary = _run_build(monkeypatch, store, workspace_id=str(WS_A))

    global_key = ("platform_list_agents", "platform_get_agent", "used_after", None, None)
    assert global_key not in store.edges, "a scoped run must not write a global row"
    assert summary.global_edges_built == 0
    # And nothing from WS_B was loaded or written.
    assert all(k[3] != str(WS_B) for k in store.edges)


def test_global_edge_layer_is_idempotent_across_runs(monkeypatch):
    """Review HIGH #1: a global row's (workspace_id NULL, agent_id NULL) key defeats
    ON CONFLICT (Postgres treats NULLs as distinct), so a plain upsert would INSERT a
    duplicate every nightly run. The delete-then-insert rebuild keeps exactly ONE
    global row per pair across repeated full runs."""
    store = _FakeStore()
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 5, WS_A, "q-a"))
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 4, WS_B, "q-b"))

    _run_build(monkeypatch, store)   # nightly run 1
    _run_build(monkeypatch, store)   # nightly run 2 — same logs

    global_rows = [k for k in store.edges if k[2] == "used_after" and k[3] is None]
    assert len(global_rows) == 1, (
        f"global used_after layer must not accumulate duplicates across runs, got {global_rows}"
    )
    assert store.edges[global_rows[0]]["sample_count"] == 9


def test_single_tenant_pair_produces_no_global_prior(monkeypatch):
    """Review MEDIUM: a full recompute where only ONE workspace exhibits a pair must
    NOT write it globally — a 'global' prior is a CROSS-tenant aggregate, and a
    one-tenant pattern relabeled global is the same leak the scoped-run guard prevents."""
    store = _FakeStore()
    store.logs.extend(_pair_logs_ws("platform_list_agents", "platform_get_agent", 5, WS_A, "q-a"))

    summary = _run_build(monkeypatch, store)  # full run, but the pair is single-tenant

    global_rows = [k for k in store.edges if k[2] == "used_after" and k[3] is None]
    assert global_rows == [], "a single-tenant pair must not become a global prior"
    assert summary.global_edges_built == 0
    # The tenant's own row is still written — only the global aggregate is withheld.
    assert ("platform_list_agents", "platform_get_agent", "used_after", str(WS_A), None) in store.edges


def test_populated_empty_hot_cluster_matches_but_artifact_is_skipped():
    """Review HIGH #2: _match_intent_cluster skips ONLY a zero-sample k-means artifact,
    not a POPULATED cluster that happens to have empty action_names_hot (e.g. a
    __tool_gap__-dominated intent whose members are all non-'success'). The populated
    cluster must still match on its centroid so US-011(c)'s gap→resolution affinity,
    scoped to its id, stays reachable."""
    from modules.tools.discovery.graph_router import GraphRouter

    store = _FakeStore()
    # A real, populated cluster with empty hot (all members non-'success').
    store.clusters.append(SimpleNamespace(
        id=7, embedding_model_key="m", centroid_embedding=[1.0, 0.0, 0.0],
        action_names_hot=[], sample_count=5,
    ))
    # A zero-member k-means artifact with a COPIED centroid (same vector) — must skip.
    store.clusters.append(SimpleNamespace(
        id=8, embedding_model_key="m", centroid_embedding=[1.0, 0.0, 0.0],
        action_names_hot=[], sample_count=0,
    ))

    match = GraphRouter._match_intent_cluster(store, [1.0, 0.0, 0.0], "m", 0.6)
    assert match is not None, "a populated (sample_count>0) cluster must match even with empty hot"
    assert match[0] == 7, "must match the populated cluster, not the zero-sample artifact"


# ===========================================================================
# AC5 — the prior factor is config-driven (documented), never hardcoded.
# ===========================================================================


def test_prior_factor_is_read_from_config():
    from config import config

    module = _load_graph_router()
    assert hasattr(config, "TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR")
    assert module.GraphRouter._global_prior_factor() == config.TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR
    # 0..1: it must reduce (never amplify) a global row's weight.
    assert 0.0 <= config.TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR <= 1.0


def test_no_hardcoded_prior_factor_in_router():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "modules" / "tools" / "discovery" / "graph_router.py").read_text()
    assert 'getattr(config, "TOOL_ROUTING_GRAPH_GLOBAL_PRIOR_FACTOR"' in src
