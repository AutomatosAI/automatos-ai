"""
P256-FIX-N14-1 (F416): the nightly EdgeBuilder never runs the backend out of memory.

Night 14 (12–13 Oct): EdgeBuilderScheduler._run_edge_builder started at 03:00 UTC,
never logged completion, and the backend container was OOMKilled at 03:50 while
chats still flowed. build_edges loaded every log of its 30-day window in one read,
embedded every query at once and broadcast an (N, k, D) tensor in the k-means.

These tests prove the bounds and the isolation:

  * a window over the cluster cap clusters only the most recent queries, embeds
    them a batch at a time, logs that it sampled, and the summary says so;
  * the log read pages through the window by id and loads only its columns;
  * the k-means' working block never exceeds its cap, and the clusters do not
    depend on the block size;
  * the run happens in a child process: a clustering MemoryError, a child the
    kernel kills, or an allocation past the child's ceiling ends the run with
    one error and a failed summary, the caller alive, and the next run works.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
import signal
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest

import core.services.edge_builder as edge_builder
import core.services.edge_builder_clusters as edge_builder_clusters
import core.services.edge_builder_logs as edge_builder_logs
import core.services.edge_builder_worker as edge_builder_worker
import core.services.intent_clustering as intent_clustering
from config import config
from core.services.edge_builder import EdgeBuildSummary
from tests.test_prd256_fix_edge_builder_off_the_loop import (
    _FakeDB,
    _FakeEmbedder,
    _seeded_logs_and_vectors,
)

_WINDOW = timedelta(days=30)
_CAP = 10
_BATCH = 4
_MB = 1024 * 1024
_WORKER_LOGGER = "core.services.edge_builder_worker"
_ON_LINUX = sys.platform.startswith("linux")


# ---------------------------------------------------------------------------
# Wiring: build_edges over the seeded fixture, no database, no network
# ---------------------------------------------------------------------------


class _BatchCountingEmbedder(_FakeEmbedder):
    async def generate_embeddings_batch(self, texts, max_concurrent=5):
        self._store["batches"].append(len(texts))
        return await super().generate_embeddings_batch(texts, max_concurrent)


def _new_store():
    return {"open": 0, "open_at_embed": [], "clusters": [], "ids": count(1),
            "batches": [], "clustered": []}


def _wire(set_attr, logs, vectors, store):
    """Fake the session, the log read, the writes and the embedder. ``set_attr``
    is monkeypatch.setattr in this process, plain setattr in a child."""

    @contextmanager
    def _session():
        yield _FakeDB(store)

    set_attr(edge_builder, "get_db_session", _session)
    set_attr(edge_builder, "_load_logs", lambda db, cutoff, workspace_id=None: logs)
    set_attr(edge_builder, "_upsert_edges", lambda db, d: len(d))
    set_attr(edge_builder, "_upsert_global_edges", lambda db, d: 0)
    set_attr(edge_builder, "_upsert_failed_after_edges", lambda db, d: len(d))
    set_attr(edge_builder, "_upsert_affinities", lambda db, affs: len(affs))
    set_attr(edge_builder, "get_embedding_manager", lambda: _BatchCountingEmbedder(vectors, store))


def _spy_clustering(set_attr, store):
    real = edge_builder_clusters.compute_intent_clusters

    def _spy(**kwargs):
        store["clustered"].append(list(kwargs["queries"]))
        return real(**kwargs)

    set_attr(edge_builder_clusters, "compute_intent_clusters", _spy)


# ---------------------------------------------------------------------------
# Child-process targets (module level: the spawned child imports them by name)
# ---------------------------------------------------------------------------


def _clustering_runs_out_of_memory(window_seconds, workspace_id):
    """In the child: the real build_edges, whose clustering raises MemoryError."""
    logs, vectors = _seeded_logs_and_vectors()
    _wire(setattr, logs, vectors, _new_store())

    def _out_of_memory(**_kwargs):
        raise MemoryError("the k-means block did not fit")

    edge_builder_clusters.compute_intent_clusters = _out_of_memory
    return edge_builder_worker.run_build_edges(window_seconds, workspace_id)


def _a_normal_night(window_seconds, workspace_id):
    """In the child: the real build_edges over the seeded fixture."""
    logs, vectors = _seeded_logs_and_vectors()
    _wire(setattr, logs, vectors, _new_store())
    return edge_builder_worker.run_build_edges(window_seconds, workspace_id)


def _killed_by_the_kernel(_window_seconds, _workspace_id):
    os.kill(os.getpid(), signal.SIGKILL)


def _allocate(mb, _window_seconds, _workspace_id):
    block = bytearray(mb * _MB)
    return EdgeBuildSummary(edges_built=len(block) // _MB)


def _own_oom_score_adj(_window_seconds, _workspace_id):
    with open("/proc/self/oom_score_adj", encoding="ascii") as handle:
        return EdgeBuildSummary(edges_built=int(handle.read().strip()))


def _run_in_worker(target):
    async def _go():
        summary = await edge_builder_worker.build_edges_in_worker(_WINDOW, target=target)
        await asyncio.sleep(0)  # the caller's loop still runs after the child ends
        return summary, os.getpid()

    return asyncio.run(_go())


def _worker_errors(caplog):
    return [r for r in caplog.records if r.name == _WORKER_LOGGER and r.levelno >= logging.ERROR]


# ---------------------------------------------------------------------------
# AC1: the clustering input is capped, embedded in batches, and says it sampled
# ---------------------------------------------------------------------------


def test_a_window_over_the_cap_clusters_a_recent_sample_and_the_summary_says_so(monkeypatch, caplog):
    monkeypatch.setattr(config, "EDGE_BUILDER_CLUSTER_MAX_QUERIES", _CAP, raising=False)
    monkeypatch.setattr(config, "EDGE_BUILDER_EMBED_BATCH_SIZE", _BATCH, raising=False)
    logs, vectors = _seeded_logs_and_vectors()
    store = _new_store()
    _wire(monkeypatch.setattr, logs, vectors, store)
    _spy_clustering(monkeypatch.setattr, store)
    query_logs = [log for log in logs if log["user_query"]]
    assert len(query_logs) > _CAP, "the fixture must be over the cap"

    with caplog.at_level(logging.WARNING):
        summary = asyncio.run(edge_builder.build_edges(window=_WINDOW))

    assert store["clustered"] == [[log["user_query"] for log in query_logs[-_CAP:]]], (
        "the cap must keep the most recent queries"
    )
    assert summary.cluster_sampled is True
    assert summary.cluster_queries == _CAP
    assert summary.cluster_queries_total == len(query_logs)
    assert summary.intent_clusters > 0 and not summary.failed
    assert sum(store["batches"]) == _CAP and max(store["batches"]) <= _BATCH
    assert any("over the cap" in r.getMessage() for r in caplog.records), "a sampled run must log it"


def test_a_window_under_the_cap_clusters_every_query_and_is_not_a_sample(monkeypatch, caplog):
    logs, vectors = _seeded_logs_and_vectors()
    query_logs = [log for log in logs if log["user_query"]]
    monkeypatch.setattr(config, "EDGE_BUILDER_CLUSTER_MAX_QUERIES", len(query_logs), raising=False)
    store = _new_store()
    _wire(monkeypatch.setattr, logs, vectors, store)
    _spy_clustering(monkeypatch.setattr, store)

    with caplog.at_level(logging.WARNING):
        summary = asyncio.run(edge_builder.build_edges(window=_WINDOW))

    assert store["clustered"] == [[log["user_query"] for log in query_logs]]
    assert summary.cluster_sampled is False
    assert summary.cluster_queries == summary.cluster_queries_total == len(query_logs)
    assert not any("over the cap" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# AC1: the logs are read in pages, the payload columns never loaded
# ---------------------------------------------------------------------------


class _PagedQuery:
    """Honours the page read: id > after, in id order, limit n."""

    def __init__(self, rows, calls):
        self._rows, self._calls = rows, calls
        self._after, self._limit = None, None

    def options(self, *opts):
        self._calls["options"] += 1
        return self

    def filter(self, *clauses):
        for clause in clauses:
            if getattr(getattr(clause, "left", None), "key", None) == "id":
                self._after = clause.right.value
        return self

    def order_by(self, *_a):
        return self

    def limit(self, n):
        self._limit = n
        return self

    def all(self):
        assert self._after is not None and self._limit is not None, "a read that is not a page"
        page = sorted((r for r in self._rows if r.id > self._after), key=lambda r: r.id)[: self._limit]
        self._calls["pages"].append([r.id for r in page])
        return page


class _PagedDB:
    def __init__(self, rows):
        self.calls = {"options": 0, "pages": []}
        self._rows = rows

    def query(self, *_a):
        return _PagedQuery(self._rows, self.calls)


def _log_rows(n):
    base = datetime.utcnow() - timedelta(days=1)
    # ids in insert order, executed_at out of order (a late-flushed batch)
    return [SimpleNamespace(
        id=i, agent_id=None, workspace_id=None, action_name=f"a{i}", app_name="PLATFORM",
        status="success", user_query=None, executed_at=base + timedelta(seconds=(i * 7) % n),
        router_decision={"turn_id": f"t{i}", "candidates": ["x"]},
    ) for i in range(1, n + 1)]


def test_the_log_read_pages_by_id_and_returns_the_window_in_time_order(monkeypatch):
    monkeypatch.setattr(config, "EDGE_BUILDER_LOG_PAGE_SIZE", 3, raising=False)
    rows = _log_rows(8)
    db = _PagedDB(rows)

    logs = edge_builder_logs.load_logs(db, datetime.utcnow() - _WINDOW)

    assert db.calls["pages"] == [[1, 2, 3], [4, 5, 6], [7, 8]]
    assert db.calls["options"] == 3, "every page loads only its columns"
    assert sorted(log["id"] for log in logs) == list(range(1, 9)), "no row skipped or repeated"
    times = [(log["executed_at"], log["id"]) for log in logs]
    assert times == sorted(times)
    assert logs[0]["turn_id"].startswith("t") and logs[0]["shown_actions"] == ["x"]


def test_the_log_read_never_loads_the_payload_columns():
    loaded = {column.key for column in edge_builder_logs._LOG_COLUMNS}
    assert not loaded & {"input_parameters", "output_result", "token_usage", "error_message"}
    assert {"id", "user_query", "router_decision", "executed_at"} <= loaded


# ---------------------------------------------------------------------------
# AC1: the k-means' working block stays under its cap; the clusters don't change
# ---------------------------------------------------------------------------


def test_the_kmeans_block_never_exceeds_its_cap_and_the_clusters_do_not_change(monkeypatch):
    rng = np.random.RandomState(3)
    data = rng.normal(size=(400, 16)).astype(np.float32)
    args = {"queries": [f"q{i}" for i in range(400)],
            "action_names": ["a"] * 400, "statuses": ["success"] * 400}
    whole = intent_clustering.compute_intent_clusters(embeddings=data, **args)

    blocks = []
    real_norm = np.linalg.norm

    def _norm_spy(x, *a, **k):
        if getattr(x, "ndim", 0) == 3:
            blocks.append(x.nbytes)
        return real_norm(x, *a, **k)

    monkeypatch.setattr(intent_clustering.np.linalg, "norm", _norm_spy)
    cap = 4096
    blocked = intent_clustering.compute_intent_clusters(embeddings=data, max_block_bytes=cap, **args)

    assert len(blocks) > 1 and max(blocks) <= cap, f"a k-means block of {max(blocks)} bytes"
    assert blocked.labels == whole.labels
    assert blocked.centroids == whole.centroids
    assert blocked.sample_counts == whole.sample_counts


# ---------------------------------------------------------------------------
# AC2: the run happens in a child process; out of memory ends the run only
# ---------------------------------------------------------------------------


def test_a_clustering_memory_error_in_the_worker_leaves_the_caller_alive_with_a_failed_summary(caplog):
    with caplog.at_level(logging.INFO):
        summary, pid = _run_in_worker(_clustering_runs_out_of_memory)

    assert pid == os.getpid()
    assert summary.failed is True and summary.error == "MemoryError"
    assert len(_worker_errors(caplog)) == 1, "a failed run logs one error"

    # the next night runs as usual: a fresh child, the real build_edges
    caplog.clear()
    with caplog.at_level(logging.INFO):
        nightly, _pid = _run_in_worker(_a_normal_night)
    assert nightly.failed is False and nightly.intent_clusters > 0 and nightly.logs_processed > 0
    assert _worker_errors(caplog) == []


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
def test_a_worker_the_kernel_kills_ends_the_run_never_the_caller(caplog):
    with caplog.at_level(logging.INFO):
        summary, pid = _run_in_worker(_killed_by_the_kernel)

    assert pid == os.getpid()
    assert summary.failed is True and summary.error == "BrokenProcessPool"
    assert len(_worker_errors(caplog)) == 1


@pytest.mark.skipif(not _ON_LINUX, reason="RLIMIT_AS and oom_score_adj are Linux behaviour")
def test_the_worker_ceiling_turns_an_oversized_allocation_into_a_failed_run(monkeypatch, caplog):
    ceiling_mb = 3072
    monkeypatch.setattr(config, "EDGE_BUILDER_WORKER_MEMORY_MB", ceiling_mb, raising=False)

    fits, _pid = _run_in_worker(functools.partial(_allocate, 64))
    assert fits.failed is False and fits.edges_built == 64, "a run under the ceiling must work"

    with caplog.at_level(logging.INFO):
        summary, pid = _run_in_worker(functools.partial(_allocate, ceiling_mb + 1024))
    assert pid == os.getpid()
    assert summary.failed is True and summary.error == "MemoryError"
    assert len(_worker_errors(caplog)) == 1


@pytest.mark.skipif(not _ON_LINUX, reason="oom_score_adj is Linux behaviour")
def test_the_worker_is_the_kernels_first_pick_when_memory_runs_out():
    summary, _pid = _run_in_worker(_own_oom_score_adj)
    assert summary.failed is False and summary.edges_built == 1000
