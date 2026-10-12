"""
P256-FIX-N13-2 (F409): the nightly EdgeBuilder's k-means never freezes the loop.

Night 13 (11–12 Oct): from ~03:20 UTC the backend sat at 105% CPU with the event
loop frozen ~25 minutes — services/edge_builder_scheduler awaited build_edges,
whose clustering (intent_clustering._kmeans) ran inside the coroutine. Every chat
and /health timed out until a restart.

build_edges now awaits only the embedding call on the loop; the log read and the
compute-and-write (the k-means included) run in worker threads, and no session is
open across the embedding await. These tests drive build_edges with a fake session
and a fake embedder (no database, no network) and prove:

  * while the clustering takes over a second, a coroutine on the same loop keeps
    completing within 100 ms;
  * the clusters and the log → cluster map equal the k-means' own on a seeded
    fixture (the clustering itself is unchanged);
  * the read and the write run off the loop thread, and no session is open while
    the embedding is awaited.
"""
from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from itertools import count

import numpy as np

import core.services.edge_builder as edge_builder
import core.services.edge_builder_clusters as edge_builder_clusters
from core.services.intent_clustering import compute_intent_clusters

_CLUSTERING_SECONDS = 1.2  # the fixture's clustering is held at least this long
_TICK_SECONDS = 0.01
_MAX_LOOP_GAP_SECONDS = 0.1
_DIMENSION = 16
_ACTIONS = ("platform_list_agents", "platform_get_agent", "composio_send_email")


# ---------------------------------------------------------------------------
# Fixture: seeded logs + embeddings, a fake session and a fake embedder
# ---------------------------------------------------------------------------


def _seeded_logs_and_vectors(n_per_blob=20, seed=7):
    """Three embedding blobs; every fourth log carries no query (index mapping)."""
    rng = np.random.RandomState(seed)
    centres = rng.normal(0, 5, size=(len(_ACTIONS), _DIMENSION))
    base = datetime.utcnow() - timedelta(days=1)
    logs, vectors = [], {}
    for blob, action in enumerate(_ACTIONS):
        for i in range(n_per_blob):
            idx = len(logs)
            query = None if idx % 4 == 3 else f"{action} ask {i}"
            logs.append({
                "id": idx + 1, "agent_id": None, "workspace_id": "ws-1",
                "action_name": action, "app_name": "PLATFORM", "status": "success",
                "user_query": query, "executed_at": base + timedelta(seconds=idx),
                "turn_id": f"t{i}", "conversation_id": None, "shown_actions": [],
            })
            if query:
                vectors[query] = (centres[blob] + rng.normal(0, 0.3, _DIMENSION)).tolist()
    return logs, vectors


class _FakeQuery:
    def filter(self, *_a, **_k):
        return self

    def all(self):
        return []


class _FakeDB:
    """Enough of a Session for the organic-cluster rebuild: no existing rows."""

    def __init__(self, store):
        self._store = store

    def query(self, *_a):
        return _FakeQuery()

    def add(self, obj):
        obj.id = next(self._store["ids"])
        self._store["clusters"].append(obj)

    def flush(self):
        return None


class _FakeEmbedder:
    def __init__(self, vectors, store):
        self._vectors = vectors
        self._store = store
        self.provider = None

    def get_provider_info(self):
        return {"provider": "fake", "model": "fake-embed", "dimension": _DIMENSION}

    def get_dimension(self):
        return _DIMENSION

    async def generate_embeddings_batch(self, texts, max_concurrent=5):
        self._store["open_at_embed"].append(self._store["open"])
        await asyncio.sleep(0)
        return [self._vectors[t] for t in texts]


def _slow_clustering(store):
    """The real k-means, held CPU-bound (pure Python, GIL-holding) past a second."""

    def _spy(**kwargs):
        store["cluster_thread"] = threading.get_ident()
        store["cluster_start"] = time.monotonic()
        result = compute_intent_clusters(**kwargs)
        while time.monotonic() - store["cluster_start"] < _CLUSTERING_SECONDS:
            pass  # busy: CPU-bound, holding the GIL between switches
        store["cluster_end"] = time.monotonic()
        return result

    return _spy


def _wire(monkeypatch, logs, vectors):
    store = {"open": 0, "open_at_embed": [], "session_threads": [],
             "clusters": [], "ids": count(1), "maps": []}

    @contextmanager
    def _session():
        store["open"] += 1
        store["session_threads"].append(threading.get_ident())
        try:
            yield _FakeDB(store)
        finally:
            store["open"] -= 1

    real_affinities = edge_builder._compute_affinities

    def _affinities_spy(logs_, cluster_map):
        store["maps"].append(dict(cluster_map))
        return real_affinities(logs_, cluster_map)

    monkeypatch.setattr(edge_builder, "get_db_session", _session)
    monkeypatch.setattr(edge_builder, "_load_logs", lambda db, cutoff, workspace_id=None: logs)
    monkeypatch.setattr(edge_builder, "_upsert_edges", lambda db, d: len(d))
    monkeypatch.setattr(edge_builder, "_upsert_global_edges", lambda db, d: 0)
    monkeypatch.setattr(edge_builder, "_upsert_failed_after_edges", lambda db, d: len(d))
    monkeypatch.setattr(edge_builder, "_upsert_affinities", lambda db, affs: len(affs))
    monkeypatch.setattr(edge_builder, "_compute_affinities", _affinities_spy)
    monkeypatch.setattr(edge_builder, "get_embedding_manager", lambda: _FakeEmbedder(vectors, store))
    monkeypatch.setattr(edge_builder_clusters, "compute_intent_clusters", _slow_clustering(store))
    return store


async def _build_while_ticking():
    """Run build_edges with a ticker coroutine on the same loop; return its ticks."""
    stop = asyncio.Event()
    ticks = []

    async def _ticker():
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(_TICK_SECONDS)
            now = time.monotonic()
            ticks.append((now, now - last))
            last = now

    ticker = asyncio.create_task(_ticker())
    summary = await edge_builder.build_edges(window=timedelta(days=30))
    stop.set()
    await ticker
    return summary, ticks, threading.get_ident()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_loop_keeps_serving_while_the_clustering_runs(monkeypatch):
    logs, vectors = _seeded_logs_and_vectors()
    store = _wire(monkeypatch, logs, vectors)

    summary, ticks, loop_thread = asyncio.run(_build_while_ticking())

    start, end = store["cluster_start"], store["cluster_end"]
    assert end - start >= _CLUSTERING_SECONDS, "the fixture's clustering must take over a second"
    assert store["cluster_thread"] != loop_thread, "the k-means ran on the event loop's thread"
    during = [gap for at, gap in ticks if start <= at <= end]
    assert len(during) >= 5, f"no coroutine ran on the loop while clustering ({len(during)} ticks)"
    assert max(during) < _MAX_LOOP_GAP_SECONDS, (
        f"a coroutine on the loop waited {max(during) * 1000:.0f} ms during the clustering"
    )
    assert summary.intent_clusters == len(store["clusters"]) > 0


def test_clusters_equal_the_kmeans_own_on_a_seeded_fixture(monkeypatch):
    logs, vectors = _seeded_logs_and_vectors()
    store = _wire(monkeypatch, logs, vectors)

    asyncio.run(edge_builder.build_edges(window=timedelta(days=30)))

    picked = [(i, log) for i, log in enumerate(logs) if log["user_query"]]
    expected = compute_intent_clusters(
        embeddings=np.array([vectors[log["user_query"]] for _, log in picked], dtype=np.float32),
        queries=[log["user_query"] for _, log in picked],
        action_names=[log["action_name"] for _, log in picked],
        statuses=[log["status"] for _, log in picked],
    )
    rows = store["clusters"]
    assert [r.centroid_embedding for r in rows] == expected.centroids
    assert [r.sample_count for r in rows] == expected.sample_counts
    assert [r.action_names_hot for r in rows] == expected.action_names_hot
    assert all(r.provenance == "organic" and r.embedding_model_key == f"fake:fake-embed:{_DIMENSION}"
               for r in rows)
    ids = [r.id for r in rows]
    assert store["maps"] == [
        {log_idx: ids[expected.labels[n]] for n, (log_idx, _) in enumerate(picked)}
    ]


def test_no_session_is_open_across_the_embedding_await(monkeypatch):
    logs, vectors = _seeded_logs_and_vectors()
    store = _wire(monkeypatch, logs, vectors)

    _summary, _ticks, loop_thread = asyncio.run(_build_while_ticking())

    assert store["open_at_embed"] == [0], "a session (and its transaction) was open while embedding"
    assert len(store["session_threads"]) == 2, "one read session, one write session"
    assert loop_thread not in store["session_threads"], "a database session ran on the loop's thread"
    assert store["open"] == 0
