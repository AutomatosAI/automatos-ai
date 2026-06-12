"""PRD-157 S4 — retrieval perf pass (instrumentation).

Pure tests proving the constructions the PRD targets happen ONCE, not per query:
* RAG settings load in a single memoized session (was up to 7 SessionLocal()s);
* the S3 backend is pooled per workspace (was rebuilt + reinitialized per query);
* document-access tracking is offloaded, never blocking the retrieve path.

The p50 latency benchmark (integration) needs the live stack and pytest-benchmark.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

import modules.rag.service as svc


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, *args, **kwargs):
        return _FakeQuery(self._rows)

    def close(self):
        pass


class TestRagConfigCaching:
    def test_settings_loaded_once_and_memoized(self, monkeypatch):
        svc.reset_rag_settings_cache()
        calls = {"n": 0}

        def fake_sessionlocal():
            calls["n"] += 1
            return _FakeDB([("max_tokens", "3000"), ("min_similarity", "0.4")])

        monkeypatch.setattr("core.database.database.SessionLocal", fake_sessionlocal)

        first = svc._load_rag_settings()
        second = svc._load_rag_settings()

        assert calls["n"] == 1                # one session, not seven
        assert first is second                # memoized object
        assert first["max_tokens"] == "3000"
        assert svc._get_rag_setting_int("max_tokens", 2000) == 3000
        assert svc._get_rag_setting_float("min_similarity", 0.5) == 0.4
        svc.reset_rag_settings_cache()

    def test_failure_does_not_poison_cache(self, monkeypatch):
        svc.reset_rag_settings_cache()

        def boom():
            raise RuntimeError("db down")

        monkeypatch.setattr("core.database.database.SessionLocal", boom)
        assert svc._load_rag_settings() == {}          # falls back to defaults
        assert svc._get_rag_setting_int("max_tokens", 2000) == 2000  # default
        # nothing cached, so a later healthy read still works
        monkeypatch.setattr("core.database.database.SessionLocal", lambda: _FakeDB([("max_tokens", "9")]))
        assert svc._load_rag_settings()["max_tokens"] == "9"
        svc.reset_rag_settings_cache()


class TestS3BackendPooling:
    @pytest.mark.asyncio
    async def test_backend_pooled_per_workspace(self, monkeypatch):
        constructed = []

        class FakeBackend:
            def __init__(self, workspace_id):
                constructed.append(workspace_id)

            async def initialize(self):
                pass

        monkeypatch.setattr(
            "modules.search.vector_store.backends.s3_vectors_backend.S3VectorsBackend",
            FakeBackend,
        )

        rag = svc.RAGService.__new__(svc.RAGService)  # bypass __init__ (no DB)
        rag._s3_backends = {}

        b1 = await rag._get_s3_backend("ws1")
        b2 = await rag._get_s3_backend("ws1")
        b3 = await rag._get_s3_backend("ws2")

        assert b1 is b2 and b1 is not b3
        assert constructed == ["ws1", "ws2"]   # one construct per workspace


class TestAccessTrackingOffloaded:
    def test_runs_inline_without_loop(self, monkeypatch):
        rag = svc.RAGService.__new__(svc.RAGService)
        called = {"n": 0}
        rag._track_document_access = lambda chunks, ws: called.__setitem__("n", called["n"] + 1)

        rag._schedule_access_tracking([{"document_id": 1}], "ws")  # no running loop
        assert called["n"] == 1

    @pytest.mark.asyncio
    async def test_offloaded_to_thread_in_loop(self):
        rag = svc.RAGService.__new__(svc.RAGService)
        ran = threading.Event()
        main_thread = threading.get_ident()
        ran_thread = {}

        def fake_track(chunks, ws):
            ran_thread["id"] = threading.get_ident()
            ran.set()

        rag._track_document_access = fake_track
        rag._schedule_access_tracking([{"document_id": 1}], "ws")

        for _ in range(40):
            if ran.is_set():
                break
            await asyncio.sleep(0.02)

        assert ran.is_set()                     # the offloaded work ran
        assert ran_thread["id"] != main_thread  # on a worker thread, off the loop


@pytest.mark.integration
@pytest.mark.benchmark
def test_retrieval_p50_benchmark(request):
    """p50 retrieval latency on a seeded corpus — needs the live stack + S3.

    Run with: pytest -m benchmark --benchmark-only (with DATABASE_URL + S3 set).
    The before/after p50 goes in the PR per the S4 acceptance.

    Guard runs BEFORE the ``benchmark`` fixture is requested: importorskip can't
    sit alongside a ``benchmark`` parameter (pytest resolves fixtures at setup,
    so a missing pytest-benchmark errors before the body). Resolve it lazily.
    """
    pytest.importorskip("pytest_benchmark")
    benchmark = request.getfixturevalue("benchmark")
    from modules.rag.service import get_rag_service

    rag = get_rag_service()

    def _run():
        return asyncio.get_event_loop().run_until_complete(
            rag.retrieve("benchmark query", max_chunks=8, workspace_id=None)
        )

    benchmark(_run)
