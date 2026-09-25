"""F105 (25 Sep): a document search never freezes the event loop.

Every document backend's ``search`` is sync, and the local one is a Postgres
full scan (F107). RAGService._get_candidates called it on the event loop, so the
backend stood still for each scan: 826 searches since the 09:00Z restart,
median 9.4 s, 209 s at worst, about three hours of a frozen loop in eleven and a
half. /health timed out at 25 s and the scheduler's jobs slipped by up to 70 s.

A fake backend blocks its thread the way a scan does; the tests watch the loop
around it. No database, no embeddings.
"""
import asyncio
import contextvars
import threading
import time

import pytest

import modules.rag.service as svc
from config import config

WS = "00000000-0000-0000-0000-0000000000c1"
_caller = contextvars.ContextVar("f105_caller", default=None)


class _Embeddings:
    async def generate_embedding(self, text):
        return [0.1, 0.2, 0.3]


class _ScanningBackend:
    """A backend whose search holds its thread for ``seconds``, like a full scan."""

    def __init__(self, seconds):
        self.seconds = seconds
        self.threads = []
        self.callers = []
        self.in_flight = 0
        self.most_in_flight = 0
        self._lock = threading.Lock()

    def search(self, query_embedding, limit=10, min_score=0.5, filters=None):
        with self._lock:
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
            self.threads.append(threading.get_ident())
            self.callers.append(_caller.get())
        time.sleep(self.seconds)
        with self._lock:
            self.in_flight -= 1
        return [{
            "key": "doc_7_chunk_0", "score": 0.9, "content": "the answer",
            "file_name": "facts.md", "external_file_id": "7", "metadata": {},
        }]


def _rag(backend):
    rag = svc.RAGService.__new__(svc.RAGService)  # no settings read, no embedding client
    rag._embedding_manager = _Embeddings()
    rag._doc_backends = {WS: backend}
    rag._workspace_id = WS
    return rag


@pytest.fixture(autouse=True)
def no_substrate_rows(monkeypatch):
    """The documents-seam telemetry row is a task of its own; not under test."""
    monkeypatch.setattr(svc, "record_substrate_search_nowait", lambda **kwargs: None)


@pytest.fixture
def two_search_threads(monkeypatch):
    monkeypatch.setattr(config, "DOCUMENT_SEARCH_THREADS", 2, raising=False)
    monkeypatch.setattr(svc, "_search_executor", None, raising=False)
    yield
    executor = getattr(svc, "_search_executor", None)
    if executor is not None:
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_the_loop_keeps_running_while_a_search_scans():
    backend = _ScanningBackend(seconds=0.6)
    gaps = []
    done = asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    candidates = await _rag(backend)._get_candidates("what is the answer", workspace_id=WS)
    done.set()
    await beat

    assert [c["content"] for c in candidates] == ["the answer"]
    assert gaps and max(gaps) < 0.3, f"the loop stood still for {max(gaps):.2f} s during one search"
    assert backend.threads == [backend.threads[0]] and backend.threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_the_search_carries_the_callers_context():
    backend = _ScanningBackend(seconds=0)
    token = _caller.set("request-42")
    try:
        await _rag(backend)._get_candidates("q", workspace_id=WS)
    finally:
        _caller.reset(token)
    assert backend.callers == ["request-42"]


@pytest.mark.asyncio
async def test_searches_run_side_by_side_up_to_the_thread_count(two_search_threads):
    backend = _ScanningBackend(seconds=0.2)
    rag = _rag(backend)

    results = await asyncio.gather(*[rag._get_candidates(f"q{i}", workspace_id=WS) for i in range(5)])

    assert [len(r) for r in results] == [1, 1, 1, 1, 1]
    assert backend.most_in_flight == 2
