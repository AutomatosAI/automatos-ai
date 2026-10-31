"""PRD-268 US-C22b (§6.11.5, K16) — what the person's answer to their calibration remembers.

``services/calibration/memory.py`` writes through the existing memory service only, with the
memory service mocked here at ``store_long_term``. Proven here:

* Sounds right stores the line as the content, fact_type ``user_fact``, owner ``user:<id>``, the
  private scope, source_type ``platform_verified``, extra {origin ``calibration``,
  calibration_id, line_id} and subject_id ``user:<id>``, and nothing else is called;
* Reject stores ``Not so: <the line>`` with the same owner, scope and source, and ``correction``
  true in its extra;
* a line that names a secret trips ``violates_exclusions`` and is never stored: ``store``
  refuses it before the service is called, and ``answers._kept`` leaves it out and logs the
  rule's name, never the line;
* a write the service refuses (its error answer, success false, no answer) or that raises is
  ``MemoryNotStored``, logged;
* ``remember`` runs the write on the app's event loop from a worker thread, as a plain ``def``
  route calls it.

Nothing here needs a database.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain (``answers`` reaches the
# models) when the suite's own environment does not set it; nothing here connects.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import anyio  # noqa: E402
import pytest  # noqa: E402

from modules.memory.write_contract import build_memory_metadata  # noqa: E402
from services.calibration import answers, memory  # noqa: E402

WS = "6f1d2c1e-8b6a-4c43-9a55-1f2f0c1d9e01"
CALIBRATION = "0b8e5f8a-2a54-4d3e-9a0c-6b1f6f0c2a11"
PERSON = 41
LINE = "You review pricing for clients in Lisbon"
SECRET_LINE = "You keep the wallet seed phrase in the launch plan"


class _Service:
    """The memory service: records each ``store_long_term`` call and answers ``answer``, or raises ``failure``."""

    def __init__(self, answer=None, failure=None):
        self.answer = {"success": True, "id": "point-1"} if answer is None else answer
        self.failure, self.calls, self.loops = failure, [], []

    async def store_long_term(self, **kwargs):
        self.calls.append(kwargs)
        self.loops.append((asyncio.get_running_loop(), threading.get_ident()))
        if self.failure is not None:
            raise self.failure
        return self.answer


@pytest.fixture
def service(monkeypatch):
    """Install a memory service whose answer the test gives."""

    def install(**kwargs):
        fake = _Service(**kwargs)
        monkeypatch.setattr(memory, "memory_service", lambda: fake)
        return fake

    return install


# --------------------------------------------------------------------------- what is stored


def test_a_confirmed_line_is_the_persons_private_verified_fact(service):
    fake = service()
    write = memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE)

    asyncio.run(memory.store(write))

    (call,) = fake.calls
    assert set(call) == {"workspace_id", "content", "metadata", "subject_id"}
    assert (call["workspace_id"], call["content"], call["subject_id"]) == (WS, LINE, f"user:{PERSON}")
    meta = call["metadata"]
    assert (meta["type"], meta["category"], meta["scope"], meta["owner"], meta["source_type"]) == (
        "user_fact", "user_fact", "private", f"user:{PERSON}", "platform_verified")
    assert (meta["origin"], meta["calibration_id"], meta["line_id"]) == ("calibration", CALIBRATION, "line-1")
    assert "correction" not in meta
    assert meta == build_memory_metadata(
        fact_type="user_fact", source_type="platform_verified", owner=f"user:{PERSON}",
        extra={"origin": "calibration", "calibration_id": CALIBRATION, "line_id": "line-1"})


def test_a_rejected_line_is_remembered_as_its_correction(service):
    fake = service()
    write = memory.correction_write(WS, PERSON, CALIBRATION, "line-2", LINE)

    asyncio.run(memory.store(write))

    (call,) = fake.calls
    assert call["content"] == f"Not so: {LINE}"
    assert call["subject_id"] == f"user:{PERSON}"
    meta = call["metadata"]
    assert (meta["type"], meta["scope"], meta["owner"], meta["source_type"]) == (
        "user_fact", "private", f"user:{PERSON}", "platform_verified")
    assert (meta["origin"], meta["calibration_id"], meta["line_id"], meta["correction"]) == (
        "calibration", CALIBRATION, "line-2", True)


def test_the_metadata_never_lets_extra_override_the_contract():
    meta = memory.memory_metadata(memory.correction_write(WS, PERSON, CALIBRATION, "line-1", LINE))

    assert meta["scope"] == "private" and meta["owner"] == f"user:{PERSON}" and meta["type"] == "user_fact"


# --------------------------------------------------------------------------- exclusions


def test_a_line_that_names_a_secret_is_never_stored(service):
    fake = service()
    for write in (memory.line_write(WS, PERSON, CALIBRATION, "line-1", SECRET_LINE),
                  memory.correction_write(WS, PERSON, CALIBRATION, "line-1", SECRET_LINE)):
        assert memory.excluded(write) == "seed_phrase"
        with pytest.raises(ValueError):
            asyncio.run(memory.store(write))
    assert fake.calls == []
    assert memory.excluded(memory.line_write(WS, PERSON, CALIBRATION, "line-2", LINE)) is None


def test_the_answer_leaves_an_excluded_line_out_and_logs_the_rule_not_the_line(caplog):
    writes = [memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE),
              memory.line_write(WS, PERSON, CALIBRATION, "line-2", SECRET_LINE)]

    with caplog.at_level(logging.INFO, logger=answers.__name__):
        kept, refused = answers._kept(writes)

    assert [write.line_id for write in kept] == ["line-1"]
    assert refused == ("line-2",)
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "seed_phrase" in logged and "line-2" in logged
    assert "wallet" not in logged and LINE not in logged


# --------------------------------------------------------------------------- a failed write


@pytest.mark.parametrize("answer", [
    {"success": False, "error": "store_long_term failed"},
    {"success": False, "skipped": True, "error": "durable memory not configured (QDRANT_URL empty)"},
    {"success": False},
    {"error": "the memory store did not answer in time, twice"},
    "not a mapping",
])
def test_a_write_the_service_refuses_is_memory_not_stored(service, answer, caplog):
    fake = service(answer=answer)

    with caplog.at_level(logging.WARNING, logger=memory.__name__):
        with pytest.raises(memory.MemoryNotStored):
            asyncio.run(memory.store(memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE)))

    assert len(fake.calls) == 1
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert LINE not in caplog.text


def test_a_write_that_raises_is_memory_not_stored_and_logged(service, caplog):
    service(failure=ConnectionError("qdrant at 10.0.0.9 refused"))

    with caplog.at_level(logging.ERROR, logger=memory.__name__):
        with pytest.raises(memory.MemoryNotStored) as raised:
            asyncio.run(memory.store(memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE)))

    assert isinstance(raised.value.__cause__, ConnectionError)
    assert "10.0.0.9" not in str(raised.value)
    (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert isinstance(record.exc_info[1], ConnectionError)


def test_a_success_answer_without_a_success_key_is_stored(service):
    fake = service(answer={"id": "point-9"})

    asyncio.run(memory.store(memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE)))

    assert len(fake.calls) == 1


# --------------------------------------------------------------------------- from a route's worker thread


def test_remember_runs_the_write_on_the_apps_loop_from_a_worker_thread(service):
    fake = service()
    write = memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE)
    seen = {}

    async def app():
        seen["loop"], seen["thread"] = asyncio.get_running_loop(), threading.get_ident()
        await anyio.to_thread.run_sync(memory.remember, write)

    anyio.run(app)

    assert fake.calls[0]["content"] == LINE
    assert fake.loops == [(seen["loop"], seen["thread"])]


def test_remember_raises_what_the_store_raises(service):
    service(answer={"success": False, "error": "store_long_term failed"})
    write = memory.line_write(WS, PERSON, CALIBRATION, "line-1", LINE)

    async def app():
        await anyio.to_thread.run_sync(memory.remember, write)

    with pytest.raises(memory.MemoryNotStored):
        anyio.run(app)
