"""F103 (night 3) — a mission-field write that times out is tried once more.

The mission field writes to the same Qdrant that answered durable memory with
408s during the F105 freezes. A field write that times out is now tried once
more after a short pause, as a durable-memory write is; a second timeout stays
the honest failure platform_field_inject already returned ("Field inject
failed: …").
"""
from __future__ import annotations

import asyncio

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse

from modules.context.adapters import vector_field as vf

FIELD = "field-7"


def _408():
    return UnexpectedResponse(408, "Request Timeout", b"", httpx.Headers())


class _Client:
    """Raises the queued outcomes in turn; None is a write that lands."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def upsert(self, collection_name, points):
        self.calls += 1
        assert collection_name == vf.SHARED_COLLECTION and len(points) == 1
        outcome = self.outcomes.pop(0) if self.outcomes else None
        if outcome is not None:
            raise outcome


class _Embedder:
    async def generate_embedding(self, text):
        return [0.0, 0.1, 0.2]


def _field(monkeypatch, *outcomes):
    from modules.context import factory

    monkeypatch.setattr(vf.config, "MEMORY_WRITE_RETRY_PAUSE_S", 0, raising=False)
    field = vf.VectorFieldSharedContext.__new__(vf.VectorFieldSharedContext)
    field._client, field._embedder = _Client(*outcomes), _Embedder()
    field._bootstrap_done, field._boundary_permeability = True, 1.0

    async def no_duplicate(context_id, content_hash):
        return None

    field._find_by_hash = no_duplicate
    monkeypatch.setattr(factory, "get_shared_context", lambda backend=None: field)
    return field


def _inject():
    from modules.tools.discovery.handlers_field import field_inject

    return asyncio.run(field_inject(None, "ws", {"key": "delivery-day", "value": "Thursdays", "_agent_id": 268},
                                    field_id=FIELD))


def test_a_timeout_then_success_is_injected(monkeypatch):
    field = _field(monkeypatch, _408(), None)
    assert _inject() == {"success": True, "message": "Pattern 'delivery-day' shared with the mission field."}
    assert field._client.calls == 2


def test_two_timeouts_are_the_honest_failure_it_was(monkeypatch):
    field = _field(monkeypatch, _408(), _408())
    result = _inject()
    assert result["success"] is False and result["error"].startswith("Field inject failed: ")
    assert field._client.calls == 2


def test_a_healthy_write_is_unchanged(monkeypatch):
    field = _field(monkeypatch)
    assert _inject()["success"] is True
    assert field._client.calls == 1


def test_any_other_failure_is_not_retried(monkeypatch):
    field = _field(monkeypatch, UnexpectedResponse(400, "Bad Request", b"wrong vector size", httpx.Headers()))
    assert _inject()["success"] is False
    assert field._client.calls == 1
