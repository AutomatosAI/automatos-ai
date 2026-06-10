"""PRD-154 S7 — Field memory: agent_id actor fix + archive-don't-destroy.

Two verified breakages (reports/PLATFORM_DEEP_REVIEW_2026-06.md §2):

1. The field handlers (handlers_field.field_query / field_inject /
   field_stability) read the acting agent from ``kwargs.get("agent_id", 0)``,
   but the platform executor calls them as ``handler(db, workspace_id, params)``
   with NO kwargs — so every field op was attributed to agent 0 ("System")
   regardless of who called it. The actor id travels in ``params["_agent_id"]``
   (the executor's actor convention, see platform_executor.py), so the handlers
   must read it from there. A field tool invoked with ``_agent_id`` must now
   succeed and attribute to that agent instead of silently collapsing to 0.

2. CoordinatorService._cleanup_terminal_fields DESTROYED a terminal mission's
   field (Qdrant delete-by-field_id) and popped ``field_id`` from run.config —
   so a completed mission's Field tab read ``not_created`` with zero patterns
   forever. Fields now share ONE collection keyed by field_id payload, so a
   terminal field needs no teardown. Archive in place (mark field_archived +
   field_expired_at, KEEP field_id + the data) so the field stays queryable
   after the mission ends (BINDING D7 stepping stone). The orphan reaper still
   removes data whose run row is gone.

All deterministic — no DB, no network (mocked Session, mocked shared context,
mocked CoordinatorService methods).
"""
from __future__ import annotations

import os
import sys
import types

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import tests.conftest as _conftest  # noqa: E402
_conftest._restore_real_app_modules()

import asyncio  # noqa: E402
import uuid  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from modules.tools.discovery.handlers_field import (  # noqa: E402
    field_inject,
    field_query,
    field_stability,
)


# --------------------------------------------------------------------------- #
# 1. Handlers read the acting agent from params["_agent_id"]
# --------------------------------------------------------------------------- #

def _fake_field():
    field = MagicMock()
    field.inject = AsyncMock()
    field.query = AsyncMock(return_value=[])
    return field


def test_field_inject_reads_agent_id_from_params():
    """An agent calling inject with _agent_id attributes the pattern to it."""
    field = _fake_field()
    with patch("modules.context.factory.get_shared_context", return_value=field):
        res = asyncio.run(
            field_inject(
                db=MagicMock(),
                workspace_id=uuid.uuid4(),
                params={"key": "k", "value": "v", "field_id": "f1", "_agent_id": 7},
            )
        )
    assert res["success"] is True
    assert field.inject.call_args.kwargs["agent_id"] == 7


def test_field_query_reads_agent_id_from_params():
    field = _fake_field()
    with patch("modules.context.factory.get_shared_context", return_value=field):
        res = asyncio.run(
            field_query(
                db=MagicMock(),
                workspace_id=uuid.uuid4(),
                params={"query": "q", "field_id": "f1", "_agent_id": 3},
            )
        )
    assert res["success"] is True
    assert field.query.call_args.kwargs["agent_id"] == 3


def test_field_inject_coerces_string_agent_id():
    """LLM/JSON callers may send _agent_id as a string — must not 400."""
    field = _fake_field()
    with patch("modules.context.factory.get_shared_context", return_value=field):
        res = asyncio.run(
            field_inject(
                db=MagicMock(),
                workspace_id=uuid.uuid4(),
                params={"key": "k", "value": "v", "field_id": "f1", "_agent_id": "5"},
            )
        )
    assert res["success"] is True
    assert field.inject.call_args.kwargs["agent_id"] == 5


def test_field_inject_without_agent_id_falls_back_to_zero():
    """Absent actor → system (agent 0), preserving prior default behaviour."""
    field = _fake_field()
    with patch("modules.context.factory.get_shared_context", return_value=field):
        res = asyncio.run(
            field_inject(
                db=MagicMock(),
                workspace_id=uuid.uuid4(),
                params={"key": "k", "value": "v", "field_id": "f1"},
            )
        )
    assert res["success"] is True
    assert field.inject.call_args.kwargs["agent_id"] == 0


def test_field_query_bad_agent_id_does_not_crash():
    """A non-numeric actor degrades to 0 rather than throwing (no 400)."""
    field = _fake_field()
    with patch("modules.context.factory.get_shared_context", return_value=field):
        res = asyncio.run(
            field_query(
                db=MagicMock(),
                workspace_id=uuid.uuid4(),
                params={"query": "q", "field_id": "f1", "_agent_id": "not-a-number"},
            )
        )
    assert res["success"] is True
    assert field.query.call_args.kwargs["agent_id"] == 0


# --------------------------------------------------------------------------- #
# 2. _cleanup_terminal_fields archives instead of destroying
# --------------------------------------------------------------------------- #

def _coordinator():
    from services.coordinator_service import CoordinatorService
    return CoordinatorService.__new__(CoordinatorService)


def _db_returning(runs):
    db = MagicMock()
    db.query.return_value.filter.return_value.limit.return_value.all.return_value = runs
    return db


def test_cleanup_terminal_fields_archives_not_destroys():
    """Terminal field is archived in place: data kept, field_id kept, marked."""
    svc = _coordinator()
    # Archiving is pure metadata — it must NOT touch the field backend at all
    # (no _get_field → no destroy_context).
    svc._get_field = MagicMock()

    run = MagicMock()
    run.id = uuid.uuid4()
    run.config = {"field_id": "f1", "agent_id": 2}
    db = _db_returning([run])

    asyncio.run(svc._cleanup_terminal_fields(db))

    svc._get_field.assert_not_called()
    # field_id retained → the /field endpoint can still query it post-completion.
    assert run.config["field_id"] == "f1"
    # Archived markers stamped.
    assert run.config["field_archived"] is True
    assert "field_expired_at" in run.config
    db.flush.assert_called()


def test_cleanup_archived_field_stays_queryable():
    """Proxy for 'archived field queryable post-completion': after archiving,
    the field_id the /field endpoint reads is still present and the backend
    was never asked to tear the field down."""
    svc = _coordinator()
    svc._get_field = MagicMock()

    run = MagicMock()
    run.id = uuid.uuid4()
    run.config = {"field_id": "abc-123"}
    db = _db_returning([run])

    asyncio.run(svc._cleanup_terminal_fields(db))

    assert run.config.get("field_id") == "abc-123"
    svc._get_field.assert_not_called()


def test_cleanup_skips_already_archived_runs():
    """An already-archived run is not re-stamped or destroyed (idempotent)."""
    svc = _coordinator()
    svc._get_field = MagicMock()

    run = MagicMock()
    run.id = uuid.uuid4()
    run.config = {
        "field_id": "f1",
        "field_archived": True,
        "field_expired_at": "2026-01-01T00:00:00+00:00",
    }
    db = _db_returning([run])

    asyncio.run(svc._cleanup_terminal_fields(db))

    svc._get_field.assert_not_called()
    assert run.config["field_expired_at"] == "2026-01-01T00:00:00+00:00"


def test_cleanup_source_does_not_call_destroy_or_pop_field_id():
    """Source guard: the archive path replaces the old destroy+pop, no shim."""
    import inspect
    from services import coordinator_service

    src = inspect.getsource(coordinator_service.CoordinatorService._cleanup_terminal_fields)
    assert "field_archived" in src
    assert "_destroy_mission_field" not in src
    assert 'pop("field_id"' not in src
