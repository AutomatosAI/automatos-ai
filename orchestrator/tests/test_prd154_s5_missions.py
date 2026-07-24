"""PRD-154 S5 — Mission handler fixes + honest create reply + duplicate route.

Four verified breakages (reports/PLATFORM_DEEP_REVIEW_2026-06.md §2):

1. ``platform_get_mission`` (handlers_missions.get_mission) crashed on EVERY call:
   it cast the UUID primary key with ``int()`` (ValueError), read ``t.result``
   which the OrchestrationTask model has no such column (real column: ``output``),
   and surfaced ``t.error_message`` (real column: ``failure_detail``). The action
   schema declared ``mission_id`` as ``integer`` though run ids are UUIDs.

2. Auto-created missions lost chat context — only the UI suggestion-card set
   ``context_messages``; the executor path (create_mission) attached nothing, so
   the planner never saw the conversation that motivated the mission.

3. A byte-duplicate ``POST /{mission_id}/resume`` route shadowed the canonical
   lifecycle resume (FastAPI keeps the first registration; the second was dead).

4. The create reply falsely claimed "the coordinator will execute them
   automatically" even though a mission defaults to ``awaiting_approval`` — the
   reply must state the real state and how to approve; ``auto_approve`` must be
   documented in the create-tool schema.

All deterministic — no DB, no network (mocked Session + mocked CoordinatorService).
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
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402

import pytest  # noqa: E402

from modules.tools.discovery import handlers_missions  # noqa: E402
from modules.tools.discovery.handlers_missions import create_mission, get_mission  # noqa: E402
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.actions_missions import register_mission_actions  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _msg(role: str, text: str):
    """Fake Message row with AI-SDK ``parts`` shape (JSONB list of dicts)."""
    return SimpleNamespace(role=role, parts=[{"type": "text", "text": text}], created_at=None)


def _db_with_messages(rows):
    """MagicMock Session whose Message query (filter→order_by→limit→all) returns *rows*."""
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.limit.return_value = q
    q.all.return_value = list(rows)
    db = MagicMock()
    db.query.return_value = q
    return db


def _db_with_run(run, tasks):
    """MagicMock Session: query().filter().first()→run, ...order_by().all()→tasks."""
    q = MagicMock()
    q.filter.return_value = q
    q.order_by.return_value = q
    q.first.return_value = run
    q.all.return_value = list(tasks)
    db = MagicMock()
    db.query.return_value = q
    return db


def _patch_coordinator(run):
    """Patch CoordinatorService so .create_mission(...) is an AsyncMock returning *run*."""
    coordinator = MagicMock()
    coordinator.create_mission = AsyncMock(return_value=run)
    cls = MagicMock(return_value=coordinator)
    return patch("services.coordinator_service.CoordinatorService", cls), coordinator


# ============================================================ AC1: get_mission


def test_get_mission_returns_seeded_mission_by_uuid():
    """A UUID mission_id must NOT crash (the old int() cast raised ValueError)
    and must surface task ``output`` + ``failure_detail`` under the real schema."""
    mid = uuid.uuid4()
    ws = uuid.uuid4()
    run = SimpleNamespace(
        id=mid, goal="Investigate the outage", state="awaiting_approval",
        config={}, plan={"tasks": []}, created_by="user_x",
        created_at=None, completed_at=None,
    )
    task = SimpleNamespace(
        id=uuid.uuid4(), title="Step 1", state="failed", agent_role="researcher",
        sequence_number=1, output="THE FULL RESULT TEXT", failure_detail="boom",
    )
    db = _db_with_run(run, [task])

    result = asyncio.run(get_mission(db, ws, {"mission_id": str(mid)}))

    assert result["success"] is True
    assert result["mission"]["id"] == mid
    t0 = result["mission"]["tasks"][0]
    assert t0["result_summary"] == "THE FULL RESULT TEXT"   # from t.output, not t.result
    assert t0["error"] == "boom"                            # from t.failure_detail


def test_get_mission_missing_id_returns_error():
    result = asyncio.run(get_mission(MagicMock(), uuid.uuid4(), {}))
    assert result["success"] is False
    assert "mission_id" in result["error"]


def test_get_mission_invalid_uuid_returns_clean_error_not_crash():
    """A non-UUID id must produce a structured error, never an uncaught ValueError."""
    result = asyncio.run(get_mission(MagicMock(), uuid.uuid4(), {"mission_id": "not-a-uuid"}))
    assert result["success"] is False
    assert "error" in result


def test_get_mission_source_uses_real_columns():
    """Source guard: the three crash sites are gone and the real columns are read."""
    src = Path(handlers_missions.__file__).read_text()
    assert "int(mission_id)" not in src, "UUID PK must not be int()-cast"
    assert "t.result" not in src, "OrchestrationTask has no 'result' column"
    assert "error_message" not in src, "OrchestrationTask has no 'error_message' column"
    assert "t.output" in src, "must read the real 'output' column"
    assert "failure_detail" in src, "must read the real 'failure_detail' column"


# ============================================ AC2: context_messages on executor


def test_auto_created_mission_carries_recent_chat_as_context_messages():
    """create_mission must attach recent workspace conversation as
    config.context_messages with source='chat' so the planner sees it."""
    ws = uuid.uuid4()
    # newest-first as the DESC query returns them
    db = _db_with_messages([
        _msg("assistant", "Sure, I can launch a mission for that."),
        _msg("user", "Please research competitor pricing"),
    ])
    run = SimpleNamespace(id=uuid.uuid4(), goal="Research pricing",
                          state="awaiting_approval", plan={"tasks": []})
    p, coordinator = _patch_coordinator(run)
    with p:
        result = asyncio.run(create_mission(db, ws, {"goal": "Research pricing"}))

    assert result["success"] is True
    cfg = coordinator.create_mission.call_args.kwargs["config"]
    assert cfg["source"] == "chat"
    msgs = cfg["context_messages"]
    # chronological (oldest first) + correct text extracted from parts
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "Please research competitor pricing"


def test_explicit_context_messages_are_not_overwritten():
    """The UI suggestion-card path already sets context_messages — respect it."""
    ws = uuid.uuid4()
    db = _db_with_messages([_msg("user", "unrelated workspace chatter")])
    run = SimpleNamespace(id=uuid.uuid4(), goal="g", state="awaiting_approval", plan={"tasks": []})
    preset = [{"role": "user", "content": "the real motivating message"}]
    p, coordinator = _patch_coordinator(run)
    with p:
        asyncio.run(create_mission(
            db, ws,
            {"goal": "g", "config": {"source": "chat", "context_messages": preset}},
        ))
    cfg = coordinator.create_mission.call_args.kwargs["config"]
    assert cfg["context_messages"] == preset


def test_create_mission_without_chat_history_omits_context():
    ws = uuid.uuid4()
    db = _db_with_messages([])
    run = SimpleNamespace(id=uuid.uuid4(), goal="g", state="awaiting_approval", plan={"tasks": []})
    p, coordinator = _patch_coordinator(run)
    with p:
        asyncio.run(create_mission(db, ws, {"goal": "g"}))
    cfg = coordinator.create_mission.call_args.kwargs["config"]
    assert "context_messages" not in cfg


# ===================================================== AC4: honest create reply


def test_create_reply_states_awaiting_approval_not_auto_started():
    """Default missions await approval — the reply must say so and how to
    approve, and must NOT claim the coordinator runs them automatically."""
    ws = uuid.uuid4()
    db = _db_with_messages([])
    run = SimpleNamespace(id=uuid.uuid4(), goal="g", state="awaiting_approval",
                          plan={"tasks": [{"title": "a"}]})
    p, _ = _patch_coordinator(run)
    with p:
        result = asyncio.run(create_mission(db, ws, {"goal": "g"}))

    assert result["state"] == "awaiting_approval"
    assert result["awaiting_approval"] is True
    msg = result["message"].lower()
    assert "approv" in msg                       # tells the user how/that to approve
    assert "automatically" not in msg            # no false "mission started"


def test_create_reply_running_when_auto_approved():
    ws = uuid.uuid4()
    db = _db_with_messages([])
    run = SimpleNamespace(id=uuid.uuid4(), goal="g", state="running",
                          plan={"tasks": [{"title": "a"}]})
    p, _ = _patch_coordinator(run)
    with p:
        result = asyncio.run(create_mission(
            db, ws, {"goal": "g", "config": {"auto_approve": True}}))

    assert result["state"] == "running"
    assert result["awaiting_approval"] is False
    assert "running" in result["message"].lower()


# ====================================== AC4: auto_approve + UUID schema in tool


def test_create_tool_schema_documents_auto_approve():
    reg = ActionRegistry()
    register_mission_actions(reg)
    create = reg._actions["platform_create_mission"]
    cfg_desc = create.parameters["properties"]["config"]["description"]
    assert "auto_approve" in cfg_desc


def test_get_tool_schema_mission_id_is_string_not_integer():
    """Run ids are UUIDs — the schema must not advertise an integer id."""
    reg = ActionRegistry()
    register_mission_actions(reg)
    get = reg._actions["platform_get_mission"]
    assert get.parameters["properties"]["mission_id"]["type"] == "string"


# ===================================================== AC3: duplicate /resume


def test_only_one_resume_route_registered():
    """Exactly one POST /{mission_id}/resume — the byte-duplicate is removed."""
    missions_src = (Path(__file__).resolve().parents[1] / "api" / "missions.py").read_text()
    # PRD-195 S4 gated the route (missions:execute) — count the decorator
    # PREFIX so the byte-duplicate guard survives dependency kwargs.
    count = missions_src.count('@router.post("/{mission_id}/resume"')
    assert count == 1, f"expected exactly one /resume route, found {count}"
