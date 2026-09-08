"""PRD-238 W1 — the honest window: every tool call closes its line, a super
admin's chat can run super-admin-only tools, a cancelled ticket reports back,
memory is honestly off without Qdrant, and the tool ranker degrades to a
lexical shortlist instead of the full enum.

Pure unit tests (db and services mocked); no network, no Postgres, no Redis.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# S3 · tool-end frames carry a summary; skipped calls close their line
# ---------------------------------------------------------------------------

def test_tool_result_summary_prefers_headline_fields_and_caps():
    from consumers.chatbot.tool_summary import SUMMARY_MAX_CHARS, tool_result_summary

    assert tool_result_summary({"message": "  Bob is   busy  "}) == "Bob is busy"
    assert tool_result_summary({"error": "boom", "message": "x"}) == "x"  # message outranks error
    assert tool_result_summary({"results": [1, 2, 3]}) == "3 results"
    assert tool_result_summary({"tasks": [1]}) == "1 task"
    assert tool_result_summary({"count": 7}) == "7 count"
    assert tool_result_summary({"success": False}) == "Failed"
    assert tool_result_summary({"success": True}) == "Done"
    assert tool_result_summary({"weird": object()}) is None
    assert tool_result_summary(None) is None
    assert tool_result_summary(42) is None
    long = tool_result_summary({"summary": "a" * 500})
    assert long is not None and len(long) <= SUMMARY_MAX_CHARS and long.endswith("…")


def test_tool_end_frame_carries_summary_and_skipped():
    import json

    from consumers.chatbot.streaming import get_streaming_handler

    h = get_streaming_handler()
    frame = h.format_aisdk_tool_end("c1", "platform_get_agent", success=False, duration_ms=3, summary="Skipped", skipped=True)
    assert frame.startswith("d:")
    payload = json.loads(frame[2:])
    assert payload["type"] == "tool-end"
    assert payload["data"]["summary"] == "Skipped"
    assert payload["data"]["skipped"] is True
    plain = json.loads(h.format_aisdk_tool_end("c2", "t", success=True)[2:])
    assert "summary" not in plain["data"] and "skipped" not in plain["data"]


# ---------------------------------------------------------------------------
# S9 · the driving principal's super-admin role reaches the executor gate
# ---------------------------------------------------------------------------

def test_caller_context_writes_only_the_literal_super_admin_role():
    try:
        from consumers.chatbot.service import build_tool_caller_context
    except Exception as e:  # env without the heavy service deps
        pytest.skip(f"chat service not importable here: {e}")

    su = build_tool_caller_context(
        user_query="q", conversation_id="c", turn_id="t", driving_clerk="u", prior_action=None,
        system_role="super_admin",
    )
    assert su["system_role"] == "super_admin"
    for role in ("admin", "user", "service", None):
        ctx = build_tool_caller_context(
            user_query="q", conversation_id="c", turn_id="t", driving_clerk="u", prior_action=None,
            system_role=role,
        )
        assert "system_role" not in ctx, role


# ---------------------------------------------------------------------------
# S5 · a cancelled ticket reports to its watch, with an honest ending line
# ---------------------------------------------------------------------------

def test_ending_summary_reads_the_session_facts_only():
    try:
        from api.board_tasks import ending_summary
    except Exception as e:
        pytest.skip(f"api.board_tasks not importable here: {e}")

    task = SimpleNamespace(runtime_ref={
        "exit_reason": "cancelled", "denials": 1, "attempt": 2,
        "recent_tools": [{"name": "Read"}, {"name": "Bash"}], "files_touched": ["a.py", "b.py"],
        "transcript_path": "/should/never/appear",
    })
    line = ending_summary(task)
    assert line == "exit: cancelled; 1 permission denial; last tool: Bash; 2 files touched; attempt 2"
    assert "transcript" not in line
    assert ending_summary(SimpleNamespace(runtime_ref=None)) is None
    assert ending_summary(SimpleNamespace(runtime_ref={})) is None
    assert ending_summary(SimpleNamespace()) is None


def test_finalize_cancelled_reports_to_the_watch(monkeypatch):
    try:
        from api.board_tasks import finalize_board_task_run
    except Exception as e:
        pytest.skip(f"api.board_tasks not importable here: {e}")

    calls = []
    monkeypatch.setattr("services.watch_hooks.watch_ingest_terminal", lambda db, **kw: calls.append(kw))
    monkeypatch.setattr("api.board_tasks.notify_board_event", lambda *a, **k: None)
    task = SimpleNamespace(id=92, status="in_progress", runtime_ref={"exit_reason": "cancelled", "denials": 1})
    db = MagicMock()
    db.query.return_value.get.return_value = task

    status = asyncio.run(finalize_board_task_run(
        db, task_id=92, workspace_id=str(uuid.uuid4()), agent_id=15,
        exec_result={"status": "cancelled"},
    ))
    assert status == "cancelled"
    assert task.status == "cancelled"
    assert len(calls) == 1
    assert calls[0]["target_type"] == "board_task" and calls[0]["target_id"] == "92"
    assert calls[0]["terminal_state"] == "cancelled"
    assert calls[0]["summary"] == "exit: cancelled; 1 permission denial"


def test_cancelled_watch_says_so_in_the_originating_chat(monkeypatch):
    try:
        from services import watch_service as ws
    except Exception as e:
        pytest.skip(f"watch_service not importable here: {e}")

    watch = SimpleNamespace(
        id=uuid.uuid4(), workspace_id=uuid.uuid4(), title="Ticket: Write basic webpage",
        origin_chat_id=uuid.uuid4(), created_by=None, status=ws.WatchStatus.WATCHING.value,
        final_verdict=None,
    )
    delivered = []
    monkeypatch.setattr(ws.WatchService, "find_live_watch", staticmethod(lambda db, **kw: watch))
    monkeypatch.setattr(ws.WatchService, "ingest", staticmethod(lambda db, w, **kw: SimpleNamespace(id="evt", **kw)))
    monkeypatch.setattr(ws.WatchService, "transition", staticmethod(lambda db, w, status, reason=None: setattr(w, "status", status.value)))
    import services.chat_messenger as messenger
    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kw: delivered.append(kw))

    event = ws.WatchService.ingest_terminal(
        MagicMock(), workspace_id=watch.workspace_id, target_type="board_task", target_id="92",
        terminal_state="cancelled", summary="exit: cancelled; 1 permission denial",
    )
    assert event is not None
    assert watch.status == ws.WatchStatus.CANCELLED.value
    assert len(delivered) == 1
    msg = delivered[0]
    assert msg["chat_id"] == str(watch.origin_chat_id)
    assert msg["source"] == {"origin": "watcher", "event": "watch_cancelled"}
    assert msg["text"] == "Ticket: Write basic webpage (#92) ended: cancelled. exit: cancelled; 1 permission denial"


def test_cancelled_report_text_shapes():
    try:
        from services.watch_service import cancelled_report_text
    except Exception as e:
        pytest.skip(f"watch_service not importable here: {e}")

    w = SimpleNamespace(title="Mission: launch")
    assert cancelled_report_text(w, "mission", "m1", None) == "Mission: launch ended: cancelled."
    assert cancelled_report_text(SimpleNamespace(title=""), "board_task", "7", "why") == "board_task 7 (#7) ended: cancelled. why"


# ---------------------------------------------------------------------------
# S10 · memory is honestly off without a Qdrant URL
# ---------------------------------------------------------------------------

def test_durable_store_is_off_without_a_url(monkeypatch):
    try:
        from modules.memory import durable_store as ds
    except Exception as e:
        pytest.skip(f"durable_store not importable here: {e}")

    monkeypatch.setattr(ds.config, "QDRANT_URL", "", raising=False)
    def _must_not_dial(**kw):
        raise AssertionError("must not dial Qdrant")

    monkeypatch.setattr(ds, "AsyncQdrantClient", _must_not_dial)
    store = ds.DurableMemoryStore()
    assert store.enabled is False

    async def scenario():
        added = await store.add(messages=[{"role": "user", "content": "x"}], user_id="u", workspace_id=str(uuid.uuid4()))
        return (
            added,
            await store.search(query="q", user_id="u"),
            await store.get_all(user_id="u"),
            await store.delete(["m"], "u"),
            await store.erase_workspace(str(uuid.uuid4())),
            await store.health(),
        )

    added, found, everything, deleted, erased, health = asyncio.run(scenario())
    assert added["success"] is False and added["skipped"] is True
    assert found == [] and everything == [] and deleted is False and erased == 0
    assert health["healthy"] is False and health["configured"] is False


def test_durable_probe_and_field_factory_honour_the_empty_url(monkeypatch):
    try:
        from services.heartbeat_service import durable_probe_enabled
        from modules.context import factory
    except Exception as e:
        pytest.skip(f"not importable here: {e}")

    assert durable_probe_enabled(SimpleNamespace(QDRANT_URL="")) is False
    assert durable_probe_enabled(SimpleNamespace(QDRANT_URL="http://qdrant:6333")) is True

    monkeypatch.setattr(factory.config, "QDRANT_URL", "", raising=False)
    monkeypatch.setattr(factory, "_instances", {})
    assert factory.get_shared_context("vector_field") is None
    assert factory.get_shared_context("vector_field") is None  # second call: no construction, no repeat log


def test_unified_service_reports_durable_unconfigured(monkeypatch):
    try:
        from modules.memory import unified_memory_service as ums
    except Exception as e:
        pytest.skip(f"unified_memory_service not importable here: {e}")
    from config import config as app_config

    monkeypatch.setattr(app_config, "QDRANT_URL", "", raising=False)
    svc = object.__new__(ums.UnifiedMemoryService)
    assert ums.UnifiedMemoryService.is_durable_configured.fget(svc) is False


# ---------------------------------------------------------------------------
# S11 · the ranker's timeout degrades to a lexical shortlist, never the full enum
# ---------------------------------------------------------------------------

def _action(name, description="", tags=(), examples=(), category="agents", admin_only=False, promoted=False, super_admin_only=False):
    return SimpleNamespace(
        name=name, description=description, tags=list(tags), examples=list(examples),
        category=category, admin_only=admin_only, promoted=promoted, super_admin_only=super_admin_only,
    )


def test_lexical_rank_orders_by_overlap_and_ignores_stopwords():
    from modules.tools.discovery.lexical_rank import lexical_rank, tokens

    assert tokens("How is the system running today?") == {"system", "running", "today"}
    actions = [
        _action("platform_get_system_health", "Check the health of the running system"),
        _action("platform_list_agents", "List every agent", tags=["fleet"]),
        _action("platform_fleet_status", "Fleet status: which agents are running", examples=["is the system ok"]),
    ]
    ranked = lexical_rank("How is the system running today?", actions, top_k=5)
    names = [n for n, _ in ranked]
    assert names[0] == "platform_get_system_health"
    assert "platform_fleet_status" in names
    assert "platform_list_agents" not in names  # no overlap at all
    assert lexical_rank("the and of", actions) == []
    assert lexical_rank("system", actions, top_k=1) == [ranked[0]]


def test_router_falls_back_to_the_lexical_shortlist(monkeypatch):
    try:
        from modules.tools import tool_router as tr
    except Exception as e:
        pytest.skip(f"tool_router not importable here: {e}")

    class _Index:
        async def rank_actions(self, *a, **k):
            return []  # the embed timed out

        def lexical_rank(self, query, top_k=15, **k):
            return ["platform_get_agent", "platform_list_agents"][:top_k]

    monkeypatch.setattr(
        "modules.tools.discovery.action_semantic_index.get_action_semantic_index", lambda: _Index()
    )
    names = asyncio.run(tr._rank_actions_for_dispatcher_async("who is bob", top_k=1, exclude_admin=True, exclude_promoted=True))
    assert names == ["platform_get_agent"]

    class _Empty(_Index):
        def lexical_rank(self, *a, **k):
            return []

    monkeypatch.setattr(
        "modules.tools.discovery.action_semantic_index.get_action_semantic_index", lambda: _Empty()
    )
    assert asyncio.run(tr._rank_actions_for_dispatcher_async("zzz", top_k=5, exclude_admin=True, exclude_promoted=True)) is None
