"""PRD-159 S2 — tool-execution outcome capture.

Failures and notable successes become typed ``tool_outcome`` memories under the
workspace namespace, written direct (infer:false) and deduped by content-hash;
trivial successes are gated out. Pure helpers are unit-tested directly; the
async write is tested with a recording fake service (no DB / durable store).
"""
import os
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

from modules.memory import tool_outcome_capture as toc  # noqa: E402
from modules.memory.tool_outcome_capture import (  # noqa: E402
    build_tool_outcome,
    should_dedupe,
    write_tool_outcome,
    capture_tool_outcome,
    TOOL_OUTCOME_TYPE,
)


@pytest.fixture(autouse=True)
def _clear_dedup():
    toc._SEEN_HASHES.clear()
    yield
    toc._SEEN_HASHES.clear()


class _FakeService:
    def __init__(self):
        self.calls = []

    async def store_two_tier(self, **kwargs):
        self.calls.append(kwargs)
        return [("global", {"success": True})]


# --- noise gate / record building ------------------------------------------

def test_failed_composio_call_builds_tool_outcome_with_app_action_errorclass():
    rec = build_tool_outcome(
        tool_name="SLACK_SEND_MESSAGE",
        parameters={"action": "SLACK_SEND_MESSAGE", "app_name": "slack",
                    "params": {"channel": "#ops"}},
        result={"success": False, "error": "not_in_channel: bot is not in #ops"},
        workspace_id="ws1",
    )
    assert rec is not None
    assert rec["type"] == TOOL_OUTCOME_TYPE
    assert rec["metadata"]["app"] == "slack"
    assert rec["metadata"]["action"] == "SLACK_SEND_MESSAGE"
    assert rec["metadata"]["error_class"] == "not_found"   # not_in_channel → not_found class
    assert "SLACK_SEND_MESSAGE" in rec["fact"]
    assert rec["metadata"]["success"] is False


def test_trivial_success_is_gated_out():
    rec = build_tool_outcome(
        tool_name="COMPOSIO_SEARCH_WEB",
        parameters={"action": "COMPOSIO_SEARCH_WEB"},
        result={"success": True, "data": {"results": ["a", "b"]}},  # no id-like keys
        workspace_id="ws1",
    )
    assert rec is None


def test_notable_success_is_captured():
    rec = build_tool_outcome(
        tool_name="SLACK_CREATE_CHANNEL",
        parameters={"action": "SLACK_CREATE_CHANNEL", "app_name": "slack"},
        result={"success": True, "data": {"channel_id": "C12345", "name": "ops"}},
        workspace_id="ws1",
    )
    assert rec is not None
    assert rec["metadata"]["success"] is True
    assert rec["metadata"]["category"] == TOOL_OUTCOME_TYPE


def test_rate_limit_and_auth_error_classes():
    rl = build_tool_outcome(
        tool_name="X", parameters={"action": "X"},
        result={"success": False, "error": "429 Too Many Requests (rate limit)"},
        workspace_id="ws1")
    assert rl["metadata"]["error_class"] == "rate_limit"
    au = build_tool_outcome(
        tool_name="Y", parameters={"action": "Y"},
        result={"success": False, "error": "401 Unauthorized: invalid token"},
        workspace_id="ws1")
    assert au["metadata"]["error_class"] == "auth"


def test_no_record_without_workspace():
    assert build_tool_outcome(
        tool_name="X", parameters={}, result={"success": False, "error": "x"},
        workspace_id="") is None


# --- content-hash dedup -----------------------------------------------------

def test_identical_outcome_dedupes_to_one():
    rec = build_tool_outcome(
        tool_name="SLACK_SEND_MESSAGE",
        parameters={"action": "SLACK_SEND_MESSAGE", "app_name": "slack"},
        result={"success": False, "error": "not_in_channel"},
        workspace_id="ws1",
    )
    h = rec["metadata"]["outcome_hash"]
    assert should_dedupe(h) is False     # first sighting → write
    assert should_dedupe(h) is True      # second identical → skip


def test_different_error_class_not_deduped():
    a = build_tool_outcome(tool_name="T", parameters={"action": "T"},
                           result={"success": False, "error": "429 rate limit"},
                           workspace_id="ws1")
    b = build_tool_outcome(tool_name="T", parameters={"action": "T"},
                           result={"success": False, "error": "401 auth"},
                           workspace_id="ws1")
    assert a["metadata"]["outcome_hash"] != b["metadata"]["outcome_hash"]


# --- async write + fire-and-forget capture ----------------------------------

@pytest.mark.asyncio
async def test_write_tool_outcome_persists_global_infer_false():
    svc = _FakeService()
    rec = build_tool_outcome(
        tool_name="SLACK_SEND_MESSAGE",
        parameters={"action": "SLACK_SEND_MESSAGE", "app_name": "slack"},
        result={"success": False, "error": "not_in_channel"},
        workspace_id="ws1",
    )
    ok = await write_tool_outcome(rec, workspace_id="ws1", agent_id=3, service=svc)
    assert ok is True
    assert len(svc.calls) == 1
    call = svc.calls[0]
    assert call["tier"] == "global"
    assert call["metadata"]["category"] == TOOL_OUTCOME_TYPE
    assert call["messages"][0]["content"] == rec["fact"]


@pytest.mark.asyncio
async def test_capture_schedules_write_for_failure(monkeypatch):
    svc = _FakeService()
    monkeypatch.setattr(
        "modules.memory.unified_memory_service.get_unified_memory_service",
        lambda: svc,
    )
    task = capture_tool_outcome(
        tool_name="SLACK_SEND_MESSAGE",
        parameters={"action": "SLACK_SEND_MESSAGE", "app_name": "slack"},
        result={"success": False, "error": "not_in_channel"},
        workspace_id="ws1",
        agent_id=3,
    )
    assert task is not None
    await task
    assert len(svc.calls) == 1


@pytest.mark.asyncio
async def test_capture_skips_trivial_success(monkeypatch):
    svc = _FakeService()
    monkeypatch.setattr(
        "modules.memory.unified_memory_service.get_unified_memory_service",
        lambda: svc,
    )
    task = capture_tool_outcome(
        tool_name="COMPOSIO_SEARCH_WEB",
        parameters={"action": "COMPOSIO_SEARCH_WEB"},
        result={"success": True, "data": {"results": []}},
        workspace_id="ws1",
        agent_id=3,
    )
    assert task is None           # gated out → nothing scheduled
    assert len(svc.calls) == 0


@pytest.mark.asyncio
async def test_capture_dedupes_identical_outcomes(monkeypatch):
    svc = _FakeService()
    monkeypatch.setattr(
        "modules.memory.unified_memory_service.get_unified_memory_service",
        lambda: svc,
    )
    args = dict(
        tool_name="SLACK_SEND_MESSAGE",
        parameters={"action": "SLACK_SEND_MESSAGE", "app_name": "slack"},
        result={"success": False, "error": "not_in_channel"},
        workspace_id="ws1",
        agent_id=3,
    )
    t1 = capture_tool_outcome(**args)
    t2 = capture_tool_outcome(**args)
    assert t1 is not None
    await t1
    assert t2 is None             # identical outcome deduped
    assert len(svc.calls) == 1
