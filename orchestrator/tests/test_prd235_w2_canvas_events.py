"""PRD-235 W2 S3: a Claude Code session's hook events become Code Canvas events."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services.cli_host_service import canvas_events_for, publish_canvas_events  # noqa: E402

WS = "00000000-0000-0000-0000-0000000000c1"
TASK = SimpleNamespace(id=71, workspace_id=WS)
REF = {"session_id": "bc25"}


def _types(events):
    return [e["event_type"] for e in events]


def test_tool_events_map_to_canvas_calls_results_and_file_edits():
    pre = canvas_events_for(TASK, REF, {"event": "PreToolUse", "tool_name": "Bash", "subject": "python3 hello.py", "at": 1.0})
    assert _types(pre) == ["canvas.tool.call"]
    assert pre[0]["data"] == {"source": "cli", "task_id": 71, "session_id": "bc25", "at": 1.0, "tool_name": "Bash", "input": {"subject": "python3 hello.py"}}
    assert pre[0]["workspace_id"] == WS and pre[0]["schema_version"] == 1 and pre[0]["timestamp"]
    post = canvas_events_for(TASK, REF, {"event": "PostToolUse", "tool_name": "Write", "subject": f"/w/{WS}/sessions/71/hello.py"})
    assert _types(post) == ["canvas.tool.result", "canvas.file.edit"]
    assert post[1]["data"]["path"] == "sessions/71/hello.py"   # workspace-relative → the tree refreshes


def test_lifecycle_and_unknown_events():
    assert _types(canvas_events_for(TASK, REF, {"event": "SessionStart"})) == ["canvas.session.status"]
    assert canvas_events_for(TASK, REF, {"event": "SessionStart"})[0]["data"]["status"] == "running"
    assert canvas_events_for(TASK, REF, {"event": "Stop"})[0]["data"]["status"] == "stopped"
    assert canvas_events_for(TASK, REF, {"event": "Notification", "message": "x"}) == []
    assert canvas_events_for(TASK, REF, {"event": "PostToolUse"}) == []   # no tool name → nothing


def test_publish_is_fail_soft(monkeypatch):
    import core.redis.client as rc
    monkeypatch.setattr(rc, "get_redis_client", lambda: None)
    assert publish_canvas_events(WS, [{"event_type": "canvas.session.status"}]) == 0

    class _C:
        def __init__(self): self.sent = []
        def publish(self, channel, message): self.sent.append((channel, message)); return True
    c = _C()
    monkeypatch.setattr(rc, "get_redis_client", lambda: c)
    assert publish_canvas_events(WS, [{"a": 1}, {"b": 2}]) == 2
    assert c.sent[0][0] == f"workspace:ws:{WS}:canvas:events"


def test_permission_questions_are_remembered_answered_and_delivered_once():
    from services.cli_host_service import note_pending_permission, record_permission_decision, take_undelivered_decisions
    ref = {"session_id": "bc25"}
    ev = {"event": "PermissionRequest", "request_id": "r1", "tool_name": "Bash", "subject": "pip --version", "reason": "outside the allowlist"}
    assert note_pending_permission(ref, ev)["request_id"] == "r1"
    assert note_pending_permission(ref, {"event": "PermissionRequest"}) is None
    card = canvas_events_for(TASK, ref, ev)
    assert _types(card) == ["canvas.permission.request"] and card[0]["data"]["command"] == "pip --version" and card[0]["data"]["request_id"] == "r1"
    assert record_permission_decision(ref, "nope", True, "user:2") is False
    assert record_permission_decision(ref, "r1", True, "user:2") is True and ref["pending_permissions"] == []
    assert take_undelivered_decisions(ref) == [{"request_id": "r1", "approved": True}]
    assert take_undelivered_decisions(ref) == []


def test_two_questions_are_two_cards_and_each_needs_its_own_answer():
    """Ticket 78 (2026-09-03): the operator approved 'pip --version'; the session then
    asked 'python3 -m pip --version' and nobody answered → an honest deny after the
    timeout. Each request id is independent."""
    from services.cli_host_service import note_pending_permission, record_permission_decision
    ref = {"session_id": "s"}
    note_pending_permission(ref, {"event": "PermissionRequest", "request_id": "a", "tool_name": "Bash", "subject": "pip --version"})
    note_pending_permission(ref, {"event": "PermissionRequest", "request_id": "b", "tool_name": "Bash", "subject": "python3 -m pip --version"})
    assert record_permission_decision(ref, "a", True, "user:2") is True
    assert [p["request_id"] for p in ref["pending_permissions"]] == ["b"]
