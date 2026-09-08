"""PRD-239 S7 v2 — the Runtime Canvas: one session ticket per chat + session
agent (never dispatched), the host's TerminalOpened/TerminalClosed events
moving it between in_progress and done without ever setting a lease, the host
the session opens on, and the honest line the chat lane returns. Pure units."""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")
_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services import cli_host_service as svc  # noqa: E402
from services import cli_ticket_lane as lane  # noqa: E402

WS = uuid4()
NOW = datetime.now(timezone.utc)


class _Query:
    def __init__(self, result):
        self._result = result

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def first(self):
        return self._result

    def all(self):
        return self._result if isinstance(self._result, list) else []

    def scalar(self):
        return self._result


class _DB:
    def __init__(self, *, ticket=None, hosts=None, agent_name=None):
        self.ticket = ticket
        self.hosts = hosts or []
        self.agent_name = agent_name
        self.added = []
        self.commits = 0

    def query(self, model):
        name = getattr(model, "__name__", None) or str(model)
        if name == "CliHost":
            return _Query(self.hosts)
        if "name" in str(model) and name != "BoardTask":  # Agent.name column
            return _Query(self.agent_name)
        return _Query(self.ticket)

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        for obj in self.added:
            if getattr(obj, "id", None) is None:
                obj.id = 501

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        pass


def _agent(**cfg):
    return SimpleNamespace(id=7, name="Bob", configuration={"runtime": "cli", **cfg})


def _host():
    return SimpleNamespace(id=uuid4(), workspace_id=WS)


# ── the session ticket ───────────────────────────────────────────────────────

def test_the_session_source_is_one_per_conversation():
    assert lane.session_source_id("chat-1") == "chat:chat-1:session"


def test_an_existing_session_ticket_is_resumed_not_duplicated(monkeypatch):
    existing = SimpleNamespace(id=93, status="done")
    db = _DB(ticket=existing)
    task, created = lane.open_session_ticket(db, workspace_id=WS, agent=_agent(), chat_id="c1", host=_host(), actor="user:1")
    assert task is existing and created is False and db.added == []


def test_a_new_session_ticket_is_in_progress_without_a_lease_and_never_dispatched(monkeypatch):
    consents = []
    notified = []
    monkeypatch.setattr("services.board_consent.consent_for_created_ticket", lambda db, **kw: consents.append(kw))
    monkeypatch.setattr("services.board_events.notify_board_event", lambda db, **kw: notified.append(kw))
    monkeypatch.setattr("services.board_dispatcher.notify_task_available", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dispatched")))
    host = _host()
    db = _DB(ticket=None)
    task, created = lane.open_session_ticket(
        db, workspace_id=WS, agent=_agent(working_directory="/Users/me/Development/repo", model="opus"),
        chat_id="c1", host=host, actor="user:1",
    )
    assert created is True and task in db.added
    assert task.status == "in_progress" and task.lease_until is None
    assert task.source_type == "chat" and task.source_id == "chat:c1:session"
    assert task.assigned_agent_id == 7 and task.created_by_type == "user" and task.created_by_id == "user:1"
    ref = task.runtime_ref
    assert ref["runtime"] == "cli" and ref["mode"] == "terminal" and ref["host_id"] == str(host.id)
    assert UUID(ref["session_id"]) and ref["cwd"] == "/Users/me/Development/repo" and ref["model"] == "opus"
    assert "explorer_root" in ref
    assert consents and consents[0]["actor"] == "user:1" and consents[0]["task"] is task
    assert notified and notified[0]["event"] == "task_created"


# ── the host's terminal events ───────────────────────────────────────────────

def _terminal_ticket(status="done", **ref):
    return SimpleNamespace(
        id=93, workspace_id=WS, status=status, lease_until=None, completed_at=NOW,
        runtime_ref={"runtime": "cli", "mode": "terminal", "session_id": "sid-1", **ref},
    )


def test_terminal_events_move_an_interactive_ticket_without_ever_setting_a_lease(monkeypatch):
    task = _terminal_ticket()
    host = _host()
    monkeypatch.setattr(svc, "_owned_task", lambda db, h, tid: task)
    monkeypatch.setattr(svc, "renew_lease", lambda *a, **k: (_ for _ in ()).throw(AssertionError("lease renewed")))
    events_out = []
    monkeypatch.setattr(svc, "notify_board_event", lambda db, **kw: events_out.append(kw))
    db = _DB()
    out = svc.record_events(db, host, 93, [{"hook_event_name": "TerminalOpened", "session_id": "sid-1", "cwd": "/Users/me/Development/repo", "resumed": True}])
    assert out == {"status": "in_progress", "lease_renewed": False, "control": {}, "decisions": []}
    assert task.status == "in_progress" and task.lease_until is None and task.completed_at is None
    ref = task.runtime_ref
    assert ref["cli_session_id"] == "sid-1" and ref["cwd"] == "/Users/me/Development/repo" and ref["terminal_resumed"] is True
    assert ref["terminal_attached_at"] and "terminal_closed_at" not in ref
    assert events_out[-1]["status"] == "in_progress"
    svc.record_events(db, host, 93, [{"hook_event_name": "TerminalClosed", "session_id": "sid-1", "exit_code": 0}])
    assert task.status == "done" and task.completed_at is not None and task.lease_until is None
    assert task.runtime_ref["terminal_closed_at"] and "terminal_attached_at" not in task.runtime_ref
    assert events_out[-1]["status"] == "done"


def test_a_host_run_ticket_reopened_in_the_terminal_keeps_its_status(monkeypatch):
    task = SimpleNamespace(id=5, workspace_id=WS, status="done", lease_until=None, completed_at=NOW,
                           runtime_ref={"runtime": "cli", "session_id": "sid-5", "cwd": "/old"})
    monkeypatch.setattr(svc, "_owned_task", lambda db, h, tid: task)
    monkeypatch.setattr(svc, "notify_board_event", lambda db, **kw: (_ for _ in ()).throw(AssertionError("no status change → no event")))
    svc.record_events(_DB(), _host(), 5, [{"hook_event_name": "TerminalOpened", "session_id": "sid-5", "cwd": "/old"}])
    assert task.status == "done" and task.runtime_ref["terminal_attached_at"]


def test_a_mixed_batch_takes_the_ordinary_hook_path(monkeypatch):
    called = []
    monkeypatch.setattr(svc, "_record_terminal_events", lambda *a: called.append("terminal"))
    monkeypatch.setattr(svc, "_owned_task", lambda db, h, tid: (_ for _ in ()).throw(RuntimeError("ordinary path")))
    try:
        svc.record_events(_DB(), _host(), 1, [{"hook_event_name": "TerminalOpened"}, {"hook_event_name": "PreToolUse"}])
    except RuntimeError as exc:
        assert "ordinary path" in str(exc)
    assert called == []


# ── where the session opens ──────────────────────────────────────────────────

def _paired(online, seen):
    return SimpleNamespace(id=uuid4(), status="paired", last_seen_at=seen, is_online=lambda: online)


def test_the_session_opens_on_the_host_that_heartbeated_last():
    older = _paired(True, NOW - timedelta(minutes=5))
    newest = _paired(True, NOW)
    offline = _paired(False, NOW + timedelta(hours=1))
    assert svc.newest_online_host(_DB(hosts=[older, offline, newest]), WS) is newest
    assert svc.newest_online_host(_DB(hosts=[offline]), WS) is None


# ── the chat lane's honest line ──────────────────────────────────────────────

def test_a_chat_message_to_a_session_agent_is_pointed_at_the_canvas():
    line = lane.session_agent_terminal_message(_DB(agent_name="Bob"), 7)
    assert line.startswith("Bob runs as a Claude Code session in the Canvas terminal")
    assert "open the session from the board" in line


def test_the_sessions_route_is_declared_in_the_mount_manifest():
    import json

    manifest = json.loads((_ORCH / "reports" / "route-manifest.json").read_text())
    assert {"path": "/api/v1/cli-hosts/sessions", "method": "POST"} in manifest["routes"]
