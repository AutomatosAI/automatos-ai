"""PRD-234 S3 — lanes that target a Claude Code agent file a ticket; run outcomes are read honestly."""
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

from services import cli_ticket_lane as lane  # noqa: E402
from services.heartbeat_outcome import read_exec_outcome, tokens_of  # noqa: E402


class _DB:
    """Records adds/commits; refresh() stamps an id like the database would."""

    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 4242


def _quiet(monkeypatch, *, existing=None, online=True):
    monkeypatch.setattr(lane, "open_ticket_for_source", lambda db, ws, st, sid: existing)
    monkeypatch.setattr(lane, "host_online", lambda db, ws: online)
    monkeypatch.setattr(lane, "_notify", lambda db, ws, task: None)


def test_files_one_assigned_ticket_in_the_lanes_shape(monkeypatch):
    _quiet(monkeypatch)
    db = _DB()
    t = lane.file_cli_ticket(db, workspace_id="ws", agent_id=15, title="Heartbeat: Bob",
                             prompt="Scheduled heartbeat check. Read the board.", source_type="heartbeat",
                             source_id="agent:15", priority="low")
    assert t.id == 4242 and t.status == "assigned" and t.assigned_agent_id == 15
    assert t.source_type == "heartbeat" and t.source_id == "agent:15" and t.priority == "low"
    assert t.created_by_type == "system" and t.blocked_reason is None and db.commits == 1
    assert lane.queued_line(t) == "queued for your Claude Code session as ticket #4242"


def test_an_open_ticket_from_the_same_source_is_reused_not_duplicated(monkeypatch):
    existing = SimpleNamespace(id=7, status="assigned", blocked_reason=None)
    _quiet(monkeypatch, existing=existing)
    db = _DB()
    t = lane.file_cli_ticket(db, workspace_id="ws", agent_id=15, title="x", prompt="y",
                             source_type="heartbeat", source_id="agent:15")
    assert t is existing and not db.added and db.commits == 0


def test_no_host_online_is_said_on_the_ticket_and_in_the_reply(monkeypatch):
    _quiet(monkeypatch, online=False)
    db = _DB()
    t = lane.file_cli_ticket(db, workspace_id="ws", agent_id=15, title="x", prompt="y",
                             source_type="schedule", source_id="task:9:20260903T1200")
    assert t.status == "assigned" and "none is online" in t.blocked_reason
    assert "no CLI host is online yet" in lane.queued_line(t)


def test_source_ids_are_per_agent_for_heartbeats_and_per_fire_for_schedules():
    from datetime import datetime, timezone
    assert lane.source_id_for("agent", 15) == "agent:15"
    at = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
    assert lane.source_id_for("task", 9, at) == "task:9:20260903T1200"


def test_is_cli_agent_reads_the_configuration(monkeypatch):
    class _Q:
        def __init__(self, agent): self._a = agent
        def filter(self, *a): return self
        def first(self): return self._a
    class _D:
        def __init__(self, agent): self._a = agent
        def query(self, *a): return _Q(self._a)
    assert lane.is_cli_agent(_D(SimpleNamespace(configuration={"runtime": "cli"})), 15) is True
    assert lane.is_cli_agent(_D(SimpleNamespace(configuration={"runtime": "api"})), 15) is False
    assert lane.is_cli_agent(_D(None), 15) is False
    assert lane.is_cli_agent(_D(SimpleNamespace(configuration={"runtime": "cli"})), None) is False


def test_run_outcomes_are_read_honestly():
    text, err, detail = read_exec_outcome({"status": "error", "error": "runtime mismatch: agent 15 is a Claude Code agent"})
    assert err and text == "" and "runtime mismatch" in detail
    text, err, _ = read_exec_outcome({"status": "success", "result": {"response": "all quiet"}})
    assert not err and text == "all quiet"
    text, err, _ = read_exec_outcome("plain string")
    assert not err and text == "plain string"
    assert tokens_of({"tokens_used": "12"}) == 12 and tokens_of(None) == 0
