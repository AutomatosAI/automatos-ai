"""PRD-234 (2026-09-07, owner: "less approval, more automation").

On the local edition the operator's standing schedule is their approval: a
heartbeat / lane ticket for a Claude Code agent, and the heartbeat's own review
ticket for Auto, are consented at filing so the dispatcher's claim finds an
active grant instead of parking them behind ``always_ask``. SaaS keeps asking.
A ticket parked for an offline host says so on the ticket; the claim clears it.

Pure tests — the consent primitive and the edition are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services import board_consent as bc


def _task(status="assigned", agent_id=15, task_id=75, blocked_reason=None):
    return SimpleNamespace(id=task_id, status=status, assigned_agent_id=agent_id, blocked_reason=blocked_reason)


def test_lane_ticket_is_consented_on_the_local_edition(monkeypatch):
    calls = []
    monkeypatch.setattr(bc, "edition", lambda: "local")
    monkeypatch.setattr(bc, "record_operator_consent", lambda db, **kw: calls.append(kw) or "created")
    out = bc.consent_for_lane_ticket(object(), workspace_id="ws", task=_task(), source_type="heartbeat")
    assert out == "created"
    assert calls[0]["task_id"] == 75 and calls[0]["agent_id"] == 15
    assert calls[0]["actor"] == "lane:heartbeat"
    assert calls[0]["why"] == bc.WHY_SCHEDULED_LANE


def test_lane_ticket_keeps_asking_on_saas(monkeypatch):
    monkeypatch.setattr(bc, "edition", lambda: "saas")
    monkeypatch.setattr(bc, "record_operator_consent", lambda db, **kw: pytest.fail("must not consent on saas"))
    assert bc.consent_for_lane_ticket(object(), workspace_id="ws", task=_task(), source_type="heartbeat") == bc.SKIPPED


def test_only_a_dispatchable_ticket_is_consented(monkeypatch):
    monkeypatch.setattr(bc, "edition", lambda: "local")
    monkeypatch.setattr(bc, "record_operator_consent", lambda db, **kw: pytest.fail("not dispatchable"))
    assert bc.consent_for_lane_ticket(object(), workspace_id="ws", task=_task(status="blocked"), source_type="heartbeat") == bc.SKIPPED
    assert bc.consent_for_lane_ticket(object(), workspace_id="ws", task=_task(agent_id=None), source_type="heartbeat") == bc.SKIPPED


def test_file_cli_ticket_records_the_standing_schedule_consent(monkeypatch):
    from services import cli_ticket_lane as lane

    seen = {}
    monkeypatch.setattr(lane, "open_ticket_for_source", lambda db, ws, st, sid: None)
    monkeypatch.setattr(lane, "host_online", lambda db, ws: False)
    monkeypatch.setattr(lane, "_notify", lambda db, ws, task: None)
    monkeypatch.setattr(bc, "consent_for_lane_ticket", lambda db, **kw: seen.update(kw) or "created")

    class _DB:
        def add(self, t): self.t = t
        def commit(self): pass
        def refresh(self, t): t.id = 75

    ticket = lane.file_cli_ticket(
        _DB(), workspace_id="ws", agent_id=15, title="Heartbeat: Bob", prompt="p",
        source_type="heartbeat", source_id="agent:15", priority="low",
    )
    assert ticket.status == "assigned" and ticket.blocked_reason == lane.NO_HOST_REASON
    assert seen["task"] is ticket and seen["source_type"] == "heartbeat"


def test_parking_with_no_host_writes_the_reason_and_a_host_clears_it(monkeypatch):
    import api.board_tasks as bt
    from services.cli_ticket_lane import NO_HOST_REASON

    task = _task(status="in_progress", task_id=92)
    committed = []

    class _Q:
        def get(self, _id): return task
    class _DB:
        def query(self, _m): return _Q()
        def commit(self): committed.append(True)

    monkeypatch.setattr(bt, "notify_board_event", lambda *a, **k: None)
    monkeypatch.setattr(bt, "notify_task_available", lambda *a, **k: None)
    monkeypatch.setattr(bt, "_agent_runtime_kind", lambda db, aid: bt.RUNTIME_CLI)
    import services.cli_ticket_lane as lane
    monkeypatch.setattr(lane, "host_online", lambda db, ws: False)
    task.workspace_id = "ws"
    bt._park_for_cli_host(_DB(), 92, "ws", 15)
    assert task.status == "assigned" and task.blocked_reason == NO_HOST_REASON and committed

    monkeypatch.setattr(lane, "host_online", lambda db, ws: True)
    bt._park_for_cli_host(_DB(), 92, "ws", 15)
    assert task.blocked_reason is None


def test_run_now_redispatch_writes_the_no_host_line_for_a_cli_agent(monkeypatch):
    import api.board_tasks as bt
    import services.cli_ticket_lane as lane
    from services.cli_ticket_lane import NO_HOST_REASON

    task = SimpleNamespace(id=92, status="failed", assigned_agent_id=15, workspace_id="ws",
                           source_type="chat", lease_until="x", attempts=3, completed_at="x",
                           started_at="x", blocked_reason=None)
    class _DB:
        def commit(self): pass
        def refresh(self, t): pass
    monkeypatch.setattr(bt, "_agent_runtime_kind", lambda db, aid: bt.RUNTIME_CLI)
    monkeypatch.setattr(bt, "notify_task_available", lambda *a, **k: None)
    monkeypatch.setattr(lane, "host_online", lambda db, ws: False)
    bt._redispatch_task(_DB(), task)
    assert task.status == "assigned" and task.attempts == 0 and task.blocked_reason == NO_HOST_REASON

    api_task = SimpleNamespace(id=93, status="failed", assigned_agent_id=2, workspace_id="ws", source_type="chat",
                               lease_until=None, attempts=0, completed_at=None, started_at=None, blocked_reason=None)
    monkeypatch.setattr(bt, "_agent_runtime_kind", lambda db, aid: bt.RUNTIME_API)
    bt._redispatch_task(_DB(), api_task)
    assert api_task.blocked_reason is None
