"""PRD-200 S3 — awaiting-approval re-notify (and optional expiry) sweep.

A parked mission fires exactly one ``mission_plan_ready`` notification and then
goes invisible to the coordinator (the tick only processes RUNNING runs), so 47%
of all missions ever created sit stranded at their approval gate.
``CoordinatorService.check_approval_renotify`` re-pings a parked plan after the
reminder interval so it does not die after one notification; an optional,
OFF-by-default expiry cancels a plan that has sat too long.

Pure — the session and the notification dispatcher are mocked at the boundary.
"""
from __future__ import annotations

import os
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import pytest  # noqa: E402


# ---------------------------------------------------------------------------
# Fake session — records the filters applied so we can prove the sweep scopes
# itself to awaiting_approval, and returns a fixed row set (the real query's
# state filter is what excludes non-AWAITING runs).
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows
        self.filters = []

    def filter(self, *conds):
        self.filters.extend(conds)
        return self

    def all(self):
        return list(self._rows)


class _FakeSession:
    def __init__(self, rows):
        self.q = _FakeQuery(rows)
        self.queried_models = []

    def query(self, model):
        self.queried_models.append(model)
        return self.q


def _run(**over) -> SimpleNamespace:
    base = dict(
        id=uuid.uuid4(),
        workspace_id=uuid.uuid4(),
        goal="Draft the Q4 board memo",
        state="awaiting_approval",
        plan={"tasks": [{"title": "t1"}, {"title": "t2"}]},
        config={},
        created_at=datetime.now(timezone.utc) - timedelta(days=30),
        updated_at=datetime.now(timezone.utc) - timedelta(days=30),
    )
    base.update(over)
    return SimpleNamespace(**base)


def _service():
    """A CoordinatorService instance without running __init__ — the sweep uses
    no instance state, only its db/workspace args + Config."""
    from services.coordinator_service import CoordinatorService

    return CoordinatorService.__new__(CoordinatorService)


def _pin_config(monkeypatch, *, renotify=86400, expiry=False, max_age=604800):
    from config import Config

    monkeypatch.setattr(Config, "COORDINATOR_APPROVAL_RENOTIFY_SECONDS", renotify)
    monkeypatch.setattr(Config, "COORDINATOR_APPROVAL_EXPIRY_ENABLED", expiry)
    monkeypatch.setattr(Config, "COORDINATOR_APPROVAL_MAX_AGE_SECONDS", max_age)


@pytest.mark.asyncio
async def test_stale_parked_run_is_renotified(monkeypatch):
    import services.coordinator_service as cs

    _pin_config(monkeypatch)
    dispatch = AsyncMock()
    monkeypatch.setattr(cs, "_dispatch_mission_event", dispatch)

    two_days_ago = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    run = _run(config={
        "approval_last_notified_at": two_days_ago,
        "approval_estimated_cost_usd": 3.5,
    })
    db = _FakeSession([run])

    acted = await _service().check_approval_renotify(db)

    assert acted == 1
    dispatch.assert_awaited_once()
    kwargs = dispatch.await_args.kwargs
    assert kwargs["event_type"] == "mission_plan_ready"
    assert kwargs["status"] == "action_required"
    assert "3.50" in kwargs["message"]  # priced cost surfaced in the reminder
    # Baseline restamped so the plan will not re-ping again until next interval.
    assert run.config["approval_last_notified_at"] != two_days_ago


@pytest.mark.asyncio
async def test_fresh_parked_run_is_not_renotified(monkeypatch):
    import services.coordinator_service as cs

    _pin_config(monkeypatch)
    dispatch = AsyncMock()
    monkeypatch.setattr(cs, "_dispatch_mission_event", dispatch)

    just_now = datetime.now(timezone.utc).isoformat()
    run = _run(config={"approval_last_notified_at": just_now})
    db = _FakeSession([run])

    acted = await _service().check_approval_renotify(db)

    assert acted == 0
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_sweep_scopes_query_to_awaiting_approval(monkeypatch):
    import services.coordinator_service as cs
    from core.models.orchestration import OrchestrationRun
    from sqlalchemy.dialects import postgresql

    _pin_config(monkeypatch)
    monkeypatch.setattr(cs, "_dispatch_mission_event", AsyncMock())

    db = _FakeSession([])
    await _service().check_approval_renotify(db)

    # The sweep filters runs by state == awaiting_approval — a non-AWAITING run
    # is never a candidate (proven structurally, not by row-fixture).
    assert OrchestrationRun in db.queried_models
    assert db.q.filters, "sweep must filter, not scan every run"
    rendered = str(
        db.q.filters[0].compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )
    assert "awaiting_approval" in rendered


@pytest.mark.asyncio
async def test_expiry_off_by_default_never_cancels(monkeypatch):
    import services.coordinator_service as cs

    _pin_config(monkeypatch, expiry=False)
    monkeypatch.setattr(cs, "_dispatch_mission_event", AsyncMock())

    # 400 days old — but expiry is OFF, so it is only re-notified, not cancelled.
    run = _run(
        config={},
        created_at=datetime.now(timezone.utc) - timedelta(days=400),
        updated_at=datetime.now(timezone.utc) - timedelta(days=400),
    )
    db = _FakeSession([run])

    acted = await _service().check_approval_renotify(db)

    assert run.state == "awaiting_approval"  # untouched terminal-wise
    assert acted == 1  # re-notified (stale), not expired


@pytest.mark.asyncio
async def test_expiry_when_enabled_cancels_old_plan(monkeypatch):
    import services.coordinator_service as cs

    _pin_config(monkeypatch, expiry=True, max_age=604800)  # 7 days
    monkeypatch.setattr(cs, "_dispatch_mission_event", AsyncMock())

    # Stub transition_run so the test stays pure (no event dual-write / DB).
    def _fake_transition(db, run, new_state, **kwargs):
        run.state = new_state.value

    monkeypatch.setattr(cs, "transition_run", _fake_transition)

    run = _run(config={}, created_at=datetime.now(timezone.utc) - timedelta(days=400))
    db = _FakeSession([run])

    acted = await _service().check_approval_renotify(db)

    assert run.state == "cancelled"
    assert acted == 1
