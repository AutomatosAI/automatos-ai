"""PRD-163 S5 — async planning (create returns immediately, plan lands later).

`async_planning` config makes create_mission park the run in PLANNING and return
without invoking the (slow, LLM-backed) planner; the coordinator tick sweep runs
the planner later and fires a mission_plan_ready notification. These tests prove
the create path is non-blocking and the notification event type is registered,
without needing a live DB or planner.
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")

from core.models.orchestration_enums import RunState  # noqa: E402


def test_mission_plan_ready_event_registered():
    """The notification dispatcher must accept the plan-ready event so the
    'plan lands via notification' half of async planning actually delivers."""
    from core.services.notification_dispatcher import VALID_EVENT_TYPES
    assert "mission_plan_ready" in VALID_EVENT_TYPES


@pytest.mark.asyncio
async def test_async_planning_returns_in_planning_without_planner(mock_db, monkeypatch):
    """create_mission(async_planning=True) returns immediately in PLANNING and
    never calls the planner — the tick sweep will plan it later."""
    from services.coordinator_service import CoordinatorService
    from modules.coordination import planner as planner_mod

    async def _must_not_run(*_a, **_k):
        raise AssertionError("planner must not run synchronously in async mode")

    monkeypatch.setattr(planner_mod.MissionPlanner, "decompose", _must_not_run)

    svc = CoordinatorService()
    run = await svc.create_mission(
        db=mock_db,
        workspace_id=uuid4(),
        goal="research the market",
        created_by="user_abc",
        config={"async_planning": True},
    )

    assert run.state == RunState.PLANNING.value
    assert run.plan in (None, {}, {"tasks": []}) or not run.plan


@pytest.mark.asyncio
async def test_sync_default_calls_planner(mock_db, monkeypatch):
    """Without the flag, create_mission plans synchronously (the planner IS
    invoked) — the default chat path that emits the approval card inline."""
    from services.coordinator_service import CoordinatorService
    from modules.coordination import planner as planner_mod

    called = {"n": 0}

    async def _decompose(*_a, **_k):
        called["n"] += 1
        raise RuntimeError("stop after proving the planner was reached")

    monkeypatch.setattr(planner_mod.MissionPlanner, "decompose", _decompose)

    svc = CoordinatorService()
    with pytest.raises(RuntimeError):
        await svc.create_mission(
            db=mock_db,
            workspace_id=uuid4(),
            goal="research the market",
            created_by="user_abc",
            config={},
        )
    assert called["n"] == 1
