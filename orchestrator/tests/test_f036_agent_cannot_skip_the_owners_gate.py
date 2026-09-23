"""F036 (night 1) — an agent cannot skip the owner's mission approval.

Night 1's persona saw a mission start that it had not approved. The trail showed
a person (user 1) had approved it, but it also showed the real edge: the create
tool's ``config.auto_approve`` "forces auto-approval regardless of policy", and
the approve tool worked from any lane. Under the owner's always-ask policy (the
default) an agent's auto_approve is now held and the mission waits; an approval
from a lane no person drives is refused. Where the policy already lets missions
start on their own, nothing changes.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from modules.tools.discovery import handlers_missions as hm

WS = uuid.uuid4()


@pytest.fixture
def policy(monkeypatch):
    chosen = {"policy": "always_ask", "approval_dollar_ceiling": None, "auto_proceed_after_seconds": None}
    monkeypatch.setattr("core.services.approval_policy.load_approval_policy", lambda db, ws: dict(chosen))
    monkeypatch.setattr(hm, "_recent_chat_context", lambda *a, **k: [])
    monkeypatch.setattr("modules.tools.discovery.handlers_watches.auto_create_watch", lambda *a, **k: None)
    return chosen


def _create(config):
    run = NS(id=uuid.uuid4(), goal="g", state="awaiting_approval", plan={"tasks": [{"title": "a"}]}, config={})
    coordinator = MagicMock()
    coordinator.create_mission = AsyncMock(return_value=run)
    with patch("services.coordinator_service.CoordinatorService", return_value=coordinator):
        out = asyncio.run(hm.create_mission(MagicMock(), WS, {"goal": "g", "config": config}))
    return out, coordinator.create_mission.call_args.kwargs["config"]


def test_under_always_ask_an_agents_auto_approve_is_held_and_says_so(policy):
    out, config = _create({"auto_approve": True, "max_retries": 2})
    assert "auto_approve" not in config and config["max_retries"] == 2
    assert out["success"] is True and "auto_approve was not applied" in out["message"]


def test_where_the_policy_already_allows_auto_the_flag_still_counts(policy):
    policy["policy"] = "auto_below_budget"
    out, config = _create({"auto_approve": True})
    assert config["auto_approve"] is True and "not applied" not in out["message"]


def _approve(params):
    run = NS(id=uuid.uuid4())
    coordinator = MagicMock()
    coordinator.approve_plan = MagicMock(return_value=NS(id=run.id, state="running", goal="g"))
    with patch.object(hm, "_resolve_run", return_value=(run, None)), \
            patch("services.coordinator_service.CoordinatorService", return_value=coordinator):
        out = asyncio.run(hm.approve_mission(MagicMock(), WS, {"mission_id": str(run.id), **params}))
    return out, coordinator.approve_plan


def test_an_agent_with_no_person_behind_it_cannot_approve_under_always_ask(policy):
    out, approve = _approve({"_agent_id": 58})
    assert out["success"] is False and "owner approves it" in out["error"]
    approve.assert_not_called()


def test_a_person_driving_the_chat_still_approves_through_auto(policy):
    out, approve = _approve({"_created_by": "user_2abc", "_agent_id": 1})
    approve.assert_called_once()


def test_an_unreadable_policy_keeps_the_gate_shut(monkeypatch):
    def boom(db, ws):
        raise RuntimeError("db down")

    monkeypatch.setattr("core.services.approval_policy.load_approval_policy", boom)
    assert hm._owner_approves_every_mission(MagicMock(), WS) is True
