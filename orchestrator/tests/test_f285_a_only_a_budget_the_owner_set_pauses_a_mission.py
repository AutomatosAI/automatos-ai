"""F285 (night 8) — a mission pauses on a budget only when its owner set one.

Night 8: #0356 stopped with "Paused: spent $0.16 of the $0.14 budget (the plan's
45,000-token estimate) — raise the budget or resume". Nobody had set a budget: the
planner's token estimate, priced, was the ceiling. #0383 paused twice the same way
(at $0.21, then at $0.52 once its estimate had grown from 70,000 to 173,271 tokens),
and #0400 at $0.24.

Now only a ceiling the owner set (``config['cost_ceiling']``) pauses a mission. A
token budget the owner types when approving the plan is one, as the dollars it is
priced at. Resume raises a ceiling the owner set, and sets none where there was none.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from uuid import uuid4

from modules.coordination.dispatcher import MissionDispatcher
from modules.policy.pricing import price_total_tokens_usd
from tests.test_f153_a_budget_pause_says_why import BUDGET, _book, _ceiling, _dispatch
from tests.test_f153_a_budget_pause_says_why import mission as _f153_mission

mission = _f153_mission  # F153's running mission and its card, on the real schema

ESTIMATE_0356 = 45_000


def test_a_mission_with_no_budget_set_never_pauses_on_its_plans_estimate(mission):
    mission.run.config = {}                                   # the owner set no ceiling
    mission.run.token_budget_estimate = ESTIMATE_0356
    _book(mission, 0.16, 50_000)                              # #0356: over the estimate's $0.14

    results = _dispatch(mission)

    assert MissionDispatcher._budget_ceiling_usd(mission.run) == 0.0
    assert [result.dispatched for result in results] == [True]
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == ("running", None, None)


def test_a_ceiling_the_owner_set_still_pauses_and_says_so_plainly(mission):
    mission.run.config = {"cost_ceiling": 0.14}
    _book(mission, 0.16, 50_000)

    assert [result.skipped_reason for result in _dispatch(mission)] == ["budget_exceeded"]
    assert mission.run.stop_detail == "Paused: spent $0.16 of the $0.14 budget — raise the budget or resume"


def test_resuming_a_mission_with_no_ceiling_sets_none(mission):
    from services.coordinator_service import CoordinatorService

    mission.run.config = {}
    _book(mission, _ceiling() * 2, 60_000)                    # twice what the estimate is priced at
    CoordinatorService().pause_mission(mission.db, mission.run.id, "user_test")

    CoordinatorService().resume_mission(mission.db, mission.run.id, "user_test")

    assert mission.run.state == "running"
    assert "cost_ceiling" not in mission.run.config
    assert mission.run.token_budget_estimate == BUDGET        # night 8: raised, and paused on again


def _approve(monkeypatch, run, body):
    """The mission page's Approve, on ``run``, with ``body``; the plan's approval itself stubbed."""
    import api.missions as missions

    monkeypatch.setattr(missions, "_get_run_for_workspace", lambda db, mission_id, workspace_id: run)
    monkeypatch.setattr(missions, "get_coordinator_service",
                        lambda: NS(approve_plan=lambda db, run_id, actor_id: run))
    monkeypatch.setattr(missions, "_run_to_response", lambda approved: approved)
    ctx = NS(workspace_id=uuid4(), user=NS(id="2"))
    return asyncio.run(missions.approve_plan(run.id, missions.MissionApproveRequest(**body), ctx, MagicMock()))


def test_a_token_budget_typed_at_approval_is_the_owners_ceiling(monkeypatch):
    run = NS(id=uuid4(), state="awaiting_approval", config={"check_each_step": True},
             token_budget_estimate=ESTIMATE_0356, max_concurrent=1)

    approved = _approve(monkeypatch, run, {"token_budget_override": 200_000})

    ceiling = price_total_tokens_usd(None, None, 200_000)
    assert approved.config == {"check_each_step": True, "cost_ceiling": ceiling}
    assert approved.token_budget_estimate == 200_000
    assert MissionDispatcher._budget_ceiling_usd(approved) == ceiling


def test_approving_without_a_budget_sets_no_ceiling(monkeypatch):
    run = NS(id=uuid4(), state="awaiting_approval", config={}, token_budget_estimate=ESTIMATE_0356,
             max_concurrent=1)

    approved = _approve(monkeypatch, run, {})

    assert approved.config == {} and MissionDispatcher._budget_ceiling_usd(approved) == 0.0
