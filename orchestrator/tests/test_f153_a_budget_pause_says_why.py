"""F153 — a mission's budget is real spend, and a budget pause says why.

Night 5: mission 3805978e (a 175,000-token budget) paused at 429,423 tokens
with stop_reason and stop_detail empty, and its card read only "Mission
paused". Most of those tokens were task 2fa5467f's (NEWSROOM, runtime=cli,
730,153 tokens by the end): a Claude Code session on the owner's
subscription, which costs the workspace nothing.

The budget now measures real spend, in dollars: the run's API calls at their
llm_usage cost, any API tokens llm_usage has not booked at the flat rate, and
a runtime (CLI session) task at nothing, its tokens kept on the run for
visibility. Both of the dispatcher's budget pauses put the reason and the
numbers on the run (stop_reason budget_exhausted) and on its board card;
resuming clears them, raises the budget to twice the spend, and says so.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import text

from core.models.orchestration_enums import RunState, TaskState
from modules.policy.pricing import flat_rate_tokens, price_total_tokens_usd

BUDGET = 175_000
NEWSROOM_TOKENS = 730_153


def _ceiling():
    return price_total_tokens_usd(None, None, BUDGET)


@pytest.fixture
def mission(db_session, seed_workspace):
    """Run 3805978e's shape: running on a 175,000-token plan, with its card."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Draft the offer letter", state=RunState.RUNNING.value,
                           created_by="user_test", token_budget_estimate=BUDGET, tokens_used=0, config={})
    db_session.add(run)
    db_session.flush()
    card = BoardTask(workspace_id=ws, title="Mission: Draft the offer letter", status="in_progress",
                     source_type="orchestration", orchestration_run_id=run.id)
    db_session.add(card)
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, card=card)


def _book(mission, dollars, tokens, lane="mission"):
    """One API call the run made, as llm_usage books it."""
    mission.db.execute(text(
        "INSERT INTO llm_usage (workspace_id, model_id, provider, tier, execution_id, request_type, input_tokens, "
        "output_tokens, total_tokens, input_cost, output_cost, total_cost) VALUES (CAST(:ws AS uuid), 'm', 'p', "
        "'direct', :ref, :lane, :tokens, 0, :tokens, :cost, 0, :cost)"),
        {"ws": str(mission.ws), "ref": f"mission:{mission.run.id}", "lane": lane, "tokens": tokens, "cost": dollars})
    mission.run.tokens_used = (mission.run.tokens_used or 0) + tokens


def _record(mission, tokens, runtime=None):
    """A finished task's result, recorded the way the coordinator records it."""
    from services.coordinator_service import CoordinatorService

    task = MagicMock(id=uuid4(), state=TaskState.COMPLETED.value, title="Newsroom brief")
    result = {"status": "success", "result": "done", "execution": {"tokens_used": tokens},
              **({"runtime": runtime} if runtime else {})}
    with patch("services.coordinator_service.MissionDispatcher.record_task_completion"), \
            patch("services.coordinator_service._dispatch_mission_event", new=AsyncMock()), \
            patch("services.coordinator_service._narrate_mission"), \
            patch.object(CoordinatorService, "_inject_task_output_into_field", new=AsyncMock()), \
            patch.object(mission.db, "refresh"):
        asyncio.run(CoordinatorService()._record_task_result(mission.db, mission.run, task, 7, result))


def _dispatch(mission):
    """One dispatch tick with one ready task; the budget gate is real, the
    task's own dispatch is stubbed."""
    from modules.coordination.dispatcher import DispatchResult, MissionDispatcher

    task = MagicMock(id=uuid4(), run_id=mission.run.id, task_type="llm_generation", input_context={})
    with patch("modules.coordination.dispatcher.DependencyResolver.get_ready_tasks", return_value=[task]), \
            patch.object(MissionDispatcher, "_dispatch_single",
                         return_value=DispatchResult(dispatched=True, task_id=task.id)):
        return MissionDispatcher.dispatch_ready(mission.db, mission.run, [MagicMock(id=1)])


def test_a_runtime_task_never_pauses_the_mission(mission):
    _book(mission, 0.02, 20_000)
    _record(mission, NEWSROOM_TOKENS, runtime="cli")
    assert mission.run.tokens_used == 20_000 + NEWSROOM_TOKENS  # recorded for visibility
    assert mission.run.config["session_tokens"] == NEWSROOM_TOKENS
    _dispatch(mission)
    assert (mission.run.state, mission.run.stop_reason) == ("running", None)


def test_an_api_task_still_counts_at_its_price(mission):
    _record(mission, NEWSROOM_TOKENS)
    assert "session_tokens" not in (mission.run.config or {})
    _dispatch(mission)
    assert mission.run.state == "paused"


def test_a_budget_pause_names_the_spend_on_the_run_and_its_card(mission):
    _book(mission, _ceiling() + 1.25, 60_000)
    _record(mission, NEWSROOM_TOKENS, runtime="cli")
    assert [result.skipped_reason for result in _dispatch(mission)] == ["budget_exceeded"]
    detail = (f"Paused: spent ${_ceiling() + 1.25:,.2f} of the ${_ceiling():,.2f} budget (the plan's "
              f"175,000-token estimate); {NEWSROOM_TOKENS:,} tokens ran in Claude Code sessions at no cost "
              "— raise the budget or resume")
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == (
        "paused", "budget_exhausted", detail)
    assert (mission.card.status, mission.card.blocked_reason) == ("blocked", detail)


def test_session_lane_rows_are_not_spend(mission):
    _book(mission, _ceiling() * 3, 30_000, lane="session")
    _dispatch(mission)
    assert mission.run.state == "running"


def test_resuming_clears_the_pause_and_raises_the_budget_to_twice_the_spend(mission):
    from modules.tools.discovery.handlers_missions import resume_mission

    spent = _ceiling() + 1.25
    _book(mission, spent, 60_000)
    _dispatch(mission)
    reply = asyncio.run(resume_mission(mission.db, mission.ws, {"mission_id": str(mission.run.id)}))
    raised = price_total_tokens_usd(None, None, flat_rate_tokens(2 * spent))
    assert reply["message"].endswith(
        f"Its budget was raised from ${_ceiling():,.2f} to ${raised:,.2f} (${spent:,.2f} spent so far).")
    assert mission.run.token_budget_estimate == flat_rate_tokens(2 * spent)
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == ("running", None, None)
    assert (mission.card.status, mission.card.blocked_reason) == ("in_progress", None)
    _dispatch(mission)
    assert mission.run.state == "running"  # no pause straight back


def test_a_dollar_ceiling_pause_and_resume_stay_in_dollars(mission):
    mission.run.config = {"cost_ceiling": 2.5}
    _book(mission, 3.1, 40_000)
    _dispatch(mission)
    assert mission.run.stop_detail == "Paused: spent $3.10 of the $2.50 budget — raise the budget or resume"
    from services.coordinator_service import CoordinatorService

    CoordinatorService().resume_mission(mission.db, mission.run.id, "user_test")
    assert mission.run.config["cost_ceiling"] == 6.2
