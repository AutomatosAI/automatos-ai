"""F153 — a budget pause says why.

Night 5: mission 3805978e (a 175,000-token budget) paused at 429,423 used with
stop_reason and stop_detail empty, and its card read only "Mission paused", so
the owner could not tell why; a resume then doubled the budget without saying
so. Both of the dispatcher's budget pauses now put the reason and the numbers
on the run (stop_reason budget_exhausted, the named StopReason) and on its
board card. Resuming clears them, and the resume reply says what the budget
was raised to.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from core.models.orchestration_enums import RunState

DETAIL = "Paused: token budget used 429,423 of 175,000 — raise the budget or resume"


@pytest.fixture
def mission(db_session, seed_workspace):
    """Run 3805978e's shape: running, 175,000 budgeted, 429,423 used, with its card."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal="Draft the offer letter", state=RunState.RUNNING.value,
                           created_by="user_test", token_budget_estimate=175_000, tokens_used=429_423, config={})
    db_session.add(run)
    db_session.flush()
    card = BoardTask(workspace_id=ws, title="Mission: Draft the offer letter", status="in_progress",
                     source_type="orchestration", orchestration_run_id=run.id)
    db_session.add(card)
    db_session.flush()
    return NS(db=db_session, ws=ws, run=run, card=card)


def _dispatch(mission):
    from modules.coordination.dispatcher import MissionDispatcher

    task = MagicMock(id=uuid4(), run_id=mission.run.id, task_type="llm_generation", input_context={})
    with patch("modules.coordination.dispatcher.DependencyResolver.get_ready_tasks", return_value=[task]):
        return MissionDispatcher.dispatch_ready(mission.db, mission.run, [MagicMock(id=1)])


def test_a_budget_pause_names_the_reason_and_the_numbers(mission):
    assert [result.skipped_reason for result in _dispatch(mission)] == ["budget_exceeded"]
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == (
        "paused", "budget_exhausted", DETAIL)
    assert (mission.card.status, mission.card.blocked_reason) == ("blocked", DETAIL)


def test_a_pause_with_every_task_deferred_names_them_too(mission):
    mission.run.tokens_used = 160_000  # critical: the heavy task defers, nothing dispatches
    _dispatch(mission)
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == (
        "paused", "budget_exhausted", "Paused: token budget used 160,000 of 175,000 — raise the budget or resume")


def test_resuming_clears_the_pause_and_says_what_the_budget_became(mission):
    from modules.tools.discovery.handlers_missions import resume_mission

    _dispatch(mission)
    reply = asyncio.run(resume_mission(mission.db, mission.ws, {"mission_id": str(mission.run.id)}))
    assert reply["success"] is True
    assert reply["message"].endswith("Its token budget was raised from 175,000 to 858,846 (429,423 used so far).")
    assert (mission.run.state, mission.run.stop_reason, mission.run.stop_detail) == ("running", None, None)
    assert (mission.card.status, mission.card.blocked_reason) == ("in_progress", None)


def test_a_dollar_ceiling_pause_names_dollars():
    from modules.coordination.dispatcher import MissionDispatcher

    run = NS(config={"cost_ceiling": 2.5}, tokens_used=429_423, token_budget_estimate=175_000)
    with patch.object(MissionDispatcher, "_cost_used_usd", return_value=3.1):
        assert MissionDispatcher._budget_pause_detail(run) == (
            "Paused: budget used $3.10 of $2.50 — raise the budget or resume")
