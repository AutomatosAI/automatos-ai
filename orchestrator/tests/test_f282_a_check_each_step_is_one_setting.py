"""F282 (night 8): whether a mission checked each step depended on how Auto spelled it.

The owner asked the same way all night. Auto wrote check_each_step (#0324),
approval_mode step_by_step (#0333), wait_for_me (#0344, #0356, #0408), review_mode
step_by_step (#0282), approval_required (#0305), require_human_approval_for_each_step
(#0323) and review_required (#0268). Only the first two stopped a step: 10 of 29
missions where the owner asked checked with them.
"""
from __future__ import annotations

from uuid import UUID

import pytest

from core.models.orchestration_enums import RunState, TaskState
from tests.test_f242_wait_for_me_holds import _held

NIGHT_8_SPELLINGS = [
    {"wait_for_me": True},
    {"review_mode": "step_by_step"},
    {"approval_required": True},
    {"require_human_approval_for_each_step": True},
    {"review_required": True},
    {"wait_for_me": "true"},
]


@pytest.mark.parametrize("config", NIGHT_8_SPELLINGS)
def test_every_spelling_night_8_saw_holds_a_step_for_the_owner(db_session, seed_workspace, config):
    run, _card, task, step_card = _held(db_session, UUID(seed_workspace()), config=config)

    assert (task.state, step_card.status, run.state) == (TaskState.VERIFYING.value, "review", RunState.PAUSED.value)


@pytest.mark.parametrize("config", [{"wait_for_me": False}, {"approval_mode": "auto"}, {"review_required": "no"}, {}])
def test_a_mission_that_says_no_runs_on(db_session, seed_workspace, config):
    run, _card, task, _step_card = _held(db_session, UUID(seed_workspace()), config=config)

    assert (task.state, run.state) == (TaskState.VERIFIED.value, RunState.RUNNING.value)


def test_the_setting_is_kept_under_one_name():
    from modules.coordination.owner_checks import with_step_checks

    tags = ["sim-night", "price-rise"]
    assert with_step_checks({"wait_for_me": True, "tags": tags}) == {"check_each_step": True, "tags": tags}
    assert with_step_checks({"review_mode": "step_by_step", "approval_required": True}) == {"check_each_step": True}
    assert with_step_checks({"approval_mode": "auto", "output_format": "markdown"}) == {"output_format": "markdown"}
    assert with_step_checks({"check_each_step": True}, on=False) == {}
    assert with_step_checks({"output_format": "markdown"}, on=True) == {"output_format": "markdown",
                                                                         "check_each_step": True}
    assert with_step_checks(None) == {}
