"""F338 (night 10): a mission's step cards carry the tags the owner put on the mission.

Missions #0195 and #0224 were made with ``card_tags: ['sim-night-2026-10-05']``. The
mission's own card had the tag; its step cards (#2057-#2059, #2089-#2092) carried
``mission`` only, so filtering the board by the owner's tag found the mission and
none of the work it did.
"""
from __future__ import annotations

from types import SimpleNamespace as NS
from uuid import UUID

import pytest

GOAL = "Moorings Cafe welcome pack"
NIGHT_TAG = "sim-night-2026-10-05"


def _mission(db_session, seed_workspace, config):
    """A mission with its card and one step's card, as dispatch files them."""
    from core.models.orchestration import OrchestrationRun, OrchestrationTask
    from services.orchestration_board_bridge import create_mission_board_task, create_task_board_task

    ws = UUID(seed_workspace())
    run = OrchestrationRun(workspace_id=ws, goal=GOAL, state="awaiting_approval", created_by="user_test", config=config)
    db_session.add(run)
    db_session.flush()
    step = OrchestrationTask(run_id=run.id, title="Write the welcome letter", description="Do it.",
                             sequence_number=1, state="pending", state_type="initial")
    db_session.add(step)
    db_session.flush()
    card = create_mission_board_task(db_session, run)
    step_card = create_task_board_task(db_session, run, step)
    return NS(card=card, step_card=step_card)


def test_a_step_card_carries_the_missions_card_tags(db_session, seed_workspace):
    made = _mission(db_session, seed_workspace, {"card_tags": [NIGHT_TAG]})

    assert made.step_card.tags == ["mission", f"mission:{GOAL}", NIGHT_TAG]
    assert made.step_card.parent_task_id == made.card.id


def test_a_step_card_carries_the_tags_the_missions_page_sends(db_session, seed_workspace):
    made = _mission(db_session, seed_workspace, {"tags": ["Christmas", "gift-subscription"]})

    assert made.step_card.tags == ["mission", f"mission:{GOAL}", "Christmas", "gift-subscription"]


def test_a_mission_without_tags_keeps_its_step_cards_as_they_were(db_session, seed_workspace):
    made = _mission(db_session, seed_workspace, {})

    assert made.step_card.tags == ["mission", f"mission:{GOAL}"]


def test_each_tag_is_on_the_step_card_once_in_the_order_given():
    from services.orchestration_board_bridge import step_card_tags

    run = NS(goal=GOAL, config={"card_tags": [NIGHT_TAG, "mission", "cafes", NIGHT_TAG, "  "]})

    assert step_card_tags(run) == ["mission", f"mission:{GOAL}", NIGHT_TAG, "cafes"]


@pytest.mark.parametrize("config", [None, "card_tags", {"card_tags": "not-a-list"}])
def test_a_config_without_a_tag_list_adds_nothing(config):
    from services.orchestration_board_bridge import step_card_tags

    assert step_card_tags(NS(goal=None, config=config)) == ["mission", "mission:Mission"]
