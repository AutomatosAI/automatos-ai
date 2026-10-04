"""F288 and F289 (night 8): no mission for a playbook the owner named, and no mission twice.

- "Run my New Cafe Onboarding playbook for …" became a mission 9 times of 9.
- "Yes, go ahead." made the price-list mission again and approved the copy, leaving
  #0393 waiting; "cancel #0433 and start it again" made "Research our top 5
  competitors…" (#0437), the example goal in Auto's skill.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

PRICE_LIST = "Prepare the wholesale price list for December, including the margin on a 1 kg bag of Christmas Blend"


@pytest.fixture
def shop(db_session, seed_workspace):
    from core.models.core import WorkflowTemplate

    ws = UUID(seed_workspace())
    db_session.add(WorkflowTemplate(template_id=f"f288-{uuid4().hex[:8]}", name="New Cafe Onboarding", workspace_id=ws,
                                    description="Onboard a cafe.", template_definition={"steps": []}, steps=[],
                                    created_by="f288"))
    db_session.flush()
    return NS(db=db_session, ws=ws)


def _refusal(shop, goal, *said):
    from modules.tools.discovery.mission_create_checks import refusal_for_mission

    return refusal_for_mission(shop.db, shop.ws, {"goal": goal}, list(said))


@pytest.mark.parametrize("said", [
    "Please run New Cafe Onboarding for a new wholesale cafe: The Lantern Room, contact Priya Shah.",
    "Run my New Cafe Onboarding playbook for Bramble & Co: contact Jess Moore.",
    "Use the new cafe onboarding playbook for a new café: The Copper Kettle.",
    "Don't make a mission this time. Run my saved playbook called New Cafe Onboarding, the one I set up.",
    "I asked for my playbook, not a mission. Please cancel #0384 and run my saved New Cafe Onboarding playbook.",
])
def test_a_playbook_the_owner_names_runs_and_no_mission_is_made(shop, said):
    refusal = _refusal(shop, "Onboard new wholesale cafe", said)
    assert "playbook 'New Cafe Onboarding'" in refusal and "platform_execute_playbook" in refusal


def test_asking_to_run_a_playbook_by_another_name_finds_it(shop):
    refusal = _refusal(shop, "Onboard the Driftwood Cafe", "Run my saved playbook for the Driftwood Cafe in Portishead.")
    assert "platform_list_playbooks" in refusal


def test_a_mission_the_owner_asks_for_outright_is_made(shop):
    assert _refusal(shop, "Improve New Cafe Onboarding's welcome email",
                    "Start a mission to improve the welcome email in New Cafe Onboarding.") is None


def _waiting_mission(shop, goal, *, made_ago):
    from core.models.orchestration import OrchestrationRun
    from services.orchestration_board_bridge import create_mission_board_task

    run = OrchestrationRun(workspace_id=shop.ws, goal=goal, state="awaiting_approval", created_by="user_test",
                           config={}, created_at=datetime.now(timezone.utc) - made_ago)
    shop.db.add(run)
    shop.db.flush()
    return create_mission_board_task(shop.db, run)


def test_a_mission_still_waiting_for_the_owner_is_not_made_twice(shop):
    """#0393: "Yes, go ahead." made the mission again."""
    card = _waiting_mission(shop, PRICE_LIST, made_ago=timedelta(minutes=3))

    refusal = _refusal(shop, PRICE_LIST + ".", "Yes, go ahead.", PRICE_LIST)

    number = f"#{card.workspace_seq:04d}"
    assert f"Mission {number}" in refusal and f'platform_approve_mission {{mission_id: "{number}"}}' in refusal


def test_the_same_goal_hours_later_is_a_new_mission(shop):
    _waiting_mission(shop, PRICE_LIST, made_ago=timedelta(hours=3))
    assert _refusal(shop, PRICE_LIST, "Start a mission: " + PRICE_LIST) is None


def test_a_goal_none_of_whose_words_the_owner_said_is_not_made(shop):
    """#0437: the owner asked to start #0433's welcome pack again."""
    said = "I pressed Pause and Resume on its page and nothing changed. So yes: cancel #0433 and start it again."
    refusal = _refusal(shop, "Research our top 5 competitors, analyze their pricing, features, and market positioning.",
                       said)
    assert "isn't what the owner asked for" in refusal
