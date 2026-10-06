"""Gerard, 7 Oct: a turn routed to the Brand designer changes no setting and starts no mission.

Night 10c: "More space between sections" reached platform_update_system_setting and "Warmer,
please" became a three-task mission with a researcher. F362 pins a brand ask to the designer's
ticket, but a directive is a prompt. Now, once the brand lane fires, the executor refuses the
setting and every action that starts a mission for the rest of the turn, and says where the
work goes; filing the designer's ticket (platform_create_task) runs as before.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from consumers.chatbot.brand_assign_lane import brand_work_goes_to_the_designer
from modules.tools.discovery.follows_the_owner import follows_the_owner
from modules.tools.discovery.platform_executor import PlatformActionExecutor

WS = UUID("0b6c2d3e-4f5a-4b6c-9d7e-8f9a0b1c2d3e")
DESIGNER = NS(id=347, name="Brand Designer", status="active")
REFUSED = ("This is a brand change: it goes to the Brand Designer on a ticket "
           "(platform_create_task assigned to Brand Designer).")
TIERS = "the tiers' assessment"


@pytest.fixture(autouse=True)
def designer(monkeypatch):
    from core.seeds import seed_brand_designer

    monkeypatch.setattr(seed_brand_designer, "find_brand_designer", lambda db, workspace_id: DESIGNER)


async def _tiers(brain, message, conversation_length=0):
    return TIERS


_assess = brand_work_goes_to_the_designer(_tiers)


def _brain(onboarding=False):
    return NS(_db=object(), _workspace_id=str(WS), _onboarding_active=lambda: onboarding)


@follows_the_owner          # the executor's check before any gate or handler (PlatformActionExecutor.execute)
async def _ran(self, action_name, params, caller_context=None):
    return {"success": True, "ran": action_name}


def _execute(action_name, params):
    """A call outside a chat a person drives (no conversation), so only the brand turn's check applies."""
    return _ran(NS(db=None, workspace_id=WS), action_name, params)


def _turn(said, *calls, onboarding=False):
    """One chat turn: AutoBrain assesses ``said``, then the turn's tool calls run in its context."""
    async def turn():
        verdict = await _assess(_brain(onboarding), said, 3)
        return verdict, [await call() for call in calls]
    return asyncio.run(turn())


def test_a_brand_turn_refuses_the_setting_and_the_mission_with_where_the_work_goes():
    executor = object()     # the refusal comes before any gate, so no executor state is read
    verdict, (setting, mission, blog) = _turn(
        "More space between sections",
        lambda: PlatformActionExecutor.execute(executor, "platform_update_system_setting", {"key": "spacing"}),
        lambda: PlatformActionExecutor.execute(executor, "platform_create_mission", {"goal": "Warmer documents"}),
        lambda: PlatformActionExecutor.execute(executor, "platform_create_blog_post", {"topic": "Our new look"}),
    )

    assert verdict.target_agent_name == "Brand Designer"
    for refused in (setting, mission, blog):
        assert refused == {"success": False, "error": REFUSED}


def test_a_brand_turn_still_files_the_designers_ticket():
    _, (filed, started) = _turn(
        "Make the orange an accent only",
        lambda: _execute("platform_create_task", {"assigned_agent_name": "Brand Designer"}),
        lambda: _execute("platform_update_task_status", {"task_id": "#0901", "status": "in_progress"}),
    )

    assert filed == {"success": True, "ran": "platform_create_task"}
    assert started == {"success": True, "ran": "platform_update_task_status"}


def test_a_turn_that_is_not_a_brand_ask_runs_the_setting_and_the_mission():
    verdict, (setting, mission) = _turn(
        "Plan next week's roasting schedule",
        lambda: _execute("platform_update_system_setting", {"key": "spacing"}),
        lambda: _execute("platform_create_mission", {"goal": "Roasting schedule"}),
    )

    assert verdict == TIERS
    assert setting["success"] is True and mission["success"] is True


def test_the_next_assessment_clears_a_brand_turn():
    async def two_turns():
        await _assess(_brain(), "Warmer, please", 1)
        await _assess(_brain(), "Plan next week's roasting schedule", 3)
        return await _execute("platform_create_mission", {"goal": "Roasting schedule"})

    assert asyncio.run(two_turns())["success"] is True


def test_mid_onboarding_a_brand_ask_is_no_brand_turn():
    verdict, (mission,) = _turn(
        "Make the orange an accent only",
        lambda: _execute("platform_create_mission", {"goal": "Warmer documents"}),
        onboarding=True,
    )

    assert verdict == TIERS and mission["success"] is True
