"""F132 (night 6) — onboarding writes where the business is as the workspace's zone.

A new workspace has no timezone, so a playbook scheduled without one fired in
UTC for a Bristol roastery. When onboarding learns where the business is, Auto
passes its IANA zone to platform_update_onboarding, which writes it to the
workspace's heartbeat timezone, the setting schedules already default to (no new
field). A zone the owner set themselves is kept.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import text


@pytest.fixture
def roastery(db_session, seed_workspace):
    return NS(db=db_session, ws=UUID(seed_workspace()))


def _onboard(roastery, **params):
    from modules.tools.discovery.handlers_onboarding import update_onboarding

    return asyncio.run(update_onboarding(roastery.db, roastery.ws, params))


def _zone(roastery):
    from services.playbook_scheduler import default_schedule_zone

    roastery.db.expire_all()
    return default_schedule_zone(roastery.db, roastery.ws)


def test_the_businesss_zone_becomes_the_default_its_schedules_fire_in(roastery):
    assert _zone(roastery) == "UTC"
    reply = _onboard(roastery, timezone="Europe/London")
    assert reply["success"] is True and reply["timezone"] == {"set": "Europe/London"}
    assert _zone(roastery) == "Europe/London"


def test_a_name_that_is_not_a_zone_is_refused_and_nothing_written(roastery):
    reply = _onboard(roastery, timezone="Bristol")
    assert reply["success"] is False and "'Europe/London' for Bristol" in reply["error"]
    assert _zone(roastery) == "UTC"


def test_a_zone_the_owner_set_is_kept(roastery):
    roastery.db.execute(text(
        "UPDATE workspaces SET settings = CAST(:s AS json) WHERE id = CAST(:w AS uuid)"),
        {"s": '{"orchestrator": {"heartbeat": {"timezone": "America/New_York"}}}', "w": str(roastery.ws)})
    reply = _onboard(roastery, timezone="Europe/London")
    assert reply["timezone"]["kept"] == "America/New_York"
    assert _zone(roastery) == "America/New_York"


def test_onboarding_asks_auto_to_pass_the_zone():
    from modules.context.sections import onboarding
    from modules.tools.discovery import get_action_registry

    assert "`timezone`" in onboarding._STAGE_QUESTIONS and "Europe/London" in onboarding._STAGE_QUESTIONS
    assert "timezone" in get_action_registry().get("platform_update_onboarding").parameters["properties"]
