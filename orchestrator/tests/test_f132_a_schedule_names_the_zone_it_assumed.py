"""F132 (night 6) — the schedule tool asks for the owner's zone and says when it assumed one.

"7am Monday is fine" from a UK owner was saved at 02:30:11Z as 7am UTC:
platform_schedule_playbook was given no zone, and a new workspace has none, so
F132's default fell back to UTC. The tool now tells Auto to pass the owner's own
zone (UK time is 'Europe/London') and to ask when it does not know; a schedule
saved without one says which zone it assumed and to check it with the owner.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest


@pytest.fixture
def roastery(db_session, seed_workspace):
    from modules.tools.discovery.handlers_playbooks import create_playbook

    ws = UUID(seed_workspace())
    made = asyncio.run(create_playbook(db_session, ws, {"name": "Tom's Monday Dispatch Checklist",
                                                         "description": "The packing list for Monday"}))
    return NS(db=db_session, ws=ws, playbook_id=made["playbook"]["id"])


def _schedule(roastery, **params):
    from modules.tools.discovery.handlers_playbooks import schedule_playbook

    return asyncio.run(schedule_playbook(roastery.db, roastery.ws, {
        "playbook_id": roastery.playbook_id, "cron_expression": "0 7 * * 1", **params}))


def test_a_schedule_saved_without_a_zone_says_it_assumed_utc(roastery):
    reply = _schedule(roastery)
    assert reply["success"] is True
    assert "No timezone was given, so it fires in UTC" in reply["message"]   # night 6: silently UTC
    assert "ask for their zone" in reply["message"] and reply["timezone_given"] is False


def test_the_owners_zone_is_kept_and_nothing_is_assumed(roastery):
    reply = _schedule(roastery, timezone="Europe/London")
    assert reply["schedule_config"]["timezone"] == "Europe/London" and reply["timezone_given"] is True
    assert "No timezone was given" not in reply["message"]


def test_the_tool_tells_auto_to_pass_the_owners_zone_and_never_assume_utc():
    from modules.tools.discovery import get_action_registry

    action = get_action_registry().get("platform_schedule_playbook")
    zone = action.parameters["properties"]["timezone"]["description"]
    for said in (action.description, zone):
        assert "never assume UTC" in said and "Europe/London" in said
