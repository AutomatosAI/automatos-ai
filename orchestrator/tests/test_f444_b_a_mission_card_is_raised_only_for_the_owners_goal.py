"""P256-FIX-N17-2 (F444, night 17): a mission's card is raised only for the owner's goal.

Night 17, chats.jsonl L373: platform_create_mission with Auto's 115-word goal raised 'start a
mission' (#2816), and the owner clicked it. Only then did the mission's word rule run
(``mission_create_checks``): 6 of the goal's words were the owner's, NOT_THEIR_GOAL refused
it, the grant was given back, and the owner learned it at L378 only because they asked. Now
the rules run before the card, on the call as its handler gets it (its server-set
``_origin_chat_id`` names the owner's chat), and the click checks only for a copy still
waiting: words said after the card do not undo it.

Each call runs through the executor's whole stack (send_back_waits' desk), onto the real
``asks_as_the_owner_says`` around a mission maker that records what it was asked to make.
"""
from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace as NS

import pytest

from tests import test_f288_a_a_named_playbook_runs_and_no_mission_is_made_twice as f288
from tests import test_prd256_fix_send_back_waits_for_the_click as waits

shop, desk = waits.shop, waits.desk
CREATE_MISSION = "platform_create_mission"
OWNERS_WORDS = "Plan the autumn launch of the Christmas Blend for the wholesale cafes, and price it."
OWNERS_GOAL = "Plan the autumn launch of the Christmas Blend for the wholesale cafes, and price it"
AUTOS_GOAL = (
    "Conduct a comprehensive competitive landscape analysis covering regional specialty roasters, national "
    "chains and subscription services; benchmark their seasonal product positioning, promotional cadence, "
    "loyalty mechanics, packaging sustainability claims, digital storefront conversion tactics, influencer "
    "partnerships and review sentiment across marketplaces; synthesise opportunities, threats and white "
    "space into a prioritised recommendations deck with quantified projections, risk registers, owner "
    "assignments, success metrics, timelines, dependencies, budget envelopes, contingency triggers, "
    "stakeholder communication plans, post-campaign retrospective templates, executive summary narratives, "
    "appendix datasets, methodology notes, source citations, glossary entries, visual mock-ups, channel "
    "playbooks, experiment backlogs, measurement frameworks, attribution models, governance checkpoints, "
    "escalation paths, approval workflows, archival procedures and continuous improvement loops for "
    "leadership review."
)
NOT_THEIR_GOAL = "The goal isn't what the owner asked for"


@pytest.fixture
def words(monkeypatch):
    """The owner's words in the chat the call came from (``mission_owner_words.owners_words``),
    and the call each read was given."""
    import modules.tools.discovery.mission_owner_words as mission_owner_words

    state = NS(said=[OWNERS_WORDS], read=[])

    def owners_words(db, workspace_id, params):
        state.read.append(dict(params or {}))
        return list(state.said)

    monkeypatch.setattr(mission_owner_words, "owners_words", owners_words)
    return state


@pytest.fixture
def maker(desk):
    """The executor's mission tool: the real owner's-words wrapper around a maker that records each call."""
    from modules.tools.discovery.mission_asks import asks_as_the_owner_says

    made = []

    async def make(db, workspace_id, params):
        made.append(dict(params))
        return {"success": True, "mission_id": "m-1", "message": "Mission made."}

    desk.executor._handlers[CREATE_MISSION] = asks_as_the_owner_says(make)
    return made


def _click(desk, reply):
    from core.models.approval_grants import ApprovalGrant
    from core.services.approval_grants import grant_grant

    grant_grant(desk.db.get(ApprovalGrant, reply["grant_id"]), granted_by=waits.CLICKER)
    desk.db.flush()


def test_night_17_2816_a_goal_not_in_the_owners_words_is_refused_before_any_card(desk, words, maker):
    chat = waits._owners_chat()

    reply = waits._call(desk, {"goal": AUTOS_GOAL}, chat, action=CREATE_MISSION)

    assert reply["success"] is False and reply["error"].startswith(NOT_THEIR_GOAL), reply
    assert "requires_confirmation" not in reply and waits._grants(desk) == []          # night 17: #2816 raised
    assert maker == []
    assert any(read.get("_origin_chat_id") == chat["conversation_id"] for read in words.read)  # the owner's chat


def test_the_owners_goal_raises_one_card(desk, words, maker):
    reply = waits._call(desk, {"goal": OWNERS_GOAL}, waits._owners_chat(), action=CREATE_MISSION)

    assert waits._asked(reply), reply
    assert len(waits._grants(desk)) == 1 and maker == []


def test_a_playbook_the_owner_named_is_refused_before_any_card(desk, words, maker):
    from core.models.core import WorkflowTemplate

    desk.db.add(WorkflowTemplate(template_id="f444-onboarding", name="New Cafe Onboarding", workspace_id=desk.ws,
                                 description="Onboard a cafe.", template_definition={"steps": []}, steps=[],
                                 created_by="f444"))
    desk.db.flush()
    words.said = ["Run my New Cafe Onboarding playbook for Bramble & Co: contact Jess Moore."]

    reply = waits._call(desk, {"goal": "Onboard Bramble & Co"}, waits._owners_chat(), action=CREATE_MISSION)

    assert reply["success"] is False and "playbook 'New Cafe Onboarding'" in reply["error"], reply
    assert waits._grants(desk) == []


def test_the_click_is_not_judged_by_words_said_after_the_card(desk, words, maker):
    params = {"goal": OWNERS_GOAL}
    asked = waits._call(desk, params, waits._owners_chat(), action=CREATE_MISSION)
    _click(desk, asked)
    words.said = ["Thanks, that's all for tonight."]                                   # said after the card

    ran = waits._call(desk, params, waits._owners_chat(), action=CREATE_MISSION)

    assert ran["success"] is True and ran["approved_via_grant_id"] == asked["grant_id"], ran
    assert len(maker) == 1 and maker[0]["goal"] == OWNERS_GOAL


def test_a_copy_still_waiting_is_refused_on_the_click(desk, words, maker):
    params = {"goal": OWNERS_GOAL}
    asked = waits._call(desk, params, waits._owners_chat(), action=CREATE_MISSION)
    _click(desk, asked)
    f288._waiting_mission(desk, OWNERS_GOAL, made_ago=timedelta(minutes=2))           # made meanwhile

    ran = waits._call(desk, params, waits._owners_chat(), action=CREATE_MISSION)

    assert ran["success"] is False and "no copy was made" in ran["error"], ran
    assert maker == []


def test_outside_a_persons_chat_nothing_is_read_before_a_card():
    """No card is raised outside a person's chat, so nothing is refused before one."""
    from modules.tools.discovery.mission_create_checks import refused_before_the_mission_card

    assert refused_before_the_mission_card(None, None, "platform_update_task", {"goal": AUTOS_GOAL}) is None
    assert refused_before_the_mission_card(None, None, CREATE_MISSION, "not a call") is None
