"""P256-FIX-T5 (F314): 'draft it here' is answered in the chat, not filed as a ticket.

Benchmark 11 Oct (eval F314-a, failed on Haiku, Sonnet and Opus 5.5): "Draft the intro for this
fortnight's Harbour Log newsletter, about the October box. Just the draft here, please." became card
#1312 for HARBOUR LOG. The hand-off table read "newsletter" as customer paperwork, whose note tells Auto
not to write it and to file it; and the tiers could take the newsletter's name, HARBOUR LOG's too, for a
named assignee. Here, through AutoBrain's two wrappers (``hands_off`` over ``auto_always_answers``), the
turn's last note (``read_note``) and the dispatch (``lane_for``):

- the F314-a ask is Auto's answer in the reply whatever the tiers said: RESPOND, no ticket tools, no
  target, the here directive (write the draft in the reply, no platform_create_task or assign), and no
  paperwork note filing it for the team;
- "Have HARBOUR LOG draft the intro" is one ticket for HARBOUR LOG; "Draft the intro and put it on the
  board" keeps the tiers' ticket and the paperwork note; long work keeps its lane.

No model is called and no database is opened: the roster is the test's.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from api.chat_dispatch import lane_for
from consumers.chatbot import handoffs
from consumers.chatbot.asked_here import HERE_DIRECTIVE, asks_for_it_here
from consumers.chatbot.auto import ASSIGN_TOOL_HINTS, Action, AutoBrain, Complexity, ComplexityAssessment
from consumers.chatbot.handoffs import asks_for_paperwork, asks_for_social_work, auto_always_answers, hands_off
from consumers.chatbot.named_template_note import read_note

WS = UUID("4d9b3a19-6c0e-4f2a-9b1c-0d1e2f3a4b51")
AUTO, HARBOUR_LOG, OPS = 1, 77, 267
ROSTER = [
    NS(id=AUTO, name="Auto", job_title="Workspace orchestrator", is_system_agent=True, status="active"),
    NS(id=HARBOUR_LOG, name="HARBOUR LOG", job_title="Newsletter editor", is_system_agent=False, status="active"),
    NS(id=OPS, name="OPS", job_title="Operations Manager", is_system_agent=False, status="active"),
]
F314_A = ("Draft the intro for this fortnight's Harbour Log newsletter, about the October box. "
          "Just the draft here, please.")
NAMED = "Have HARBOUR LOG draft the intro for this fortnight's newsletter."
ON_THE_BOARD = "Draft the intro and put it on the board"
PAPERWORK_ON_A_CARD = "Draft the Harbour Log newsletter intro and put it on a card."
TEAM_NOTE_START = "The owner asked for a"
TICKET_TOOLS = ("platform_create_task", "platform_assign_task")
TIERS_SAID = [Action.RESPOND, Action.ASSIGN, Action.DELEGATE]
LINEAR_S = 2.0


class _Brain:
    """What the table and the lane read of AutoBrain: the roster, AutoBrain's own roster match."""

    _db, _workspace_id = object(), str(WS)
    _match_roster_agent = AutoBrain._match_roster_agent

    def _onboarding_active(self):
        return False

    def _active_agents(self):
        return ROSTER


class _ChatDb:
    """The chat's session, faked: a savepoint only."""

    def begin_nested(self):
        return contextlib.nullcontext()


@pytest.fixture(autouse=True)
def workspace(monkeypatch):
    """No Social Media Director, no templates, no board: only the words and the roster decide."""
    from modules.documents import template_service

    class _Templates:
        def __init__(self, db):
            pass

        def list_templates(self, workspace_id, format=None, category=None):
            return []

    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    monkeypatch.setattr(handoffs, "find_social_media_director", lambda db, workspace_id: None)
    monkeypatch.setattr("services.ticket_numbers.resolve_ticket_ref", lambda db, ws, ref: None)


def _tiers(action, complexity=Complexity.MOLECULE, **extra):
    """The tiers' verdict; an ASSIGN one names HARBOUR LOG, the newsletter's name (the F314-a failure)."""
    if action == Action.ASSIGN:
        extra = {"target_agent_id": HARBOUR_LOG, "target_agent_name": "HARBOUR LOG", **extra}
    return ComplexityAssessment(complexity=complexity, action=action, reasoning="tiers", **extra)


def _turn(said, tiers):
    """One owner turn: AutoBrain's two wrappers over ``tiers``, the turn's last note, the dispatch."""
    async def classifier(brain, message, conversation_length=0):
        return tiers

    async def run():
        verdict = await hands_off(auto_always_answers(classifier))(_Brain(), said, 3)
        return verdict, read_note(_ChatDb(), WS, [said, "Hello."])

    verdict, note = asyncio.run(run())
    return lane_for(AUTO, verdict, said), note


# ── the F314-a ask: Auto writes the draft in the reply ────────────────────────

@pytest.mark.parametrize("action", TIERS_SAID)
def test_the_f314_a_ask_is_autos_answer_in_the_reply_and_files_no_ticket(action):
    lane, note = _turn(F314_A, _tiers(action))
    verdict = lane.assessment

    assert lane.agent_id == AUTO and not lane.suggest_mission
    assert verdict.action == Action.RESPOND
    assert (verdict.target_agent_id, verdict.target_agent_name) == (None, None)
    assert not set(ASSIGN_TOOL_HINTS) & set(verdict.tool_hints or [])
    assert verdict.context_directive == HERE_DIRECTIVE
    assert note is None or not note.startswith(TEAM_NOTE_START)


def test_the_here_directive_asks_for_the_draft_in_the_reply_and_forbids_the_ticket_tools():
    assert "in full in this reply" in HERE_DIRECTIVE and "the draft" in HERE_DIRECTIVE
    for tool in TICKET_TOOLS:
        assert f"no {tool}" in HERE_DIRECTIVE
    assert "Do NOT file it as a card or hand it to an agent" in HERE_DIRECTIVE


def test_the_newsletter_asked_for_here_is_no_paperwork_for_the_team():
    assert asks_for_paperwork("Draft the intro for the Harbour Log newsletter.") == "newsletter"
    assert asks_for_paperwork(F314_A) is None


@pytest.mark.parametrize("said", [
    F314_A,
    "Draft the October newsletter intro here, please.",
    "Write the intro for the newsletter in the chat.",
    "Can you write me a short thank-you letter to Rosa? Reply with it.",
    "Draft the price list blurb and paste it.",
    "Write the welcome sheet intro and show me.",
    "Just the text for the flyer, please.",
    "Draft it here.",
    "Put the letter in your reply.",
    "Just the draft here, no need to create a card.",
    "Draft the newsletter intro here, don't file it.",
    "Write the intro here please, without a ticket.",
])
def test_every_way_of_asking_for_it_here_keeps_it_with_auto(said):
    assert asks_for_it_here(said)
    lane, note = _turn(said, _tiers(Action.ASSIGN))
    assert lane.assessment.action == Action.RESPOND and lane.assessment.context_directive == HERE_DIRECTIVE
    assert note is None or not note.startswith(TEAM_NOTE_START)


@pytest.mark.parametrize("said", [
    "The new barista is here.", "Here's the address for the Quay letter.", "Draft the newsletter intro.",
    "Come over here", "What's here?", "Where is the October box?", "show me my playbooks",
    "Show me the cards waiting on me.", "Create a card for the roaster delivery and show me.",
    "Make a ticket for the Quay letter here.", "Make the invoice for Rosa. Here, the amounts: 8 kg at £22.",
    "", None,
])
def test_a_here_that_asks_for_nothing_back_is_not_the_rule(said):
    assert not asks_for_it_here(said)


def test_a_post_asked_for_here_is_autos_text_and_a_named_role_still_gets_it():
    assert asks_for_social_work("Draft an Instagram post for the October box.")
    assert not asks_for_social_work("Draft an Instagram caption for the October box here, please.")
    assert asks_for_social_work("Have our social media team draft a LinkedIn post here, please.")


# ── named, on a card, or long: the ticket as today ────────────────────────────

@pytest.mark.parametrize("action", TIERS_SAID)
@pytest.mark.parametrize("said", [NAMED, NAMED.replace(".", " here, please.")])
def test_naming_the_agent_files_one_ticket_for_it(action, said):
    lane, _note = _turn(said, _tiers(action))
    verdict = lane.assessment

    assert verdict.action == Action.ASSIGN
    assert (verdict.target_agent_id, verdict.target_agent_name) == (HARBOUR_LOG, "HARBOUR LOG")
    assert set(ASSIGN_TOOL_HINTS) <= set(verdict.tool_hints)
    assert "platform_create_task" in verdict.context_directive and verdict.context_directive != HERE_DIRECTIVE


def test_draft_it_and_put_it_on_the_board_keeps_the_ticket():
    assert not asks_for_it_here(ON_THE_BOARD)
    lane, _note = _turn(ON_THE_BOARD, _tiers(Action.ASSIGN))
    verdict = lane.assessment

    assert verdict.action == Action.ASSIGN and verdict.target_agent_id == HARBOUR_LOG
    assert "platform_create_task" in verdict.context_directive


@pytest.mark.parametrize("said", [PAPERWORK_ON_A_CARD, PAPERWORK_ON_A_CARD.replace("a card.", "a card here.")])
def test_paperwork_put_on_a_card_still_gets_the_team_note(said):
    assert not asks_for_it_here(said)
    _lane, note = _turn(said, _tiers(Action.RESPOND))
    assert note is not None and note.startswith(TEAM_NOTE_START) and "platform_create_task" in note


@pytest.mark.parametrize("tiers", [
    _tiers(Action.MISSION, Complexity.ORGAN),
    _tiers(Action.ASSIGN, Complexity.ORGANISM),
    _tiers(Action.ASSIGN, needs_multi_agent=True),
])
def test_long_work_asked_for_here_keeps_its_lane(tiers):
    said = "Write the whole autumn campaign: the newsletter, three posts and the flyer. Show me."
    lane, _note = _turn(said, tiers)
    assert lane.assessment.action == tiers.action
    assert lane.assessment.context_directive != HERE_DIRECTIVE


def test_a_reply_with_no_here_keeps_the_tiers_verdict():
    tiers = _tiers(Action.RESPOND)
    lane, _note = _turn("What's on the board today?", tiers)
    assert lane.assessment == tiers


# ── the rule reads owner text in linear time ──────────────────────────────────

def test_the_rule_reads_a_long_run_of_spaces_in_linear_time():
    started = time.perf_counter()
    assert not asks_for_it_here("x" + " " * 20_000 + "is here")
    assert asks_for_it_here("x" + " " * 20_000 + "draft it here")
    assert not asks_for_it_here("no" + " " * 20_000 + "x")
    assert time.perf_counter() - started < LINEAR_S
