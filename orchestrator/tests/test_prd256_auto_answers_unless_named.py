"""PRD-256 US-010 (Decision D2): Auto always answers; work goes to agents on tickets.

72% of the Tier-3 verdicts were DELEGATE, and api/chat.py sent each one through the
Universal Router to a specialist, which answered the owner's chat in its own persona with
its own tools. On the dispatch now:

- an unnamed ask is Auto's (a DELEGATE verdict becomes RESPOND, the platform tools kept);
- an ask that names an agent is that agent's ticket (ASSIGN), filed by Auto;
- "Give #0192 to the Support Agent" is the ASSIGN lane on #0192 itself: no copy;
- an agent the owner chose in the UI (``request.agentId``) answers, and only then;
- no DELEGATE verdict is cached, and the Jev shadow still records the classifier's own.

No model is called and no database is opened: AutoBrain's tiers are replaced by the
verdict each test gives, while its roster match and the D2 decorator are the real ones.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from consumers.chatbot.auto import (
    ASSIGN_TOOL_HINTS,
    Action,
    AutoBrain,
    Complexity,
    ComplexityAssessment,
    build_assessment_prompt,
)
from consumers.chatbot.auto_answers import auto_always_answers, needs_no_apps

ORCH = Path(__file__).resolve().parents[1]
AUTO = 11
SUPPORT, JIM = 21, 22
ROSTER = [
    NS(id=AUTO, name="Auto", status="active", is_system_agent=True),
    NS(id=SUPPORT, name="Shopify Support Agent", status="active"),
    NS(id=JIM, name="Jim", status="active"),
    NS(id=23, name="Researcher", status="active"),
    NS(id=24, name="Customer Support", status="active"),
]


def _verdict(action, *, complexity=Complexity.MOLECULE, hints=(), **over):
    return ComplexityAssessment(complexity=complexity, action=action, reasoning="tiers",
                                tool_hints=list(hints), **over)


class _Brain:
    """AutoBrain with its tiers replaced by the test's verdict; its roster match is real."""

    verdict = None

    def __init__(self, db, workspace_id):
        self._db, self._workspace_id = db, workspace_id

    def _active_agents(self):
        return ROSTER

    _match_roster_agent = AutoBrain._match_roster_agent

    @auto_always_answers
    async def assess(self, message, conversation_length=0):
        return type(self).verdict


def _lane(monkeypatch, message, verdict):
    import api.chat_dispatch as dispatch

    monkeypatch.setattr(dispatch, "AutoBrain", type("Brain", (_Brain,), {"verdict": verdict}))
    return asyncio.run(dispatch.auto_lane(None, "ws-1", auto_agent_id=AUTO, message_text=message,
                                          history_length=1))


# ── an unnamed ask is Auto's ──────────────────────────────────────────────────

def test_an_unnamed_ask_is_answered_by_auto_with_its_platform_tools(monkeypatch):
    lane = _lane(monkeypatch, "Send an email to John", _verdict(Action.DELEGATE, hints=["email"]))

    assert lane.agent_id == AUTO
    assert lane.assessment.action == Action.RESPOND
    assert lane.assessment.tool_hints == ["email", "platform"]
    assert lane.skip_composio is False, "Auto keeps the workspace's apps for the old DELEGATE work"
    assert lane.suggest_mission is False


def test_an_unnamed_ask_with_no_hint_keeps_every_tool(monkeypatch):
    lane = _lane(monkeypatch, "Search my docs for the Q4 report", _verdict(Action.DELEGATE))

    assert (lane.agent_id, lane.assessment.action, lane.assessment.tool_hints) == (AUTO, Action.RESPOND, [])
    assert lane.skip_composio is False


def test_a_mission_is_suggested_and_auto_answers(monkeypatch):
    lane = _lane(monkeypatch, "Research competitors, write a report, build a deck and email the team",
                 _verdict(Action.MISSION, complexity=Complexity.ORGAN))

    assert lane.agent_id == AUTO and lane.suggest_mission is True


def test_a_platform_hint_is_autos_own_work(monkeypatch):
    lane = _lane(monkeypatch, "What agents do I have?", _verdict(Action.MISSION, hints=["platform"]))

    assert (lane.agent_id, lane.assessment.action, lane.suggest_mission) == (AUTO, Action.RESPOND, False)
    assert lane.skip_composio is True


# ── a named agent gets a ticket ───────────────────────────────────────────────

def test_an_ask_that_names_an_agent_is_its_ticket(monkeypatch):
    lane = _lane(monkeypatch, "Ask Jim to draft the board pack", _verdict(Action.DELEGATE))

    assert lane.agent_id == AUTO, "Auto files the ticket; Jim never answers the chat"
    assert lane.assessment.action == Action.ASSIGN
    assert (lane.assessment.target_agent_id, lane.assessment.target_agent_name) == (JIM, "Jim")
    assert set(ASSIGN_TOOL_HINTS) <= set(lane.assessment.tool_hints)
    directive = lane.assessment.context_directive
    assert 'assigned_agent_name="Jim"' in directive and "Confirm in ONE line with the task id" in directive


@pytest.mark.parametrize("said", [
    "Hey Auto, send an email to John",         # Auto is never its own ticket's agent
    "What did Jim say about the invoice?",     # Jim is mentioned, not handed work
    "Compare the Researcher's notes with the stock list",
])
def test_a_name_only_mentioned_hands_nothing_over(monkeypatch, said):
    lane = _lane(monkeypatch, said, _verdict(Action.DELEGATE))

    assert (lane.agent_id, lane.assessment.action) == (AUTO, Action.RESPOND)
    assert lane.assessment.target_agent_id is None


def test_the_researcher_asked_by_role_gets_the_ticket(monkeypatch):
    lane = _lane(monkeypatch, "Ask the researcher to find three roasters", _verdict(Action.DELEGATE))

    assert (lane.assessment.action, lane.assessment.target_agent_id) == (Action.ASSIGN, 23)


def test_a_role_with_no_roster_match_asks_once(monkeypatch):
    lane = _lane(monkeypatch, "Have my accountant agent chase the invoices",
                 _verdict(Action.ASSIGN, target_agent_name="accountant"))

    assert lane.agent_id == AUTO and lane.assessment.target_agent_id is None
    assert "confirm the agent first" in lane.assessment.context_directive


@pytest.mark.parametrize("said", [
    "Give #0192 to the Support Agent",
    "Please give #0192 to the Shopify Support Agent.",
    "Auto, give #0192 to the Support Agent",                  # Auto is named, never the receiver
    "Give #0192 to the Support Agent to handle by Friday",    # the receiver, not the purpose
])
def test_giving_a_card_to_an_agent_assigns_that_card_and_makes_no_copy(monkeypatch, said):
    """The F263 fast path classifies a card message MOLECULE/RESPOND with the platform hint."""
    lane = _lane(monkeypatch, said, _verdict(Action.RESPOND, hints=["platform"]))

    assert lane.agent_id == AUTO
    assert lane.assessment.action == Action.ASSIGN
    assert lane.assessment.target_agent_id == SUPPORT
    directive = lane.assessment.context_directive
    assert 'platform_assign_task with task_id "#0192" and agent_name "Shopify Support Agent"' in directive
    assert "do NOT create a new card" in directive and "platform_create_task" not in directive


@pytest.mark.parametrize("said", [
    "Give #0192 to Bob",                       # nobody on the roster: Auto asks
    "Give #0192 to Bob, Jim already has too many",   # Jim is mentioned, Bob is the receiver
    "Give #0192 to me",                        # a pronoun is nobody
    "Approve #0177 with this note: Going with Kestrel's 250-box run.",
    "Approve #0177, tell Jim to give the customer a discount to keep them",
    "Move #0192 to review",                    # a status, not a hand-off
    "Can you close all the blocked tickets for VECTOR?",
])
def test_every_other_board_message_stays_autos(monkeypatch, said):
    lane = _lane(monkeypatch, said, _verdict(Action.RESPOND, hints=["platform"]))

    assert (lane.agent_id, lane.assessment.action) == (AUTO, Action.RESPOND)


# ── the owner's own choice, through api/chat.py ───────────────────────────────

def test_an_agent_the_owner_chose_answers(monkeypatch):
    import api.chat as chat
    import services.cli_ticket_lane as cli

    async def never(*a, **k):
        raise AssertionError("Auto must not classify a turn the owner gave to an agent")

    monkeypatch.setattr(chat, "_explicitly_chosen_agent", lambda db, ws, agent_id: agent_id)
    monkeypatch.setattr(cli, "is_cli_agent", lambda db, agent_id: False)
    monkeypatch.setattr(chat, "auto_lane", never)

    lane = asyncio.run(chat._turn_lane(None, NS(workspace_id="ws-1"), NS(agentId=77), "hi", 1))

    assert (lane.agent_id, lane.assessment, lane.session_agent) == (77, None, False)


def test_no_agent_chosen_is_auto(monkeypatch):
    import api.chat as chat
    import api.chat_dispatch as dispatch

    monkeypatch.setattr(chat, "get_default_agent_id", lambda db, ws: AUTO)
    monkeypatch.setattr(dispatch, "AutoBrain", type("Brain", (_Brain,), {"verdict": _verdict(Action.DELEGATE)}))

    lane = asyncio.run(chat._turn_lane(None, NS(workspace_id="ws-1"), NS(agentId=None), "Check the stock", 1))

    assert (lane.agent_id, lane.assessment.action) == (AUTO, Action.RESPOND)


def test_the_chat_dispatch_never_routes_to_another_agent():
    for rel in ("api/chat.py", "api/chat_dispatch.py"):
        source = (ORCH / rel).read_text(encoding="utf-8")
        assert "UniversalRouter" not in source and "ChatbotIngestor" not in source, rel
    assert not (ORCH / "core" / "routing" / "ingestors" / "chatbot.py").exists()


def test_the_classifier_is_no_longer_offered_delegate():
    prompt = build_assessment_prompt("Send an email to John", 0, "")

    assert "**delegate**" not in prompt and "Most molecule/cell/organ work" not in prompt
    assert "/ delegate" not in prompt
    assert '"action": "respond|assign|mission"' in prompt


# ── which turns may skip the connected apps ───────────────────────────────────

@pytest.mark.parametrize("verdict, skips", [
    (_verdict(Action.RESPOND, complexity=Complexity.ATOM), True),
    (_verdict(Action.RESPOND, hints=["platform"]), True),
    (_verdict(Action.RESPOND, complexity=Complexity.CELL, needs_memory=True), True),
    (_verdict(Action.RESPOND), False),
    (_verdict(Action.RESPOND, hints=["email", "platform"]), False),
    (_verdict(Action.ASSIGN, hints=["platform"]), False),
    (None, False),
])
def test_only_chitchat_platform_and_memory_turns_skip_the_apps(verdict, skips):
    assert needs_no_apps(verdict) is skips


# ── the cache and the shadow ──────────────────────────────────────────────────

class _Redis:
    def __init__(self):
        self.stored = []

    def setex(self, key, ttl, value):
        self.stored.append(key)

    def get(self, key):
        return None


def _real_brain(monkeypatch, tier_two_verdict, shadowed):
    brain = AutoBrain.__new__(AutoBrain)
    brain._db, brain._workspace_id, brain._redis = None, "ws-1", _Redis()
    monkeypatch.setattr(brain, "_onboarding_active", lambda: False)
    monkeypatch.setattr(brain, "_active_agents", lambda: ROSTER)
    monkeypatch.setattr(brain, "_start_shadow", lambda message, length: "shadow-task")
    monkeypatch.setattr(brain, "_run_fast_heuristics", lambda msg_lower: tier_two_verdict)

    def record(assessment, tier, task, message, started):
        shadowed.append((assessment.action, tier, task))
        return assessment

    monkeypatch.setattr(brain, "_with_shadow", record)
    return brain


def test_a_delegate_verdict_is_never_cached_and_the_shadow_still_records_it(monkeypatch):
    shadowed = []
    brain = _real_brain(monkeypatch, _verdict(Action.DELEGATE, hints=["email"]), shadowed)

    out = asyncio.run(brain.assess("Send an email to John", 1))

    assert out.action == Action.RESPOND
    assert brain._redis.stored == [], "a routed-away verdict is never kept for 24 hours"
    assert shadowed == [(Action.DELEGATE, 2, "shadow-task")], "the shadow records the classifier unchanged"


def test_a_verdict_auto_keeps_is_still_cached(monkeypatch):
    brain = _real_brain(monkeypatch, _verdict(Action.RESPOND, hints=["platform"]), [])

    asyncio.run(brain.assess("What agents do I have?", 1))

    assert len(brain._redis.stored) == 1
