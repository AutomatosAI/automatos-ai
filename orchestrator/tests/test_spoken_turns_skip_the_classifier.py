"""Spoken turns skip the classifier's model call (voice latency C-1).

A turn sent with ``spokenMode`` and no ``agentId`` was classified by the whole AutoBrain before
the response headers: its last tier is a model call. With ``SPOKEN_MODE_SKIP_CLASSIFIER`` on,
``auto_lane`` builds ``SpokenAutoBrain``: the tiers that cost no model call, and nothing else.

* A spoken turn no tier matched asks no model and no decision engine, and carries no assessment
  (the shape Auto Live and ``agentId`` turns run).
* The tiers that cost nothing still decide: the hand-off table, the onboarding pin, a cached
  verdict, a greeting, a platform phrase.
* The unclassified verdict is never cached; a real one still is.
* The lane's rewrites of an unclassified verdict are kept, with memory left to the turn's
  intent: a named teammate's ticket (ASSIGN) and "answer it here".
* Given up on purpose and pinned here: a hand-off by role ("get the accountant to ...") is
  Auto's own answer.
* With the switch off, or on a typed turn, ``AutoBrain`` is built and Tier 3 is asked.
* ``_turn_lane`` reads ``spokenMode`` with ``getattr``; a spoken turn with ``agentId`` builds no
  brain at all.

DB-free: no database, model, decision engine, cache server or socket. The brains under test are
the real classes with their roster, onboarding flag and cache given by the test.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from api import chat, chat_dispatch
from api.chat_models import ChatRequest
from config import config
from consumers.chatbot import auto as auto_mod
from consumers.chatbot import spoken_lane
from consumers.chatbot.asked_here import HERE_DIRECTIVE
from consumers.chatbot.auto import Action, AutoBrain, Complexity, ComplexityAssessment
from consumers.chatbot.spoken_lane import UNCLASSIFIED, SpokenAutoBrain, is_unclassified, skips_the_classifier
from consumers.chatbot.streaming import StreamingHandler
from consumers.chatbot.verdict_parser import lost_the_lane
from core.llm.decisions import MODE_OFF, MODE_SHADOW

ORCH = Path(__file__).resolve().parents[1]
WS = "6a0f3c1e-2b4d-4e8f-9a1b-c10000000001"
CHAT_ID = "6a0f3c1e-2b4d-4e8f-9a1b-c1000000c4a7"
USER_PK = 7
AUTO = 11
BOB = 22
ACCOUNTANT = 25
DESIGNER = 347
CHOSEN = 23
ROSTER = [
    NS(id=AUTO, name="Auto", status="active", is_system_agent=True),
    NS(id=BOB, name="Bob", status="active"),
    NS(id=ACCOUNTANT, name="Penny", job_title="Accountant", status="active"),
]
GENERAL = "What's a good subject line for a welcome email?"  # no tier below the model reads it
GREETING = "hi"
PLATFORM_ASK = "list my agents"
NAMED_HAND_OFF = "ask Bob to draft the newsletter"
ASKED_HERE = "draft the reply and show me"
ROLE_HAND_OFF = "get the accountant to file this"
BRAND_ASK = "Redesign our brand kit from the logo."
ONBOARDING_ANSWER = "We're a coffee roaster in Cork"
DISPATCH_LOGGER = "api.chat_dispatch"
MIN_CONFIDENCE = 0.8


# --------------------------------------------------------------------------- what must not be asked


@pytest.fixture(autouse=True)
def model(monkeypatch):
    """The LLM manager's factory: every call is kept in ``model.asked``, and none succeeds."""
    import core.llm as llm

    state = NS(asked=[])

    def _create(*args, **kwargs):
        state.asked.append(kwargs.get("service_name"))
        raise AssertionError("a spoken turn asked the classifier's model")

    monkeypatch.setattr(llm, "create_llm_manager", _create)
    return state


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    """The decision engine: its dial is ``engine.mode``; every read of the dial and every
    question asked is counted."""
    state = NS(mode=MODE_OFF, dialled=0, asked=[])

    def _dials():
        state.dialled += 1
        return NS(classifier_mode=state.mode, min_confidence=MIN_CONFIDENCE)

    async def _decide(**kwargs):
        state.asked.append(kwargs.get("purpose"))
        return None

    monkeypatch.setattr(auto_mod, "get_decision_engine",
                        lambda: NS(dials=_dials, decide=_decide, record_shadow=lambda row: None))
    return state


@pytest.fixture
def switch_on(monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_CLASSIFIER", True)


# --------------------------------------------------------------------------- the brains


class _Redis:
    """Tier 1's store: ``held`` answers every lookup; every write is kept in ``stored``."""

    def __init__(self, held=None):
        self.held, self.stored = held, []

    def get(self, key):
        return self.held

    def setex(self, key, ttl, value):
        self.stored.append(json.loads(value))


class _NoDatabase:
    """What a brain reads from the database and Redis, given by the test instead."""

    roster = ROSTER
    onboarding = False
    redis = None

    def __init__(self, db, workspace_id):
        self._db, self._workspace_id, self._redis = db, workspace_id, type(self).redis

    def _active_agents(self):
        return list(type(self).roster)

    def _onboarding_active(self):
        return type(self).onboarding


class _Spoken(_NoDatabase, SpokenAutoBrain):
    """The real spoken brain: every tier and both ``assess`` wrappers are its own."""


class _Typed(_NoDatabase, AutoBrain):
    """The real AutoBrain with Tier 3's model call answered by ``verdict``; each call is kept."""

    asked: list = []
    verdict = None

    async def _llm_classify(self, message, conversation_length):
        type(self).asked.append(message)
        return type(self).verdict


class _NeverBuilt:
    def __init__(self, *args, **kwargs):
        raise AssertionError("this brain must not be built for this turn")


def _spoken_brain(monkeypatch, **given):
    """``auto_lane`` builds this spoken brain, and can build no other."""
    monkeypatch.setattr(chat_dispatch, "SpokenAutoBrain", type("Spoken", (_Spoken,), given))
    monkeypatch.setattr(chat_dispatch, "AutoBrain", _NeverBuilt)


def _typed_brain(monkeypatch, verdict):
    """``auto_lane`` builds this full brain, and can build no other. Returns its class."""
    brain = type("Typed", (_Typed,), {"asked": [], "verdict": verdict})
    monkeypatch.setattr(chat_dispatch, "AutoBrain", brain)
    monkeypatch.setattr(chat_dispatch, "SpokenAutoBrain", _NeverBuilt)
    return brain


def _lane(message, *, spoken=True):
    return asyncio.run(chat_dispatch.auto_lane(
        None, WS, auto_agent_id=AUTO, message_text=message, history_length=1, spoken=spoken))


def _tier_three(**over):
    values = {"complexity": Complexity.ATOM, "action": Action.RESPOND, "reasoning": "tier three"}
    return ComplexityAssessment(**{**values, **over})


# --------------------------------------------------------------------------- the setting


def test_the_switch_is_off_unless_the_environment_turns_it_on():
    source = (ORCH / "config.py").read_text()

    assert 'os.getenv("SPOKEN_MODE_SKIP_CLASSIFIER", "false").strip().lower() == "true"' in source
    assert isinstance(config.SPOKEN_MODE_SKIP_CLASSIFIER, bool)


@pytest.mark.parametrize("setting,spoken,skips", [
    (True, True, True), (True, False, False), (False, True, False), (False, False, False),
    (True, None, False),
])
def test_only_a_spoken_turn_with_the_switch_on_skips_the_classifier(monkeypatch, setting, spoken, skips):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_CLASSIFIER", setting)

    assert skips_the_classifier(spoken) is skips


# --------------------------------------------------------------------------- the marker


def test_the_unclassified_verdict_keeps_the_tools_and_says_what_it_is():
    brain = SpokenAutoBrain.__new__(SpokenAutoBrain)

    verdict = asyncio.run(brain._llm_classify(GENERAL, 1))

    assert verdict == dataclasses.replace(lost_the_lane(), reasoning=UNCLASSIFIED)
    assert (verdict.complexity, verdict.action) == (Complexity.MOLECULE, Action.RESPOND)


def test_the_marker_is_read_by_containment_because_the_lane_appends_to_the_reasoning():
    assert is_unclassified(NS(reasoning=UNCLASSIFIED)) is True
    assert is_unclassified(NS(reasoning=f"{UNCLASSIFIED} (PRD-256 D2: the named agent gets a ticket)")) is True


@pytest.mark.parametrize("assessment", [
    None, NS(reasoning=None), NS(reasoning=""), NS(reasoning="Greeting or chitchat"), lost_the_lane(), NS(),
])
def test_any_other_verdict_is_not_the_unclassified_one(assessment):
    assert is_unclassified(assessment) is False


def test_the_spoken_brain_inherits_assess_and_its_two_wrappers():
    assert "assess" not in vars(SpokenAutoBrain)
    assert SpokenAutoBrain.assess is AutoBrain.assess


# --------------------------------------------------------------------------- no model call


def test_a_spoken_turn_no_tier_matched_asks_no_model_and_carries_no_assessment(monkeypatch, switch_on, model, engine):
    _spoken_brain(monkeypatch)

    lane = _lane(GENERAL)

    assert (lane.agent_id, lane.assessment) == (AUTO, None)
    assert lane.skip_composio is False
    assert (lane.suggest_mission, lane.session_agent) == (False, False)
    assert model.asked == []
    assert (engine.dialled, engine.asked) == (0, [])


def test_an_unclassified_turn_sends_no_auto_headers(monkeypatch, switch_on):
    _spoken_brain(monkeypatch)

    headers = chat_dispatch.response_headers(_lane(GENERAL).assessment)

    assert [name for name in headers if name.startswith("x-auto-")] == []


def test_the_unclassified_line_names_the_agent_and_none_of_the_words(monkeypatch, switch_on, caplog):
    _spoken_brain(monkeypatch)

    with caplog.at_level(logging.INFO, logger=DISPATCH_LOGGER):
        _lane(GENERAL)

    said = [r.getMessage() for r in caplog.records if r.name == DISPATCH_LOGGER]
    assert said == [f"[Auto] spoken turn, no tier matched: Auto answers unclassified: agent_id={AUTO}"]


def test_no_shadow_call_is_started_in_shadow_mode(monkeypatch, switch_on, engine):
    engine.mode = MODE_SHADOW
    _spoken_brain(monkeypatch)

    async def _run():
        before = set(auto_mod._SHADOW_TASKS)
        lane = await chat_dispatch.auto_lane(
            None, WS, auto_agent_id=AUTO, message_text=GENERAL, history_length=1, spoken=True)
        return lane, set(auto_mod._SHADOW_TASKS) - before

    lane, started = asyncio.run(_run())

    assert lane.assessment is None
    assert started == set()
    assert (engine.dialled, engine.asked) == (0, [])


# --------------------------------------------------------------------------- the tiers that cost nothing


def test_a_spoken_greeting_is_still_the_light_path(monkeypatch, switch_on, model):
    _spoken_brain(monkeypatch)

    lane = _lane(GREETING)

    assert (lane.assessment.complexity, lane.assessment.action) == (Complexity.ATOM, Action.RESPOND)
    assert lane.skip_composio is True
    assert model.asked == []


def test_a_spoken_platform_phrase_keeps_its_platform_hint(monkeypatch, switch_on, model):
    _spoken_brain(monkeypatch)

    lane = _lane(PLATFORM_ASK)

    assert (lane.assessment.complexity, lane.assessment.action) == (Complexity.MOLECULE, Action.RESPOND)
    assert lane.assessment.tool_hints == ["platform"]
    assert model.asked == []


def test_a_verdict_cached_from_a_typed_turn_is_used(monkeypatch, switch_on, model):
    typed = ComplexityAssessment(complexity=Complexity.CELL, action=Action.RESPOND, reasoning="typed",
                                 confidence=0.9, needs_memory=True, tool_hints=["email"])
    redis = _Redis(held=json.dumps(typed.to_dict()))
    _spoken_brain(monkeypatch, redis=redis)

    lane = _lane(GENERAL)

    assert (lane.assessment.complexity, lane.assessment.reasoning) == (Complexity.CELL, "typed (cached)")
    assert (lane.assessment.needs_memory, lane.assessment.tool_hints) == (True, ["email"])
    assert redis.stored == [] and model.asked == []


def test_mid_onboarding_is_still_the_onboarding_pin(monkeypatch, switch_on, model):
    _spoken_brain(monkeypatch, onboarding=True)

    lane = _lane(ONBOARDING_ANSWER)

    assert lane.assessment.reasoning.startswith("Onboarding active")
    assert (lane.assessment.action, lane.assessment.tool_hints) == (Action.RESPOND, ["platform"])
    assert model.asked == []


def test_a_brand_ask_is_still_the_designers_ticket(monkeypatch, switch_on, model):
    from core.seeds import seed_brand_designer

    designer = NS(id=DESIGNER, name="Brand Designer", status="active")
    monkeypatch.setattr(seed_brand_designer, "find_brand_designer", lambda db, workspace_id: designer)
    _spoken_brain(monkeypatch)

    lane = _lane(BRAND_ASK)

    assert lane.agent_id == AUTO
    assert (lane.assessment.action, lane.assessment.target_agent_id) == (Action.ASSIGN, DESIGNER)
    assert not is_unclassified(lane.assessment)
    assert model.asked == []


# --------------------------------------------------------------------------- the cache


def test_the_unclassified_verdict_is_never_cached(monkeypatch, switch_on):
    redis = _Redis()
    _spoken_brain(monkeypatch, redis=redis)

    _lane(GENERAL)
    _lane(NAMED_HAND_OFF)
    _lane(ASKED_HERE)

    assert redis.stored == [], "it would answer typed turns from Tier 1 for a day"


def test_a_tier_two_verdict_is_still_cached(monkeypatch, switch_on):
    redis = _Redis()
    _spoken_brain(monkeypatch, redis=redis)

    _lane(GREETING)

    assert [stored["complexity"] for stored in redis.stored] == ["atom"]


def test_the_typed_brain_caches_what_its_model_said(monkeypatch):
    """The contrast: the full AutoBrain keeps its Tier 3 verdict, as before."""
    redis = _Redis()
    brain = _typed_brain(monkeypatch, _tier_three())
    brain.redis = redis

    _lane(GENERAL, spoken=False)

    assert [stored["reasoning"] for stored in redis.stored] == ["tier three"]


# --------------------------------------------------------------------------- the lane's rewrites


def test_a_named_hand_off_keeps_the_assign_lane_and_does_not_skip_memory(monkeypatch, switch_on, model):
    _spoken_brain(monkeypatch)

    lane = _lane(NAMED_HAND_OFF)

    assert lane.agent_id == AUTO, "Auto files the ticket; Bob never answers the chat"
    assert lane.assessment.action == Action.ASSIGN
    assert (lane.assessment.target_agent_id, lane.assessment.target_agent_name) == (BOB, "Bob")
    assert lane.assessment.needs_memory is True
    assert is_unclassified(lane.assessment)
    assert model.asked == []


def test_work_asked_for_here_keeps_the_here_directive_and_does_not_skip_memory(monkeypatch, switch_on, model):
    _spoken_brain(monkeypatch)

    lane = _lane(ASKED_HERE)

    assert lane.assessment.action == Action.RESPOND
    assert lane.assessment.context_directive == HERE_DIRECTIVE
    assert lane.assessment.needs_memory is True
    assert lane.skip_composio is True, "stated side effect: the lane reads it as a recall turn"
    assert model.asked == []


def test_a_hand_off_by_role_is_given_up_and_auto_answers(monkeypatch, switch_on, model):
    """Pinned on purpose: only Tier 3 resolved a role ("the accountant" is Penny) to an agent."""
    _spoken_brain(monkeypatch)

    lane = _lane(ROLE_HAND_OFF)

    assert (lane.agent_id, lane.assessment) == (AUTO, None)
    assert model.asked == []


# --------------------------------------------------------------------------- off, and typed turns


@pytest.mark.parametrize("setting,spoken", [(False, True), (True, False), (False, False)])
def test_with_the_switch_off_or_a_typed_turn_the_full_brain_is_built_and_tier_three_is_asked(
        monkeypatch, setting, spoken):
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_CLASSIFIER", setting)
    brain = _typed_brain(monkeypatch, _tier_three())

    lane = _lane(GENERAL, spoken=spoken)

    assert brain.asked == [GENERAL]
    assert (lane.assessment.complexity, lane.assessment.reasoning) == (Complexity.ATOM, "tier three")


def test_auto_lane_without_the_keyword_is_a_typed_turn(monkeypatch, switch_on):
    brain = _typed_brain(monkeypatch, _tier_three())

    lane = asyncio.run(chat_dispatch.auto_lane(
        None, WS, auto_agent_id=AUTO, message_text=GENERAL, history_length=1))

    assert brain.asked == [GENERAL] and lane.assessment is not None


def test_with_the_switch_off_a_verdict_that_carries_the_marker_is_kept_as_it_is(monkeypatch):
    """Off is exactly as before: nothing reads the marker."""
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_CLASSIFIER", False)
    _typed_brain(monkeypatch, dataclasses.replace(lost_the_lane(), reasoning=UNCLASSIFIED))

    lane = _lane(GENERAL)

    assert lane.assessment is not None
    assert (lane.assessment.reasoning, lane.assessment.needs_memory) == (UNCLASSIFIED, False)


# --------------------------------------------------------------------------- the wiring


def test_the_dispatch_builds_the_spoken_brain_from_its_module():
    assert chat_dispatch.SpokenAutoBrain is spoken_lane.SpokenAutoBrain
    assert chat_dispatch.skips_the_classifier is spoken_lane.skips_the_classifier
    assert issubclass(SpokenAutoBrain, AutoBrain)


def test_auto_live_reads_neither_the_module_nor_the_setting():
    source = (ORCH / "api" / "voice_retell.py").read_text()

    assert "spoken_lane" not in source
    assert "SPOKEN_MODE_SKIP_CLASSIFIER" not in source


@pytest.mark.parametrize("request_,spoken", [
    (NS(agentId=None), False),  # a caller's request object with no spokenMode at all
    (NS(agentId=None, spokenMode=None), False),
    (NS(agentId=None, spokenMode=False), False),
    (NS(agentId=None, spokenMode=True), True),
])
def test_the_route_tells_the_lane_whether_the_turn_is_spoken(monkeypatch, request_, spoken):
    asked = []

    async def _auto_lane(db, workspace_id, **kwargs):
        asked.append(kwargs)
        return "the lane"

    monkeypatch.setattr(chat, "auto_lane", _auto_lane)
    monkeypatch.setattr(chat, "get_default_agent_id", lambda db, workspace_id: AUTO)

    lane = asyncio.run(chat._turn_lane(None, NS(workspace_id=WS), request_, GENERAL, 1))

    (kwargs,) = asked
    assert lane == "the lane"
    assert kwargs["spoken"] is spoken
    assert (kwargs["auto_agent_id"], kwargs["message_text"], kwargs["history_length"]) == (AUTO, GENERAL, 1)


# --------------------------------------------------------------------------- through the route


@pytest.fixture
def route(monkeypatch):
    """``POST /api/chat`` with its collaborators patched (the fixture of
    ``test_a_named_app_brings_the_typed_tools_back.py``), but with the real ``_turn_lane``;
    the streaming service keeps the kwargs it was asked with."""
    asked = []
    handler = StreamingHandler()

    class _ChatService:
        def __init__(self, db):
            self.db = db

        def save_message(self, **kwargs):
            return None

    class _StreamingService:
        def __init__(self, db, workspace_id=None):
            self.streaming_handler = handler

        async def stream_response_with_agent(self, **kwargs):
            asked.append(kwargs)
            yield handler.format_aisdk_text("Done.")
            yield handler.format_aisdk_finish()

    class _Queue:
        @asynccontextmanager
        async def acquire(self, key):
            yield

    async def _inline(*, chat_id, produce, on_complete=None):
        async for chunk in produce():
            yield chunk

    monkeypatch.setattr(chat, "ChatService", _ChatService)
    monkeypatch.setattr(chat, "StreamingChatService", _StreamingService)
    monkeypatch.setattr(chat, "SessionLocal", lambda: NS(close=lambda: None))
    monkeypatch.setattr(chat, "get_session_queue", lambda: _Queue())
    monkeypatch.setattr(chat, "get_user_id", lambda db, ctx=None: USER_PK)
    monkeypatch.setattr(chat, "_open_chat", lambda *args, **kwargs: CHAT_ID)
    monkeypatch.setattr(chat, "turn_history", lambda *args, **kwargs: [])
    monkeypatch.setattr(chat, "get_default_agent_id", lambda db, workspace_id: AUTO)
    monkeypatch.setattr(chat, "run_detached_turn", _inline)
    monkeypatch.setattr(config, "SPOKEN_MODE_SKIP_COMPOSIO", False)  # C-2 is its own test file
    return NS(asked=asked)


def _ask(body: dict) -> list:
    """The frames ``POST /api/chat`` streams for ``body``."""
    ctx = NS(workspace_id=WS, user=NS(id="owner@c1.test", system_role="user"))

    async def _run() -> list:
        response = await chat.stream_chat(request=ChatRequest(**body), ctx=ctx, db=NS(rollback=lambda: None))
        return [chunk async for chunk in response.body_iterator]

    return asyncio.run(_run())


def _body(text: str = GENERAL, **over) -> dict:
    return {"message": {"role": "user", "parts": [{"type": "text", "text": text}]}, **over}


def test_a_spoken_auto_turn_reaches_the_service_with_no_assessment_and_no_model_call(
        route, monkeypatch, switch_on, model):
    _spoken_brain(monkeypatch)

    _ask(_body(spokenMode=True))

    (kwargs,) = route.asked
    assert (kwargs["agent_id"], kwargs["spoken_mode"]) == (AUTO, True)
    assert kwargs["complexity_assessment"] is None
    assert kwargs["skip_composio"] is False
    assert model.asked == []


def test_a_typed_auto_turn_still_reaches_the_service_with_tier_threes_verdict(route, monkeypatch, switch_on):
    brain = _typed_brain(monkeypatch, _tier_three())

    _ask(_body(spokenMode=False))

    (kwargs,) = route.asked
    assert brain.asked == [GENERAL]
    assert kwargs["complexity_assessment"].reasoning == "tier three"


def test_a_spoken_turn_for_a_chosen_agent_builds_no_brain(route, monkeypatch, switch_on, model):
    import services.cli_ticket_lane as cli

    monkeypatch.setattr(chat, "_explicitly_chosen_agent", lambda db, workspace_id, agent_id: agent_id)
    monkeypatch.setattr(cli, "is_cli_agent", lambda db, agent_id: False)
    monkeypatch.setattr(chat_dispatch, "SpokenAutoBrain", _NeverBuilt)
    monkeypatch.setattr(chat_dispatch, "AutoBrain", _NeverBuilt)

    _ask(_body(spokenMode=True, agentId=CHOSEN))

    (kwargs,) = route.asked
    assert (kwargs["agent_id"], kwargs["complexity_assessment"]) == (CHOSEN, None)
    assert model.asked == []
