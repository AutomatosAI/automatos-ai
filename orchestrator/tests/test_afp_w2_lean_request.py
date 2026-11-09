"""Wave 2, PR B: a lean turn's request may pick its model from the agent's allow-list and
send app context for that turn's prompt only (PRD draft "Automatos for products", §5.3–5.4).

Both are refused (422) before the message is saved for any agent that is not lean, a model
off the list, and app context over ``APP_CONTEXT_MAX_CHARS``. The model moves the turn the
way a spoken turn moves to the spoken model, held to the agent's policy, not Auto's seat.
The app context is never in the conversation; the turn's trace keeps its size only.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from consumers.chatbot import lean_request as lr
from consumers.chatbot import spoken_model

ALLOWED = [{"provider": "OpenRouter", "model_id": "acme/fast-1"}, {"provider": "anthropic", "model_id": "acme/deep-2"}]
CART = {"cart": {"items": 3, "total": "48.00 GBP"}, "page": "checkout"}


def _agent(lean=True, **configuration):
    configuration = {"runtime_profile": "lean" if lean else "full", "model_allow_list": ALLOWED, **configuration}
    return SimpleNamespace(id=7, is_system_agent=False, configuration=configuration)


@pytest.fixture(autouse=True)
def lean_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)
    monkeypatch.setattr(config, "APP_CONTEXT_MAX_CHARS", 200)


# --- the allow-list ---------------------------------------------------------------------------

def test_the_allow_list_is_a_list_of_provider_and_model():
    assert lr.allow_list_errors({}) == []
    assert lr.allow_list_errors({"model_allow_list": ALLOWED}) == []
    for bad in ("acme/fast-1", ["acme/fast-1"], [{"model_id": "acme/fast-1"}], [{"provider": "", "model_id": "x"}]):
        assert lr.allow_list_errors({"model_allow_list": bad}), bad


def test_saving_an_agent_with_a_bad_allow_list_is_refused():
    from api.agents import _reject_invalid_runtime

    _reject_invalid_runtime({"runtime_profile": "lean", "model_allow_list": ALLOWED})
    with pytest.raises(HTTPException) as refused:
        _reject_invalid_runtime({"model_allow_list": ["acme/fast-1"]})
    assert refused.value.status_code == 400


# --- the request ------------------------------------------------------------------------------

def test_a_request_that_asks_for_nothing_is_unchanged_for_any_agent():
    assert lr.checked_lean_request(None, None, None) is lr.NOTHING_ASKED
    assert lr.checked_lean_request(_agent(lean=False), None, {}) is lr.NOTHING_ASKED


@pytest.mark.parametrize("agent", [None, _agent(lean=False)])
def test_model_and_app_context_need_a_lean_agent(agent):
    for model, context in (("acme/fast-1", None), (None, CART)):
        with pytest.raises(lr.LeanRequestRefused, match="lean agent"):
            lr.checked_lean_request(agent, model, context)


def test_the_model_comes_from_the_agents_allow_list():
    asked = lr.checked_lean_request(_agent(), "acme/fast-1", None)
    assert asked.model == lr.TurnModel(provider="openrouter", model="acme/fast-1") and asked.app_context is None
    with pytest.raises(lr.LeanRequestRefused, match="not on this agent's model allow-list"):
        lr.checked_lean_request(_agent(), "acme/other-3", None)


def test_app_context_is_rendered_and_capped():
    assert lr.checked_lean_request(_agent(), None, CART).app_context == (
        '{"cart": {"items": 3, "total": "48.00 GBP"}, "page": "checkout"}'
    )
    assert lr.checked_lean_request(_agent(), None, "Ticket #881, priority high").app_context == "Ticket #881, priority high"
    with pytest.raises(lr.LeanRequestRefused, match="the limit is 200"):
        lr.checked_lean_request(_agent(), None, "x" * 201)


def test_the_route_refuses_with_422_before_it_saves_the_message(monkeypatch):
    from api import chat

    monkeypatch.setattr(chat, "_explicitly_chosen_agent", lambda db, ws, agent_id: agent_id)
    full = _agent(lean=False)
    db = SimpleNamespace(query=lambda model: SimpleNamespace(filter=lambda *a: SimpleNamespace(first=lambda: full)))
    request = SimpleNamespace(agentId=7, model="acme/fast-1", appContext=None)
    with pytest.raises(HTTPException) as refused:
        chat._checked_lean_request(db, SimpleNamespace(workspace_id="ws"), request)
    assert refused.value.status_code == 422
    source = inspect.getsource(chat.stream_chat)
    assert source.index("_checked_lean_request(") < source.index("save_message(")


# --- the prompt and the trace -----------------------------------------------------------------

def test_the_app_context_is_in_the_system_prompt_only():
    said = {"role": "user", "content": "Can I pay later?"}
    messages = [{"role": "system", "content": "You are Quill."}, said]
    shown = lr.with_app_context(messages, '{"cart": 3}')
    assert shown[0]["content"].startswith("You are Quill.") and lr.APP_CONTEXT_HEADING in shown[0]["content"]
    assert shown[1] is said and messages[0]["content"] == "You are Quill."  # the conversation is untouched
    assert lr.with_app_context(messages, None) is messages


def test_the_trace_keeps_the_size_never_the_words():
    trace = lr.with_app_context_size({"mode": "lean"}, '{"cart": 3}')
    assert trace == {"mode": "lean", "app_context_chars": 11}
    assert lr.with_app_context_size(None, None) is None


# --- the model the turn answers on --------------------------------------------------------------

def _turn_agents(monkeypatch, *, spoken, requested):
    moved = []

    async def _runtime_on(factory, runtime, workspace_id, wanted, *, orchestrator_seat, label):
        moved.append((wanted, orchestrator_seat, label))
        return runtime

    async def _activate(agent_id, **kwargs):
        return SimpleNamespace(llm_manager=SimpleNamespace(config=SimpleNamespace(model="agent/own")))

    monkeypatch.setattr(spoken_model, "runtime_on", _runtime_on)
    agents = spoken_model.TurnAgents(SimpleNamespace(activate_agent=_activate), spoken=spoken, requested=requested)
    asyncio.run(agents.activate_agent(7, workspace_id="ws"))
    return moved


def test_a_requested_model_moves_the_turn_on_the_agents_policy(monkeypatch):
    wanted = lr.TurnModel(provider="openrouter", model="acme/fast-1")
    assert _turn_agents(monkeypatch, spoken=True, requested=wanted) == [(wanted, False, "requested")]


def test_without_one_a_spoken_turn_moves_as_before(monkeypatch):
    moved = _turn_agents(monkeypatch, spoken=True, requested=None)
    assert [(seat, label) for _, seat, label in moved] == [(True, "spoken")]


def test_only_a_lean_turn_has_a_requested_model():
    asked = lr.LeanRequest(model=lr.TurnModel("openrouter", "acme/fast-1"))
    assert spoken_model._requested_model(SimpleNamespace(_lean_actions=[], _lean_request=asked)) == asked.model
    assert spoken_model._requested_model(SimpleNamespace(_lean_actions=None, _lean_request=asked)) is None


def test_a_lean_turns_prompt_carries_the_app_context_and_its_trace_the_size():
    from dataclasses import dataclass

    from consumers.chatbot import lean_turn

    @dataclass
    class Orchestrated:
        context_trace: dict

    @lean_turn.a_lean_turns_prompt
    async def prepare(chat, messages, agent_runtime, agent_ctx, all_tools, *rest, **kwargs):
        return [{"role": "system", "content": "You are Quill."}, {"role": "user", "content": "hi"}], None, \
            Orchestrated({"mode": "lean"})

    chat = SimpleNamespace(_lean_actions=[], _lean_request=lr.LeanRequest(app_context='{"cart": 3}'))
    llm_messages, _, orchestrated = asyncio.run(prepare(chat, [], None, {}, [], "chat"))
    assert lr.APP_CONTEXT_HEADING in llm_messages[0]["content"] and llm_messages[1] == {"role": "user", "content": "hi"}
    assert orchestrated.context_trace == {"mode": "lean", "app_context_chars": 11}
    assert '{"cart": 3}' not in str(orchestrated.context_trace)
