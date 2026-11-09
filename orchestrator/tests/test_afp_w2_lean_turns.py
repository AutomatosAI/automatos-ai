"""Lean turns (PRD draft "Automatos for products", Wave 2, §5.3).

A lean agent's chat turn assembles only its own identity, what it recalls, the date,
the owner's standing rules and the conversation; it is offered only the tools its
session tool groups grant, and the executor refuses anything else. Off by default
(``LEAN_TURNS_ENABLED``), and never for a system agent.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from consumers.chatbot import lean_turn, turn_timing
from core.security.lean_grants import LEAN_REFUSAL, lean_grants, lean_may_call, lean_turn as is_lean_turn

PERSONA = "You are Quill, the support agent for Acme Billing. You answer customers' invoice questions."


def _agent(**configuration):
    return SimpleNamespace(
        id=None, name="Quill", agent_type="support", description="Answers Acme Billing's customers.",
        workspace_id="00000000-0000-0000-0000-0000000000c1", is_system_agent=False, slug="quill",
        use_custom_persona=True, custom_persona_prompt=PERSONA, persona=None, configuration=configuration,
    )


@pytest.fixture
def lean_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)


# --- the profile -----------------------------------------------------------------------

def test_the_profile_is_full_or_lean():
    assert lean_turn.profile_errors({}) == []
    assert lean_turn.profile_errors({"runtime_profile": "lean"}) == []
    assert lean_turn.profile_errors({"runtime_profile": "full"}) == []
    assert "runtime_profile" in lean_turn.profile_errors({"runtime_profile": "fast"})[0]


def test_saving_an_agent_with_an_unknown_profile_is_refused():
    from api.agents import _reject_invalid_runtime

    _reject_invalid_runtime({"runtime_profile": "lean"})
    with pytest.raises(HTTPException) as refused:
        _reject_invalid_runtime({"runtime_profile": "fast"})
    assert refused.value.status_code == 400


def test_a_lean_agent_is_lean_only_with_the_dial_on(monkeypatch):
    from config import config

    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", False)
    assert not lean_turn.is_lean(_agent(runtime_profile="lean"))
    monkeypatch.setattr(config, "LEAN_TURNS_ENABLED", True)
    assert lean_turn.is_lean(_agent(runtime_profile="lean"))
    assert not lean_turn.is_lean(_agent())
    assert not lean_turn.is_lean(_agent(runtime_profile="full"))


def test_a_system_agent_is_never_lean(lean_on):
    auto = _agent(runtime_profile="lean")
    auto.is_system_agent = True
    assert not lean_turn.is_lean(auto)


# --- the grants ----------------------------------------------------------------------------

def test_the_grants_follow_the_agents_session_tool_groups():
    core = ["platform_search_memory", "platform_search_documents", "platform_read_document"]
    only_playbooks = lean_turn.granted_names(_agent(session_tool_groups=["playbooks"]))
    assert only_playbooks == [*core, "platform_list_playbooks", "platform_get_playbook", "platform_execute_playbook"]
    assert lean_turn.granted_names(_agent(session_tool_groups=[])) == core
    everything = lean_turn.granted_names(_agent())  # no saved list: every group, as its sessions
    assert {"platform_query_data", "platform_list_missions", "generate_document"} <= set(everything)
    assert not {"platform_ask_human", "platform_update_task_status", "composio_execute"} & set(everything)


def test_a_lean_turn_is_offered_only_its_grants():
    granted = lean_turn.granted_names(_agent(session_tool_groups=["playbooks"]))
    surface = lean_turn.lean_tool_surface(granted, is_super_admin=False)
    offered = lean_turn.offered_names(surface)
    assert offered and set(offered) <= set(granted)
    assert "platform_list_playbooks" in offered and "platform_create_agent" not in offered


def test_the_offered_actions_are_described_in_the_prompt():
    surface = lean_turn.lean_tool_surface(["platform_list_playbooks"], is_super_admin=False)
    messages = [{"role": "system", "content": PERSONA}, {"role": "user", "content": "hi"}]
    described = lean_turn.with_granted_actions(messages, surface)
    assert described[0]["content"].startswith(PERSONA)
    assert lean_turn.GRANTED_ACTIONS_HEADING in described[0]["content"]
    assert "platform_list_playbooks" in described[0]["content"]
    assert messages[0]["content"] == PERSONA  # a new list


def test_the_executor_refuses_what_a_lean_turn_was_not_granted():
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    executor = object.__new__(UnifiedToolExecutor)
    executor.composio_actions = {}
    call = lambda action: executor._widget_gate("platform_execute", {"action": action, "params": {}}, "t")  # noqa: E731
    assert call("platform_create_agent") is None  # not a lean turn: nothing to judge
    with lean_grants(["platform_list_playbooks"]):
        assert is_lean_turn() and lean_may_call("platform_list_playbooks")
        assert call("platform_list_playbooks") is None
        refused = call("platform_create_agent")
        assert refused["permission_denied"] is True and refused["error"] == LEAN_REFUSAL
    assert not is_lean_turn()


def test_a_turn_that_is_not_lean_has_no_grants(lean_on):
    assert lean_turn.lean_grant_for(None, 1) is None


# --- the context ---------------------------------------------------------------------------

def _build(mode, agent):
    from core.database.database import SessionLocal
    from modules.context import ContextService

    db = SessionLocal()
    try:
        return asyncio.run(ContextService(db).build_context(
            mode=mode, agent=agent, workspace_id=agent.workspace_id,
            messages=[{"role": "user", "content": "When is my next invoice due?"}],
            query="When is my next invoice due?", skip_memory=True,
        ))
    finally:
        db.close()


def test_a_lean_turns_prompt_is_the_agents_own_not_autos():
    from modules.context import ContextMode

    agent = _agent(runtime_profile="lean")
    lean, full = _build(ContextMode.LEAN, agent), _build(ContextMode.CHATBOT, agent)
    assert PERSONA in lean.system_prompt and PERSONA in full.system_prompt
    assert "orchestrator brain" in full.system_prompt and "orchestrator brain" not in lean.system_prompt
    assert lean.tools == [] and lean.messages  # the turn passes the granted tools; the conversation is there
    assert lean.token_estimate < full.token_estimate / 4


def test_a_lean_turns_chat_context_is_built_in_the_lean_mode():
    """``ContextService.build_context`` builds CHATBOT as LEAN while a lean turn's grants are set."""
    from modules.context import ContextMode
    from modules.context.lean_mode import builds_a_lean_turn_in_lean_mode

    @builds_a_lean_turn_in_lean_mode
    async def build(service, mode=None, **kwargs):
        return mode

    run = lambda *a, **k: asyncio.run(build(None, *a, **k))  # noqa: E731
    assert run(mode=ContextMode.CHATBOT) == ContextMode.CHATBOT
    with lean_grants([]):
        assert run(mode=ContextMode.CHATBOT) == ContextMode.LEAN
        assert run(ContextMode.CHATBOT) == ContextMode.LEAN
        assert run(mode=ContextMode.TASK_EXECUTION) == ContextMode.TASK_EXECUTION


def test_the_brand_kit_is_not_a_lean_turns():
    from consumers.chatbot.brand_turn import _owners_turn

    chat = SimpleNamespace(workspace_id="ws", widget_mode=False, _lean_actions=None)
    assert _owners_turn(chat)
    chat._lean_actions = ["platform_list_playbooks"]
    assert not _owners_turn(chat)


# --- time to first token -------------------------------------------------------------------

def test_the_turn_records_its_time_to_first_token():
    chat = SimpleNamespace()
    turn_timing.start(chat)

    async def _stream():
        await asyncio.sleep(0.02)
        yield "0:\"Hello\""
        yield {"_response": object()}

    async def _drain():
        return [item async for item in turn_timing.first_token(chat, _stream())]

    assert len(asyncio.run(_drain())) == 2
    assert chat._first_token_ms >= 20
    trace = turn_timing.with_first_token({"mode": "lean"}, chat)
    assert trace == {"mode": "lean", "first_token_ms": chat._first_token_ms}
    assert turn_timing.with_first_token(None, SimpleNamespace()) is None


# --- the turn's steps ----------------------------------------------------------------------

def _chat(lean: bool):
    return SimpleNamespace(_lean_actions=["platform_list_playbooks"] if lean else None, db=None)


def test_a_lean_turn_is_offered_its_grants_and_a_full_turn_its_own_tools():
    @lean_turn.offers_a_lean_turn_its_grants
    async def get_tools(chat, agent_id, **kwargs):
        return ["the full surface"]

    assert asyncio.run(get_tools(_chat(False), 1, is_super_admin=False)) == ["the full surface"]
    offered = lean_turn.offered_names(asyncio.run(get_tools(_chat(True), 1, is_super_admin=False)))
    assert offered == ["platform_list_playbooks"]


def test_a_lean_turns_call_carries_its_granted_tools_described():
    surface = lean_turn.lean_tool_surface(["platform_list_playbooks"], is_super_admin=False)

    @lean_turn.a_lean_turns_prompt
    async def prepare(chat, messages, agent_runtime, agent_ctx, all_tools, *rest, **kwargs):
        return [{"role": "system", "content": PERSONA}], None, "orchestrated"  # the LEAN mode loads no tools

    llm_messages, use_tools, orchestrated = asyncio.run(prepare(_chat(True), [], None, {}, surface, "chat"))
    assert use_tools == surface and orchestrated == "orchestrated"
    assert lean_turn.GRANTED_ACTIONS_HEADING in llm_messages[0]["content"]
    assert asyncio.run(prepare(_chat(False), [], None, {}, surface, "chat"))[1] is None  # a full turn as it was


def test_a_lean_turn_skips_pinned_documents_app_search_and_the_prefetch():
    ran = []

    @lean_turn.no_pinned_documents_on_a_lean_turn
    def pin(chat, llm_messages, chat_id):
        ran.append("pinned")

    @lean_turn.no_app_search_on_a_lean_turn
    async def apps(chat, llm_messages, use_tools, *rest):
        ran.append("apps")
        return use_tools + ["composio"], "result"

    @lean_turn.no_prefetch_on_a_lean_turn
    async def prefetch(chat, *args):
        ran.append("prefetch")
        yield "frame"

    async def frames(chat):
        return [f async for f in prefetch(chat, "q")]

    lean = _chat(True)
    pin(lean, [], "c")
    assert asyncio.run(apps(lean, [], ["dispatcher"], "q")) == (["dispatcher"], None)
    assert asyncio.run(frames(lean)) == [] and ran == []
    full = _chat(False)
    pin(full, [], "c")
    assert asyncio.run(apps(full, [], ["dispatcher"], "q")) == (["dispatcher", "composio"], "result")
    assert asyncio.run(frames(full)) == ["frame"] and ran == ["pinned", "apps", "prefetch"]


def test_the_saved_reply_carries_the_turns_first_token():
    saved = []
    chats = SimpleNamespace(save_message=lambda **kwargs: saved.append(kwargs))
    chat = SimpleNamespace(chat_service=chats)

    @turn_timing.records_the_first_token
    async def stream_llm_call(chat):
        await asyncio.sleep(0.01)
        yield "0:\"Hi\""
        yield {"_response": object()}

    @turn_timing.keeps_the_turns_timing
    async def turn(chat):
        async for frame in stream_llm_call(chat):
            yield frame
        chat.chat_service.save_message(role="user", context_trace=None)
        chat.chat_service.save_message(role="assistant", context_trace={"mode": "lean"})

    async def drain():
        return [f async for f in turn(chat)]

    assert len(asyncio.run(drain())) == 2
    assert saved[0] == {"role": "user", "context_trace": None}
    assert saved[1]["context_trace"]["mode"] == "lean" and saved[1]["context_trace"]["first_token_ms"] >= 10
    assert chat.chat_service is chats  # put back when the turn ends
