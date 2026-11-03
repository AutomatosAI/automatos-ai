"""PRD-268 US-C12b (§6.6, FR-12, K12): chips end the turn, with no further model call.

Auto writes its question and calls ``platform_offer_replies`` last. The round's own text is the
reply, so the loop ends on it with no model call, and the run's response carries the chips; it is the
model's own answer, so it is not marked a forced synthesis (the chat's answer checks read it). A
round with no text goes on as it always has, and the next answer carries the chips; a later round of
other work leaves them off. A turn that called ``platform_get_brief`` is never ended early and
carries no chips; a refused call changes nothing; a caller hook's own ``force_final`` wins, and its
own words carry no chips. A run without the tool is unchanged. The model below counts its calls, the
turn's first response included, as the chat makes it before the loop.
"""
from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
from types import SimpleNamespace

import pytest

from consumers.chatbot.claims_backed import NOT_DONE_PREFIX
from core.llm.clients.base import LLMResponse
from modules.tools.discovery import follow_up_context, handlers_brief, handlers_replies
from modules.tools.execution import reply_chips_turn
from modules.tools.execution.reply_chips_turn import REPLY_CHIPS, with_reply_chips
from modules.tools.execution.tool_loop import ToolLoopExecutor, ToolPostResult

QUESTION = "Shall I book the 15:10 slot or the 16:00 one?"
CHIPS = ["15:10", "16:00", "Neither"]
OFFER = "platform_offer_replies"
BRIEF = "platform_get_brief"
LIST_TASKS = "platform_list_tasks"
TOOLS = [{"type": "function", "function": {"name": name}}
         for name in ("platform_execute", OFFER, BRIEF, LIST_TASKS)]


class _Model:
    """A model that answers from a queue and counts its calls."""

    def __init__(self, *responses):
        self.queue, self.calls = list(responses), 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.queue.pop(0)


async def _answer(action, params):
    """Each action's own answer: the real chips handler, a brief, an empty board."""
    if action == OFFER:
        return await handlers_replies.offer_replies(None, "ws", params)
    if action == BRIEF:
        return {"success": True, "brief": {"sections": []}}
    return {"success": True, "tasks": []}


def _tools(envelope):
    """The tool callback: the chat's envelope (the answer as ``raw_result``), or the bare answer."""
    async def call(name, args, call_id, workspace_id):
        dispatched = name == "platform_execute"
        answer = await _answer(args.get("action") if dispatched else name,
                               (args.get("params") if dispatched else args) or {})
        if not envelope:
            return answer
        return {"success": answer.get("success", True), "llm_context": json.dumps(answer), "raw_result": answer}
    return call


def _direct(action, **params):
    return {"id": f"call_{action}", "type": "function",
            "function": {"name": action, "arguments": json.dumps(params)}}


def _dispatched(action, **params):
    return {"id": f"call_execute_{action}", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": params})}}


def _say(text, *calls):
    return LLMResponse(content=text, tool_calls=list(calls) or None)


def _turn(model, *, envelope=True, **hooks):
    """One chat turn: the model's first response, then the loop over it. Returns (result, messages)."""
    async def go():
        messages = [{"role": "user", "content": "Book me a slot for the delivery call."}]
        loop = ToolLoopExecutor(llm_callback=model, tool_callback=_tools(envelope), max_iterations=5,
                                promises=True)
        first = await model(messages, TOOLS)
        result = await loop.run(initial_response=first, messages=messages, tools=TOOLS, workspace_id="ws",
                                **hooks)
        return result, messages
    return asyncio.run(go())


def _chips_of(response):
    return getattr(response, REPLY_CHIPS, None)


# ── text and chips: the run ends on the round's own text, no further model call ──

@pytest.mark.parametrize("offer", [_direct(OFFER, replies=CHIPS), _dispatched(OFFER, replies=CHIPS)],
                         ids=["direct", "platform_execute"])
@pytest.mark.parametrize("envelope", [True, False], ids=["chat-envelope", "bare-answer"])
def test_text_and_chips_end_the_run_after_one_model_call(offer, envelope):
    model = _Model(_say(QUESTION, offer))
    result, messages = _turn(model, envelope=envelope)
    assert model.calls == 1 and result.iterations == 1
    assert result.response.content == QUESTION and not result.response.tool_calls
    # The model's own answer, not a forced synthesis: F187's and F205's checks in the chat read it.
    assert not result.forced_final
    assert _chips_of(result.response) == CHIPS
    # The round, its call and the tool's answer stay in the history, as any round's do.
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]


def test_the_chips_are_the_tools_replies_as_it_shows_them():
    model = _Model(_say(QUESTION, _direct(OFFER, replies=["  15:10 ", "16:00", "Neither  one"])))
    result, _ = _turn(model)
    assert model.calls == 1 and _chips_of(result.response) == ["15:10", "16:00", "Neither one"]


def test_chips_with_other_calls_in_the_same_round_still_end_it():
    model = _Model(_say(QUESTION, _direct(LIST_TASKS), _direct(OFFER, replies=CHIPS)))
    result, _ = _turn(model)
    assert model.calls == 1 and result.response.content == QUESTION and _chips_of(result.response) == CHIPS


# ── a round with no text is not ended early; the chips ride the answer it ends with ──

def test_a_round_with_no_text_calls_the_model_again_and_the_chips_ride_its_answer():
    answer = _say(QUESTION)
    model = _Model(_say("", _direct(OFFER, replies=CHIPS)), answer)
    result, _ = _turn(model)
    assert model.calls == 2
    assert not result.forced_final
    assert result.response.content == QUESTION and _chips_of(result.response) == CHIPS
    assert answer.reply_chips is None  # a copy carries them; the model's response is never changed


def test_a_round_of_whitespace_is_a_round_with_no_text():
    model = _Model(_say("  \n ", _direct(OFFER, replies=CHIPS)), _say(QUESTION))
    result, _ = _turn(model)
    assert model.calls == 2 and _chips_of(result.response) == CHIPS


def test_a_later_round_of_other_work_ends_on_an_answer_without_the_chips():
    model = _Model(_say("", _direct(OFFER, replies=CHIPS)), _say("Checking the board first.", _direct(LIST_TASKS)),
                   _say("Your board is empty."))
    result, _ = _turn(model)
    assert model.calls == 3 and not result.forced_final
    assert result.response.content == "Your board is empty." and _chips_of(result.response) is None


def test_a_later_round_that_offers_again_ends_on_its_own_chips():
    model = _Model(_say("", _direct(OFFER, replies=CHIPS)), _say(QUESTION, _dispatched(OFFER, replies=["Yes", "No"])))
    result, _ = _turn(model)
    assert model.calls == 2 and result.response.content == QUESTION and _chips_of(result.response) == ["Yes", "No"]


# ── never on the brief ──

@pytest.mark.parametrize("brief", [_direct(BRIEF), _dispatched(BRIEF), _dispatched(f" {BRIEF}"), _dispatched(f"{BRIEF}\n")],
                         ids=["direct", "platform_execute", "leading-space", "trailing-newline"])
def test_a_turn_that_called_the_brief_is_not_ended_early_and_carries_no_chips(brief):
    model = _Model(_say("Here is your brief.", brief, _direct(OFFER, replies=CHIPS)), _say("Your brief is above."))
    result, _ = _turn(model)
    assert model.calls == 2 and not result.forced_final
    assert result.response.content == "Your brief is above." and _chips_of(result.response) is None


def test_a_brief_from_an_earlier_round_still_keeps_the_chips_off():
    model = _Model(_say("Here is your brief.", _direct(BRIEF)), _say(QUESTION, _dispatched(OFFER, replies=CHIPS)),
                   _say("Your brief is above."))
    result, _ = _turn(model)
    assert model.calls == 3 and not result.forced_final and _chips_of(result.response) is None


# ── a refused call changes nothing ──

@pytest.mark.parametrize("replies", [["Yes"], ["Yes", "No", "Maybe", "Later"], ["Yes", "yes"],
                                     ["Book the 15:10 slot please", "No"]],
                         ids=["one", "four", "duplicate", "too-long"])
def test_a_refused_offer_changes_nothing(replies):
    model = _Model(_say(QUESTION, _direct(OFFER, replies=replies)), _say(QUESTION))
    result, messages = _turn(model)
    assert model.calls == 2 and not result.forced_final
    assert _chips_of(result.response) is None
    refused = json.loads(messages[2]["content"])
    assert refused["success"] is False and refused["error_code"]


def test_a_refused_offer_in_a_later_round_offers_nothing_new():
    model = _Model(_say("", _direct(OFFER, replies=CHIPS)), _say(QUESTION, _dispatched(OFFER, replies=["Yes"])),
                   _say(QUESTION))
    result, _ = _turn(model)
    assert model.calls == 3 and not result.forced_final and _chips_of(result.response) is None


# ── a caller hook's own force_final wins ──

def test_a_callers_round_end_force_final_wins():
    rounds = []

    async def stop(state):
        rounds.append(state.iteration)
        return ToolPostResult(force_final=True, final_content="The caller's own answer.")

    model = _Model(_say(QUESTION, _direct(OFFER, replies=CHIPS)))
    result, _ = _turn(model, on_round_end=stop)
    assert rounds == [1] and model.calls == 1 and result.forced_final
    assert result.response.content == "The caller's own answer."
    assert _chips_of(result.response) is None  # the caller's own words (a fixed line) carry no chips


def test_a_callers_round_end_synthesis_wins_and_calls_the_model():
    async def synthesize(_state):
        return ToolPostResult(force_final=True)

    model = _Model(_say(QUESTION, _direct(OFFER, replies=CHIPS)), _say("Which slot suits you?"))
    result, _ = _turn(model, on_round_end=synthesize)
    assert model.calls == 2 and result.forced_final and result.response.content == "Which slot suits you?"
    assert _chips_of(result.response) == CHIPS  # the model's answer to the round: the chips ride it


def test_a_callers_tool_result_force_final_wins():
    async def stop(name, args, result):
        return ToolPostResult(force_final=True, final_content="The caller's answer.")

    model = _Model(_say(QUESTION, _direct(OFFER, replies=CHIPS)))
    result, _ = _turn(model, on_tool_result=stop)
    assert model.calls == 1 and result.forced_final and result.response.content == "The caller's answer."
    assert _chips_of(result.response) is None


def test_the_claim_check_still_reads_the_rounds_text_and_the_chips_survive_it():
    model = _Model(_say("I've sent it to Declan.", _direct(OFFER, replies=CHIPS)))
    result, _ = _turn(model)
    assert model.calls == 1
    assert result.response.content.startswith(NOT_DONE_PREFIX)
    assert _chips_of(result.response) == CHIPS


def test_the_callers_hooks_see_every_call_and_round_as_before():
    seen, rounds = [], []

    async def on_tool(name, args, result):
        seen.append((name, args, result["success"]))
        return None

    async def on_round(state):
        rounds.append(state.iteration)
        return None

    model = _Model(_say(QUESTION, _direct(OFFER, replies=CHIPS)))
    _turn(model, on_tool_result=on_tool, on_round_end=on_round)
    assert seen == [(OFFER, {"replies": CHIPS}, True)] and rounds == [1]


# ── a run without the tool is unchanged ──

def test_a_run_without_the_tool_is_unchanged():
    seen, rounds = [], []

    async def on_tool(name, args, result):
        seen.append(name)
        return ToolPostResult(followup_messages=[{"role": "system", "content": "A hint from the caller."}])

    async def on_round(state):
        rounds.append(state.iteration)
        return None

    answer = _say("Your board is empty.")
    model = _Model(_say("Checking the board.", _direct(LIST_TASKS)), answer)
    result, messages = _turn(model, on_tool_result=on_tool, on_round_end=on_round)
    assert model.calls == 2 and not result.forced_final and result.iterations == 1
    assert result.response is answer and answer.reply_chips is None
    assert seen == [LIST_TASKS] and rounds == [1]
    assert messages[-1] == {"role": "system", "content": "A hint from the caller."}


def test_an_agent_run_with_no_hooks_without_the_tool_is_unchanged():
    answer = _say("Your board is empty.")
    model = _Model(_say("Checking the board.", _direct(LIST_TASKS)), answer)
    result, _ = _turn(model, envelope=False)
    assert model.calls == 2 and result.response is answer and not result.forced_final


def test_a_first_reply_with_no_call_is_unchanged():
    answer = _say("Your board is empty.")
    model = _Model(answer)
    result, _ = _turn(model)
    assert model.calls == 1 and result.response is answer and result.iterations == 0


# ── the seam ──

def test_the_terminal_turn_is_a_decorator_on_run():
    run, chain = ToolLoopExecutor.run, []
    while run is not None:
        chain.append(run.__code__.co_qualname)
        run = getattr(run, "__wrapped__", None)
    assert chain[0] == "its_answer_is_checked.<locals>.checked" and chain[-1] == "ToolLoopExecutor.run"
    assert "ends_on_reply_chips.<locals>.wrapped" in chain


def test_the_names_are_the_handlers():
    assert reply_chips_turn.OFFER_REPLIES == follow_up_context.OFFER_REPLIES == handlers_replies.OFFER_REPLIES
    assert reply_chips_turn.REPLIES == handlers_replies.REPLIES_PARAM
    assert reply_chips_turn.GET_BRIEF == handlers_brief.GET_BRIEF
    assert REPLY_CHIPS in {f.name for f in dataclasses.fields(LLMResponse)}


def test_a_copy_of_the_response_keeps_its_chips():
    for response in (LLMResponse(content=QUESTION), SimpleNamespace(content=QUESTION, tool_calls=None)):
        carried = with_reply_chips(response, CHIPS)
        assert carried is not response and _chips_of(response) is None and _chips_of(carried) == CHIPS
        assert _chips_of(copy.copy(carried)) == CHIPS
    assert dataclasses.replace(with_reply_chips(LLMResponse(content=QUESTION), CHIPS),
                               content="Corrected.").reply_chips == CHIPS


def test_a_response_that_cannot_carry_chips_is_sent_as_it_was():
    frozen = dataclasses.make_dataclass("Frozen", [("content", str)], frozen=True)(content=QUESTION)
    assert with_reply_chips(frozen, CHIPS) is frozen
