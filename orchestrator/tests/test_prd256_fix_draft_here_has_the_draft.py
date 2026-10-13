"""PRD-256 P256-FIX-N14-5 (F419, night 14): a reply to a here-ask holds the draft.

Night 14 (Auto on Haiku 5.5): the owner asked for a draft here (T5 routes the ask to Auto's
answer, with ``asked_here.HERE_DIRECTIVE``) and the whole reply was "The draft is above, in chat
only.", twice. Now a reply to a here-ask that points to a draft ("above", "below", "in chat") with
no body besides that sentence is retried once with the directive, through the FX-017 retry seam
(modules/tools/execution/draft_here, on the loop's final-reply check), and never sent as a bare
pointer: a retry that still gives no draft becomes a plain line saying so. A first reply that ran
no tool is routed through the loop for the check. A real draft passes untouched.
"""
from __future__ import annotations

import asyncio
import inspect

import pytest

from core.llm.clients.base import LLMResponse
from core.llm.usage_context import LANE_CHAT, usage_scope
from modules.tools.execution import draft_here
from modules.tools.execution.nudges import is_nudge
from modules.tools.execution.tool_loop import ToolLoopExecutor
from tests.test_prd256_fix_one_answer_on_screen import TOOLS, _Provider, _retractions, _screen_after, _service, _turn

ASK = ("Draft the intro for this fortnight's Harbour Log newsletter, about the October box. "
       "Just the draft here, please.")
NOT_HERE = "What did we say about the October box last week?"
POINTER = "The draft is above, in chat only."
DRAFT = (   # no claim of work done in it, so only the pointer rule can nudge it
    "October's box brings a washed Colombian from Huila that tastes of red apple and panela, with a long "
    "cocoa finish. Brew it a little coarser than usual and let it bloom for forty seconds: it opens up as "
    "it cools, and the sweetness lasts well into the afternoon."
)
DRAFTED = f"Here's the intro:\n\n{DRAFT}"


@pytest.fixture(autouse=True)
def _chat_budgets(monkeypatch):
    """The CHATBOT_* config properties read system_settings (DB): pinned, as the loop tests pin them."""
    from config import config as _cfg

    monkeypatch.setattr(type(_cfg), "CHATBOT_MAX_TOOL_ITERATIONS", 5)
    monkeypatch.setattr(type(_cfg), "CHATBOT_ACTION_RETRY_BUDGET", 2)
    monkeypatch.setattr(type(_cfg), "CHATBOT_PARAM_RETRY_BUDGET", 2)


class _Model:
    def __init__(self, *responses):
        self.queue = list(responses)
        self.calls = 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.queue.pop(0)


def _run(first, *replies, said=ASK, promises=True):
    """The loop from a first reply that made no call, over the owner's words ``said``."""
    model = _Model(*replies)

    async def tools(name, args, call_id, workspace_id):
        raise AssertionError("no tool runs in these turns")

    executor = ToolLoopExecutor(llm_callback=model, tool_callback=tools, max_iterations=5, promises=promises)
    messages = [{"role": "user", "content": said}]
    with usage_scope(request_type=LANE_CHAT if promises else "agent"):
        result = asyncio.run(executor.run(initial_response=LLMResponse(content=first, tool_calls=None),
                                          messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, model, [m["content"] for m in messages if is_nudge(m)]


# ── what a bare pointer is ─────────────────────────────────────────────────────────

def test_the_night_14_reply_is_a_bare_pointer():
    assert draft_here.bare_pointer(POINTER) == POINTER
    assert draft_here.bare_pointer("I've written the draft above. Let me know if you'd like changes.") \
        == "I've written the draft above."
    assert draft_here.bare_pointer("You'll find it in the chat.") == "You'll find it in the chat."


def test_a_reply_that_holds_a_draft_or_points_nowhere_is_no_pointer():
    assert draft_here.bare_pointer(DRAFTED) is None
    assert draft_here.bare_pointer(f"Here's the draft below:\n\n{DRAFT}") is None
    assert draft_here.bare_pointer(f"Here it is, in the chat as you asked:\n\n{DRAFT}") is None
    assert draft_here.bare_pointer("Sales are above target this week.") is None     # "above" is no place
    assert draft_here.bare_pointer("") is None


def test_the_retry_carries_the_here_directive():
    from consumers.chatbot.asked_here import HERE_DIRECTIVE

    said = draft_here.draft_nudge(POINTER)
    assert said.startswith(f"Your previous reply points to a draft ({POINTER}), but there is no draft there")
    assert HERE_DIRECTIVE.strip().splitlines()[-1] in said and "## Manager directive" not in said


# ── the loop: one retry, then the draft ─────────────────────────────────────────────

def test_a_bare_pointer_is_retried_once_and_the_retry_holds_the_draft():
    result, model, nudges = _run(POINTER, LLMResponse(content=DRAFTED, tool_calls=None))
    assert model.calls == 1 and len(nudges) == 1
    assert "points to a draft (The draft is above, in chat only.)" in nudges[0]
    assert "Write the work itself" in nudges[0] and "no platform_create_task" in nudges[0]
    assert result.response.content == DRAFTED


def test_a_real_draft_passes_untouched():
    result, model, nudges = _run(DRAFTED)
    assert model.calls == 0 and nudges == []
    assert result.response.content == DRAFTED


def test_a_retry_that_points_again_is_never_sent_as_a_pointer():
    result, model, nudges = _run(POINTER, LLMResponse(content="It's above, in the chat.", tool_calls=None))
    assert model.calls == 1 and len(nudges) == 1                          # asked once only
    assert result.response.content == draft_here.NO_DRAFT_SAID
    assert result.response.streamed is False and not result.response.tool_calls


def test_a_blank_retry_is_never_sent_as_a_pointer():
    result, model, _nudges = _run(POINTER, LLMResponse(content="", tool_calls=None))
    assert model.calls == 1
    assert result.response.content == draft_here.NO_DRAFT_SAID


def test_an_ask_that_is_not_for_it_here_is_not_nudged():
    result, model, nudges = _run(POINTER, said=NOT_HERE)
    assert model.calls == 0 and nudges == [] and result.response.content == POINTER


def test_an_agents_run_is_not_nudged():
    result, model, nudges = _run(POINTER, promises=False)
    assert model.calls == 0 and nudges == [] and result.response.content == POINTER


# ── the chat: the pointer comes off the screen, the draft stays ─────────────────────

def test_the_screen_shows_the_draft_and_not_the_pointer():
    provider = _Provider(([POINTER], None), ([DRAFTED], None))
    frames, final = _turn(provider, chat_lane=True, owner=ASK)

    assert _retractions(frames) == [POINTER]
    assert final.content == DRAFTED
    assert _screen_after(frames) == DRAFTED


def _first_reply(reply, *, owner=ASK, tools=TOOLS):
    svc = _service(True)
    return asyncio.run(svc._first_reply_goes_through_the_loop(LLMResponse(content=reply, tool_calls=None),
                                                              tools, [], owner))


def test_a_first_reply_that_only_points_goes_through_the_loop():
    assert _first_reply(POINTER) is True
    assert _first_reply(DRAFTED) is False
    assert _first_reply(POINTER, owner=NOT_HERE) is False
    assert _first_reply(POINTER, tools=None) is False


def test_both_checks_are_wired():
    from consumers.chatbot.service import StreamingChatService

    assert "@shows_the_work" in inspect.getsource(ToolLoopExecutor._recover_claimed_action)
    assert "@a_bare_pointer_goes_through_the_loop" in inspect.getsource(
        StreamingChatService._first_reply_goes_through_the_loop)
