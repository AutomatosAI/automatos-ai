"""P256-FIX-N13-5 (F412, night 13): no phantom continuity in a fresh chat, and no honesty line
on a reply that claims nothing.

Six brand-new chats opened with "my last reply…" / "as I said" with no turn before them: the
recalled thread checkpoints ("Autumn launch — where we left off: …") reached the model as bare,
undated lines, and it read them as its own earlier turns. And "Just to be clear: I haven't done
that yet, and nothing has changed" sat on replies that claimed nothing (a greeting, a fact).

- What is recalled is labelled as an earlier conversation, with its day, on both memory lanes
  (modules/context/earlier_conversations.py), and the memory rule says none of it was said here.
- The opening of a reply in a chat with no earlier reply is checked by one rule
  (core/llm/fresh_reply.without_a_back_reference), on what streams and what is saved.
- The not-done line is the receipts' alone: a reply with no unbacked claim gets none, a claim
  gets it once, and a copy the model wrote itself is taken off its opening.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

from consumers.chatbot.claim_check import NOT_DONE, NOTHING_DONE_SAID
from consumers.chatbot.receipts import honesty_lines
from core.llm.clients.base import LLMResponse
from core.llm.fresh_reply import (
    EARLIER_CONVERSATION, opening_for, opens_the_conversation, without_a_back_reference, without_the_platforms_line,
)
from core.llm.turn_order import reprompts_in_the_users_turn
from modules.context.earlier_conversations import EARLIER, labelled, recalled_block
from modules.context.remembered_figures import EARLIER_NOT_THIS_CHAT, REMEMBERED_RULE
from modules.tools.execution.nudges import PLATFORM_CHECK
from tests.helpers_receipts_rule import line

NOTHING_DONE_LINE = NOT_DONE.format(said=NOTHING_DONE_SAID)   # the plain not-done line (a bare "Done.")

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
CHECKPOINT = {
    "memory": "Autumn launch — where we left off: the launch copy for #1403 was sent to Rosa for review.",
    "created_at": "2026-10-11T22:14:05+00:00",
    "metadata": {"type": "thread_summary", "source": "thread_checkpoint", "chat_id": "c-night13-old"},
}
SYSTEM = {"role": "system", "content": "You are Auto."}
FIRST_ASK = {"role": "user", "content": "Morning. Where are we with the Autumn launch?"}
PHANTOM = ["My last reply said ", "the launch copy went to Rosa for review. ", "Want me to chase her?"]
REWRITTEN = f"{EARLIER_CONVERSATION}, I said the launch copy went to Rosa for review. Want me to chase her?"
GREETING = "Evening Gerard. What do you need tonight?"


# ── what is recalled reaches the model as an earlier conversation ─────────────

def test_a_recalled_checkpoint_is_labelled_as_an_earlier_conversation_on_its_day():
    from consumers.chatbot.smart_memory import SmartMemoryManager, UserContext

    manager = SmartMemoryManager.__new__(SmartMemoryManager)
    block = manager._format_memories_for_llm([CHECKPOINT], UserContext(name="Gerard"))

    assert block.splitlines()[0] == "User's name: Gerard"
    assert f"  - {EARLIER} on 11 Oct: Autumn launch — where we left off:" in block
    assert block == recalled_block([CHECKPOINT], "Gerard")


def test_the_context_router_lane_labels_its_memories_too():
    from modules.context.sections.base import SectionContext
    from modules.context.sections.memory import MemorySection
    from modules.memory.context_router import ContextBundle

    transcript = {"content": "User: send the copy to Rosa", "created_at": "2026-10-10T09:00:00Z"}
    bundle = ContextBundle(long_term_memories=(CHECKPOINT,), temporal_results=(transcript,))
    ctx = SectionContext(agent=SimpleNamespace(id=322), workspace_id=WS, context_mode="chatbot", kwargs={})
    rendered = MemorySection()._format_context_bundle(bundle, ctx)

    assert f"- {EARLIER} on 11 Oct: Autumn launch — where we left off:" in rendered
    assert f"- {EARLIER} on 10 Oct: User: send the copy to Rosa" in rendered
    assert bundle.long_term_memories == (CHECKPOINT,)          # the bundle itself is never changed


def test_an_undated_memory_still_says_it_is_from_an_earlier_conversation():
    assert labelled([{"memory": "Prefers short emails."}, "a plain string"]) == (
        {"memory": f"{EARLIER}: Prefers short emails."}, "a plain string")


def test_the_memory_rule_says_none_of_it_was_said_in_this_chat():
    assert EARLIER_NOT_THIS_CHAT in REMEMBERED_RULE
    assert "'my last reply'" in EARLIER_NOT_THIS_CHAT and "'as I said'" in EARLIER_NOT_THIS_CHAT


# ── the first reply of a chat never refers back to a reply it never gave ─────

class _Streaming:
    """A provider behind LLMManager's wrapper: each delta goes out live, the response is the whole text."""

    def __init__(self, deltas):
        self.deltas = deltas

    async def generate_response(self, messages, tools=None, on_delta=None):
        for delta in self.deltas:
            if on_delta is not None:
                await on_delta("text", delta)
        return LLMResponse(content="".join(self.deltas), streamed=on_delta is not None, finish_reason="stop")


def _reply(deltas, messages, streams=True):
    """What reached the screen and what is saved, through the generate wrapper."""
    shown = []

    async def on_delta(kind, text):
        shown.append(text)

    generate = reprompts_in_the_users_turn(_Streaming.generate_response)
    kwargs = {"on_delta": on_delta} if streams else {}
    saved = asyncio.run(generate(_Streaming(deltas), messages, None, **kwargs))
    return "".join(shown), saved.content


def test_my_last_reply_in_a_fresh_chat_is_rewritten_on_the_screen_and_in_the_saved_answer():
    screen, saved = _reply(PHANTOM, [SYSTEM, FIRST_ASK])
    assert screen == saved == REWRITTEN


def test_as_i_said_loses_its_lead_in_after_a_greeting_too():
    screen, saved = _reply(["Morning Gerard! As I ", "said, the Taster plan ", "is £14 a month."], [SYSTEM, FIRST_ASK])
    assert screen == saved == "Morning Gerard! The Taster plan is £14 a month."


def test_a_chat_with_an_earlier_reply_keeps_its_words():
    earlier = [SYSTEM, FIRST_ASK, {"role": "assistant", "content": "The copy went to Rosa."},
               {"role": "user", "content": "And what did you say about the price?"}]
    screen, saved = _reply(["As I said, the Taster plan is £14."], earlier)
    assert screen == saved == "As I said, the Taster plan is £14."


def test_an_agent_run_or_a_distiller_that_does_not_stream_is_not_checked():
    _screen, saved = _reply(PHANTOM, [SYSTEM, FIRST_ASK], streams=False)
    assert saved == "".join(PHANTOM)
    assert opening_for([SYSTEM, FIRST_ASK], reprompt=False, streams=False) is None


def test_the_first_turn_stays_fresh_through_its_tool_rounds_and_a_platform_check():
    tool_round = [SYSTEM, FIRST_ASK, {"role": "assistant", "content": "", "tool_calls": [{"id": "1"}]},
                  {"role": "tool", "tool_call_id": "1", "content": "{}"}]
    reprompted = [*tool_round, {"role": "assistant", "content": "Done."},
                  {"role": "user", "content": f"{PLATFORM_CHECK}\n\nSay it again."}]
    assert opens_the_conversation([SYSTEM, FIRST_ASK]) and opens_the_conversation(tool_round)
    assert opens_the_conversation(reprompted)
    assert not opens_the_conversation([SYSTEM, FIRST_ASK, {"role": "assistant", "content": "Hi."},
                                       {"role": "user", "content": "Thanks."}])


def test_one_rule_for_the_lead_ins():
    assert without_a_back_reference("As I mentioned earlier, the copy is with Rosa.") == "The copy is with Rosa."
    assert without_a_back_reference("Picking up where we left off: Rosa has the copy.") == "Rosa has the copy."
    assert without_a_back_reference("In my previous message, I flagged the price.") == (
        f"{EARLIER_CONVERSATION}, I flagged the price.")
    assert without_a_back_reference(GREETING) == GREETING


# ── the not-done line: only for a claim the receipts do not back, and once ────
# P256-FIX-N15-2: said only in place of the claim it answers, never above the reply.

def test_a_greeting_or_a_fact_with_no_receipts_gets_no_honesty_line():
    assert line(GREETING) is None and honesty_lines([], GREETING) == []
    assert line("The Taster plan is £14 a month and ships on the first Monday.") is None


def test_i_have_sent_it_with_no_receipt_gets_the_line_once():
    assert honesty_lines([], "I've sent it.") == []
    assert line("I've sent it.") == "Just to be clear: I haven't sent anything in this reply. Ask me again if you want it done."


def test_a_copy_of_the_line_the_model_wrote_never_reaches_the_screen():
    earlier = [SYSTEM, FIRST_ASK, {"role": "assistant", "content": f"{NOTHING_DONE_LINE}\n\nShall I draft it?"},
               {"role": "user", "content": "Evening."}]
    copied = [NOTHING_DONE_LINE[:30], NOTHING_DONE_LINE[30:] + "\n\n", GREETING]
    screen, saved = _reply(copied, earlier)

    assert screen == saved == GREETING
    assert line(saved) is None                               # the greeting claims nothing: no line at all


def test_a_claim_after_a_copied_line_is_answered_by_the_receipts_line_alone():
    answer = without_the_platforms_line(f"{NOTHING_DONE_LINE}\n\nI've sent it.")
    assert answer == "I've sent it."
    assert line(answer) == "Just to be clear: I haven't sent anything in this reply. Ask me again if you want it done."   # once, in the claim's place


def test_a_reply_that_is_only_the_line_is_left_as_it_is():
    assert without_the_platforms_line(NOTHING_DONE_LINE) == NOTHING_DONE_LINE
