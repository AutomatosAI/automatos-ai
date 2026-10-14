"""P256-FIX-N15-2 (F423, night 15): the not-done line is said only in place of a claim, never above a reply.

"Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done."
sat above 148 of night 15's 458 replies, most with no write call at all, pure questions among them
("why does a natural Ethiopian taste different", "what's on my board"). The receipts' second path
(``receipts.honesty_lines``) read every past participle as a claim and put the line above the text.
Now the line has one producer, N15-1's: an unbacked claim's sentence is taken out and the line is put
in its place. Above the text go only the refused writes, and none when the reply's line names them.

The shapes below are the night's, each from the real sentence, with the reads its turn made; after
N15-1's detector fixes, every shape still drawing the line was a verb in no family ("cut off",
"I've used", "It's been run 18 times"): read, never taken out, so it now draws no line.
"""
from __future__ import annotations

import asyncio
import inspect
import json

import pytest

from consumers.chatbot import claim_check, claims_backed, receipts
from consumers.chatbot.claim_removed import without_unbacked_claims
from consumers.chatbot.claims_backed import NOT_DONE_PREFIX
from consumers.chatbot.receipts import build_receipts, honesty_lines, says_nothing_was_done
from core.llm.clients.base import LLMResponse
from modules.tools.execution.claims_in_place import its_answer_is_checked
from modules.tools.execution.tool_loop import ToolLoopExecutor, ToolPostResult
from tests.helpers_receipts_rule import call, line, tracker_of

OK = {"success": True}
TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]
SENT = "I've sent it to Declan."
SENT_LINE = "Just to be clear: I haven't sent anything in this reply. Ask me again if you want it done."
REFUSED = {"success": False, "error": "Missing required params: status"}


def _search():
    return ("search_knowledge", {"query": "night 15"}, OK)


AUTOMATIC = ("automatic_reads", {}, OK)
AGENTS, TASKS, FEED = call("platform_list_agents"), call("platform_list_tasks"), call("platform_get_activity_feed")
SCHEDULES = [call("platform_get_schedule"), FEED, call("platform_list_scheduled_tasks"), _search(),
             call("platform_list_playbooks"), call("platform_get_playbook")]
DOCUMENTS = [call("platform_list_deliverables"), call("platform_get_deliverable"), call("platform_search_documents")]

# (the sentence as night 15 delivered it, the reads its turn made); none of these turns wrote anything.
NIGHT_15 = [
    ("I count roughly 40 active ones from the roster, and the list was cut off after about 45 entries, so there "
     "may be a few more.", [AUTOMATIC, AGENTS, AGENTS]),
    ("Our setup sends customer newsletters to the team rather than me drafting them in chat, and you asked for no "
     "ticket, so I've stopped there.", [call("platform_recommend_agent")]),
    ("I've taken Thursday as 15 October, since today is Tuesday 13 October.", []),
    ("The roster was truncated, and my search of chat history for agent creation found nothing, which doesn't "
     "prove none was made.", [AUTOMATIC, AGENTS, FEED, call("platform_search_chat_history")]),
    ("The wholesale terms were written after that ruling, so I've used them.", [AUTOMATIC, _search()]),
    ("It's been run 18 times so far.", [call("platform_list_playbooks"), call("platform_get_schedule"),
                                        call("platform_get_playbook")]),
    ("I've used the database prices.", [call("query_database"), _search()]),
    ("- Banned-word lists: I've merged them above.", [AUTOMATIC]),
    ("I can't find the reorder draft you mean, so I don't want to send Hollis a message I've guessed at.",
     [_search(), call("platform_search_documents"), call("platform_list_deliverables")]),
    ("I described the letter's contact line and its payment terms, but I never read the file, so those "
     "descriptions were invented.", DOCUMENTS),
    ("My search for the letter found nothing, and the results it did return were unrelated.", DOCUMENTS),
    ("Where they differ, I've treated the brand kit as the final word, so the safe approach is to avoid the whole "
     "combined list.", [AUTOMATIC]),
    ("That's not a request from you, so I've ignored it.", []),
    ("If you want, I can set up a reminder or a task so Callum's Wednesday and the Parcelwise pickup are sorted "
     "before it changes.", [AUTOMATIC]),
    ("Tonight I made five agents, all tagged for the night of 12 October:", [AUTOMATIC, AGENTS, FEED]),
    ("It was set.", [AUTOMATIC, *SCHEDULES]),
    ("So it either fired and was cleared, or it never got stored properly.", [AUTOMATIC, *SCHEDULES]),
    ("- Guji lot: I've assumed the new lot is the Guji Shakiso Natural and priced it at the £13 single-origin "
     "250 g bag from the 19 September retail board.", [_search()]),
    ("40 in the 13 October price note is the landed cost of the green coffee, so I've kept it out of a customer "
     "newsletter.", [_search()]),
    ("In our own list, the Guji Shakiso natural was cupped as \"blueberry and jasmine\", and the Yirgacheffe Konga "
     "washed as \"bergamot lemon\".", [AUTOMATIC]),
    ("What I haven't done: I couldn't see the full high-priority list in one pull, so there may be high-priority "
     "cards I've missed.", [AUTOMATIC, TASKS, TASKS]),
    ("The board shows 66 cards in review and I've named owners for about 30 of them.",
     [AUTOMATIC, TASKS, call("platform_board_summary")]),
]


@pytest.mark.parametrize("said, reads", NIGHT_15, ids=[said[:48] for said, _ in NIGHT_15])
def test_a_read_only_answer_carries_no_line(said, reads):
    """Night 15: each of these drew the line above a read-only answer; none draws a line now."""
    assert honesty_lines(build_receipts(tracker_of(reads)), said) == []        # nothing above
    assert without_unbacked_claims(said, tracker_of(reads).outcomes) is None   # and nothing taken out
    assert line(said, *reads) is None


def test_a_pure_question_answered_from_the_documents_carries_no_line():
    """'Why does a natural process Ethiopian taste different from a washed one?', answered from the reads."""
    answer = ("A natural is dried with the fruit still on the bean, so it picks up sweet, winey, berry notes; a "
              "washed coffee has the fruit removed before drying, so it tastes cleaner and brighter. In our own "
              "list, the Guji Shakiso natural was cupped as \"blueberry and jasmine\", and the Yirgacheffe Konga "
              "washed as \"bergamot lemon\".")
    assert honesty_lines(build_receipts(tracker_of([AUTOMATIC])), answer) == []
    assert without_unbacked_claims(answer, tracker_of([AUTOMATIC]).outcomes) is None


def test_a_claim_no_call_did_is_replaced_by_the_line_and_nothing_goes_above():
    answer = f"Here's the reorder for Hollis: 60 kg of the Kibirizi. {SENT}"
    removed = without_unbacked_claims(answer, tracker_of([_search()]).outcomes)
    assert removed.text == f"Here's the reorder for Hollis: 60 kg of the Kibirizi. {SENT_LINE}"
    assert honesty_lines(build_receipts(tracker_of([_search()])), answer) == []          # never above
    assert honesty_lines(build_receipts(tracker_of([_search()])), removed.text) == []


def test_a_refused_write_goes_above_only_when_the_reply_does_not_name_it():
    refused = call("platform_update_task_status", {"task_id": "#0422"}, REFUSED)
    (tried,) = honesty_lines(build_receipts(tracker_of([refused])), "Here is where #0422 stands.")
    assert tried == "I tried to move the card #0422 and it didn't go through: Missing required params: status."
    in_place = without_unbacked_claims("I've approved #0422.", tracker_of([refused]).outcomes).text
    assert "I tried to move the card #0422 and it didn't go through" in in_place
    assert honesty_lines(build_receipts(tracker_of([refused])), in_place) == []          # said once, in place


def test_another_just_to_be_clear_line_keeps_the_refusal_above():
    """Only the not-done line names the refusals: an id line that opens the same way does not."""
    refused = call("platform_update_task_status", {"task_id": "#0422"}, REFUSED)
    answer = "Here is where it stands.\n\nJust to be clear: task 1100 does not exist — I named it without looking it up."
    assert not says_nothing_was_done(answer)
    assert honesty_lines(build_receipts(tracker_of([refused])), answer)[0].startswith("I tried to move the card #0422")


def test_the_line_is_what_says_nothing_was_done():
    """FX-005: the turn's notice defers to the line, which is now in the reply itself."""
    in_place = without_unbacked_claims(SENT, []).text
    assert in_place == SENT_LINE
    assert says_nothing_was_done(f"Here are your cards. {in_place}") and not says_nothing_was_done(SENT)
    assert not says_nothing_was_done("I count roughly 40 active ones from the roster.")


def test_taking_the_claims_out_twice_changes_nothing():
    refused = call("platform_update_task_status", {"task_id": "#0422"}, REFUSED)
    ask = call("platform_update_agent", {"agent_name": "Scout"},
               {"success": False, "requires_confirmation": True, "grant_id": 7, "act": "change an agent 'Scout'"})
    outcomes = tracker_of([refused, ask]).outcomes
    once = without_unbacked_claims("I've approved #0422 and changed Scout's model. Done.", outcomes).text
    assert once.count(NOT_DONE_PREFIX) == 1
    assert without_unbacked_claims(once, outcomes) is None


# ── the loop: a run that ends any way holds no unbacked claim ───────────────

class _Model:
    def __init__(self, *responses):
        self.queue, self.calls = list(responses), 0

    async def __call__(self, messages, tools):
        self.calls += 1
        return self.queue.pop(0)


async def _tools(name, args, call_id, workspace_id):
    return {"success": True, "tasks": [{"number": "#1571"}, {"number": "#1572"}]}


def _call(action):
    return {"id": f"call_{action}", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": {}})}}


def _run(model, **hooks):
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_tools, max_iterations=5, promises=True)
    first = LLMResponse(content="", tool_calls=[_call("platform_list_tasks")])
    return asyncio.run(executor.run(initial_response=first, messages=[{"role": "user", "content": "Send it."}],
                                    tools=TOOLS, workspace_id="ws", **hooks))


def test_a_forced_synthesis_has_its_claim_taken_out():
    """The loop's own check never sees a forced synthesis (the chat's dedup or fatal-error stop): the
    receipts' line above answered it; now its claim is taken out in place."""
    async def stop(_state):
        return ToolPostResult(force_final=True)

    model = _Model(LLMResponse(content=f"#1571 and #1572 are in Review. {SENT}", tool_calls=None))
    result = _run(model, on_round_end=stop)
    assert result.forced_final and model.calls == 1
    assert result.response.content == f"#1571 and #1572 are in Review. {SENT_LINE}"


def test_an_explicit_final_answer_has_its_claim_taken_out():
    async def stop(_state):
        return ToolPostResult(force_final=True, final_content=SENT)

    result = _run(_Model(), on_round_end=stop)
    assert result.forced_final and result.response.content == SENT_LINE


def test_a_backed_claim_is_left_as_it_is():
    async def stop(_state):
        return ToolPostResult(force_final=True, final_content="#1571 and #1572 are in Review.")

    assert _run(_Model(), on_round_end=stop).response.content == "#1571 and #1572 are in Review."


def test_the_run_is_checked_however_it_ends():
    run = ToolLoopExecutor.run
    chain = []
    while run is not None:
        chain.append(run.__code__.co_qualname)
        run = getattr(run, "__wrapped__", None)
    assert chain[0] == "its_answer_is_checked.<locals>.checked" and chain[-1] == "ToolLoopExecutor.run"
    assert inspect.getsource(its_answer_is_checked).count("in_place(") == 1


# ── one producer: nothing above the reply says it ───────────────────────────

def test_the_above_the_reply_line_is_gone():
    assert not hasattr(receipts, "NOTHING_DONE_LINE") and "NOTHING_DONE_LINE" not in receipts.__all__
    assert not hasattr(claims_backed, "not_done_line") and not hasattr(claim_check, "NOTHING_DONE")
    assert "not_done_line" not in inspect.getsource(receipts) and "NOTHING_DONE" not in inspect.getsource(receipts)


@pytest.mark.parametrize("answer", [SENT, "Done.", "The card has been approved.", "I've cancelled Mission #0365.",
                                    "Each step will pause for your approval.", "I've rejected the card."])
def test_honesty_lines_never_says_the_not_done_line(answer):
    for calls in ([], [_search()], [call("platform_store_memory")]):
        assert not any(said.startswith(NOT_DONE_PREFIX) for said in honesty_lines(build_receipts(tracker_of(calls)),
                                                                                  answer))
