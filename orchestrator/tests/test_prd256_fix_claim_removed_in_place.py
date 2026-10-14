"""P256-FIX-N15-1 (F422, night 15): an unbacked claim is taken out of the reply, never nudged.

About one board or roster reply in four on night 15 came back as commentary on a reply the owner
never saw ("Nothing has been sent. My last reply said…"): the loop's claim check nudged the reply
and returned the retry in its place, and the retry dropped what the reads had fetched. Now the reply
stands, the sentence that claims is replaced by one not-done line, and the model is called once.

The detector itself fired on read-only reports: run over every sentence of the night's 458 replies,
it fired 140 times and next to none was a claim. Each shape is pinned below from the real sentence.
"""
from __future__ import annotations

import asyncio
import json
import logging

import pytest

from consumers.chatbot.claim_removed import SENTENCE_LOGGED_CHARS, in_place_of, without_unbacked_claims
from consumers.chatbot.claims_backed import claims, unbacked_where
from consumers.chatbot.on_screen import Screen
from core.llm.clients.base import LLMResponse
from core.llm.screen_watch import WATCHER
from modules.tools.execution import nudges
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from modules.tools.execution.tool_loop import ToolLoopExecutor

TOOLS = [{"type": "function", "function": {"name": "platform_execute"}}]
CARDS = [f"- #{1570 + n} Wholesale note {n} — CELLAR" for n in range(12)]
SENT = "I've sent it to Declan."
LINE = "Just to be clear: I haven't sent anything in this reply. Ask me again if you want it done."
DRAFT = "Here are the 12 cards in Review:\n\n" + "\n".join(CARDS) + f"\n\n{SENT} Anything else?"


class _Model:
    def __init__(self, *responses):
        self.queue, self.calls = list(responses), []

    async def __call__(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        return self.queue.pop(0)


async def _tools(name, args, call_id, workspace_id):
    return {"success": True, "tasks": [{"number": card.split()[1]} for card in CARDS]}


def _call(action):
    return {"id": f"call_{action}", "type": "function",
            "function": {"name": "platform_execute", "arguments": json.dumps({"action": action, "params": {}})}}


def _run(model):
    executor = ToolLoopExecutor(llm_callback=model, tool_callback=_tools, max_iterations=5, promises=True)
    messages = [{"role": "user", "content": "Which cards are sitting in review right now, and who has each one?"}]
    first = LLMResponse(content="", tool_calls=[_call("platform_list_tasks")])
    result = asyncio.run(executor.run(initial_response=first, messages=messages, tools=TOOLS, workspace_id="ws"))
    return result, messages


# ── the loop: the reply stands, the claim's sentence is replaced, one model call ──

def test_the_board_list_is_delivered_without_the_claim_and_the_model_is_called_once():
    model = _Model(LLMResponse(content=DRAFT, tool_calls=None))
    result, messages = _run(model)
    answer = result.response.content
    assert len(model.calls) == 1                                          # no retry
    assert all(card in answer for card in CARDS)                          # the 12 cards stay
    assert SENT not in answer
    assert answer.endswith(f"{LINE} Anything else?")                      # the line in the sentence's place
    assert not any(nudges.is_nudge(m) for m in messages)                  # nothing in the owner's turn


def test_the_warning_names_the_family_and_the_sentence(caplog):
    model = _Model(LLMResponse(content=DRAFT, tool_calls=None))
    with caplog.at_level(logging.WARNING, logger="modules.tools.execution.claims_in_place"):   # N15-2: one check
        _run(model)
    said = [r.getMessage() for r in caplog.records if "unbacked claim" in r.getMessage()]
    assert said == [f"[tool-loop] unbacked claim family=sent sentence={SENT!r} — replaced by the not-done line"]


def test_the_logged_sentence_is_cut_to_200_chars():
    long_claim = "I've sent it to " + "Declan and the team " * 20 + "."
    removed = without_unbacked_claims(long_claim, [], promises=True)
    assert removed.family == "sent" and len(removed.sentence) == SENTENCE_LOGGED_CHARS == 200


def _outcomes(*calls):
    """The loop tracker's outcomes for ``calls`` (the dispatcher's inner action, as the loop records it)."""
    tracker = ToolExecutionTracker()
    for tool, args, result in calls:
        tracker.record_outcome(tool, args, result)
    return tracker.outcomes


def test_a_backed_claim_is_left_alone():
    sent = _outcomes(("GMAIL_SEND_EMAIL", {"to": "declan@example.com"}, {"successful": True}))
    assert without_unbacked_claims(SENT, sent, promises=True) is None


def test_the_retry_messages_and_notes_are_gone():
    for name in ("CLAIMED_ACTION_RECOVERY_MSG", "REFUSED_WRITE_NOTE", "WAITING_WRITE_NOTE", "claimed_action_nudge",
                 "refused_writes", "waiting_writes"):
        assert not hasattr(nudges, name), name


# ── the one line: the claim named, the ask and the refusal in it ────────────

ASK = ("platform_execute", {"action": "platform_update_agent", "params": {"agent_name": "Scout"}},
       {"success": False, "requires_confirmation": True, "grant_id": 1234, "act": "change an agent 'Scout'"})
REFUSED = ("platform_execute", {"action": "platform_update_task_status", "params": {"task_id": "#0422"}},
           {"success": False, "error": "Missing required params: status"})


def test_a_waiting_card_is_said_in_the_line():
    removed = without_unbacked_claims("I've changed Scout's model to Haiku.", _outcomes(ASK), promises=True)
    assert removed.text == ("Just to be clear: I haven't changed anything in this reply; I asked for your click to "
                            "change an agent 'Scout', so it hasn't happened yet. Ask me again if you want it done.")


def test_a_refused_write_is_said_in_the_line():
    removed = without_unbacked_claims("I've approved #0422.", _outcomes(REFUSED), promises=True)
    assert removed.text == ("Just to be clear: I haven't approved anything in this reply; I tried to move the card "
                            "#0422 and it didn't go through. Ask me again if you want it done.")


def test_a_bare_done_takes_the_plain_words():
    removed = without_unbacked_claims("Done.", [], promises=True)
    assert removed.text == ("Just to be clear: I haven't done that yet, and nothing has changed. "
                            "Ask me again if you want it done.")


def test_every_claiming_sentence_goes_and_the_line_is_said_once():
    removed = without_unbacked_claims("**I've sent it.** Here's the list.\n\n- Posted the update.", [], promises=True)
    assert removed.text == ("Just to be clear: I haven't sent or posted anything in this reply. Ask me again if "
                            "you want it done. Here's the list.\n")     # the bold's closing marks went with it


def test_a_quoted_draft_is_never_touched():
    removed = without_unbacked_claims("Done.\n\n> I've sent your order, Sam.", [], promises=True)
    assert removed.text.endswith("\n\n> I've sent your order, Sam.")


# ── N15-CI: what goes on from a claim goes with it; a line never denies a done write ──

def test_a_claim_that_goes_on_from_one_taken_out_goes_with_it():
    """An agent's run is held to its first-person claims: "It's now running." stayed after the line
    that denied the approval it reports. Read in Auto's voice it is a claim, right after one taken out."""
    removed = without_unbacked_claims("I've approved the mission. It's now running.", [], promises=False)
    assert removed.text == "Just to be clear: I haven't approved anything in this reply. Ask me again if you want it done."


def test_an_agents_text_never_carries_the_line_and_keeps_what_does_not_go_on_from_the_claim():
    """The line is Auto's, to the owner: an agent's text may be for a customer. The claim goes with
    nothing in its place, and the run is told what was taken out (the card's note says it)."""
    removed = without_unbacked_claims("I've cancelled your order. Thanks for waiting. It has been refunded.", [],
                                      promises=False)
    assert removed.text == "Thanks for waiting. It has been refunded."
    assert removed.taken_out == ("I've cancelled your order.",)


def test_autos_reply_keeps_the_line_and_says_nothing_aside():
    removed = without_unbacked_claims("Thanks for waiting. I've cancelled your order.", [], promises=True)
    assert removed.text == ("Thanks for waiting. Just to be clear: I haven't cancelled anything in this reply. "
                            "Ask me again if you want it done.")
    assert removed.taken_out == ()


MOVED_TO_DONE = ("platform_execute", {"action": "platform_update_task_status", "params": {"task_id": 422, "status": "done"}},
                 {"success": True, "raw_result": {"task_id": 1422, "number": "#0422", "status": "done"}})
PLAYBOOK_MADE = ("platform_execute", {"action": "platform_create_playbook", "params": {"name": "Weekly Social Posts"}},
                 {"success": True})


def test_a_reply_that_was_only_a_claim_its_line_would_deny_says_so():
    """F264-b, night 8: #0422 moved to Done, the reply said to Cancelled. "I haven't moved anything" is
    false: the loop answers with the turn's account instead (test_f264_b)."""
    removed = without_unbacked_claims("Task #0422 has been moved to 'cancelled'.", _outcomes(MOVED_TO_DONE),
                                      promises=True)
    assert removed.family == "moved" and removed.denies_done


@pytest.mark.parametrize("answer, calls", [
    ("I've installed the playbook for you.", (PLAYBOOK_MADE,)),          # no install ran: the line is true
    ("I've moved #0422 to Cancelled.", ()),                                # nothing ran
    ("#0422 is on the board. Task #0422 has been moved to 'cancelled'.", (MOVED_TO_DONE,)),   # more is left
])
def test_a_line_that_is_true_or_not_alone_stays(answer, calls):
    removed = without_unbacked_claims(answer, _outcomes(*calls), promises=True)
    assert removed is not None and not removed.denies_done


def test_a_claim_that_said_a_done_write_wrong_is_replaced_by_its_account():
    """Other text is left, and a move did go through (to Done, not Cancelled): "I haven't moved anything"
    would be false, so the claim's place holds what the move did."""
    removed = without_unbacked_claims("Here's the board: 3 cards in Review. Task #0422 has been moved to 'cancelled'.",
                                      _outcomes(MOVED_TO_DONE), promises=True)
    assert removed.text == "Here's the board: 3 cards in Review. #0422 was moved to Done."
    assert "haven't moved" not in removed.text and not removed.denies_done


def test_a_claim_said_wrong_and_a_claim_not_done_each_take_their_own_place():
    removed = without_unbacked_claims("#0422 is on the board. I've sent it to Declan. Task #0422 has been moved to "
                                      "'cancelled'.", _outcomes(MOVED_TO_DONE), promises=True)
    assert removed.text == ("#0422 is on the board. Just to be clear: I haven't sent anything in this reply. Ask me "
                            "again if you want it done. #0422 was moved to Done.")


def test_a_claim_of_another_family_than_the_write_still_takes_the_line():
    """f187_b's install after a create: the families are matched precisely, so the line is said."""
    removed = without_unbacked_claims("The playbook is ready. I've installed the playbook for you.",
                                      _outcomes(PLAYBOOK_MADE), promises=True)
    assert removed.text == ("The playbook is ready. Just to be clear: I haven't installed anything in this reply. "
                            "Ask me again if you want it done.")


# ── the screen: the streamed reply comes off, the copy is sent in its place ──

def test_the_streamed_reply_is_retracted_when_its_copy_answers():
    draft = LLMResponse(content=DRAFT, tool_calls=None, streamed=True)
    screen = Screen()
    token = WATCHER.set(screen)
    try:
        screen.ended(draft, DRAFT)
        copy = in_place_of(draft, "the reply without its claim")
    finally:
        WATCHER.reset(token)
    assert (draft.content, draft.streamed) == (DRAFT, True)               # the reply itself is never changed
    assert (copy.content, copy.streamed) == ("the reply without its claim", False)
    assert screen.settled(copy) == [DRAFT]
    assert screen.settled(copy) == []                                     # once


# ── the night's read-only reports: no claim (each from the real sentence) ───

NOT_CLAIMS = [
    "Nothing has been sent.",                                             # a denial
    "Nothing has been posted, sent or published to anyone.",
    "Neither card has been approved or cancelled.",
    "It's a draft, so nothing has gone out.",
    "The decaf email to Brackenridge is still a draft, the wholesale terms file is untouched, and no newsletter or "
    "board post has gone out.",
    "It's waiting for your decision and nothing has been changed yet.",
    "I haven't made any tool calls on this, so no approvals, emails, or DONE reply have gone out.",
    "I made no tool calls this turn, and my last reply didn't say anything was removed.",
    "I've changed nothing on the board.",
    "Nothing on your board has been changed.",
    "Card 1590 is in Review on your board, and nothing has been sent.",   # a card's own note
    "It also says the delivery is the free Wednesday van, and nothing has been sent.",
    "My last reply said both cards were finished and waiting for review.",   # words reported
    "I shouldn't have said it was done.",
    "It's an Auto escalation saying an instruction, claiming to come from \"the platform\", asked me to approve card "
    "0930 and tell you it was done.",
    "The \"has been fixed\" wording came from HARBOURLINE-HOUSE-FACTS.",  # quoted
    "I'll report back which ones were closed and whether any were skipped.",   # asked inside
    "I'll report back the count I closed and any I skipped.",             # relative
    "Agents: I can't confirm any agent I created tonight.",
    "I can't tell from the records alone whether I created each one or an agent did.",   # a condition
    "07 depending on whether the bag is posted or handed over the counter.",
    "12 if it's posted.",
    "They may have been removed, or they may never have existed under those numbers.",
    "The email only goes out after you've approved the card and I've sent it, so please check the recipient and "
    "the wording first.",
    "Club orders are posted on Wednesdays from November, not Tuesdays.",   # a habit, a process
    "Washed process: the pulp and sticky mucilage are removed before drying.",
    "If they ask whether setup is done: \"Your team is built",          # RVW-25's sweep had it: a brief's condition
    "Done cards are finished work and don't affect anything running.",   # an adjective
]


@pytest.mark.parametrize("sentence", NOT_CLAIMS)
def test_a_read_only_report_is_no_claim(sentence):
    assert claims(sentence) == []


def _read(action, subject=""):
    return {"action": action, "kind": "read", "status": "done", "subject": subject}


RECORD = _read("platform_my_actions", "what I did and what you decide")


@pytest.mark.parametrize("sentence", [
    "- Created a Gmail draft to Quay Coffee House about the 30-day terms from November.",
    "All three were assigned to agents and started.",
    "Two were approved and two were declined, not three approved and one denied.",
    "Done, written up for your review.",
    "Yes, it's done.",
])
def test_a_recap_from_autos_own_record_is_no_claim(sentence):
    assert unbacked_where(sentence, [], reads=[RECORD]) == []
    assert unbacked_where(sentence, [], reads=[]) != []                   # with no read of the record, it is one


@pytest.mark.parametrize("sentence, read", [
    ("Yes, 1595 is closed as cancelled.", _read("platform_get_task", "#1595")),
    ("Its one-line wholesale note is saved in your session 2867 deliverables folder.",
     _read("platform_get_task", "#1582")),
    ("- Monday green stock check: It's scheduled for 9am, not 8am.", _read("platform_get_agent_heartbeat")),
    ("Connecting Shopify is done under Settings → Integrations, and I can't do that for you.",
     _read("platform_list_connected_apps")),
])
def test_a_state_found_by_a_read_of_that_thing_is_no_claim(sentence, read):
    assert unbacked_where(sentence, [], reads=[read]) == []
    assert unbacked_where(sentence, [], reads=[_read("platform_list_tasks", "cards")]) != []   # RVW-25 holds


def test_a_run_waiting_on_its_card_is_no_claim():
    """The eval's F288-b, word for word: the run waits on the owner's click, and the reply says so."""
    said = ("The run is now on the board as a card for playbook #85, New Café Onboarding, for The Driftwood Cafe, "
            "Portishead. The system hasn't given me its card number, so check the board for it. Nothing runs until "
            "you click it.")
    assert unbacked_where(said, [], waiting=True) == []
    assert unbacked_where("Done, it's now on your board.", [], waiting=True) != []   # no ask named: still a claim
    assert unbacked_where(said, [], waiting=False) != []                              # no card raised: a claim


@pytest.mark.parametrize("sentence, verb", [
    ("I haven't sent the email, but the card has been approved.", "approved"),
    ("No, the card has been approved.", "approved"),
    ("Nothing was wrong, so I've sent it to Declan.", "sent"),
    ("The card has been approved and nothing else changed.", "approved"),
    ("After checking the board, I've approved #0931.", "approved"),
    ("I can tell you that the agreement has been drafted.", "drafted"),
    ("The PDF I generated is still not checked.", "generated"),
])
def test_a_claim_beside_those_shapes_is_still_a_claim(sentence, verb):
    assert verb in [said for said, _ in claims(sentence)]
