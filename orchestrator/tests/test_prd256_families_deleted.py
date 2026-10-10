"""PRD-256 FX-007 (US-012, unblocked by D10): the regex claim families are deleted.

The pre-PRD claim checker was regex families one night behind (12 of 22 real sentences caught,
false denials F319, F337, F363), and its tier-2 id correction fired wrongly twice on night 12
("Just to be clear: task 0930 does not exist", A439/A593: #0930 was a card number quoted from
ticket #0931's own title). The receipts carry every call, and a claim no receipt backs is said
from them (FX-006), so the families, their note lanes and the ``_retrieval_first`` decorators
that served only them are gone. The in-loop nudge (F108) reads the receipts' rule.

Each family's sentence is read by the receipts' rule: CAUGHT (the nudge names the claim's verb)
or CLEARED (no claim: a plan, a figure, a place, or a write of its kind went through; the
receipts above the answer show what ran).

P256-FIX-RVW-37: FX-007 had inverted four rows the families caught. They are caught again by the
receipts rule (``claims_by_kind``), each cleared by its own backing call: "Each step will pause"
over an unchecked mission (F308), a reply ending "I will now send …" with nothing sent (F261), "all
agents know" over a saved memory (F324), and a letter said made over a card made (F351).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from consumers.chatbot import claim_check
from consumers.chatbot.claims_by_kind import STEPS_LABEL, TEAM_LABEL
from modules.tools.execution.tool_execution_tracker import TRACKERS_MADE, ToolExecutionTracker
from tests.helpers_receipts_rule import call, line, nudged

ORCHESTRATOR = Path(__file__).resolve().parents[1]
CHATBOT = ORCHESTRATOR / "consumers" / "chatbot"
EXECUTION = ORCHESTRATOR / "modules" / "tools" / "execution"
DELETED = [EXECUTION / "action_claims.py", EXECUTION / "document_claims.py", EXECUTION / "shop_and_team_claims.py",
           EXECUTION / "social_post_claims.py", CHATBOT / "figure_disputes.py", CHATBOT / "shop_figures.py",
           CHATBOT / "team_corrections.py"]
SERVICE_CEILING = 3200       # service.py had 3,277 lines on the fix wave's base; it may only shrink

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
EMAIL_SENT = ("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {}}, {"successful": True})
REFUSED = {"success": False, "error": "The owner said cancel, not approve. Nothing was done."}


def _moved(task_id, status, result=None):
    return call("platform_update_task_status", {"task_id": task_id, "status": status}, result)


LETTER_REFUSED = ("generate_document", {"title": "Letter to Maya Osei"},
                  {"success": False, "raw_result": {"success": False, "error": "data is not a complete object"}})
UNCHECKED_MISSION = call("platform_create_mission", {"goal": "September takings"},
                         {"success": True, "mission_id": "x", "checks_each_step": False})
AT_17_58 = ("My apologies, Gerard. I've now created a task on your board for the Brand Designer. You should now see "
            "this on your board, with task number #0891.\n\nI've noted your decision.")
GAVE_WAY = ("The Analyst's count of 11 club cancellations between April and September on card 1886 is correct. My "
            "previous answer of 4 was based on an incomplete query of the database.")
B33 = "Looking through the entire CSV file, I can now give you the exact numbers Tom needs for Monday's club post:"

# (family, the sentence, the calls of its turn, what the nudge names: its verb, or None when cleared)
SENTENCES = [
    ("F108 approved, no call", "I've approved the mission. It's now running.", [], "approved"),
    ("F108 approved, the approval ran", "I've approved the mission. It's now running.",
     ["platform_approve_mission"], None),
    ("F108 emailed through Composio", "I've emailed Declan the invoice.", [EMAIL_SENT], None),
    ("F187 installed over a create", "I've installed the playbook for you.",
     ["search_knowledge", "platform_create_playbook"], "installed"),
    ("F187 checked: a read, no work", "I've checked the board for you.", ["search_knowledge"], None),
    ("F187 exact numbers: a figure", B33, ["platform_read_document"], None),
    ("F187 under way: a promise", "Please bear with me while I fix this.", [], None),
    ("F261 a promise to send", "I will now send this updated brief to the agent.", [], "sent"),
    ("F261 moved to cancelled over a move to done", "Task #0422 has been moved to 'cancelled'.",
     [_moved(422, "done")], "moved"),
    ("F261 sent back over an edit", "I've sent it back to the agent to redo.",
     [call("platform_update_task", {"task_id": 451, "description": "To: first"})], "sent"),
    ("F261 initiated, no call", "I have now correctly initiated the \"New Cafe Onboarding\" playbook.", [],
     "initiated"),
    ("F261 approved by the move to done", "Alright, I've approved ticket #0231 and marked it as done.",
     [_moved(231, "done")], None),
    ("F308 steps said to pause", "Each step will pause for your approval.", [UNCHECKED_MISSION], STEPS_LABEL),
    ("F314 saved in passing, nothing saved", "This draft has now been saved as a social post.", [], "saved"),
    ("F314 saved, the post was made", "This draft has now been saved as a social post.",
     ["platform_create_social_post"], None),
    ("F303 a figure said right", GAVE_WAY, [], None),
    ("F316 a shop figure", "There are 87 subscribers on the Harvest Club.", ["search_knowledge"], None),
    ("F324 stored for the team", "I've stored it in my memory so that all agents know.",
     ["platform_store_memory"], TEAM_LABEL),
    ("F319 sent back by its number", "I've sent card 67.1 back to the Analyst.", [_moved(67.1, "assigned")], None),
    ("F319 cancelled behind a quoted title", 'Card #0044, "Minimum wholesale order and cut-off," has been cancelled.',
     [_moved("0044", "done", REFUSED)], "cancelled"),
    ("F351 a letter over a refused document and a card",
     "I've generated the letter to Maya Osei and saved it to Deliverables.", [LETTER_REFUSED, "platform_create_task"],
     "generated"),
    ("F351 where a document is", 'You can find the letter in your Deliverables as "Payment Terms Update.docx".', [],
     None),
    ("F363 noted on the card the turn made", AT_17_58, ["platform_list_tasks", "platform_create_task"], None),
    ("F363 a hand-off said as a plan", "I'll get the Brand Designer to update the brand kit with Option A.", [], None),
    ("F379 removed from a post by its edit", "I've removed the tasting notes from the carousel.",
     ["platform_update_social_post"], None),
    ("F379 posted over a draft", "I've posted it to Instagram.", ["platform_create_social_post"], "posted"),
]


@pytest.mark.parametrize("path", DELETED, ids=[path.name for path in DELETED])
def test_the_family_modules_and_their_lanes_are_gone(path):
    assert not path.exists(), f"{path.name} is back: the families are gone (PRD-256 D10, FX-007)"


def test_the_table_is_the_families_26_sentences():
    assert len(SENTENCES) == 26 and len({case[0] for case in SENTENCES}) == 26


@pytest.mark.parametrize("family, said, calls, claim", SENTENCES, ids=[case[0] for case in SENTENCES])
def test_each_sentence_is_caught_or_cleared_by_the_receipts_rule(family, said, calls, claim):
    assert nudged(said, *calls) == claim, family
    assert (line(said, *calls) is None) is (claim is None), family     # caught: the line above the answer too


# ── P256-FIX-RVW-37: the four rows FX-007 inverted, each cleared by its backing call ──

CHECKED_MISSION = call("platform_create_mission", {"goal": "September takings"},
                       {"success": True, "mission_id": "x", "checks_each_step": True})
LETTER_AND_SAVED = "I've created the letter to Maya and saved it to your Deliverables."
BACKED = [
    ("F308 steps said to pause over a mission that checks each step", "Each step will pause for your approval.",
     [CHECKED_MISSION]),
    ("F261 a promise to send, the email sent", "I will now send this updated brief to the agent.", [EMAIL_SENT]),
    ("F324 stored for the team, a note in the documents", "I've stored it in my memory so that all agents know.",
     ["platform_store_memory", "platform_upload_document"]),
    ("F351 a letter made", LETTER_AND_SAVED, ["generate_document"]),
    ("F351 a letter uploaded", LETTER_AND_SAVED, ["platform_create_task", "platform_upload_document"]),
]


@pytest.mark.parametrize("family, said, calls", BACKED, ids=[case[0] for case in BACKED])
def test_each_inverted_row_is_cleared_by_its_backing_call(family, said, calls):
    assert nudged(said, *calls) is None and line(said, *calls) is None, family


def test_a_letter_said_made_and_saved_over_a_card_is_caught():
    """F351: create_task backs no letter "created" and no letter "saved to your Deliverables"."""
    assert nudged(LETTER_AND_SAVED, "platform_create_task") == "created"
    assert line(LETTER_AND_SAVED, "platform_create_task") == (
        "Just to be clear: I haven't created or saved anything in this reply. Ask me again if you want it done.")
    assert nudged(LETTER_AND_SAVED, "platform_create_task", "generate_document") is None


def test_the_steps_and_the_team_are_said_in_their_own_words():
    assert line("Each step will pause for your approval.", UNCHECKED_MISSION) == (
        "Just to be clear: nothing in this reply set the mission's steps to wait for your OK. Ask me again if you "
        "want it done.")
    assert line("I've stored it in my memory so that all agents know.", "platform_store_memory").startswith(
        "Just to be clear: your agents haven't been told: I only keep this in my own memory")


def test_a_promise_ending_the_reply_gets_the_announced_step_nudge():
    from modules.tools.execution.nudges import announced_step
    from modules.tools.execution.tool_loop import ToolLoopExecutor
    from tests.helpers_receipts_rule import tracker_of

    said = "I will now send this updated brief to the agent."
    assert announced_step(said, tracker_of([]).outcomes) == said
    assert announced_step(said, tracker_of([EMAIL_SENT]).outcomes) is None
    assert announced_step(said) is None                                     # outside a loop: the cap and the card
    assert announced_step("I will now send it. Is there anything else?", tracker_of([]).outcomes) is None
    assert "announced_step(text, self.tracker.outcomes)" in inspect.getsource(ToolLoopExecutor._recover_claimed_action)


# ── F187 tier 2 stays: a number the turn's own tool results quote is backed ──

TICKET_0931 = {"success": True, "tasks": [{"number": "#0931", "title": "Card #0930 Approval Request",
                                           "status": "review"}]}
NAMES_0930 = "Ticket #0931 is the approval request for card #0930: it waits for your OK."


@pytest.fixture
def board(monkeypatch):
    """#0931 is a card of the workspace; #0930 is no card number (it is quoted in #0931's title)."""
    monkeypatch.setattr(claim_check, "existing_ids", lambda ws, named: {p for p in named if p == ("task", "0931")})


def _in_a_loop(*results):
    tracker = ToolExecutionTracker.__new__(ToolExecutionTracker)
    tracker.outcomes = [("platform_list_tasks", {}, result) for result in results]
    token = TRACKERS_MADE.set([tracker])
    try:
        return claim_check.invented_ids(NAMES_0930, "What is waiting for me?", WS)
    finally:
        TRACKERS_MADE.reset(token)


def test_a_card_number_quoted_from_a_tool_result_of_the_turn_is_backed(board):
    """Night 12 A439/A593: 'task 0930 does not exist' never fires on '#0931 … Card #0930 Approval Request'."""
    assert _in_a_loop(TICKET_0931) == []


def test_the_same_number_with_no_tool_result_behind_it_is_still_corrected(board):
    assert _in_a_loop() == [("task", "0930")]
    assert claim_check.invented_ids(NAMES_0930, "", WS) == [("task", "0930")]       # outside a loop


def _looked_up(text, *outcomes):
    tracker = ToolExecutionTracker.__new__(ToolExecutionTracker)
    tracker.outcomes = list(outcomes)
    token = TRACKERS_MADE.set([tracker])
    try:
        return claim_check.invented_ids(text, "What is waiting for me?", WS)
    finally:
        TRACKERS_MADE.reset(token)


def test_a_failed_lookup_that_repeats_the_id_quotes_nothing(board):
    """RVW-32: ID_NUDGE's lookup answers 'Task #1100 not found'; that echo never backs #1100."""
    not_found = ("platform_get_task", {"task_id": "#1100"}, {"success": False, "error": "Task #1100 not found"})
    assert _looked_up("Task #1100 is in Review.", not_found) == [("task", "1100")]
    errored = ("platform_get_task", {"task_id": "#1100"}, {"error": "Task #1100 not found"})
    assert _looked_up("Task #1100 is in Review.", errored) == [("task", "1100")]
    refused = ("composio_execute", {}, {"successful": False, "data": {"message": "ticket 1100 unknown"}})
    assert _looked_up("Task #1100 is in Review.", refused) == [("task", "1100")]


def test_a_succeeded_result_never_quotes_its_own_params(board):
    """RVW-32: a search that echoes the asked number back has not found it."""
    echoed = ("platform_list_tasks", {"search": "#1100"}, {"success": True, "query": "#1100", "tasks": []})
    assert _looked_up("Task #1100 is in Review.", echoed) == [("task", "1100")]
    by_number = ("platform_get_task", {"task_id": 1100}, {"success": True, "note": "task 1100 has no card"})
    assert _looked_up("Task #1100 is in Review.", by_number) == [("task", "1100")]


def test_the_quoted_card_number_still_clears_beside_a_failed_call(board):
    """The '#0931 … Card #0930 Approval Request' case keeps clearing #0930 next to a refused lookup."""
    listed = ("platform_list_tasks", {}, TICKET_0931)
    missing = ("platform_get_task", {"task_id": "#0930"}, {"success": False, "error": "Task #0930 not found"})
    assert _looked_up(NAMES_0930, missing, listed) == []
    assert _looked_up(NAMES_0930, missing) == [("task", "0930")]


def test_a_number_inside_a_longer_one_or_a_count_is_not_quoted(board):
    assert _in_a_loop({"success": True, "title": "Order 109300 shipped"}) == [("task", "0930")]
    assert _in_a_loop({"success": True, "count": 930, "limit": 930}) == [("task", "0930")]   # numbers, not text


# ── the families' guards the receipts' rule keeps ────────────────────────────

@pytest.mark.parametrize("said, calls, claim", [
    ("I've noted that you're happy to raise the budget.", [], None),                       # the owner heard
    ("I've started reading the October note.", [], None),                                  # a read begun
    ("I've created digest playbook for Mondays.", [], "created"),                          # "dig…" is no read
    ("I've built search filters for the board.", [], "built"),
    ("As I've noted before, the price is £12.", [], None),                                 # a back-reference
    ("I've sent it before Friday's cut-off.", [], "sent"),                                 # "before Friday" is no past
    ("Here's the draft:\n\n> Dear Maya, I've sent the invoice.", [], None),                # quoted: its writer's voice
    ("```\nI've sent the invoice.", [], "sent"),                                            # a fence never closed
    ("Task #0422 has been moved to Done.", ["platform_approve_task"], None),               # an approval moves it
    ("Task #0422 has been moved to Done.", [_moved(422, "cancelled")], "moved"),
])
def test_the_rule_keeps_the_families_guards(said, calls, claim):
    assert nudged(said, *calls) == claim


# ── the turn: what stays, what went ─────────────────────────────────────────

def test_service_py_is_shorter_and_holds_no_family():
    from consumers.chatbot import service

    source = inspect.getsource(service)
    assert len(source.splitlines()) <= SERVICE_CEILING
    for gone in ("claimed_action_not_done", "passive_claim", "rechecks_disputed_figures", "counts_from_the_shop",
                 "tells_the_team_honestly", "class ToolExecutionTracker"):
        assert gone not in source, gone


def test_the_in_loop_nudge_and_the_first_reply_read_the_receipts_rule():
    from consumers.chatbot.service import StreamingChatService
    from modules.tools.execution.tool_loop import ToolLoopExecutor

    nudge = inspect.getsource(ToolLoopExecutor._recover_claimed_action)
    assert "unbacked_claim(text, self.tracker.outcomes, promises=self.promises)" in nudge   # RVW-7: the run's voice
    assert "claims_work_done(response.content)" in inspect.getsource(
        StreamingChatService._first_reply_goes_through_the_loop)


def test_a_customer_draft_is_checked_by_the_receipts_rule():
    from services.draft_guides import check_before_sending

    brief = "Email from Rosie Tanner, club member: she was charged twice. Please draft a reply."
    draft = "Dear Rosie, I have also updated your subscription to filter grind."
    assert "says something was updated" in check_before_sending(brief, draft, ["platform_load_skill"])
    assert check_before_sending(brief, draft, ["platform_update_subscription"]) is None
