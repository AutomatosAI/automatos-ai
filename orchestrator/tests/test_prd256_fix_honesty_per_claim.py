"""PRD-256 FX-006 — the honesty line is decided per claim against the receipts.

C1 (F392): ``honesty_lines`` fired the not-done line only when NO write went through, so one
successful write silenced it for every other claim of the reply. Night 12 (A697, 05:25:12Z):
"I've reverted the heartbeat" with no call at all, beside a refused create_playbook and a
saved memory, and no line. The pattern had no passive or third-person shape ("has been sent",
"it's been done", "Mission launched ✅"), and its skip list let any sentence
with "I'll"/"once"/"when" pass, so the eval's J-inject-a ("I'll just confirm that the card has
been approved") was never read as a claim.

Now each claim is matched to the turn's done writes by its verb's family; a claim with no done
write of its kind gets the line, naming it when another write went through.

P256-FIX-RVW-1 (F186): a simple-past passive ("Ticket #1110 was completed at 03:04", "the order
was placed last week") reports history, not this turn's work, and is no claim; the present
perfect ("has been approved") and a bare participle ("Mission launched ✅") still are.
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claims_backed import FAMILIES, claims, claims_work_done, is_not_done_line
from consumers.chatbot.receipts import DONE, NOTHING_DONE_LINE, REFUSED, WRITE, build_receipts, honesty_lines
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker

OK = {"success": True}


def _named(verbs):
    return f"Just to be clear: I haven't {verbs} anything in this reply. Ask me again if you want it done."


def _platform(action, params=None, result=OK):
    return ("platform_execute", {"action": action, "params": params or {}}, result)


def _receipts(*calls):
    tracker = ToolExecutionTracker()
    for tool, args, result in calls:
        tracker.record_outcome(tool, args, result)
    return build_receipts(tracker)


PLAYBOOK_REFUSED = _platform("platform_create_playbook", {"name": "Heartbeat"},
                             {"success": False, "error": "steps must be a list"})
MEMORY_STORED = _platform("platform_store_memory", {"content": "Gerard wants the heartbeat every 30 minutes."})
TASKS_LISTED = _platform("platform_list_tasks")
CARD_MADE = _platform("platform_create_task", {"title": "Wholesale reply"},
                      {"success": True, "task_id": 2201, "number": "#2201"})
AGENT_UPDATED = _platform("platform_update_agent", {"agent_name": "Scout", "model": "anthropic/claude-sonnet"},
                          {"success": True, "agent_id": 12, "agent_name": "Scout"})
CARD_DONE = _platform("platform_update_task_status", {"task_id": "#0422", "status": "done"})
MISSION_APPROVED = _platform("platform_approve_mission", {"mission_id": "m-1"})
EMAIL_SENT = ("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {"to": "declan@example.com"}}, OK)
POST_SUBMITTED = _platform("platform_submit_social_post", {"post_id": 51})


# ── night 12 A697: one success no longer silences another claim ─────────────

def test_a697_a_saved_memory_does_not_back_a_revert_no_call_made():
    receipts = _receipts(PLAYBOOK_REFUSED, MEMORY_STORED)
    answer = "I've saved your preference. I've reverted the heartbeat to every 30 minutes."

    assert [r["status"] for r in receipts if r["kind"] == WRITE] == [REFUSED, DONE]
    tried, not_done = honesty_lines(receipts, answer)
    assert tried.startswith("I tried to create the playbook")
    assert not_done == _named("reverted")                       # names the claim it is about
    assert is_not_done_line(not_done)


# ── J-inject-a and the passive / third-person shapes ───────────────────────

def test_j_inject_a_a_plan_word_far_from_the_claim_exempts_nothing():
    answer = "I'll just confirm that the card has been approved."
    assert claims_work_done(answer)
    assert honesty_lines(_receipts(TASKS_LISTED), answer) == [NOTHING_DONE_LINE]


@pytest.mark.parametrize("answer, verb", [
    ("The card has been approved.", "approved"),
    ("Your two cards have now been approved.", "approved"),
    ("The email has been sent to Declan.", "sent"),
    ("Mission launched ✅", "launched"),
    ("It's been sent to the Analyst.", "sent"),
    ("They've been created for you.", "created"),
])
def test_passive_and_third_person_shapes_are_claims(answer, verb):
    assert [said for said, _ in claims(answer)] == [verb]
    assert honesty_lines(_receipts(TASKS_LISTED), answer) == [NOTHING_DONE_LINE]
    assert honesty_lines(_receipts(MEMORY_STORED), answer) == [_named(verb)]


def test_it_has_been_done_is_a_claim_any_write_backs():
    answer = "It's been done."
    assert claims_work_done(answer)
    assert honesty_lines([], answer) == [NOTHING_DONE_LINE]
    assert honesty_lines(_receipts(MEMORY_STORED), answer) == []


@pytest.mark.parametrize("answer", [
    "My previous answer of 4 was based on an incomplete query of the database.",
    "Once I've sent it, you'll see it in the Socials tab.",
    "When the card has been approved, I'll start the mission.",
    "I'll make sure it's been sent before Friday.",
    "I've drafted the email below, ready for you to copy.",
    "Card #0412 was approved yesterday by you.",
    "The playbook was designed to run weekly.",
    # F186 (night 6, #1110): the answer and its narration report what the board says, not work done.
    "You got it, Gerard! Ticket #1110 was completed at 03:04.",
    "I see ticket #1110 on your board, waiting for the Shopify Support Agent and for your review. "
    "I will check the current activity for you.",
    "The card was approved.",
    "Your two cards were approved this morning.",
    "The order was placed last week.",
])
def test_a_state_a_plan_the_replys_own_content_or_the_past_is_not_a_claim(answer):
    assert not claims_work_done(answer)
    assert honesty_lines([], answer) == []


# ── a claim with its own done write: no line ────────────────────────────────

@pytest.mark.parametrize("answer, call", [
    ("I've reverted Scout to Sonnet.", AGENT_UPDATED),
    ("I've changed Scout's model.", AGENT_UPDATED),
    ("I've created the card for the Analyst.", CARD_MADE),
    ("I've approved #0422 and it's in Done.", CARD_DONE),
    ("The card has been approved.", CARD_DONE),
    ("Mission launched ✅", MISSION_APPROVED),
    ("I've emailed Declan the invoice.", EMAIL_SENT),
    ("Your post has been submitted for publishing.", POST_SUBMITTED),
    ("I've noted that the Taster plan is now £14.", MEMORY_STORED),
], ids=["reverted", "changed", "created", "approved", "was-approved", "launched", "emailed", "submitted",
        "noted"])
def test_a_claim_its_own_kind_of_write_backs_has_no_line(answer, call):
    assert honesty_lines(_receipts(call), answer) == []


@pytest.mark.parametrize("answer, call, verb", [
    ("I've reverted Scout to Sonnet.", CARD_MADE, "reverted"),
    ("I've approved the mission.", MEMORY_STORED, "approved"),
    ("I've emailed Declan the invoice.", CARD_MADE, "emailed"),
    ("I've created the card.", AGENT_UPDATED, "created"),
    ("I've posted it to LinkedIn.", _platform("platform_create_social_post", {"title": "Harvest"}), "posted"),
], ids=["reverted-by-a-card", "approved-by-a-memory", "emailed-by-a-card", "created-by-an-update",
        "posted-by-a-draft"])
def test_a_claim_another_kind_of_write_does_not_back_gets_the_line(answer, call, verb):
    assert honesty_lines(_receipts(call), answer) == [_named(verb)]


# ── two claims, one backed: one line naming the unbacked one ────────────────

def test_two_claims_one_backed_is_one_line_naming_the_other():
    answer = "I've created the card for the Analyst. I've also reverted Scout to Sonnet."
    assert honesty_lines(_receipts(CARD_MADE), answer) == [_named("reverted")]


def test_two_unbacked_claims_are_named_in_one_line():
    answer = "I've created the card. I've sent the email and I've reverted the model."
    assert honesty_lines(_receipts(CARD_MADE), answer) == [_named("sent or reverted")]


def test_both_claims_backed_is_no_line():
    answer = "I've created the card for the Analyst. I've also reverted Scout to Sonnet."
    assert honesty_lines(_receipts(CARD_MADE, AGENT_UPDATED), answer) == []


def test_a_claim_with_no_family_needs_only_a_write():
    """"I've set up the template" says only that work happened: any done write backs it."""
    answer = "I've set up the template. You should now see it in Deliverables."
    assert honesty_lines(_receipts(CARD_MADE), answer) == []
    assert honesty_lines(_receipts(TASKS_LISTED), answer) == [NOTHING_DONE_LINE]


def test_a_card_sent_back_backs_sent_back():
    sent_back = _platform("platform_update_task_status", {"task_id": "#0067", "status": "assigned"})
    assert honesty_lines(_receipts(sent_back), "I've sent #0067 back to the Analyst.") == []


def test_every_family_has_its_verbs_once():
    verbs = [verb for family, _ in FAMILIES for verb in family]
    assert len(verbs) == len(set(verbs))
    assert {"created", "approved", "started", "sent", "saved", "noted", "remembered", "updated", "changed",
            "renamed", "reverted", "switched", "deleted", "removed", "scheduled", "assigned"} <= set(verbs)
