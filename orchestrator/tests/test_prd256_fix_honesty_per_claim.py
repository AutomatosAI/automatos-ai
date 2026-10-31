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

P256-FIX-RVW-6: added, paused, "set up", booked, … were in no family, so a saved memory backed
"I've paused the heartbeat" (A697's shape); one verb was read per "I've" ("I've created the task
and sent it to Declan" was only created); and "here's" exempted its whole sentence ("Here's the
update: the card has been approved." was no claim).

P256-FIX-RVW-20: RVW-1 had removed the whole "was/were" shape, so "The card was approved." with no
call was no claim at all; it is again, and only a simple-past passive with a past time in its
sentence ("at 03:04", "this morning") is history. Replied, texted, shared, invited and refunded
were in no family, so a mail draft or a saved memory backed "I've replied to Declan." (D7's
"reply").

P256-FIX-RVW-36: platform_load_skill (a registered read: its permission_level says so, though no read
stem is in its name) was a done write and backed "Your post is now live on LinkedIn."; and any done
write backed a verb in no family, so a saved memory backed "I've rejected the card.".

P256-FIX-RVW-25: with the families deleted (FX-007, D10) this rule is the only guard, and "The task
is done.", "The post is published.", "The post got sent.", "It went out.", "Email sent.", "Posted!",
"I sent the email to Declan." and "I emailed the supplier and created the ticket." were no claim;
"I've created the ticket, emailed Sam." was only created.

P256-FIX-N15-2: the not-done line is said only in place of the claims it answers (taken out of the reply,
N15-1), and names them whether or not another write went through; a verb in no family ("sorted",
"rejected", "made the changes") is still read as unbacked here (``unbacked``) but never taken out, so it
draws no line. Nothing puts the line above the reply.
P256-FIX-N17-5 (F446): after a read only (``TASKS_LISTED``), a third-person shape says what it found: no line; with
no call it is caught, and Auto's own reports ("Email sent.") are caught after a read too (recall: the owner's call).
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claims_backed import FAMILIES, claims, claims_work_done, is_not_done_line
from consumers.chatbot.receipts import DONE, READ, REFUSED, WRITE, build_receipts, honesty_lines
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from tests.helpers_receipts_rule import line, nudged, unbacked

OK = {"success": True}


def _named(verbs):
    return f"Just to be clear: I haven't {verbs} anything in this reply. Ask me again if you want it done."


NOTHING_DONE_LINE = "Just to be clear: I haven't done that yet, and nothing has changed. Ask me again if you want it done."


def _in_place_line(verbs):
    """The line in place of claims of ``verbs`` ("" a bare "Done."), when no write of their kind went through."""
    named = list(dict.fromkeys(verb for verb in verbs if verb))
    if not named:
        return NOTHING_DONE_LINE
    return _named(named[0] if len(named) == 1 else f"{', '.join(named[:-1])} or {named[-1]}")


def _platform(action, params=None, result=OK):
    return ("platform_execute", {"action": action, "params": params or {}}, result)


def _receipts(*calls):
    tracker = ToolExecutionTracker()
    for tool, args, result in calls:
        tracker.record_outcome(tool, args, result)
    return build_receipts(tracker)


def _said(answer, *calls):
    """P256-FIX-N15-2: the not-done line the answer carries in place of its claims ([] when none). It is the
    line's only place: a refused write alone goes above the text (``honesty_lines``)."""
    in_place = line(answer, *calls)
    return [in_place] if in_place else []


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
    (not_done,) = _said(answer, PLAYBOOK_REFUSED, MEMORY_STORED)
    assert not_done.startswith("Just to be clear: I haven't reverted anything in this reply")   # names the claim
    assert "I tried to create the playbook" in not_done and is_not_done_line(not_done)          # and the refusal
    (tried,) = honesty_lines(receipts, answer)                  # above the reply as composed: the refusal alone
    assert tried.startswith("I tried to create the playbook")
    assert honesty_lines(receipts, f"I've saved your preference. {not_done}") == []   # the line names it: once


# ── J-inject-a and the passive / third-person shapes ───────────────────────

def test_j_inject_a_a_plan_word_far_from_the_claim_exempts_nothing():
    answer = "I'll just confirm that the card has been approved."
    assert claims_work_done(answer)
    assert _said(answer) == [_named("approved")]
    assert _said(answer, TASKS_LISTED) == []                 # N17-5: after a read, a third-person shape is a finding


@pytest.mark.parametrize("answer, verb, after_a_read", [
    ("The card has been approved.", "approved", False),
    ("Your two cards have now been approved.", "approved", False),
    ("The email has been sent to Declan.", "sent", False),
    ("Mission launched ✅", "launched", True),                      # N17-5: a tick is Auto's own report
    ("It's been sent to the Analyst.", "sent", False),
    ("They've been created for you.", "created", False),
])
def test_passive_and_third_person_shapes_are_claims(answer, verb, after_a_read):
    assert [said for said, _ in claims(answer)] == [verb]
    assert _said(answer) == [_named(verb)]                           # no call at all: caught
    assert _said(answer, TASKS_LISTED) == ([_named(verb)] if after_a_read else [])   # N17-5: else a read's finding
    assert _said(answer, MEMORY_STORED) == [_named(verb)]


def test_it_has_been_done_is_a_claim_any_write_backs():
    answer = "It's been done."
    assert claims_work_done(answer)
    assert _said(answer) == [NOTHING_DONE_LINE]
    assert _said(answer, MEMORY_STORED) == []


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
    "Your two cards were approved this morning.",
    "The order was placed last week.",
])
def test_a_state_a_plan_the_replys_own_content_or_the_past_is_not_a_claim(answer):
    assert not claims_work_done(answer)
    assert _said(answer) == []


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
    assert _said(answer, call) == []


@pytest.mark.parametrize("answer, call, verb", [
    ("I've reverted Scout to Sonnet.", CARD_MADE, "reverted"),
    ("I've approved the mission.", MEMORY_STORED, "approved"),
    ("I've emailed Declan the invoice.", CARD_MADE, "emailed"),
    ("I've created the card.", AGENT_UPDATED, "created"),
    ("I've posted it to LinkedIn.", _platform("platform_create_social_post", {"title": "Harvest"}), "posted"),
], ids=["reverted-by-a-card", "approved-by-a-memory", "emailed-by-a-card", "created-by-an-update",
        "posted-by-a-draft"])
def test_a_claim_another_kind_of_write_does_not_back_gets_the_line(answer, call, verb):
    assert _said(answer, call) == [_named(verb)]


# ── two claims, one backed: one line naming the unbacked one ────────────────

def test_two_claims_one_backed_is_one_line_naming_the_other():
    answer = "I've created the card for the Analyst. I've also reverted Scout to Sonnet."
    assert _said(answer, CARD_MADE) == [_named("reverted")]


def test_two_unbacked_claims_are_named_in_one_line():
    answer = "I've created the card. I've sent the email and I've reverted the model."
    assert _said(answer, CARD_MADE) == [_named("sent or reverted")]


def test_both_claims_backed_is_no_line():
    answer = "I've created the card for the Analyst. I've also reverted Scout to Sonnet."
    assert _said(answer, CARD_MADE, AGENT_UPDATED) == []


def test_a_claim_with_no_family_needs_a_write_its_stem_names():
    """"I've sorted the template" is in no family: a card made backed it until P256-FIX-RVW-36, now
    only a write named for sorting would; "You should now see it" says only that work happened, and
    any done write backs it ("set up" has its family since P256-FIX-RVW-6, below). P256-FIX-N15-2:
    "sorted" is still read as unbacked, but a verb in no family is never taken out, so it draws no line;
    with no write at all, "You should now see it" is taken out (any write backs it, none ran)."""
    answer = "I've sorted the template. You should now see it in Deliverables."
    assert unbacked(answer, CARD_MADE) == ["sorted"] and _said(answer, CARD_MADE) == []
    assert _said("You should now see it in Deliverables.", CARD_MADE) == []
    assert unbacked(answer, TASKS_LISTED) == ["sorted", ""] and _said(answer, TASKS_LISTED) == [NOTHING_DONE_LINE]


def test_a_card_sent_back_backs_sent_back():
    sent_back = _platform("platform_update_task_status", {"task_id": "#0067", "status": "assigned"})
    assert _said("I've sent #0067 back to the Analyst.", sent_back) == []


def test_every_family_has_its_verbs_once():
    verbs = [verb for family, _ in FAMILIES for verb in family]
    assert len(verbs) == len(set(verbs))
    assert {"created", "approved", "started", "sent", "saved", "noted", "remembered", "updated", "changed",
            "renamed", "reverted", "switched", "deleted", "removed", "scheduled", "assigned"} <= set(verbs)


# ── P256-FIX-RVW-6: the verbs that said only "work happened" have families ──

def _composio(slug):
    return ("composio_execute", {"action": slug, "params": {}}, {"successful": True})


HEARTBEAT_SET = _platform("platform_configure_agent_heartbeat", {"agent_name": "Scout", "enabled": False})
CONTACT_MADE = _composio("HUBSPOT_CREATE_CONTACT")
REPORT_SCHEDULED = _platform("platform_schedule_playbook", {"playbook_name": "Weekly report"})
BOOKING_MADE = _composio("CALCOM_CREATE_BOOKING")
ORDER_MADE = _composio("SHOPIFY_CREATE_ORDER")
PAYMENT_MADE = _composio("STRIPE_CREATE_PAYMENT_INTENT")
DOCUMENT_UPLOADED = _platform("platform_upload_document", {"filename": "price-list.pdf"})
CHANNEL_CONNECTED = _platform("platform_connect_channel", {"channel": "instagram"})
TOOL_ADDED = _platform("platform_assign_tool_to_agent", {"agent_name": "Scout", "tool": "DROPBOX"})

# (the sentence, its verb, a done write of its own kind)
OWN_KIND = [
    ("I've paused the heartbeat.", "paused", HEARTBEAT_SET),                         # verified at b7fcc419f
    ("I've added the supplier to your contacts.", "added", CONTACT_MADE),             # verified
    ("I've set up the weekly report.", "set up", REPORT_SCHEDULED),                   # verified
    ("I've booked the courier for Friday.", "booked", BOOKING_MADE),                  # verified
    ("I've turned off the heartbeat for Scout.", "turned off", HEARTBEAT_SET),
    ("I've turned on the heartbeat for Scout.", "turned on", HEARTBEAT_SET),
    ("I've disabled Scout's heartbeat.", "disabled", HEARTBEAT_SET),
    ("I've enabled Scout's heartbeat.", "enabled", HEARTBEAT_SET),
    ("I've activated the weekly report.", "activated", REPORT_SCHEDULED),
    ("I've deactivated the weekly report.", "deactivated", REPORT_SCHEDULED),
    ("I've added the Dropbox tool to Scout.", "added", TOOL_ADDED),
    ("I've uploaded the price list.", "uploaded", DOCUMENT_UPLOADED),
    ("I've ordered two sacks of Guji.", "ordered", ORDER_MADE),
    ("I've purchased two sacks of Guji.", "purchased", ORDER_MADE),
    ("I've paid the roaster's invoice.", "paid", PAYMENT_MADE),
    ("I've connected your Instagram account.", "connected", CHANNEL_CONNECTED),
    ("I've linked your Instagram account.", "linked", CHANNEL_CONNECTED),
]


@pytest.mark.parametrize("answer, verb, _own", OWN_KIND, ids=[case[1] + ":" + case[0][5:20] for case in OWN_KIND])
def test_rvw6_a_saved_memory_does_not_back_a_verb_that_had_no_family(answer, verb, _own):
    assert [said for said, _ in claims(answer)] == [verb]
    assert _said(answer, MEMORY_STORED) == _said(answer, TASKS_LISTED) == [_named(verb)]


@pytest.mark.parametrize("answer, verb, own", OWN_KIND, ids=[case[1] + ":" + case[0][5:20] for case in OWN_KIND])
def test_rvw6_its_own_kind_of_write_backs_it(answer, verb, own):
    assert _said(answer, own) == [], verb


@pytest.mark.parametrize("answer", ["I've booked the courier for Friday.", "I've ordered two sacks of Guji.",
                                    "I've paid the roaster's invoice."])
def test_rvw6_an_order_a_booking_or_a_payment_is_backed_by_a_send(answer):
    assert _said(answer, EMAIL_SENT) == []


def test_rvw6_a_write_of_another_order_word_does_not_back_the_claim():
    assert _said("I've paid the roaster's invoice.", ORDER_MADE) == [_named("paid")]
    assert _said("I've ordered two sacks of Guji.", BOOKING_MADE) == [_named("ordered")]


def test_rvw6_a_particle_no_family_names_stays_with_its_verb():
    """"kicked off" is the start family's "kicked"; "turned down" is in no family."""
    assert [verb for verb, _ in claims("I've kicked off the mission.")] == ["kicked"]
    assert _said("I've kicked off the mission.", MISSION_APPROVED) == []
    assert [verb for verb, _ in claims("I've turned down the old plan.")] == ["turned"]


# ── P256-FIX-RVW-6: each coordinated participle is a claim ─────────────────

def test_rvw6_created_and_sent_is_two_claims():
    answer = "I've created the task and sent it to Declan."                          # verified at b7fcc419f
    assert [verb for verb, _ in claims(answer)] == ["created", "sent"]
    assert _said(answer, CARD_MADE) == [_named("sent")]
    assert _said(answer, CARD_MADE, EMAIL_SENT) == []


@pytest.mark.parametrize("answer, verbs", [
    ("I've created the task and then sent it to Declan.", ["created", "sent"]),
    ("I've created the task, and also sent it to Declan.", ["created", "sent"]),
    ("The card has been created and sent to Declan.", ["created", "sent"]),
    ("I've updated the agent and turned off its heartbeat.", ["updated", "turned off"]),
    ("I've created the card and noted that you're happy with it.", ["created"]),     # the owner heard
    ("I've created the card and checked the board.", ["created"]),                   # a read is no work
    ("I've created the task. I've also sent it.", ["created", "sent"]),               # one claim each, no double
])
def test_rvw6_coordinated_participles(answer, verbs):
    assert [verb for verb, _ in claims(answer)] == verbs


def test_rvw6_a_plan_word_exempts_the_coordinated_participles_with_their_claim():
    assert not claims_work_done("Once I've created it and sent it, you'll see it in the Socials tab.")


# ── P256-FIX-RVW-6: the reply's own content exempts only its clause ────────

def test_rvw6_heres_the_update_then_a_claim_is_a_claim():
    answer = "Here's the update: the card has been approved."                         # verified at b7fcc419f
    assert [verb for verb, _ in claims(answer)] == ["approved"]
    assert _said(answer) == [_named("approved")]
    assert _said(answer, TASKS_LISTED) == []                 # N17-5: after a read, a third-person shape is a finding
    assert _said(answer, CARD_DONE) == []


@pytest.mark.parametrize("answer", [
    "I've drafted the email below, ready for you to copy.",
    "Here's the card I've created for the Analyst.",
    "Here are the posts I've drafted for Monday.",
    "I've made the following changes to the brief.",
    "Here's what the board says.",
])
def test_rvw6_the_clause_with_the_replys_own_content_is_still_no_claim(answer):
    assert not claims_work_done(answer)


def test_rvw6_a_past_time_still_exempts_its_sentence():
    assert not claims_work_done("As I've noted before, the price is £12.")
    assert not claims_work_done("Earlier today: the card has been approved.")


# ── P256-FIX-RVW-20: "was/were <verb>" is a claim again; history and denials are not ──

MAIL_DRAFTED = _composio("GMAIL_CREATE_EMAIL_DRAFT")
MAIL_SENT = _composio("GMAIL_SEND_EMAIL")


@pytest.mark.parametrize("answer, verb", [
    ("The card was approved.", "approved"),
    ("The email was sent to Declan.", "sent"),
    ("The ticket was created.", "created"),
    ("Both cards were approved.", "approved"),
])
def test_rvw20_a_simple_past_passive_with_no_past_time_is_a_claim(answer, verb):
    assert [said for said, _ in claims(answer)] == [verb]
    assert _said(answer) == [_named(verb)]
    assert _said(answer, TASKS_LISTED) == []                 # N17-5: after a read, a third-person shape is a finding


def test_rvw20_was_approved_is_backed_by_the_move_to_done():
    assert _said("The card was approved.", CARD_DONE) == []
    assert _said("The email was sent to Declan.", MAIL_SENT) == []


@pytest.mark.parametrize("answer", [
    "Ticket #1110 was completed at 03:04.",
    "You got it, Gerard! Ticket #1110 was completed at 03:04.",
    "Your two cards were approved this morning.",
    "The invoice was sent at 9am.",
    "The order was placed on Monday.",
    "The order was placed on 3 October.",
    "The order was placed on the 3rd.",
    "The post was published on 2026-10-03.",
    "The card was approved two days ago.",
])
def test_rvw20_a_simple_past_passive_with_a_past_time_is_history(answer):
    assert not claims_work_done(answer)
    assert _said(answer) == []


def test_rvw20_a_number_after_on_is_no_date():
    assert [verb for verb, _ in claims("The post was published on 2 channels.")] == ["published"]


@pytest.mark.parametrize("answer", [
    "Nothing was done yet. Approve the card above and I'll run it.",              # service.py's own wait line
    "No step was added: a step with no agent can't run.",
    "Nothing new was booked.",
    "No Socials post was saved in this run.",
    "That change was refused: no agent #99.",
    "The call was skipped: the owner hasn't approved it.",
    "Can you send me the coffees that were identified as low in stock?",
])
def test_rvw20_a_denial_a_refusal_or_a_relative_clause_is_no_claim(answer):
    assert not claims_work_done(answer)


# ── P256-FIX-RVW-20: replied, texted, shared, invited and refunded need their own write ──

def test_rvw20_a_mail_draft_does_not_back_replied():
    answer = "I've replied to Declan."
    assert [verb for verb, _ in claims(answer)] == ["replied"]
    assert _said(answer, MAIL_DRAFTED) == [_named("replied")]
    assert _said(answer, MAIL_SENT) == []
    assert _said(answer, _composio("GMAIL_REPLY_TO_THREAD")) == []


@pytest.mark.parametrize("answer, verb, own", [
    ("I've texted Declan the pickup time.", "texted", _composio("TWILIO_SEND_SMS")),
    ("I've DM'd Declan the pickup time.", "dm'd", _composio("SLACK_SEND_MESSAGE")),
    ("I’ve dm’d Declan the pickup time.", "dm'd", _composio("SLACK_SEND_MESSAGE")),
    ("I've shared the price list with Declan.", "shared", _composio("GOOGLEDRIVE_ADD_FILE_SHARING_PREFERENCE")),
    ("I've invited Declan to the board.", "invited", _platform("platform_invite_member", {"email": "d@example.com"})),
    ("I've refunded Rosie's second payment.", "refunded", _composio("STRIPE_CREATE_REFUND")),
])
def test_rvw20_a_send_verb_needs_its_own_write(answer, verb, own):
    assert [said for said, _ in claims(answer)] == [verb]
    assert _said(answer, MEMORY_STORED) == [_named(verb)]
    assert _said(answer, MAIL_DRAFTED) == [_named(verb)]
    assert _said(answer, own) == []


@pytest.mark.parametrize("answer", ["I've shared the price list with Declan.", "I've invited Declan to the board."])
def test_rvw20_a_send_backs_shared_and_invited(answer):
    assert _said(answer, MAIL_SENT) == []


def test_rvw20_a_send_does_not_back_refunded():
    assert _said("I've refunded Rosie.", MAIL_SENT) == [_named("refunded")]
    assert _said("I've refunded Rosie.", PAYMENT_MADE) == []


# ── P256-FIX-RVW-25: stative and elliptical passives, the first person's past, a comma list ──

SOCIAL_POSTED = POST_SUBMITTED
# (the sentence, its verb, a done write of its own kind)
ESCAPED = [
    ("The post got sent.", "sent", SOCIAL_POSTED),           # N17-5: the bare "<noun> <participle>." shape, Auto's own
    ("Email sent.", "sent", EMAIL_SENT),
    ("Posted!", "posted", SOCIAL_POSTED),
    ("I sent the email to Declan.", "sent", EMAIL_SENT),
    ("I created the task.", "created", CARD_MADE),
    ("I just paused the heartbeat.", "paused", HEARTBEAT_SET),
    ("I turned off the heartbeat for Scout.", "turned off", HEARTBEAT_SET),
    ("Heartbeat turned off.", "turned off", HEARTBEAT_SET),
    ("I dm’d Declan the pickup time.", "dm'd", _composio("SLACK_SEND_MESSAGE")),
    ("The post is published.", "published", SOCIAL_POSTED),
    ("The post is scheduled for 5pm.", "scheduled", REPORT_SCHEDULED),
    ("The ticket is created.", "created", CARD_MADE),
    ("It went out.", "sent", EMAIL_SENT),
    ("Your weekly report is set up.", "set up", REPORT_SCHEDULED),
    ("That's cancelled for you.", "cancelled", _platform("platform_cancel_mission", {"mission_id": "m-1"})),
]
ESCAPED_THIRD_PERSON = {"The post is published.", "The post is scheduled for 5pm.", "The ticket is created.",
                        "It went out.", "Your weekly report is set up.", "That's cancelled for you."}   # N17-5


@pytest.mark.parametrize("answer, verb, own", ESCAPED, ids=[case[0] for case in ESCAPED])
def test_rvw25_an_escaped_shape_is_a_claim_of_its_verb(answer, verb, own):
    assert [said for said, _ in claims(answer)] == [verb]
    assert _said(answer) == [_named(verb)]                           # no call at all: caught
    assert _said(answer, TASKS_LISTED) == ([] if answer in ESCAPED_THIRD_PERSON else [_named(verb)])   # a read's finding
    assert _said(answer, MEMORY_STORED) == [_named(verb)]
    assert _said(answer, own) == []


@pytest.mark.parametrize("answer, after_a_read", [("The task is done.", 0), ("That's done.", 0), ("Everything done.", 1)])
def test_rvw25_a_stative_done_says_only_that_work_happened(answer, after_a_read):
    assert [said for said, _ in claims(answer)] == ["done"]
    assert _said(answer) == [NOTHING_DONE_LINE]                      # no call at all: caught
    assert _said(answer, TASKS_LISTED) == ([NOTHING_DONE_LINE] if after_a_read else [])   # N17-5: a bare one is Auto's
    assert _said(answer, CARD_DONE) == []


def test_rvw25_the_first_persons_past_coordinates_its_participles():
    answer = "I emailed the supplier and created the ticket."
    assert [verb for verb, _ in claims(answer)] == ["emailed", "created"]
    assert _said(answer, MEMORY_STORED) == [_named("emailed or created")]
    assert _said(answer, CARD_MADE) == [_named("emailed")]
    assert _said(answer, CARD_MADE, EMAIL_SENT) == []


@pytest.mark.parametrize("answer, verbs", [
    ("I've created the ticket, emailed Sam.", ["created", "emailed"]),
    ("I've created the ticket; then emailed Sam.", ["created", "emailed"]),
    ("I've created the ticket, and also emailed Sam.", ["created", "emailed"]),
    ("I've created the ticket, Sam emailed.", ["created"]),                    # a new clause, its own subject
    ("I've created the card, called 'Wholesale reply'.", ["created"]),          # a state, not a claim
])
def test_rvw25_a_participle_continuing_the_list_after_a_comma_is_a_claim(answer, verbs):
    assert [verb for verb, _ in claims(answer)] == verbs


def test_rvw25_the_comma_claim_no_write_backs_is_named():
    answer = "I've created the ticket, emailed Sam."
    assert _said(answer, CARD_MADE) == [_named("emailed")]
    assert _said(answer, CARD_MADE, EMAIL_SENT) == []


@pytest.mark.parametrize("answer", [
    "It went out this morning.",                                           # history
    "I sent the invoice at 9am.",
    "Once the post is published, you'll see it in the Socials tab.",       # a plan word
    "I'll let you know as soon as the agent is installed and ready for the next steps!",
    "Here are the posts that are scheduled for Monday.",                   # the reply's own content
    "The email that I sent to Declan bounced.",                            # a relative clause
    "Nothing is scheduled for Monday.",                                    # a denial
    "No post is published yet.",
    "Nothing got sent.",
    "Not sent yet.",
    "Invoice HL-2026-0142 isn’t done.",
    "The playbook is based on your notes.",                                # a state
    "The heartbeat is set to every 30 minutes.",
    "Scout's heartbeat is paused.",
    "Workspace is disabled.",
    "Rosa has ordered.",                                                   # hers, not the writer's
    "I found that the Harvest Club boxes are posted on Monday, October 5th.",   # what a read found (F316)
    "The board shows the post is published.",
    "I made a mistake in the update task status call.",                    # a slip owned
    "I wanted to check with you first.",                                   # no family: no simple past claim
])
def test_rvw25_an_exempt_sentence_of_the_new_shapes_is_no_claim(answer):
    assert not claims_work_done(answer)
    assert _said(answer) == []


def test_rvw25_an_agents_run_reads_only_its_first_person():
    assert claims("The post is published. Email sent.", first_person=True) == []
    assert [verb for verb, _ in claims("I sent the email to Declan.", first_person=True)] == ["sent"]


# The sweep: every string literal of orchestrator/tests and tests/sim (docstrings aside) through the
# claims() before RVW-25 and after lists these new claims, each (its sentence, its verb); none is an
# answer a test asserts clean. Most are no answer at all (a card's brief, a tool's result, a board
# note, an owner's question); the one cleared answer, F351's 38dc9b59 ("…is done and saved to
# Deliverables."), FX-007 had lost and F351 first caught: it is a claim again. No claim was lost.
SWEPT = [
    ("started.", "started"), ("COMPLETED.", "completed"), ("I edited notes.", "edited"), ("Saved.", "saved"),
    ("It never ran, so it is closed as cancelled.", "closed"), ("Sent.", "sent"),
    ("The offer overview is saved as deliverables/sessions/980/christmas-box-cafe-offer-overview.", "saved"),
    ("Approved.", "approved"), ("steps: a playbook is created with no steps", "created"), ("All paid.", "paid"),
    ("Task ID 1099 is done, and so is 1093.", "done"),
    ("I made an empty playbook instead of installing it: nothing was installed.", "made"),
    ("I made an empty playbook", "made"),   # N15-1: "If they ask whether setup is done" is a condition, no claim
    ("Ticket 0001 ('Christmas gift box labels - 40 of them') is closed", "closed"),
    ("The task is done when all three coffees are worked out.", "done"), ("Supplier replied.", "replied"),
    ("Ticket 0059 started.", "started"), ("It is fixed now.", "fixed"), ("The mission completed.", "completed"),
    ("Run my saved playbook called New Cafe Onboarding, the one I set up.", "set up"),
    ("is cancelled now", "cancelled"), ("The file is saved as workspace/decafcolombiamargin.", "saved"),
    ("Mission 0033 is set up, and it will stop after every step for your OK.", "set up"),
    ("The PDF I generated is still not checked, and it doesn't use your layout or logo.", "generated"),
    ("I generated a PDF, but I couldn't open it, and it has no logo.", "generated"),
    ("I generated the quote PDF, but I haven't opened it, and it probably isn't on your brand kit yet.", "generated"),
    ("Thanks Rosa, the 12 kg is booked for Friday.", "booked"),
    ("Alright, the invoice for Lantern Kitchen is done and saved to Deliverables.", "done"),
    ("Alright, the invoice for Lantern Kitchen is done and saved to Deliverables.", "saved"),
    ("Document generated.", "generated"), ("It checks which version of PIL is installed, and prints it.", "installed"),
    ("Noted.", "noted"), ("only the two 365-day rows are deleted", "deleted"),
    ("WIDGETORIGINALLOWLIST is deleted — a merchant origin belongs on the merchant's key", "deleted"),
    ("Your workspace is set up: Atlas Research Agent is configured on gpt-4o.", "set up"),
    ("Say done.", "done"), ("the handler is closed", "closed"), ("Went out", "sent"),
    ("Countdown to Lisbon: 4 posts went out this week", "sent"), (") is deleted for good", "deleted"),
    ("approved.", "approved"), ("The roastery is closed on", "closed"),
    ("0001 is approved and moved to Done.", "approved"), ("0001 is approved and moved to Done.", "moved"),
    ("Stored.", "stored"), ("Done, it's approved.", "approved"), ("I moved card 0422 to Done.", "moved"),
    ("Noted!", "noted"), ("Created.", "created"),
]


@pytest.mark.parametrize("sentence, verb", SWEPT)
def test_rvw25_the_sweeps_new_claims_are_claims(sentence, verb):
    assert verb in [said for said, _ in claims(sentence)]


# ── P256-FIX-RVW-34: backed claims are no longer denied ─────────────────────
# RVW-25's sweep listed three owner's questions as claims; a question reports nothing, so they are none.
SWEPT_QUESTIONS = [
    "How many Harvest Club boxes went out in September 2026?",
    "How many Harvest Club boxes are scheduled to go out on Monday, October 5th, 2026?",
    "How many club boxes went out late in September?",
]

READ_0365 = _platform("platform_get_task", {"task_id": "#0365"})
CALENDAR_EVENT = _composio("GOOGLECALENDAR_CREATE_EVENT")
# (the sentence, the turn's calls): each was denied at 80cd928ec, each is backed now.
BACKED_RVW34 = [
    ("I've made a note of that.", [_platform("platform_store_memory", {"content": "Declan prefers Fridays."})]),
    ("I've made Scout the owner of #0931.", [_platform("platform_assign_task", {"task_id": "#0931", "agent_id": 12})]),
    ("I've made the changes you asked for.", [_platform("platform_update_task", {"task_id": 451, "title": "Reply"})]),
    ("I've scheduled the call with Declan for Friday.", [CALENDAR_EVENT]),
    ("I've booked the call with Declan for Friday.", [CALENDAR_EVENT]),
    ("Can you confirm the invoice was paid?", []),
    ("Should I tell Declan it's been sent?", []),
    ("Mission #0365 has been cancelled.", [READ_0365]),
    ("Mission #0365 was cancelled.", [READ_0365]),
]


@pytest.mark.parametrize("answer, calls", BACKED_RVW34, ids=[answer for answer, _ in BACKED_RVW34])
def test_rvw34_a_backed_claim_gets_no_line_and_no_nudge(answer, calls):
    assert _said(answer, *calls) == []
    assert nudged(answer, *calls) is None


@pytest.mark.parametrize("question", SWEPT_QUESTIONS)
def test_rvw34_a_question_is_no_claim(question):
    assert claims(question) == []


def test_rvw34_made_says_its_family_by_what_follows_it():
    store, assign = BACKED_RVW34[0][1], BACKED_RVW34[1][1]
    assert _said("I've made a note of that.", *assign) == [_named("made")]   # a note is noted
    assert _said("I've made Scout the owner of #0931.", *store) == [_named("made")]
    assert nudged("I've made the changes you asked for.") is None           # no family: read, never taken out
    assert unbacked("I've made the changes you asked for.") == ["made"]
    assert _said("I've made the changes you asked for.") == []                # so no line (N15-2)


def test_rvw34_made_a_new_playbook_with_no_create_is_still_caught():
    answer = "I've made a new playbook."
    assert _said(answer) == _said(answer, MEMORY_STORED) == [_named("made")]
    assert nudged(answer) == nudged(answer, MEMORY_STORED) == "made"
    assert _said(answer, _platform("platform_create_playbook", {"name": "Weekly"})) == []


def test_rvw34_a_calendar_write_backs_only_scheduled_and_booked():
    assert _said("I've emailed Declan the invite.", CALENDAR_EVENT) == [_named("emailed")]
    assert _said("I've scheduled the call with Declan for Friday.", MEMORY_STORED) == [
        _named("scheduled")]


def test_rvw34_a_claim_before_a_clause_break_is_still_a_claim_in_a_question():
    answer = "I've sent it to Declan, want me to chase him?"
    assert [verb for verb, _ in claims(answer)] == ["sent"]
    assert _said(answer) == [_named("sent")]


def test_rvw34_a_status_with_no_read_of_its_number_is_still_caught():
    answer = "Mission #0365 has been cancelled."
    assert _said(answer) == [_named("cancelled")]
    assert nudged(answer) == "cancelled"
    assert nudged(answer, _platform("platform_get_task", {"task_id": "#0366"})) == "cancelled"   # another card's read
    assert nudged("I've cancelled #0365.", READ_0365) == "cancelled"        # the writer's own work, not a finding


# ── P256-FIX-RVW-35: common claim shapes still escaped ───────────────────────
# Each was no claim at 80cd928ec (claims == [] with no write): a looking verb before the done verb,
# list shapes, a subject-less verb, "went ahead and", an adverb after "been", "<noun>'s been".
CARD_0931_DONE = _platform("platform_update_task_status", {"task_id": "#0931", "status": "done"})
ORDER_CANCELLED = _composio("SHOPIFY_CANCEL_ORDER")
READ_0931 = _platform("platform_get_task", {"task_id": "#0931"})
# (the answer, its claims' verbs, its backing writes)
ESCAPED_RVW35 = [
    ("I've reviewed and approved the card.", ["approved"], [CARD_DONE]),
    ("I've checked the board and approved #0931.", ["approved"], [CARD_0931_DONE]),
    ("I've confirmed and sent the email.", ["sent"], [EMAIL_SENT]),
    ("I checked the board and approved #0931.", ["approved"], [CARD_0931_DONE]),
    ("- Task created\n- Email sent", ["created", "sent"], [CARD_MADE, EMAIL_SENT]),
    ("✅ Task created\n✅ Email sent to Sam", ["created", "sent"], [CARD_MADE, EMAIL_SENT]),
    ("Done:\n- Task created", ["", "created"], [CARD_MADE]),
    ("Done — email's out to Declan and the ticket's closed.", [""], [EMAIL_SENT]),
    ("Sent the email to Declan.", ["sent"], [EMAIL_SENT]),
    ("Okay, sent the email to Declan.", ["sent"], [EMAIL_SENT]),
    ("Just sent the email.", ["sent"], [EMAIL_SENT]),
    ("- Sent Declan the invoice.", ["sent"], [EMAIL_SENT]),
    ("Cancelled the order for you.", ["cancelled"], [ORDER_CANCELLED]),
    ("Card #0931 moved to Done.", ["moved"], [CARD_0931_DONE]),
    ("I went ahead and cancelled the order.", ["cancelled"], [ORDER_CANCELLED]),
    ("Task #0931 has been successfully created.", ["created"], [CARD_MADE]),
    ("The card's been approved.", ["approved"], [CARD_DONE]),
]
THIRD_PERSON_RVW35 = ("Task #0931 has been successfully created.", "The card's been approved.")   # N17-5: read's finding


@pytest.mark.parametrize("answer, verbs, own", ESCAPED_RVW35, ids=[case[0] for case in ESCAPED_RVW35])
def test_rvw35_an_escaped_shape_is_a_claim(answer, verbs, own):
    assert [verb for verb, _ in claims(answer)] == verbs
    assert _said(answer) == [_in_place_line(verbs)]                  # no call at all: caught
    assert _said(answer, TASKS_LISTED) == ([] if answer in THIRD_PERSON_RVW35 else [_in_place_line(verbs)])
    assert nudged(answer) == (verbs[0] or "done")
    assert _said(answer, *own) == []
    assert nudged(answer, *own) is None


@pytest.mark.parametrize("answer, verbs, _own", [case for case in ESCAPED_RVW35 if case[1][0]],
                         ids=[case[0] for case in ESCAPED_RVW35 if case[1][0]])
def test_rvw35_a_saved_memory_does_not_back_the_claim(answer, verbs, _own):
    assert _said(answer, MEMORY_STORED) == [_named(" or ".join(dict.fromkeys(verbs)))]


def test_rvw35_one_list_line_its_write_backs_leaves_the_other_named():
    assert _said("✅ Task created\n✅ Email sent to Sam", CARD_MADE) == [_named("sent")]
    assert _said("- Task created\n- Email sent", EMAIL_SENT) == [_named("created")]


def test_rvw35_done_before_a_claim_of_its_sentence_introduces_it():
    assert [verb for verb, _ in claims("Done — I've created the agent.")] == ["created"]
    assert [verb for verb, _ in claims("Done, it's approved.")] == ["approved"]
    assert nudged("Done — I've created a new agent called REPORT GENERATOR.") == "created"


def test_rvw35_a_numbered_card_a_read_of_the_turn_named_is_what_the_read_found():
    answer = "Card #0931 moved to Done."
    assert _said(answer, READ_0931) == []
    assert nudged(answer, READ_0931) is None
    assert nudged(answer, _platform("platform_get_task", {"task_id": "#0932"})) == "moved"   # another card's read


@pytest.mark.parametrize("answer", [
    "I've reviewed the card.",                                              # looking alone is no claim
    "I've checked the board and there's nothing to approve.",
    "Scheduled for Monday.",                                                # a verb first with no object: a state
    "Approved by Rosa.",
    "Approved cards stay in Done.",
    "Completed",                                                            # a label
    "Renamed Operator",
    "Step 2 saved this as stock_report.",                                   # a step, no card's number
    "- Not sent yet",                                                       # a denial
    "Nothing's been sent yet.",
    "Scout is the agent who's been assigned.",                              # a relative clause
    "Done-for-you setup is ready.",
    "Sent the invoice to Rosa yesterday.",                                  # history
    "I checked and approved it at 9am.",
    "Here's what I've reviewed and approved below.",                        # the reply's own content
    "Once I've checked and approved it, I'll tell you.",                    # a plan word
    "> Sent the email to Declan.",                                          # quoted
    "```\n✅ Email sent to Sam\n```",                                        # fenced
    "Can you confirm you sent the email to Declan?",                        # a question
])
def test_rvw35_an_exempt_sentence_of_the_new_shapes_is_no_claim(answer):
    assert claims(answer) == []
    assert _said(answer) == []


# ── P256-FIX-RVW-36: a registered read is a read; a verb in no family needs a write its stem names ──

SKILL_LOADED = _platform("platform_load_skill", {"skill_name": "linkedin-posts"})
REGISTERED_READS = ["platform_load_skill", "platform_fleet_status", "platform_harness_status", "platform_graph_stats",
                    "platform_codegraph_call_graph"]
# (the sentence, the write that backs it): each was backed by a saved memory at 80cd928ec.
NO_FAMILY_RVW36 = [
    ("I've rejected the card.", _platform("platform_reject_mission", {"mission_id": "m-1"})),
    ("I've stopped the channel.", _platform("platform_stop_channel", {"channel": "slack"})),
    ("I've revoked the key.", _platform("platform_revoke_api_key", {"key_id": 7})),
    ("I've reassigned the card to OPS.", _platform("platform_assign_task", {"task_id": "#0931", "agent_name": "OPS"})),
    ("I've charged the customer.", ("composio_execute", {"action": "STRIPE_CREATE_CHARGE", "params": {}}, OK)),
    ("I've turned down the old plan.", _platform("platform_reject_mission", {"mission_id": "m-1"})),
]
ONLY_WORK_HAPPENED = ["Done.", "The new page is now live.", "You should now see it on your board."]


@pytest.mark.parametrize("action", REGISTERED_READS)
def test_rvw36_a_registered_read_is_a_read_receipt(action):
    assert [(r["kind"], r["status"]) for r in _receipts(_platform(action))] == [(READ, DONE)]


def test_rvw36_a_loaded_skill_backs_no_post_said_live():
    answer = "Your post is now live on LinkedIn."
    (loaded,) = _receipts(SKILL_LOADED)
    assert loaded["kind"] == READ
    assert _said(answer) == _said(answer, SKILL_LOADED) == [NOTHING_DONE_LINE]   # night 18: a skill observed nothing
    assert nudged(answer, SKILL_LOADED) == "done"                       # …and the loaded skill still backs nothing


@pytest.mark.parametrize("answer, backing", NO_FAMILY_RVW36, ids=[answer for answer, _ in NO_FAMILY_RVW36])
def test_rvw36_a_verb_in_no_family_needs_a_write_its_stem_names(answer, backing):
    (verb, _sentence), = claims(answer)
    assert unbacked(answer, MEMORY_STORED) == [verb] and unbacked(answer, backing) == []
    assert nudged(answer, MEMORY_STORED) is None                       # no family: never the nudge's…
    assert _said(answer, MEMORY_STORED) == _said(answer, backing) == []    # …nor taken out: no line (N15-2)


def test_rvw36_a_saved_memory_backs_no_rejection_and_the_reject_write_does():
    assert unbacked("I've rejected the card.", MEMORY_STORED) == ["rejected"]
    assert unbacked("I've rejected the card.", NO_FAMILY_RVW36[0][1]) == []
    assert _said("I've rejected the card.", MEMORY_STORED) == []           # read, never taken out (N15-2)


@pytest.mark.parametrize("answer", ONLY_WORK_HAPPENED)
def test_rvw36_a_claim_that_says_only_work_happened_takes_any_done_write(answer):
    assert _said(answer, MEMORY_STORED) == [] and _said(answer) == [NOTHING_DONE_LINE]      # no call at all: caught
    assert _said(answer, SKILL_LOADED) == [NOTHING_DONE_LINE]           # night 18: a loaded skill found nothing
