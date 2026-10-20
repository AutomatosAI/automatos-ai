"""PRD-256 P256-FIX-RVW-5 — a Composio call backs only the claims its own action says.

A composio_execute call was recorded under the dispatcher's name, so its receipt was a write
whatever ran (GMAIL_FETCH_EMAILS was a DONE write), and ``claims_backed._integration`` let any
Composio receipt back a claim of any family: composio_execute fetching mail + "I've sent the
reply to Declan." had no line and no nudge. The reverse held too: a per-action tool recorded
under its own name (HUBSPOT_CREATE_CONTACT) matched no family, so "I've created the contact"
was denied.

Now the call is recorded under the action that ran (composio_execute's ``action``), a slug with
a read word is a read receipt, and a write backs only the claims the words of its slug say,
through the same FAMILIES. The cap still counts every composio_execute call as one tool.
"""
from __future__ import annotations

import inspect

import pytest

import consumers.chatbot.claims_backed as claims_backed
from consumers.chatbot.receipts import DONE, READ, WRITE, build_receipts, honesty_lines, unbacked_claim
from modules.tools.execution.composio_action import action_that_ran, is_slug, slug_reads, slug_stems
from modules.tools.execution.tool_execution_tracker import ToolExecutionTracker
from modules.tools.execution.turn_account import is_read
from tests.helpers_receipts_rule import line, unbacked

OK = {"successful": True}
SAID_SENT = "I've sent the reply to Declan."


def _named(verbs):
    return f"Just to be clear: I haven't {verbs} anything in this reply. Ask me again if you want it done."


def _composio(action, params=None, result=OK):
    return ("composio_execute", {"action": action, "params": params or {}}, result)


def _tracker(*calls):
    tracker = ToolExecutionTracker()
    for tool, args, result in calls:
        tracker.record_outcome(tool, args, result)
    return tracker


def _receipts(*calls):
    return build_receipts(_tracker(*calls))


def _said(answer, *calls):
    """P256-FIX-N15-2: the not-done line the answer carries in place of its claims, as a list ([] when none)."""
    in_place = line(answer, *calls)
    return [in_place] if in_place else []


MAIL_FETCHED = _composio("GMAIL_FETCH_EMAILS", {"max_results": 5})
MAIL_SENT = _composio("GMAIL_SEND_EMAIL", {"to": "declan@example.com"})
CONTACT_MADE = ("HUBSPOT_CREATE_CONTACT", {"email": "declan@example.com"}, OK)


# ── the story's four sentences ───────────────────────────────────────────────

def test_a_mail_fetch_is_a_read_and_backs_no_send():
    receipts = _receipts(MAIL_FETCHED)
    assert [(r["action"], r["kind"], r["status"]) for r in receipts] == [("GMAIL_FETCH_EMAILS", READ, DONE)]
    assert honesty_lines(receipts, SAID_SENT) == []                      # P256-FIX-N15-2: nothing above
    assert _said(SAID_SENT, MAIL_FETCHED) == [_named("sent")]           # the line in the claim's place


def test_a_mail_send_backs_the_send():
    receipts = _receipts(MAIL_SENT)
    assert [(r["action"], r["kind"], r["status"]) for r in receipts] == [("GMAIL_SEND_EMAIL", WRITE, DONE)]
    assert honesty_lines(receipts, SAID_SENT) == [] and _said(SAID_SENT, MAIL_SENT) == []


def test_a_per_action_create_backs_the_create():
    assert _said("I've created the contact.", CONTACT_MADE) == []


def test_a_send_does_not_back_a_delete():
    assert _said("I've deleted the draft.", MAIL_SENT) == [_named("deleted")]


# ── the nudge reads the same receipts ────────────────────────────────────────

def test_the_loop_nudges_a_send_claimed_after_a_fetch():
    assert unbacked_claim(SAID_SENT, _tracker(MAIL_FETCHED).outcomes) == "sent"
    assert unbacked_claim(SAID_SENT, _tracker(MAIL_SENT).outcomes) is None


# ── a write backs only the claims its slug's words say ───────────────────────

@pytest.mark.parametrize("call, answer", [
    (_composio("GMAIL_REPLY_TO_THREAD"), SAID_SENT),
    (_composio("GMAIL_FORWARD_MESSAGE"), "I've forwarded it to Declan."),
    (_composio("INSTAGRAM_PUBLISH_MEDIA"), "I've posted it to Instagram."),
    (_composio("LINKEDIN_CREATE_LINKED_IN_POST"), "I've posted it to LinkedIn."),
    (_composio("GOOGLECALENDAR_UPDATE_EVENT"), "I've updated the event."),
    (_composio("GOOGLECALENDAR_DELETE_EVENT"), "I've deleted the event."),
    (("SHOPIFY_CREATE_ORDER", {}, OK), "I've created the order."),
    (_composio("gmail-send-email"), SAID_SENT),                  # written the way Composio writes it
], ids=["reply", "forward", "publish", "linkedin-post", "update", "delete", "per-action-create", "lower-case"])
def test_a_write_backs_the_claim_its_verb_says(call, answer):
    assert _said(answer, call) == []


@pytest.mark.parametrize("call, answer, verb", [
    (_composio("GMAIL_CREATE_EMAIL_DRAFT"), "I've sent the email to Declan.", "sent"),   # "email" is no "mail"
    (_composio("GMAIL_ADD_LABEL_TO_EMAIL"), SAID_SENT, "sent"),
    (CONTACT_MADE, "I've updated the contact.", "updated"),
    (_composio("GOOGLECALENDAR_CREATE_EVENT"), "I've deleted the old event.", "deleted"),
], ids=["draft-is-no-send", "label-is-no-send", "create-is-no-update", "create-is-no-delete"])
def test_a_write_does_not_back_another_verb(call, answer, verb):
    assert _said(answer, call) == [_named(verb)]


def test_a_claim_with_no_family_needs_the_composio_write_its_stem_names():
    """P256-FIX-RVW-36: a send backed "set" (no family) until any write stopped backing such a verb;
    a write whose slug says SET does. P256-FIX-N15-2: a verb in no family is read, never taken out, so
    it draws no line."""
    assert unbacked("I've set it up for you.", MAIL_SENT) == ["set"]
    assert unbacked("I've set it up for you.", _composio("SLACK_SET_USER_PRESENCE")) == []
    assert unbacked("I've set it up for you.", MAIL_FETCHED) == ["set"]
    assert _said("I've set it up for you.", MAIL_SENT) == _said("I've set it up for you.", MAIL_FETCHED) == []


# ── read words are whole words ───────────────────────────────────────────────

@pytest.mark.parametrize("slug, reads", [
    ("GMAIL_FETCH_EMAILS", True), ("GMAIL_LIST_THREADS", True), ("GOOGLECALENDAR_FIND_EVENT", True),
    ("GOOGLEDRIVE_DOWNLOAD_FILE", True), ("SHOPIFY_LIST_ORDERS", True),
    ("GMAIL_REPLY_TO_THREAD", False),                              # "thread" holds "read", it is no read
    ("GMAIL_SEND_EMAIL", False), ("HUBSPOT_CREATE_CONTACT", False),
])
def test_a_slug_reads_by_its_whole_words(slug, reads):
    assert is_slug(slug)
    assert slug_reads(slug) is reads
    assert is_read(slug) is reads


def test_platform_calls_are_read_as_before():
    assert not is_slug("platform_list_tasks") and is_read("platform_list_tasks")
    assert not is_slug("composio_execute") and not is_read("composio_execute")


def test_the_slug_is_the_action_composio_execute_named():
    assert action_that_ran("composio_execute", {"action": "gmail-send-email"}) == "GMAIL_SEND_EMAIL"
    assert action_that_ran("composio_execute", {"action_name": "GMAIL_FETCH_EMAILS"}) == "GMAIL_FETCH_EMAILS"
    assert action_that_ran("composio_execute", {"params": {}}) is None
    assert action_that_ran("platform_execute", {"action": "GMAIL_SEND_EMAIL"}) is None
    assert slug_stems("GMAIL_SEND_EMAIL") == ("send_", "email_")


# ── the tracker: recorded by what ran, counted as one tool ───────────────────

def test_the_tracker_records_the_action_that_ran_and_counts_the_dispatcher():
    tracker = _tracker(MAIL_FETCHED, MAIL_SENT)
    assert [action for action, _params, _result in tracker.outcomes] == ["GMAIL_FETCH_EMAILS", "GMAIL_SEND_EMAIL"]
    assert tracker.succeeded == {"GMAIL_FETCH_EMAILS", "GMAIL_SEND_EMAIL"}
    tracker.record_execution(*MAIL_FETCHED[:2])
    tracker.record_execution(*MAIL_SENT[:2])
    assert tracker.tool_counts == {"composio_execute": 2}               # the cap is unchanged


def test_a_refused_composio_send_is_named_by_its_action():
    refused = _composio("GMAIL_SEND_EMAIL", result={"successful": False, "error": "no recipient"})
    tracker = _tracker(refused)
    assert tracker.failed == {"GMAIL_SEND_EMAIL"}
    (tried,) = honesty_lines(build_receipts(tracker), SAID_SENT)          # above: the refused send alone
    assert tried == "I tried to send the email and it didn't go through: no recipient."     # N17-5: its slug's words
    assert _said(SAID_SENT, refused) == [
        "Just to be clear: I haven't sent anything in this reply; I tried to send the email and it didn't go through. "
        "Ask me again if you want it done."]


def test_a_skipped_composio_call_is_named_by_its_action():
    tracker = ToolExecutionTracker()
    args = MAIL_SENT[1]
    tracker.record_execution("composio_execute", args)
    skip, _why = tracker.should_skip_execution("composio_execute", args)
    assert skip and [action for action, _params, _why in tracker.skipped] == ["GMAIL_SEND_EMAIL"]


# ── a customer draft's check reads the run's slugs the same way ──────────────

def test_a_customer_draft_is_backed_by_the_runs_composio_send():
    from services.draft_guides import check_before_sending

    brief = "Email from Rosie Tanner, club member: she was charged twice. Please draft a reply."
    draft = "Dear Rosie, I have sent you the corrected invoice."
    assert check_before_sending(brief, draft, ["GMAIL_SEND_EMAIL"]) is None
    assert "says something was sent" in check_before_sending(brief, draft, ["GMAIL_FETCH_EMAILS"])
    assert "says something was sent" in check_before_sending(brief, draft, ["GMAIL_CREATE_EMAIL_DRAFT"])


# ── the blanket pass is gone ─────────────────────────────────────────────────

def test_the_blanket_integration_pass_is_deleted():
    assert not hasattr(claims_backed, "_integration") and not hasattr(claims_backed, "COMPOSIO")
    assert "_integration" not in inspect.getsource(claims_backed.unbacked_claims)
