"""P256-FIX-RVW-37: the four claims a verb's family alone cannot read (``consumers/chatbot/claims_by_kind``).

The rows FX-007 inverted are in ``test_prd256_families_deleted``; this file holds what each rule must
NOT catch (a truthful answer never gets the not-done line) and the writes that back each one: a
document made through Composio or edited in place, a note in the documents, a send that waits on
the owner's card.
"""
from __future__ import annotations

import pytest

from consumers.chatbot.claims_by_kind import STEPS_LABEL, TEAM_LABEL, document_backs, promised_now, said_in
from tests.helpers_receipts_rule import call, line, nudged

MEMORY = "platform_store_memory"
SEND_WAITING = ("composio_execute", {"action": "GMAIL_SEND_EMAIL", "params": {}},
                {"success": False, "requires_confirmation": True, "grant_id": 42,
                 "message": "Approve sending the email to Declan."})


def _composio(slug):
    return ("composio_execute", {"action": slug, "params": {}}, {"successful": True})


# ── F324: what the team knows is a claim only when the reply says it told them ──

@pytest.mark.parametrize("said", [
    "Everyone knows the price is £12.",
    "Your agents know how to read documents.",
    "The team knows about the change, as I told them in the document.",
    "Your agents don't know yet.",
    "I'll write it into a note so every agent knows.",
])
def test_what_the_team_knows_is_no_claim_unless_the_reply_told_them(said):
    assert not [label for label, _ in said_in(said) if label == TEAM_LABEL]
    assert nudged(said, MEMORY) is None and line(said, MEMORY) is None


@pytest.mark.parametrize("said", [
    "This will ensure that all agents, including the Support Agent, are aware of this change going forward.",
    "I've stored it in my memory so that all agents know.",
    "Everyone now knows the new terms.",
    "Your whole team has been told.",
])
def test_the_team_said_told_needs_a_write_other_than_memory(said):
    assert nudged(said, MEMORY) == TEAM_LABEL
    assert nudged(said, MEMORY, "platform_upload_document") is None


# ── F351: a document said made needs a write that makes one ──────────────────

@pytest.mark.parametrize("said, calls", [
    ("I've created the report in Notion.", [_composio("NOTION_CREATE_PAGE")]),
    ("I've saved the letter to your Drive.", [_composio("GOOGLEDRIVE_UPLOAD_FILE")]),
    ("I've saved the letter to your Drive.", [_composio("GOOGLEDRIVE_CREATE_FILE_FROM_TEXT")]),
    ("I've saved the report with your changes.", ["platform_update_document"]),
    ("I've saved the letter with your changes.", ["workspace_edit_file"]),
    ("I've generated the PDF for Salt Kitchen.", ["generate_document"]),
])
def test_a_document_made_through_any_writer_backs_the_claim(said, calls):
    assert nudged(said, *calls) is None and line(said, *calls) is None


@pytest.mark.parametrize("said, calls", [
    ("I've created the report in Notion.", ["platform_create_task"]),
    ("I've saved the letter to your Drive.", [MEMORY, "platform_create_task"]),
])
def test_a_write_that_makes_no_document_backs_no_document(said, calls):
    assert nudged(said, *calls) is not None and line(said, *calls) is not None


@pytest.mark.parametrize("said, verb, calls", [
    ("Done — I've created a new agent called REPORT GENERATOR.", "created", ["platform_create_agent"]),
    ("The report shows two overdue cards and the follow-up task has been created.", "created",
     ["platform_create_task"]),
    ("The report request has been saved.", "saved", ["platform_create_task"]),
    ("I've created the report task for the Analyst.", "created", ["platform_create_task"]),
    ("I've saved your report preferences.", "saved", ["platform_update_auto_reporting_prefs"]),
])
def test_a_document_word_that_is_not_what_was_made_asks_no_document(said, verb, calls):
    assert document_backs(verb, said) is None
    assert nudged(said, *calls) is None and line(said, *calls) is None


def test_a_documents_own_phrase_is_its_subject():
    said = ("The letter to Maya Osei at Quay Coffee House regarding the payment terms update has been generated "
            "and saved to Deliverables.")
    assert document_backs("generated", said) is not None
    assert nudged(said, "platform_create_task") == "generated"
    assert nudged(said, "generate_document") is None


# ── F308: each step said to wait ─────────────────────────────────────────────

def test_steps_said_to_wait_read_only_a_claim_of_how_the_steps_run():
    assert [label for label, _ in said_in("Each step will pause for your approval.")] == [STEPS_LABEL]
    assert said_in("Should each step pause for your approval?") == []
    assert said_in("Each step will not pause: they run on their own.") == []
    assert said_in("The first step drafts the email.") == []


# ── F261-A: work announced now ───────────────────────────────────────────────

def test_work_announced_for_later_or_on_a_card_is_no_promise_now():
    assert promised_now("I will now send the email once you approve.") == []
    assert promised_now("Would you like me to send it now?") == []
    said = "I will now send this updated brief to the agent."
    assert nudged(said, SEND_WAITING) is None                                 # the send waits on its card
    assert nudged(said) == "sent"


def test_work_announced_now_reads_its_document_noun():
    said = "I will now create the letter to Maya."
    assert nudged(said, "platform_create_task") == "created"                  # a card is no letter
    assert nudged(said, "generate_document") is None


def test_a_mission_that_checks_each_step_backs_the_claim_on_its_receipt():
    checked = call("platform_create_mission", {"goal": "x"}, {"success": True, "mission_id": "x",
                                                                "checks_each_step": True})
    assert nudged("Each step will pause for your approval.", checked) is None
