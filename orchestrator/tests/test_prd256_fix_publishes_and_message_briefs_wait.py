"""PRD-256 P256-FIX-RVW-19 (Decisions D1, D7): a channel's publish action and a brief that messages wait for a person.

D7 is "send/publish/order". The click gate (``owner_only.is_composio_send``) asked only for a
slug carrying a send word, and TIKTOK_UPLOAD_VIDEO and YOUTUBE_UPLOAD_VIDEO carry none, though
the Socials channel registry classes them ``publish``: an agent on Auto's ticket, or Auto in
the owner's chat, published a video with no card. Now every action the registry classes as a
seeded channel's publish step asks too (the slugs stay in ``modules/socials/channel_adapters.py``).

And a ticket filed from a person's chat whose brief sends is reviewed by a person
(``brief_sends``); "Email the supplier", "Post the spring menu" and "Reply to Declan" kept
``review_mode`` auto. A message verb now counts in the brief's verb position, while a brief
that only drafts the message keeps auto. P256-FIX-RVW-27: the verb is read past a lead-in
("Please email", "Also post", "Can you"), after "ask/get/have <someone> to", and the
outbound verbs and phrases (contact, notify, reach out, let <x> know, get back to, …) count.

The gate's chain is the RVW-3 test's: the real owner's-click gate over the real
ComposioToolExecutor.execute; the SDK call is recorded and must not happen.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from modules.socials.capabilities import SEEDED_CHANNELS
from modules.tools.discovery.agent_sends import AGENT_SEND
from modules.tools.discovery.brief_sends import REVIEW_HELD, reviewed_by_a_person
from modules.tools.discovery.owner_only import is_composio_send
from tests import test_prd256_fix_orders_and_resolved_sends_wait_for_the_click as sends

desk = sends.desk   # the workspace, Auto's ticket and the recorded SDK (the RVW-3 fixture)

PUBLISHES = sorted(set().union(*(adapter.publish_actions for adapter in SEEDED_CHANNELS.values())))
VIDEO = {"caption": "The spring menu is here", "file_to_upload": "spring-menu.mp4"}
OWNERS_CHAT = {"driving_user_id": "7", "conversation_id": "chat-1"}


def test_the_registry_classes_both_video_uploads_as_publish():
    assert {"TIKTOK_UPLOAD_VIDEO", "YOUTUBE_UPLOAD_VIDEO"} <= set(PUBLISHES)


# ── Every publish-class action asks before it runs ─────────────────────────────────────

@pytest.mark.parametrize("action", PUBLISHES)
def test_on_autos_ticket_a_channel_publish_raises_one_card_and_nothing_runs(desk, action):
    reply = sends._call(desk, sends._session(desk), action, VIDEO)

    assert sends._ran(desk) == []
    assert reply["requires_confirmation"] is True and reply["message"].startswith("Card raised: ")
    (grant,) = sends._grants(desk)
    assert grant.details[AGENT_SEND]["task_id"] == desk.ticket.id
    assert action in grant.question_md


def test_in_the_owners_chat_a_video_upload_asks_the_owner(desk):
    reply = sends._call(desk, OWNERS_CHAT, "TIKTOK_UPLOAD_VIDEO", VIDEO)

    assert sends._ran(desk) == []
    assert reply["requires_confirmation"] is True and reply["owner_only"] is True
    assert len(sends._grants(desk)) == 1


@pytest.mark.parametrize("slug, sends_it", [
    ("TIKTOK_UPLOAD_VIDEO", True), ("tiktok-upload-video", True), ("YOUTUBE_UPLOAD_VIDEO", True),
    ("TIKTOK_PUBLISH_VIDEO", True),                # never offered, still a publish
    ("TIKTOK_QUERY_CREATOR_INFO", False),          # the registry's read step
    ("TIKTOK_FETCH_PUBLISH_STATUS", False),        # its status step
    ("YOUTUBE_UPDATE_THUMBNAIL", False),           # its upload step: the publish is the video's
    ("GMAIL_SEND_EMAIL", True), ("GMAIL_FETCH_EMAILS", False), ("", False),
])
def test_the_gate_reads_the_registrys_publish_class(slug, sends_it):
    assert is_composio_send(slug) is sends_it


# ── A brief whose verb sends a message is reviewed by a person ────────────────────────

def _brief(title, description=""):
    return {"title": title, "description": description, "_user_id": "user_owner"}


@pytest.mark.parametrize("title", [
    "Email the supplier to confirm delivery",
    "Post the spring menu on Instagram",
    "Reply to Declan about the box",
])
def test_a_brief_whose_verb_sends_is_reviewed_by_a_person(title):
    params, held = reviewed_by_a_person(_brief(title))
    assert held is True and params["review_mode"] == "human"


@pytest.mark.parametrize("title, description", [
    ("Check the stock", "Count the boxes. Then text Declan the total."),   # a sentence's verb, after 'then'
    ("Draft the reply and forward it to Kerbside", ""),                   # the verb after 'and'
])
def test_a_message_verb_later_in_the_brief_sends_too(title, description):
    assert reviewed_by_a_person(_brief(title, description))[1] is True


@pytest.mark.parametrize("title, description", [
    ("Draft a reply to Declan", ""),
    ("Draft the email to the supplier", "Put it in Deliverables."),
    ("Summarise the text of the supplier's email", ""),
])
def test_a_brief_that_only_drafts_the_message_is_unchanged(title, description):
    brief = _brief(title, description)
    params, held = reviewed_by_a_person(brief)
    assert held is False and params == brief


# ── P256-FIX-RVW-27: a lead-in word or a common outbound verb still sends ─────────────

LEAD_IN_AND_OUTBOUND = [
    "Please email the supplier to confirm delivery", "Also post the menu on Instagram",
    "Just message Declan the times", "Contact the supplier to confirm the order", "Reach out to Declan",
    "Let Declan know the box is late", "Get back to the customer", "Respond to the review", "Notify the team",
    "Invite Sam to the call", "Submit the form",
]
ONLY_DRAFTS_OR_READS = ["Draft a reply to Declan", "Draft the email", "Prepare a post for review",
                        "Read the supplier's email"]


def _filed(monkeypatch, title):
    """The params ``platform_create_task`` files for ``title`` from the owner's chat, and its receipt."""
    from consumers.chatbot.receipts import receipt
    from modules.tools.discovery import handlers_board_task_review, handlers_board_tasks

    filed = {}

    async def create(db, workspace_id, params):
        filed.update(params)
        return {"success": True, "task_id": 2318, "status": "assigned", "title": params["title"],
                "review_mode": params.get("review_mode", "auto")}

    monkeypatch.setattr(handlers_board_tasks, "create_board_task", create)
    params = _brief(title)
    result = asyncio.run(handlers_board_task_review.create_board_task(None, uuid4(), params))
    return filed, result, receipt("platform_create_task", params, result)["effect"]


@pytest.mark.parametrize("title", LEAD_IN_AND_OUTBOUND)
def test_a_brief_with_a_lead_in_or_an_outbound_verb_is_reviewed_and_its_receipt_says_so(monkeypatch, title):
    filed, result, effect = _filed(monkeypatch, title)

    assert filed["review_mode"] == "human" and result[REVIEW_HELD] is True
    assert effect == "card created, reviewed by you before it closes"


@pytest.mark.parametrize("title", ONLY_DRAFTS_OR_READS)
def test_a_brief_that_drafts_or_reads_still_closes_by_itself(monkeypatch, title):
    filed, result, effect = _filed(monkeypatch, title)

    assert filed.get("review_mode", "auto") == "auto" and REVIEW_HELD not in result
    assert "reviewed by you" not in effect


@pytest.mark.parametrize("title, description", [
    ("Can you email Declan the times", ""), ("Could you please text Sam", ""),
    ("Remember to reply to Declan", ""), ("Make sure to forward the invoice to Kerbside", ""),
    ("Kindly notify the team", ""), ("First, share the menu with Kerbside", ""),
    ("Ask Sam to email the supplier", ""), ("Get the team to reach out to Kerbside", ""),
    ("Have Declan to let the supplier know", ""),
    ("Check the stock", "Count the boxes and then let the team know."),
])
def test_the_verb_is_read_past_a_lead_in_and_after_an_ask(title, description):
    assert reviewed_by_a_person(_brief(title, description))[1] is True


@pytest.mark.parametrize("title", [
    "Update the contact list", "Draft the invite for Sam", "Ask Sam about the menu", "Get the van to the depot",
    "Draft a response to the review",
])
def test_a_noun_or_an_ask_with_no_message_verb_keeps_auto(title):
    brief = _brief(title)
    assert reviewed_by_a_person(brief) == (brief, False)
