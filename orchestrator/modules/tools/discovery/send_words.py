"""PRD-256 Decision D7: the words of an act whose work leaves the workspace, in one list.

A Composio action whose slug carries one of them, and no read word, waits for the owner's
click (``owner_only.is_composio_send``); a ticket whose brief carries one of the order words
is reviewed by a person before it closes (``brief_sends``). P256-FIX-RVW-3: D7's
"send/publish/order" covers an order, a purchase, a booking, a payment, a charge and an
invoice, as it covers a message or a post; SHOPIFY_CREATE_ORDER on Auto's "Confirm the
order" ticket ran with no card while only the message words were listed.

P256-FIX-RVW-38: a message made or timed (SLACK_SCHEDULE_MESSAGE, DISCORDBOT_CREATE_MESSAGE,
TWITTER_CREATE_A_NEW_DM_CONVERSATION, GITHUB_CREATE_AN_ISSUE_COMMENT) and money paid out
(STRIPE_CREATE_REFUND, PAYPAL_CREATE_PAYOUT) leave the workspace too: an agent told 'Card
raised' on GMAIL_SEND_EMAIL retried with SLACK_SCHEDULE_MESSAGE and nothing asked.
"""
from __future__ import annotations

import re
from typing import FrozenSet

# A message or a post.
SEND_WORDS = frozenset({"send", "sends", "publish", "post", "reply", "forward", "tweet", "broadcast"})
# Placing an order, booking or paying: a slug's verb or object, and a brief's verb (shared with brief_sends).
ORDER_WORDS = frozenset({"order", "orders", "purchase", "purchases", "book", "booking", "pay", "payment"})
# What a slug creates when it charges (STRIPE_CREATE_INVOICE). Not a brief's word: "Draft a reminder for
# invoice HL-2291" only drafts.
CHARGE_OBJECTS = frozenset({"payments", "bookings", "charge", "charges", "invoice", "invoices"})
# Money paid out of the workspace: a slug's object (STRIPE_CREATE_PAYOUT); "charge" is a charge object above.
MONEY_OUT = frozenset({"refund", "refunds", "payout", "payouts", "transfer", "transfers"})
# A message a slug makes or times: it leaves as a send does (TWILIO_CREATE_MESSAGE, SLACK_SCHEDULE_MESSAGE).
MESSAGE_OBJECTS = frozenset({"message", "messages", "dm", "dms", "comment", "comments"})
MAKING_VERBS = frozenset({"create", "creates"})
TIMING_VERBS = frozenset({"schedule", "schedules"})
# What a send by reference sends: a thing the app keeps, whose recipients and body can change after
# the card (GMAIL_SEND_DRAFT, MAILCHIMP_SEND_CAMPAIGN). A draft made is not sent (GMAIL_CREATE_EMAIL_DRAFT);
# a draft or a campaign timed is.
BY_REFERENCE = frozenset({"draft", "drafts", "campaign", "campaigns"})
# A draft's id sends a kept draft whatever the action is named; a campaign's send always names its campaign
# (an ad post may carry a campaign_id and still name its own body).
BY_REFERENCE_KEYS = ("draft_id", "draftId")
# A slug with one of these only reads, whatever else it names (GMAIL_FETCH_EMAILS, SHOPIFY_LIST_ORDERS).
READ_WORDS = frozenset({"get", "list", "fetch", "search", "find", "retrieve", "read", "lookup", "count",
                        "download"})
LEAVES_THE_WORKSPACE = SEND_WORDS | ORDER_WORDS | CHARGE_OBJECTS | MONEY_OUT
_SLUG_WORD = re.compile(r"[^a-z0-9]+")


def slug_words(slug: str) -> FrozenSet[str]:
    """A slug's words in lower case, split on any non-alphanumeric: 'gmail-send-email' → send, email, gmail."""
    return frozenset(word for word in _SLUG_WORD.split(str(slug or "").lower()) if word)


def makes_or_times_a_message(words: FrozenSet[str]) -> bool:
    """A message made (create + message/dm/comment, not a draft of one) or timed (schedule + a message,
    a draft or a campaign): it goes out as a send does."""
    if words & READ_WORDS:
        return False
    made = bool(words & MAKING_VERBS and words & MESSAGE_OBJECTS and not words & BY_REFERENCE)
    return made or bool(words & TIMING_VERBS and words & (MESSAGE_OBJECTS | BY_REFERENCE))


__all__ = ["BY_REFERENCE", "BY_REFERENCE_KEYS", "CHARGE_OBJECTS", "LEAVES_THE_WORKSPACE", "MESSAGE_OBJECTS",
           "MONEY_OUT", "ORDER_WORDS", "READ_WORDS", "SEND_WORDS", "makes_or_times_a_message", "slug_words"]
