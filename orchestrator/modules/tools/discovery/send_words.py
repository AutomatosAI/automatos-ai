"""PRD-256 Decision D7: the words of an act whose work leaves the workspace, in one list.

A Composio action whose slug carries one of them, and no read word, waits for the owner's
click (``owner_only.is_composio_send``); a ticket whose brief carries one of the order words
is reviewed by a person before it closes (``brief_sends``). P256-FIX-RVW-3: D7's
"send/publish/order" covers an order, a purchase, a booking, a payment, a charge and an
invoice, as it covers a message or a post; SHOPIFY_CREATE_ORDER on Auto's "Confirm the
order" ticket ran with no card while only the message words were listed.
"""
from __future__ import annotations

# A message or a post.
SEND_WORDS = frozenset({"send", "sends", "publish", "post", "reply", "forward", "tweet", "broadcast"})
# Placing an order, booking or paying: a slug's verb or object, and a brief's verb (shared with brief_sends).
ORDER_WORDS = frozenset({"order", "orders", "purchase", "purchases", "book", "booking", "pay", "payment"})
# What a slug creates when it charges (STRIPE_CREATE_INVOICE). Not a brief's word: "Draft a reminder for
# invoice HL-2291" only drafts.
CHARGE_OBJECTS = frozenset({"payments", "bookings", "charge", "charges", "invoice", "invoices"})
# A slug with one of these only reads, whatever else it names (GMAIL_FETCH_EMAILS, SHOPIFY_LIST_ORDERS).
READ_WORDS = frozenset({"get", "list", "fetch", "search", "find", "retrieve", "read", "lookup", "count",
                        "download"})
LEAVES_THE_WORKSPACE = SEND_WORDS | ORDER_WORDS | CHARGE_OBJECTS

__all__ = ["CHARGE_OBJECTS", "LEAVES_THE_WORKSPACE", "ORDER_WORDS", "READ_WORDS", "SEND_WORDS"]
