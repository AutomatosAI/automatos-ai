"""P256-FIX-RVW-5: a Composio call is read by the action that ran, never by its dispatcher.

The tracker recorded a ``composio_execute`` call under the dispatcher's name, so its receipt
was a write whatever ran (GMAIL_FETCH_EMAILS, a read, was a DONE write), and a blanket rule
let any Composio receipt back a claim of any family: "I've sent the reply to Declan." after a
mail fetch had no line and no nudge. A per-action tool recorded under its own name
(HUBSPOT_CREATE_CONTACT) matched no family, so "I've created the contact" was denied.

Now the call is recorded under the action that ran (``action_that_ran``: composio_execute's
``action``, written the way Composio writes it), a per-action tool keeps its own name, and the
slug is read by its words, whole: one of D7's read words and it only reads (``slug_reads``,
GMAIL_FETCH_EMAILS; GMAIL_REPLY_TO_THREAD is no read for its "thread"), else each word after
its toolkit is read as a platform call's name begins (``slug_stems``: "create_", "contact_"),
a send word as ``send_``, so the claim families match it as they match a platform call.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple

from modules.tools.discovery.send_words import READ_WORDS, SEND_WORDS

COMPOSIO_EXECUTE = "composio_execute"
# Where composio_execute names its action (exec_composio reads the same two keys).
ACTION_KEYS = ("action", "action_name")
SEND_STEM = "send_"
# A Composio action's slug: the toolkit, then its words (GMAIL_SEND_EMAIL); platform calls are lower case.
_SLUG = re.compile(r"^[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+$")
_NOT_A_WORD = re.compile(r"[^A-Z0-9]+")


def _as_composio_writes_it(said: str) -> str:
    """"gmail-send-email" → GMAIL_SEND_EMAIL."""
    return "_".join(word for word in _NOT_A_WORD.split(said.upper()) if word)


def action_that_ran(tool_name: str, tool_args: Any) -> Optional[str]:
    """composio_execute's own action as a slug; None for any other call, or one naming none."""
    if tool_name != COMPOSIO_EXECUTE or not isinstance(tool_args, dict):
        return None
    said = next((tool_args.get(key) for key in ACTION_KEYS if isinstance(tool_args.get(key), str)), "")
    slug = _as_composio_writes_it(said)
    return slug if is_slug(slug) else None


def is_slug(action: str) -> bool:
    """Whether ``action`` is a Composio action's slug (GMAIL_SEND_EMAIL), not a platform call."""
    return bool(_SLUG.match(str(action or "")))


def _words(slug: str) -> Tuple[str, ...]:
    """The slug's words after its toolkit, lower case: GMAIL_SEND_EMAIL → ("send", "email")."""
    return tuple(slug.lower().split("_")[1:])


def slug_reads(slug: str) -> bool:
    """Whether the action only reads: a whole word of D7's read list (fetch, list, get, …)."""
    return bool(set(_words(slug)) & READ_WORDS)


def slug_stems(slug: str) -> Tuple[str, ...]:
    """Each word of the slug as a call's name begins ("create_", "contact_"); a send word is
    ``send_`` (D7's list: SEND, REPLY, FORWARD, PUBLISH, POST, …)."""
    return tuple(dict.fromkeys(SEND_STEM if word in SEND_WORDS else f"{word}_" for word in _words(slug)))


__all__ = ["ACTION_KEYS", "COMPOSIO_EXECUTE", "SEND_STEM", "action_that_ran", "is_slug", "slug_reads", "slug_stems"]
