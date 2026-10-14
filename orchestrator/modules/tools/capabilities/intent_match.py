"""F428 (night 15): what the execution-time intent gate matches a Composio action against.

``ActionCapabilityFilter.check_action_eligibility`` maps the owner's latest words to capability
tags (``INTENT_TO_CAPABILITIES``, by keyword) and refuses an action whose capabilities, as the
daily sync stored them, share none. Night 15 hit two holes in that match:

- A write intent refused the read it needs first. 'Draft me a reply to the last email I got
  from a supplier' maps to [email.draft, email.reply, message.reply], and
  GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID is stored as [data.query, message.read]: refused five
  times. Reading is never a write, so an action that only reads passes any intent. "Only
  reads" takes both witnesses: every stored capability is a read, and the slug carries a read
  word and no word of a send, an order or a payment (``send_words``). A send misclassified as a
  read (GMAIL_SEND_DRAFT stored as data.query) still meets the match.
- A draft action was never classed as one. The sync's classifier has no draft pattern, so
  GMAIL_CREATE_EMAIL_DRAFT is stored as a generic create (data.create), and whether it passed
  hung on an incidental word in the owner's message ('make', 'create') and on whether the sync
  had classified Gmail yet (an empty table allows any non-destructive action, PRD-177): the
  same request was allowed at 04:05 and refused at 04:35. The slug now says it drafts
  (email.draft), and an intent to reply, send or forward covers a draft, which sends nothing.
  A draft SENT (GMAIL_SEND_DRAFT) is a send, never a draft made.

The destructive check is not here: it stays in the filter, unchanged.
"""
from __future__ import annotations

import re
from typing import FrozenSet, Iterable, Optional

from modules.tools.discovery.send_words import LEAVES_THE_WORKSPACE, READ_WORDS, slug_words

DRAFT = "email.draft"

# The taxonomy's capabilities that only read. Not notification.read (marks read), user.status
# (gets or sets), admin.audit or the exports.
READ_CAPABILITIES: FrozenSet[str] = frozenset({
    "message.read", "channel.list", "channel.read", "user.lookup", "user.list",
    "file.read", "file.list", "data.query", "data.read",
    "repo.read", "repo.list", "branch.read", "commit.read", "commit.list",
    "issue.read", "issue.list", "pr.read", "pr.list", "release.read",
    "event.read", "event.list", "email.read", "email.list",
    "task.read", "task.list", "project.read", "project.list", "notification.list", "webhook.list",
    "search.messages", "search.files", "search.users", "search.general",
    "analytics.read", "sheet.read", "database.query",
})

# An intent to send a message covers drafting it: a draft is the same message, not sent.
DRAFT_COVERED_BY: FrozenSet[str] = frozenset({DRAFT, "email.reply", "email.send", "email.forward"})

# A slug that makes or changes a draft (GMAIL_CREATE_EMAIL_DRAFT, OUTLOOK_CREATE_DRAFT).
_DRAFT_WORDS = frozenset({"draft", "drafts"})
_DRAFT_MAKING_WORDS = frozenset({"create", "creates", "new", "save", "update"})
# A slug that writes, whatever it also reads: never "only reads", even when the classifier's top three
# capabilities kept only its reads (X_GET_OR_CREATE_LABEL).
_WRITE_WORDS = _DRAFT_MAKING_WORDS | frozenset({"add", "insert", "edit", "modify", "patch", "set", "move", "upload",
                                                "write", "delete", "remove", "trash", "archive", "mark"})
_SLUG_SPLIT = re.compile(r"[^a-z0-9]+")


def makes_a_draft(action_id: str) -> bool:
    """Whether the slug makes or changes a draft, and sends nothing (GMAIL_SEND_DRAFT sends)."""
    words = slug_words(action_id)
    return bool(words & _DRAFT_WORDS and words & _DRAFT_MAKING_WORDS
                and not words & (LEAVES_THE_WORKSPACE | READ_WORDS))


def the_slugs_verb(action_id: str) -> str:
    """The word after the app in an APP_VERB_OBJECT slug ('fetch' in GMAIL_FETCH_EMAILS), else ''."""
    words = [word for word in _SLUG_SPLIT.split(str(action_id or "").lower()) if word]
    return words[1] if len(words) > 1 else ""


def only_reads(action_id: str, stored: Optional[Iterable[str]]) -> bool:
    """Whether the action only reads: every stored capability a read, a read verb, and nothing sent or written."""
    caps = frozenset(stored or ())
    return bool(caps) and caps <= READ_CAPABILITIES and the_slugs_verb(action_id) in READ_WORDS \
        and not slug_words(action_id) & (LEAVES_THE_WORKSPACE | _WRITE_WORDS)


def the_actions_capabilities(action_id: str, stored: Optional[Iterable[str]]) -> FrozenSet[str]:
    """The stored capabilities, and email.draft when the slug makes a draft."""
    caps = frozenset(stored or ())
    return caps | {DRAFT} if makes_a_draft(action_id) else caps


def the_intents_capabilities(required: Iterable[str]) -> FrozenSet[str]:
    """The intent's capabilities, and email.draft when it replies, sends or forwards."""
    caps = frozenset(required or ())
    return caps | {DRAFT} if caps & DRAFT_COVERED_BY else caps


def intent_allows(action_id: str, stored: Optional[Iterable[str]], required: Iterable[str]) -> bool:
    """Whether the action's capabilities meet the intent's: a read always does (F428)."""
    if not required or only_reads(action_id, stored):
        return True
    return bool(the_actions_capabilities(action_id, stored) & the_intents_capabilities(required))


__all__ = ["DRAFT", "DRAFT_COVERED_BY", "READ_CAPABILITIES", "intent_allows", "makes_a_draft", "only_reads",
           "the_actions_capabilities", "the_intents_capabilities", "the_slugs_verb"]
