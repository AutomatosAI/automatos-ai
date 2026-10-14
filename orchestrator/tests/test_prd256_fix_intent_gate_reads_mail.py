"""P256-FIX-N15-7 (F428, night 15): the intent gate lets a reply or draft read mail, and a draft is a draft.

'Draft me a reply to the last email I got from a supplier': GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID was
refused five times ('Action capabilities [data.query, message.read] don't match intent capabilities
[email.draft, email.reply, …]'), and GMAIL_CREATE_EMAIL_DRAFT passed at 04:05 and 04:08 and was
refused at 04:35 on the same request shape. The cause, from the gate's code: the sync's classifier
has no draft pattern, so the draft action is stored as a generic create, and whether it passed hung
on the sync's state (an empty table allows any non-destructive action) and on incidental words.

Pure unit tests: the eligibility check runs against a fake session, the stored capabilities come
from the sync's own classifier.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.tools.capabilities.classifier import ActionClassifier  # noqa: E402
from modules.tools.capabilities.intent_match import (  # noqa: E402
    DRAFT, intent_allows, makes_a_draft, only_reads,
)
from modules.tools.services.action_capability_filter import get_action_capability_filter  # noqa: E402

THE_NIGHTS_ASK = "Draft me a reply to the last email I got from a supplier"
FETCH = "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID"
CREATE_DRAFT = "GMAIL_CREATE_EMAIL_DRAFT"
SEND_DRAFT = "GMAIL_SEND_DRAFT"
SEND_EMAIL = "GMAIL_SEND_EMAIL"
DRAFT_DESCRIPTION = "Creates a gmail email draft, supporting to/cc/bcc, subject, plain/html body and attachments."


class _Result:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row

    def first(self):
        return self._row


class _Session:
    """execute(select(...)) yields the one stored row, or an empty table when there is none."""

    def __init__(self, row=None):
        self._row = row

    def execute(self, *_a, **_k):
        return _Result(self._row)


def _row(action_id, capabilities, destructive=False):
    return SimpleNamespace(action_id=action_id, capabilities=list(capabilities), destructive=destructive)


def _check(row, intent=THE_NIGHTS_ASK, action_id=None):
    svc = get_action_capability_filter(_Session(row))
    return svc.check_action_eligibility(action_id=action_id or row.action_id, intent=intent)


def _as_the_sync_stores(action_id, description=None):
    """The capabilities the daily sync writes for an action (heuristics, no LLM client)."""
    return ActionClassifier().classify(action_id, description=description).capabilities


def test_the_nights_reply_intent_maps_to_writes_only():
    """The ask's capabilities are the night's: a draft and a reply, no read."""
    caps = get_action_capability_filter(_Session())._intent_to_capabilities(THE_NIGHTS_ASK)
    assert {"email.draft", "email.reply"} <= set(caps)
    assert not {"message.read", "data.query", "email.read"} & set(caps)


@pytest.mark.parametrize("intent", [
    THE_NIGHTS_ASK,
    "Reply to the supplier's last email",
    "draft a reply to the last supplier email, draft only",
])
def test_an_email_reply_intent_allows_the_fetch(intent):
    """Reading is never a write: a reply or draft intent may fetch the mail it answers."""
    eligible, reason = _check(_row(FETCH, ["data.query", "message.read"]), intent=intent)
    assert eligible is True, reason


def test_the_fetch_as_the_sync_classifies_it_is_allowed():
    caps = _as_the_sync_stores(FETCH)
    assert set(caps) <= {"data.query", "message.read", "data.read", "email.read"}, caps
    assert _check(_row(FETCH, caps))[0] is True


@pytest.mark.parametrize("stored", [
    ["data.create"],                       # heuristic: the slug's 'create', no draft pattern
    None,                                  # computed below from the sync's own classifier
])
def test_the_draft_gets_the_same_answer_whatever_the_sync_stored(stored):
    """04:05 allowed, 04:35 refused, same request: now allowed on an empty table and on a synced one."""
    caps = stored if stored is not None else _as_the_sync_stores(CREATE_DRAFT, DRAFT_DESCRIPTION)
    assert DRAFT not in caps, "the classifier has no draft pattern: the gate must not need one"
    unsynced, _ = get_action_capability_filter(_Session()).check_action_eligibility(CREATE_DRAFT, THE_NIGHTS_ASK)
    synced, reason = _check(_row(CREATE_DRAFT, caps))
    assert unsynced is True and synced is True, reason


@pytest.mark.parametrize("intent", [
    THE_NIGHTS_ASK,
    "Reply to the supplier's last email",
    "Make me a draft reply to the supplier",
    "draft an email to the supplier about the late delivery",
])
def test_the_draft_is_allowed_however_the_ask_is_worded(intent):
    eligible, reason = _check(_row(CREATE_DRAFT, ["data.create"]), intent=intent)
    assert eligible is True, reason


def test_a_draft_intent_never_covers_a_send():
    """A draft asked for is never sent: the send of a draft, and a plain send, still meet the match."""
    assert _check(_row(SEND_DRAFT, ["data.query"]))[0] is False
    assert _check(_row(SEND_EMAIL, ["email.send"]))[0] is False


def test_a_send_misclassified_as_a_read_is_not_a_read():
    assert only_reads(SEND_DRAFT, ["data.query"]) is False
    assert only_reads(FETCH, ["data.query", "message.read"]) is True
    assert only_reads(FETCH, ["data.query", "data.create"]) is False
    assert only_reads(FETCH, []) is False


@pytest.mark.parametrize("slug", ["SHOPIFY_LIST_ORDERS", "GMAIL_GET_OR_CREATE_LABEL", "GMAIL_FETCH_AND_MARK_READ",
                                  "GMAIL_RESOLVE_MESSAGE", "GOOGLE_DRIVE_LIST_FILES"])
def test_a_slug_that_also_orders_or_writes_never_only_reads(slug):
    """Its stored reads are not enough: an order or write word, or a verb not known to read (resolve; a
    two-word app's 'drive'), leaves it to the match: 'only reads' fails closed."""
    assert only_reads(slug, ["data.query", "message.read"]) is False
    assert _check(_row(slug, ["data.query", "message.read"]), intent="draft a reply")[0] is False


def test_only_a_draft_made_is_a_draft():
    assert makes_a_draft(CREATE_DRAFT) is True
    assert makes_a_draft("OUTLOOK_CREATE_DRAFT_REPLY") is False   # 'reply' is a send word: left to the match
    assert makes_a_draft(SEND_DRAFT) is False
    assert makes_a_draft("GMAIL_LIST_DRAFTS") is False


def test_a_write_still_needs_a_matching_intent():
    """The gate still refuses a write the words do not ask for."""
    assert intent_allows("SLACK_SEND_MESSAGE", ["message.send"], ["event.create"]) is False
    assert _check(_row("SLACK_SEND_MESSAGE", ["message.send"]), intent="schedule a meeting")[0] is False


def test_a_destructive_read_is_still_refused_for_a_plain_intent():
    """The destructive check is unchanged: a read flagged destructive needs a destructive intent."""
    eligible, reason = _check(_row(FETCH, ["data.query", "message.read"], destructive=True))
    assert eligible is False and "Destructive" in reason


def test_an_unknown_action_on_a_synced_table_is_still_refused():
    class _SyncedWithoutIt(_Session):
        def execute(self, *_a, **_k):
            self._calls = getattr(self, "_calls", 0) + 1
            return _Result(None if self._calls == 1 else ("SOME_OTHER_ACTION",))

    svc = get_action_capability_filter(_SyncedWithoutIt())
    eligible, reason = svc.check_action_eligibility(FETCH, THE_NIGHTS_ASK)
    assert eligible is False and "Unknown action" in reason
