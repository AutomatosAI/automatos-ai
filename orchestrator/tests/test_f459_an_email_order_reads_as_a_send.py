"""F459 (night 18): "Email <customer> <thing>" reads as a send, so the owner's approval card is raised.

"Email Quay Coffee House the new terms." mapped to [issue.create, channel.create, data.create,
file.write] (the creates of 'new'), and "Email Saskia at Northgate asking where the Guji shipment
is." to [data.query, search.general]: no keyword read 'email' as the verb. The intent gate refused
GMAIL_SEND_EMAIL before execute_tool, where the owner's click gate raises the card (rows 290, 401,
551, 589, 764), so the owner could never approve the send, and Auto said "Gmail refused". When the
intent allowed it ("Send Saskia…", row 390) the card was raised and waited.

Now 'email' used as the verb maps to email.send (and so covers a draft), a noun ('my email', 'the
email from Saskia', 'the email address') does not, and the gate's refusal says it was Automatos's
check, not the app's, for the owner and for the model.

Pure unit tests: the eligibility check runs against a fake session, as the F428 tests do.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.tools.capabilities.taxonomy import emails_someone, get_capabilities_for_intent  # noqa: E402
from modules.tools.execution.owner_reasons import OWNER_REASON, intent_gate_error, intent_refused  # noqa: E402
from modules.tools.services.action_capability_filter import get_action_capability_filter  # noqa: E402

SEND_EMAIL = "GMAIL_SEND_EMAIL"
CREATE_DRAFT = "GMAIL_CREATE_EMAIL_DRAFT"
THE_TERMS = "Email Quay Coffee House the new terms."                                   # rows 290, 401, 551, 764
THE_SHIPMENT = "Email Saskia at Northgate asking where the Guji shipment is."           # row 589
SENT = {"action": SEND_EMAIL, "params": {"recipient_email": "orders@quaycoffeehouse.example",
                                         "subject": "Your account terms from 1 November 2026", "body": "Hi Zara, …"}}


class _Result:
    def __init__(self, row):
        self._row = row

    def scalar_one_or_none(self):
        return self._row

    def first(self):
        return self._row


class _Session:
    """execute(select(...)) yields the one stored row (the sync's), or an empty table when there is none."""

    def __init__(self, row=None):
        self._row = row

    def execute(self, *_a, **_k):
        return _Result(self._row)

    def close(self):
        return None


def _row(action_id, capabilities):
    return SimpleNamespace(action_id=action_id, capabilities=list(capabilities), destructive=False)


def _check(action_id, intent, stored):
    return get_action_capability_filter(_Session(_row(action_id, stored))).check_action_eligibility(
        action_id=action_id, intent=intent)


# ── 'email' as the verb ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("said", [
    THE_TERMS, THE_SHIPMENT, "Can you email Zara the new terms?", "please email Kit at Marlstone",
    "Check the stock and then email Northgate", "I want to email the Taunton café", "  Email me the report",
    "“Email Quay the terms”", "E-mail Saskia today",
])
def test_email_used_as_the_verb_is_a_send(said):
    assert emails_someone(said) is True
    assert "email.send" in get_capabilities_for_intent(said)


@pytest.mark.parametrize("said", [
    "Check my email", "What did the email from Saskia say?", "Draft an email to Kit at Marlstone, draft only",
    "What is Quay's email address?", "Don't email Quay yet", "Subject: Email from Kit",
    "Summarise my inbox and email threads", "Is there an email draft for Quay?", "Reply to the email from Kit",
    "the email is wrong",
])
def test_email_used_as_a_noun_is_no_send(said):
    assert emails_someone(said) is False
    assert "email.send" not in get_capabilities_for_intent(said)


def test_the_nights_two_asks_read_as_sends_and_keep_what_they_read_before():
    terms = set(get_capabilities_for_intent(THE_TERMS))
    assert {"email.send", "data.create"} <= terms                             # 'new' still maps as it did
    assert set(get_capabilities_for_intent(THE_SHIPMENT)) == {"email.send"}   # was [data.query, search.general]


def test_the_filter_reads_the_taxonomys_one_mapping():
    """The filter's mapper is the taxonomy's (it was a copy, and only the taxonomy's would have learned)."""
    svc = get_action_capability_filter(_Session())
    for said in (THE_TERMS, THE_SHIPMENT, "Check my email", "zzz"):
        assert sorted(svc._intent_to_capabilities(said)) == sorted(get_capabilities_for_intent(said))


# ── The gate lets the send on to the owner's card ───────────────────────────────────

@pytest.mark.parametrize("intent", [THE_TERMS, THE_SHIPMENT])
def test_the_send_is_eligible_for_the_nights_asks(intent):
    eligible, reason = _check(SEND_EMAIL, intent, ["email.send"])
    assert eligible is True, reason


def test_a_draft_is_eligible_for_an_email_order_too():
    """A send covers a draft (F428), so "draft it instead" never hits the gate on the same words."""
    eligible, reason = _check(CREATE_DRAFT, THE_SHIPMENT, ["data.create"])
    assert eligible is True, reason


def test_a_read_of_the_inbox_still_refuses_a_send():
    eligible, reason = _check(SEND_EMAIL, "Check my email", ["email.send"])
    assert eligible is False and "don't match intent capabilities" in reason


def test_the_router_hands_the_send_on_to_execute_tool_where_the_card_is_raised():
    from modules.tools import tool_router

    asked = {"success": False, "requires_confirmation": True, "owner_only": True}
    with patch.object(tool_router, "composio_available", lambda: True), \
            patch.object(tool_router, "SessionLocal", lambda: _Session(_row(SEND_EMAIL, ["email.send"]))), \
            patch.object(tool_router, "execute_tool", new=AsyncMock(return_value=asked)) as ran:
        out = asyncio.run(tool_router.execute_tool_with_validation("composio_execute", SENT, original_intent=THE_TERMS))

    ran.assert_awaited_once()
    assert out == asked


# ── A refusal names the gate, never the app ─────────────────────────────────────────

def test_a_refusal_says_automatos_stopped_it_before_the_app():
    from modules.tools import tool_router

    with patch.object(tool_router, "composio_available", lambda: True), \
            patch.object(tool_router, "SessionLocal", lambda: _Session(_row(SEND_EMAIL, ["email.send"]))), \
            patch.object(tool_router, "execute_tool", new=AsyncMock()) as ran:
        out = asyncio.run(tool_router.execute_tool_with_validation("composio_execute", SENT,
                                                                   original_intent="Check my email"))

    ran.assert_not_called()
    assert out["success"] is False and out["error"].startswith("Action not allowed for this intent: ")
    assert "never reached Gmail, so Gmail did not refuse it" in out["error"]
    assert out[OWNER_REASON] == intent_refused(SEND_EMAIL)


def test_the_owners_words_name_the_action_and_why():
    said = intent_refused(SEND_EMAIL)
    assert said.startswith("Automatos stopped the Gmail 'send email' action before it reached Gmail")
    assert "didn't read as asking for that" in said and "nothing ran" in said
    assert intent_refused("").startswith("Automatos stopped that action before it ran")
    assert "the app did not refuse it" in intent_gate_error("", "the reason")
