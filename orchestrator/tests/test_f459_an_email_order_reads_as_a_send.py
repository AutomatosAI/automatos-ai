"""F459 (night 18): in a person's chat, a send reaches the owner's card whatever words asked for it.

"Email Quay Coffee House the new terms." mapped to [issue.create, channel.create, data.create,
file.write] (the creates of 'new'), and "Email Saskia at Northgate asking where the Guji shipment
is." to [data.query, search.general]. The intent gate (tool_router.execute_tool_with_validation)
refused GMAIL_SEND_EMAIL before execute_tool, where the owner's click gate raises the card (rows 290,
401, 551, 589, 764): the owner could never approve the send, and Auto said "Gmail refused". When the
words held 'send' ("Send Saskia…", row 390) the card was raised and waited.

A reading of the owner's words can't cover every phrasing ("drop Quay a line", "mail Quay", another
language), and D10 retired phrase families. In a chat a person drives, a Composio send waits for their
click on a card that shows to whom, about what and its first line, so the card decides it and the intent
gate leaves it alone (``owner_only.the_card_decides``). An agent's run, a playbook step or a heartbeat
drives no user: the intent gate keeps every action there, so an agent steered into a send stays refused.
Any other action in a person's chat (not a send) is still read against their words.

The gate's refusal says it was Automatos's check, not the app's, for the owner and for the model.
"""
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from modules.tools.capabilities.taxonomy import get_capabilities_for_intent  # noqa: E402
from modules.tools.discovery.owner_only import the_card_decides  # noqa: E402
from modules.tools.execution.owner_reasons import OWNER_REASON, intent_gate_error, intent_refused  # noqa: E402
from modules.tools.services.action_capability_filter import get_action_capability_filter  # noqa: E402

SEND_EMAIL = "GMAIL_SEND_EMAIL"
RELABEL = "GMAIL_MODIFY_THREAD_LABELS"
THE_TERMS = "Email Quay Coffee House the new terms."                                   # rows 290, 401, 551, 764
THE_SHIPMENT = "Email Saskia at Northgate asking where the Guji shipment is."           # row 589
OWNERS_CHAT = {"driving_user_id": "7", "conversation_id": "c-1"}
AN_AGENTS_RUN = {"board_task_id": 85}
ASKED = {"success": False, "requires_confirmation": True, "owner_only": True}


def _sent(action=SEND_EMAIL):
    return {"action": action, "params": {"recipient_email": "orders@quaycoffeehouse.example",
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


def _route(intent, context, *, action=SEND_EMAIL, stored=("email.send",)):
    """The router call as the chat makes it, in ``context``; (its answer, execute_tool's mock, the gate's spy)."""
    from modules.tools import tool_router

    gate = MagicMock(wraps=tool_router.validate_action_for_intent)
    with patch.object(tool_router, "composio_available", lambda: True), \
            patch.object(tool_router, "the_callers_context", lambda: context), \
            patch.object(tool_router, "SessionLocal", lambda: _Session(_row(action, stored))), \
            patch.object(tool_router, "validate_action_for_intent", gate), \
            patch.object(tool_router, "execute_tool", new=AsyncMock(return_value=ASKED)) as ran:
        out = asyncio.run(tool_router.execute_tool_with_validation("composio_execute", _sent(action),
                                                                   original_intent=intent))
    return out, ran, gate


# ── Who decides a send ──────────────────────────────────────────────────────────────

def test_the_card_decides_a_send_in_a_persons_chat_only():
    assert the_card_decides(SEND_EMAIL, OWNERS_CHAT) is True
    assert the_card_decides("gmail-send-email", OWNERS_CHAT) is True
    assert the_card_decides(SEND_EMAIL, AN_AGENTS_RUN) is False                  # an agent's run: the gate stays
    assert the_card_decides(SEND_EMAIL, None) is False
    assert the_card_decides("GMAIL_FETCH_EMAILS", OWNERS_CHAT) is False          # a read is no send
    assert the_card_decides(RELABEL, OWNERS_CHAT) is False                       # nor is a label change
    assert the_card_decides("", OWNERS_CHAT) is False


@pytest.mark.parametrize("said", [
    THE_TERMS, THE_SHIPMENT, "Send Saskia a note about the Guji", "drop Quay a line with the new terms",
    "mail Quay the terms", "write to Zara and send it", "Envoie à Quay les nouvelles conditions",
    "Check my email",
])
def test_in_a_persons_chat_the_send_goes_on_to_its_card_whatever_the_words(said):
    out, ran, gate = _route(said, OWNERS_CHAT)

    gate.assert_not_called()                                                     # their words decide nothing here
    ran.assert_awaited_once()                                                    # execute_tool raises the card
    assert out == ASKED


# ── Where the intent gate stays ─────────────────────────────────────────────────────

@pytest.mark.parametrize("said", [THE_TERMS, "Check my email"])
def test_an_agents_run_keeps_the_intent_gate_for_a_send(said):
    out, ran, gate = _route(said, AN_AGENTS_RUN)

    gate.assert_called_once()
    ran.assert_not_called()                                                      # refused: no card in an agent's run
    assert out["success"] is False and out["blocked_by"] == "capability_validation"
    assert out[OWNER_REASON] == intent_refused(SEND_EMAIL)


def test_with_no_caller_context_the_send_is_still_checked():
    out, ran, gate = _route(THE_TERMS, None)
    gate.assert_called_once()
    ran.assert_not_called()
    assert out["success"] is False


def test_an_action_that_is_no_send_is_still_read_against_the_persons_words():
    out, ran, gate = _route("Check my email", OWNERS_CHAT, action=RELABEL, stored=("email.label",))

    gate.assert_called_once()
    ran.assert_not_called()
    assert out["success"] is False and "never reached Gmail, so Gmail did not refuse it" in out["error"]


# ── The words the refusal says ──────────────────────────────────────────────────────

def test_a_refusal_says_automatos_stopped_it_before_the_app():
    out, ran, _gate = _route("Check my email", AN_AGENTS_RUN)

    ran.assert_not_called()
    assert out["error"].startswith("Action not allowed for this intent: ")
    assert "never reached Gmail, so Gmail did not refuse it" in out["error"]


def test_the_owners_words_name_the_action_and_why():
    said = intent_refused(SEND_EMAIL)
    assert said.startswith("Automatos stopped the Gmail 'send email' action before it reached Gmail")
    assert "didn't read as asking for that" in said and "nothing ran" in said
    assert intent_refused("").startswith("Automatos stopped that action before it ran")
    assert "the app did not refuse it" in intent_gate_error("", "the reason")


# ── The filter reads one mapping ────────────────────────────────────────────────────

def test_the_filter_reads_the_taxonomys_one_mapping():
    """The filter's mapper is the taxonomy's (it was a copy of the same loop)."""
    svc = get_action_capability_filter(_Session())
    for said in (THE_TERMS, THE_SHIPMENT, "Check my email", "zzz"):
        assert sorted(svc._intent_to_capabilities(said)) == sorted(get_capabilities_for_intent(said))
