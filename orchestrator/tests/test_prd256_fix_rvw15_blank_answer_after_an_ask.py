"""PRD-256 P256-FIX-RVW-15: a blank answer after an owner's-click ask says the card waits.

F264's account (``turn_account.account_of``) counted every result whose ``success`` was False
as failed. An ask in the chat's envelope is ``{success: False, raw_result: {requires_confirmation:
True, grant_id …}}``, so after an ask and a blank reply the owner read "nothing was changed: what
I tried didn't go through. Please ask again." while the card waited for their click, and beside a
create, "1 other request didn't go through". Now an ask is neither a change nor a failure: the
account names its card once, in card_raised's words, and a genuine refusal is still not through.
"""
from __future__ import annotations

from modules.tools.execution.turn_account import (
    ACCOUNT_HEADER, NOT_THROUGH, NOTHING_CHANGED, WHY_ALL_FAILED, account_of,
)
from tests.test_prd256_fix_waiting_receipt import UPDATE_AGENT, _agent_ask, _chats, _routed

CARD = "- Card raised: change an agent 'Scout' (agent #12). Nothing changes until you click."
NOT_THROUGH_WORDS = "didn't go through"
ASK_AGAIN = "ask again"
NO_AGENT = "No agent #99 in this workspace."


def _ask_envelope():
    """What the chat's tool callback hands the loop for owner_only's ask (service.py _tool_callback)."""
    return _chats(_routed(_agent_ask()))


def _refusal_envelope():
    return _chats(_routed({"success": False, "error": NO_AGENT}))


def _made_card():
    raw = {"success": True, "task_id": 1420, "number": "#0220", "status": "inbox", "title": "Draft the brief"}
    return {"success": True, "llm_context": "made", "raw_result": raw}


def test_the_chat_envelope_of_an_ask_is_a_failure_at_its_top_level():
    """The shape the account misread: the envelope's own flag says False."""
    envelope = _ask_envelope()
    assert envelope["success"] is False
    assert envelope["raw_result"]["requires_confirmation"] is True


def test_an_ask_alone_says_the_card_waits_for_the_click():
    said = account_of([(UPDATE_AGENT, {}, _ask_envelope())])
    assert CARD in said.splitlines()
    assert said.count("Card raised") == 1
    assert NOT_THROUGH_WORDS not in said and ASK_AGAIN not in said.lower()
    assert "waits for your click" in said


def test_the_executors_own_ask_reads_the_same():
    said = account_of([(UPDATE_AGENT, {}, _agent_ask())])
    assert CARD in said.splitlines()
    assert NOT_THROUGH_WORDS not in said and ASK_AGAIN not in said.lower()


def test_the_same_ask_twice_is_named_once():
    said = account_of([(UPDATE_AGENT, {}, _ask_envelope()), (UPDATE_AGENT, {}, _ask_envelope())])
    assert said.count("Card raised") == 1


def test_beside_a_done_create_the_create_is_listed_and_nothing_is_not_through():
    said = account_of([("platform_create_task", {"title": "Draft the brief"}, _made_card()),
                       (UPDATE_AGENT, {}, _ask_envelope())])
    lines = said.splitlines()
    assert lines[0] == ACCOUNT_HEADER
    assert "- #0220 (Draft the brief): in the Inbox" in lines
    assert CARD in lines
    assert "other request" not in said
    assert NOT_THROUGH_WORDS not in said


def test_a_genuine_refusal_still_counts_as_not_through():
    said = account_of([(UPDATE_AGENT, {}, _refusal_envelope())])
    assert said == NOTHING_CHANGED.format(why=WHY_ALL_FAILED)
    assert NO_AGENT not in said


def test_a_refusal_beside_a_create_and_an_ask_is_one_not_through():
    said = account_of([("platform_create_task", {"title": "Draft the brief"}, _made_card()),
                       (UPDATE_AGENT, {}, _ask_envelope()),
                       (UPDATE_AGENT, {"agent_id": 99}, _refusal_envelope())])
    lines = said.splitlines()
    assert CARD in lines
    assert lines[-1] == NOT_THROUGH.format(count=1, s="", they="it")


def test_an_ask_beside_a_refusal_waits_and_names_the_refusal():
    said = account_of([(UPDATE_AGENT, {}, _ask_envelope()), (UPDATE_AGENT, {"agent_id": 99}, _refusal_envelope())])
    lines = said.splitlines()
    assert CARD in lines
    assert ASK_AGAIN not in said.lower()
    assert lines[-1] == NOT_THROUGH.format(count=1, s="", they="it")
