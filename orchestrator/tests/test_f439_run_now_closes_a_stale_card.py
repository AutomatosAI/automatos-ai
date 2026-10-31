"""P256-FIX-N17-3 (F439, nights 16–17): run-now never reuses a stale card from another chat.

Night 16: five run-nows of playbook #81 from four chats each got "Card raised", and each was
#2607, raised the night before in another chat; night 17 reused it 13 hours after it expired.
``issue_tool_grant`` reused any pending row of the same call whatever its chat or age.

Now a lapsed row is closed (``system:expired``) and a fresh card raised; a chat turn that finds
another chat's card closes it (``system:superseded``) and raises its own, so the click's line goes
to this chat; an agent's send card, and any row an agent's or a playbook's call finds, is reused as
today. A row a click decided a moment before (the close matched none) is reused.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from uuid import UUID, uuid4

import pytest

from core.services.approval_grants import is_lapsed
from modules.tools.execution.card_raised import ACT, for_the_model
from modules.tools.execution.stale_cards import ALREADY_WAITING, EXPIRED_BY, SUPERSEDED_BY

ACTION = "platform_execute_playbook"
RUN_NOW = {"playbook_id": 81}
RUN_ACT = "run a playbook 'Monday green-coffee reorder' (playbook #81)"
ASK = {"success": False, "requires_confirmation": True, "owner_only": True, ACT: RUN_ACT,
       "message": f"Waiting for the owner's click: {RUN_ACT}"}
PLAYBOOK = {"playbook_execution_id": "exec-f439", "playbook_step": 1}
AN_AGENT = {"mission_id": "m-f439"}


def _chat(conversation=None, turn=None):
    """A person's chat turn, as build_tool_caller_context makes it (a turn id is unique per turn)."""
    return {"driving_user_id": "7", "user_id": "user_owner", "conversation_id": conversation or str(uuid4()),
            "turn_id": turn or uuid4().hex}


@pytest.fixture
def desk(db_session, seed_workspace, monkeypatch):
    import modules.tools.execution.tool_grants as tool_grants

    monkeypatch.setattr(tool_grants, "_notify_approval_pending", lambda grant, ws: None)
    return NS(db=db_session, ws=UUID(seed_workspace()))


@pytest.fixture
def closes(monkeypatch):
    """Each close the run-now makes (its ids and revoked_by), through the real UPDATE."""
    import core.services.approval_grants as service

    made, real = [], service.expire_pending_grants

    def spy(db, workspace_id, grant_ids, *, revoked_by, now=None):
        made.append((list(grant_ids), revoked_by))
        return real(db, workspace_id, grant_ids, revoked_by=revoked_by, now=now)

    monkeypatch.setattr(service, "expire_pending_grants", spy)
    return made


def _ask(desk, ctx):
    from modules.tools.execution.tool_grants import attach_ask_grant

    return attach_ask_grant(desk.db, desk.ws, action=ACTION, params=RUN_NOW, ask=dict(ASK), permission_level="write",
                            description="Run a playbook", caller_context=ctx)


def _row(desk, grant_id):
    from core.models.approval_grants import ApprovalGrant

    desk.db.expire_all()
    return desk.db.get(ApprovalGrant, grant_id)


def _lapse(desk, grant_id, hours=13):
    row = _row(desk, grant_id)
    row.expires_at = datetime.now(timezone.utc) - timedelta(hours=hours)
    desk.db.flush()


def _mark_as_an_agents_send(desk, grant_id, task_id):
    from modules.tools.discovery.agent_sends import AGENT_SEND

    row = _row(desk, grant_id)
    row.details = {**row.details, AGENT_SEND: {"lane": "session", "task_id": task_id,
                                               "context": {"session_task_id": task_id}}}
    desk.db.flush()


# ── A card past its expiry is closed, and a fresh one raised ─────────────────────────

def test_an_expired_chat_card_is_closed_and_chat_b_gets_a_fresh_one_bound_to_it(desk, closes):
    stale = _ask(desk, _chat())["grant_id"]
    _lapse(desk, stale)
    b = str(uuid4())

    fresh = _ask(desk, _chat(b))

    assert isinstance(fresh["grant_id"], int) and fresh["grant_id"] != stale
    assert closes == [([stale], EXPIRED_BY)]
    old, new = _row(desk, stale), _row(desk, fresh["grant_id"])
    assert (old.status, old.revoked_by) == ("expired", EXPIRED_BY)
    assert new.status == "pending" and new.details["conversation_id"] == b and new.details["lane"] == "chat"
    assert ALREADY_WAITING not in fresh and for_the_model(fresh).startswith("Card raised: ")


def test_an_expired_card_met_by_a_playbook_step_is_closed_and_a_fresh_one_raised(desk, closes):
    stale = _ask(desk, _chat())["grant_id"]
    _lapse(desk, stale)

    fresh = _ask(desk, PLAYBOOK)

    assert fresh["grant_id"] != stale and closes == [([stale], EXPIRED_BY)]
    assert _row(desk, fresh["grant_id"]).status == "pending"
    assert for_the_model(fresh).startswith("Card raised: ")


def test_an_expired_agents_send_card_is_reused_never_closed(desk, closes):
    """A session ticket is parked on an agent's send card (cli_host_service._park_for_answer):
    closing it would strand the ticket, so the same send on another ticket still meets it."""
    from modules.tools.discovery.agent_sends import ON_ANOTHER_TICKET, waits_on_the_card

    stale = _ask(desk, PLAYBOOK)["grant_id"]
    _mark_as_an_agents_send(desk, stale, 41)
    _lapse(desk, stale)

    again = _ask(desk, {"session_task_id": 42})

    assert again["grant_id"] == stale and closes == [] and _row(desk, stale).status == "pending"
    ticket = {"lane": "session", "task_id": 42, "context": {"session_task_id": 42}}
    told = waits_on_the_card(desk.db, desk.ws, ticket, again, sent=RUN_NOW, agent_id=None)
    assert told["message"] == ON_ANOTHER_TICKET.format(task_id="41")


# ── Another chat's live card is replaced by this chat's own ─────────────────────────

def test_chat_b_replaces_chat_as_live_card_and_the_click_is_told_in_b(desk, closes, monkeypatch):
    from core.services.approval_grants import grant_grant
    from services.click_results import said_in_the_chat
    import services.chat_messenger as messenger

    a, b = str(uuid4()), str(uuid4())
    first = _ask(desk, _chat(a))["grant_id"]

    second = _ask(desk, _chat(b))

    assert second["grant_id"] != first and ALREADY_WAITING not in second
    assert closes == [([first], SUPERSEDED_BY)]
    old = _row(desk, first)
    assert (old.status, old.revoked_by) == ("expired", SUPERSEDED_BY)
    told = []
    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kwargs: told.append(kwargs))
    new = _row(desk, second["grant_id"])
    grant_grant(new, granted_by="user:7")
    new.details = {**new.details, "executed_result": {"success": True, "ids": {"execution_id": "exec-9"}}}
    said_in_the_chat(desk.db, new)
    assert [line["chat_id"] for line in told] == [b]


def test_a_playbook_step_reuses_a_chats_live_card_as_today(desk, closes):
    first = _ask(desk, _chat())["grant_id"]

    again = _ask(desk, PLAYBOOK)

    assert again["grant_id"] == first and closes == [] and ALREADY_WAITING not in again
    assert for_the_model(again).startswith("Card raised: ")


def test_a_chat_meets_an_agents_live_card_as_already_waiting_never_replacing_it(desk, closes):
    first = _ask(desk, AN_AGENT)["grant_id"]

    again = _ask(desk, _chat())

    assert again["grant_id"] == first and closes == [] and _row(desk, first).status == "pending"
    assert again[ALREADY_WAITING]["grant_id"] == first
    assert for_the_model(again).startswith("Not run: this exact call already waits on its approval card")


def test_another_persons_card_from_their_chat_is_left_to_them(desk, closes):
    """Two people in one workspace ask for the same call: the second meets the first's card as
    already waiting (it is under Approvals) and never closes it."""
    theirs = _ask(desk, {**_chat(), "driving_user_id": "8"})["grant_id"]

    mine = _ask(desk, _chat())

    assert mine["grant_id"] == theirs and closes == [] and _row(desk, theirs).status == "pending"
    assert mine[ALREADY_WAITING]["grant_id"] == theirs


def test_a_turn_that_lost_the_close_to_another_turn_raises_its_own_card(desk, monkeypatch):
    """Two chats close the same stale card at once: the one whose close matched nothing never
    returns the closed card (no click could run it); it raises a fresh one."""
    import core.services.approval_grants as service

    stale = _ask(desk, _chat())["grant_id"]
    _lapse(desk, stale)
    real = service.expire_pending_grants

    def closed_by_the_other_turn(db, workspace_id, grant_ids, *, revoked_by, now=None):
        real(db, workspace_id, grant_ids, revoked_by=SUPERSEDED_BY, now=now)
        return 0

    monkeypatch.setattr(service, "expire_pending_grants", closed_by_the_other_turn)

    fresh = _ask(desk, _chat())

    assert fresh["grant_id"] != stale and _row(desk, fresh["grant_id"]).status == "pending"


def test_a_click_that_won_the_race_keeps_the_row(desk, monkeypatch):
    """The close is guarded on pending: when it matched no row (a click decided it a moment
    before), the row is returned as today and no second card is raised."""
    import core.services.approval_grants as service
    from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant

    stale = _ask(desk, _chat())["grant_id"]
    _lapse(desk, stale)
    monkeypatch.setattr(service, "expire_pending_grants", lambda *a, **k: 0)

    again = _ask(desk, _chat())

    assert again["grant_id"] == stale
    rows = desk.db.query(ApprovalGrant).filter(ApprovalGrant.workspace_id == desk.ws,
                                                ApprovalGrant.subject_type == SUBJECT_TOOL_CALL).all()
    assert [row.id for row in rows] == [stale]


# ── is_lapsed: the clock, whatever the status ───────────────────────────────────────

def test_is_lapsed_reads_the_clock_naive_as_utc_and_a_row_with_no_expiry_never_lapses():
    now = datetime(2026, 10, 15, 18, 0, tzinfo=timezone.utc)

    assert is_lapsed(NS(expires_at=datetime(2026, 10, 15, 5, 12)), now=now) is True        # naive: UTC
    assert is_lapsed(NS(expires_at=now + timedelta(minutes=1)), now=now) is False
    assert is_lapsed(NS(expires_at=None), now=now) is False
    assert is_lapsed(NS(id=None), now=now) is False                 # a stand-in that keeps no expiry
