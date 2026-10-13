"""P256-FIX-N14-7 (F421): an approval card Auto files for an order is approved by the owner
without a manual move.

Night 14: Auto filed #1440, an order for the owner to approve, from the owner's chat with no
agent on it. D7 held it for a person, but it was filed in the Inbox: the board's Approve
refuses an Inbox card ("no one has worked on it yet"), and Auto's move to Review was refused
too (nothing on the card to review). Such a card (a person's chat, no agent, held for a person,
a brief that orders or sends and asks for an approval) carries the server's mark and is filed
in Review now, where the board's Approve moves it to Done with no artifact asked of it. A
model's own ``status: review`` is still dropped (F289), a mark the call carries is dropped, and
any other unassigned card keeps the Inbox.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from modules.tools.discovery.approval_cards import (
    APPROVAL_NOTE, FILED_FOR_APPROVAL, OWNERS_DECISION, asks_for_approval, filed_for_approval, is_an_approval_card,
    says_it_waits_for_approve,
)

CREATE = "platform_create_task"
TITLE = "Approve the order: 20 kg Guji natural from Harbour Imports"
BRIEF = ("For the owner's approval: order 20 kg of Guji natural at £9.40/kg from Harbour Imports. "
         "GREEN BUYER places it once approved.")


def _held(title=TITLE, description=BRIEF, **more):
    """A card from the owner's chat that D7 held for a person (``review_mode: human``)."""
    return {"title": title, "description": description, "_user_id": "user_owner", "review_mode": "human", **more}


# ── Which card is filed for approval ───────────────────────────────────────────────

def test_an_order_for_the_owners_approval_with_no_agent_is_marked_and_starts_in_review():
    from modules.tools.discovery.handlers_board_tasks import initial_board_status

    params, filed = filed_for_approval(_held())

    assert filed is True and params["planning_data"] == {OWNERS_DECISION: True} and "status" not in params
    assert initial_board_status(None, None, params["planning_data"]) == "review"
    assert initial_board_status(None, None, None) == "inbox"                  # an unmarked card as before


def test_a_mark_the_call_carries_is_dropped():
    forged = _held(assigned_agent_name="GREEN BUYER", planning_data={OWNERS_DECISION: True, "step": 1})

    params, filed = filed_for_approval(forged)

    assert filed is False and params["planning_data"] == {"step": 1}
    assert not is_an_approval_card(params["planning_data"]) and not is_an_approval_card({OWNERS_DECISION: "yes"})


@pytest.mark.parametrize("params", [
    _held(assigned_agent_name="GREEN BUYER"),                                # its agent works it, then Review
    _held(agent_id=267),
    _held("Email the supplier to confirm delivery", "Harbour Imports, Friday's drop."),  # work, not an approval
    _held("Approve the cupping notes", "Read them before Friday."),         # no order or send
    {**_held(), "review_mode": "auto"},                                     # the call said it closes by itself
    _held(approval_action={"type": "create_blog", "topic": "Guji"}),        # in Review by its own action
    {key: value for key, value in _held().items() if key != "_user_id"},    # an agent's own run files it
], ids=["agent-named", "agent-id", "no-approval-asked", "no-order-or-send", "auto", "approval-action", "no-person"])
def test_any_other_card_keeps_its_column(params):
    assert filed_for_approval(params) == (params, False)


@pytest.mark.parametrize("brief, asks", [
    ("Approve the order", True), ("Needs your approval", True), ("Authorise the payment", True),
    ("Sign off the supplier order", True), ("Order 20 kg of Guji", False), ("Sign the delivery note", False),
])
def test_what_asks_for_an_approval(brief, asks):
    assert asks_for_approval(brief) is asks


def test_done_asks_an_approval_card_for_no_artifact():
    """A brief that names paperwork asks Done for a Deliverable; the owner's approval card does not."""
    from services.done_needs_an_artifact import artifact_need

    title, brief = "Approve the draft price list and send it to Kerbside", "For the owner's approval before it goes."
    params, filed = filed_for_approval(_held(title, brief))
    card = NS(source_type="user", title=title, description=brief, planning_data=None)

    assert filed is True and artifact_need(card) is not None
    assert artifact_need(NS(**{**vars(card), "planning_data": params["planning_data"]})) is None


def test_the_answer_says_where_it_was_filed_only_when_it_was():
    filed = says_it_waits_for_approve({"success": True, "task_id": 1440, "status": "review"}, True)
    assert filed[FILED_FOR_APPROVAL] is True and filed["approval_note"] == APPROVAL_NOTE
    inbox = {"success": True, "task_id": 1441, "status": "inbox"}
    assert says_it_waits_for_approve(inbox, False) == inbox


# ── #1440 end to end: filed from the owner's chat, approved on the board ─────────────

@pytest.fixture
def roastery(db_session, seed_workspace, monkeypatch):
    """A workspace whose owner drives the chat; the board's fan-out is quiet. The dial is on."""
    from sqlalchemy import text

    import modules.tools.discovery.follows_the_owner as guard
    from modules.tools.discovery.platform_executor import PlatformActionExecutor
    from tests.helpers_mission_lane import quiet_board

    ws = UUID(seed_workspace())
    tag = uuid4().hex[:8]
    owner = db_session.execute(text("INSERT INTO users (email, username) VALUES (:e, :u) RETURNING id"),
                               {"e": f"n147b-{tag}@harbourline.test", "u": f"n147b-{tag}"}).scalar()
    db_session.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                            "VALUES (CAST(:ws AS uuid), :user, 'owner', TRUE)"), {"ws": str(ws), "user": owner})
    monkeypatch.setattr(guard, "owner_turn", lambda db, ws, ctx: None)
    quiet_board(monkeypatch)
    executor = PlatformActionExecutor(db_session, ws)
    executor._full_autonomy = lambda: True
    chat = {"driving_user_id": str(owner), "user_id": "user_owner", "conversation_id": str(uuid4()), "turn_id": "t-1"}
    return NS(db=db_session, ws=ws, owner=owner, executor=executor, chat=chat)


def _file(roastery, title=TITLE, description=BRIEF):
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        return asyncio.run(roastery.executor.execute(CREATE, {"title": title, "description": description},
                                                     roastery.chat))


def _card(roastery, task_id):
    from core.models.core import BoardTask

    return roastery.db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == roastery.ws).one()


def _body(payload):
    async def _json():
        return payload
    return NS(json=_json)


def test_the_owner_approves_the_card_auto_filed_without_moving_it(roastery):
    from api.board_tasks import approve_task

    filed = _file(roastery)

    assert filed["success"] is True and filed["status"] == "review" and filed[FILED_FOR_APPROVAL] is True, filed
    card = _card(roastery, filed["task_id"])
    assert (card.status, card.assigned_agent_id, card.review_mode) == ("review", None, "human")   # F421: not Inbox
    assert card.planning_data == {OWNERS_DECISION: True}
    owner = NS(workspace_id=roastery.ws, user_id=str(roastery.owner), auth_type="anonymous",
               user=NS(id=str(roastery.owner)))

    approved = asyncio.run(approve_task(card.id, _body({}), ctx=owner, db=roastery.db))

    assert approved["success"] is True and approved["status"] == "done", approved
    roastery.db.refresh(card)
    assert card.status == "done"


def test_an_unassigned_card_that_is_work_keeps_the_inbox(roastery):
    filed = _file(roastery, "Email the supplier to confirm delivery", "Harbour Imports, Friday's drop.")

    assert filed["success"] is True and filed["status"] == "inbox" and FILED_FOR_APPROVAL not in filed, filed
    assert _card(roastery, filed["task_id"]).review_mode == "human"            # D7 still holds it for a person


def test_a_models_own_status_review_is_still_dropped(roastery):
    """F289: a card nobody has worked on is never filed in Review on the model's word."""
    with patch("core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)):
        filed = asyncio.run(roastery.executor.execute(
            CREATE, {"title": "Count the Guji stock", "description": "Shelf and back room.", "status": "review"},
            roastery.chat))

    assert filed["success"] is True and filed["status"] == "inbox", filed
