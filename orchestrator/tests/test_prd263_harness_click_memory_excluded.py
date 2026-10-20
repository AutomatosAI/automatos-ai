"""PRD-263 US-055 — approving a HARNESS card writes no decision memory.

P256-FIX-N13-4 stores each click on a call's approval card as one workspace decision
memory (services.click_memory, wrapping api/approval_grants' approve and deny routes).
A HARNESS change's card, the tool_call grant for ``platform_harness_apply``, comes back
every week a change is proposed, so its decisions would pile up in recall and in the
eval's memory; the HARNESS ledger and the audit rows are their record (section 11-Q18,
built at its default: exclude). Any other call's card is remembered as before, even one
that makes the same change from the owner's chat. api/approval_grants.py is unchanged.

Real database (``db_session``, rolled back); the click's resume and the memory store
are recorded at their boundaries.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from tests import helpers_harness_grants as hgx

OWNER = NS(id=7, clerk_user_id=None, email="owner@harness-memory.test")
HEARTBEAT = "platform_configure_agent_heartbeat"


@pytest.fixture
def routes(monkeypatch):
    """The approval routes with the resume, audit and chat note recorded or off, and every
    memory the click would store recorded."""
    import api.approval_grants as ag
    import services.click_memory as click_memory

    stored = []

    async def _resume(db, grant):
        return True

    async def _store(db, grant, content, owner):
        stored.append((grant.id, content))
        return {"success": True}

    monkeypatch.setattr(ag, "_requeue_subject", _resume)
    monkeypatch.setattr(ag, "_audit", lambda *a, **k: None)
    monkeypatch.setattr(ag, "said_in_the_chat", lambda *a, **k: None)
    monkeypatch.setattr(click_memory, "_store", _store)
    return NS(ag=ag, stored=stored)


@pytest.fixture
def ws(db_session, seed_workspace):
    return seed_workspace(name="harness-click-memory")


def _ctx(ws):
    return NS(workspace_id=UUID(ws), user=OWNER)


def _heartbeat_card(db, ws) -> int:
    """The owner's chat asked Auto to change a heartbeat: that call's own card."""
    from core.models.approval_grants import SUBJECT_TOOL_CALL, ApprovalGrant
    from modules.tools.execution.tool_grants import tool_call_subject_id

    params = {"agent_id": 1, "interval_minutes": 120}
    grant = ApprovalGrant(workspace_id=UUID(ws), subject_type=SUBJECT_TOOL_CALL, tool_name=HEARTBEAT, status="pending",
                          subject_id=tool_call_subject_id(ws, HEARTBEAT, params),
                          question_md="Change Scout's heartbeat to every 120 minutes",
                          details={"action": HEARTBEAT, "params": params, "lane": "chat"})
    db.add(grant)
    db.flush()
    return grant.id


@pytest.mark.parametrize("route", ["grant_approval", "deny_approval"])
def test_a_click_on_a_harness_card_writes_no_memory(db_session, ws, routes, route):
    grant_id = hgx.grant(db_session, ws, "rx-1-001", "pending")

    asyncio.run(getattr(routes.ag, route)(grant_id, ctx=_ctx(ws), db=db_session))

    assert hgx.grant_status(db_session, grant_id) == ("granted" if route == "grant_approval" else "denied")
    assert routes.stored == []


def test_approving_another_calls_card_still_writes_one_memory(db_session, ws, routes):
    grant_id = _heartbeat_card(db_session, ws)

    asyncio.run(routes.ag.grant_approval(grant_id, ctx=_ctx(ws), db=db_session))

    ((remembered, content),) = routes.stored
    assert remembered == grant_id and content.startswith("The owner approved the card")


def test_a_harness_card_is_known_by_its_stored_call_or_its_tool_name():
    from services.click_memory import _a_harness_change

    assert _a_harness_change(NS(details={"action": hgx.APPLY}, tool_name=None))
    assert _a_harness_change(NS(details=None, tool_name=hgx.APPLY))
    assert not _a_harness_change(NS(details={"action": HEARTBEAT}, tool_name=HEARTBEAT))
