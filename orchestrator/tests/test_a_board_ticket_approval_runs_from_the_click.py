"""Approving a board ticket's old request authorises the ticket from the moment of the yes.

Grant #87 was requested on 8 Nov and approved on 10 Nov at 16:47:11. A pending
approval keeps the ``expires_at`` it was asked with (24 h), and the yes flipped only its
status, so the granted row was already past its expiry: ``is_authorising`` refused it,
the gate asked again and the ticket was back in Blocked with request #171 at 16:47:16.
``record_operator_consent`` (Run Now, a drag to In Progress) granted such a lapsed row
the same way, since ``find_pending_grant`` returns it.

Both ways of saying yes write ``approval_values``: a board ticket's approval runs a full
window from the yes. A tool call's card keeps its expiry, so a lapsed card still cannot
be approved (P256-FIX-N17-3).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.services.approval_grants as grants  # noqa: E402
from core.models.approval_grants import (  # noqa: E402
    ApprovalGrant, GrantStatus, KIND_APPROVAL, SUBJECT_BOARD_TASK, SUBJECT_TOOL_CALL,
)

ASKED = datetime(2026, 11, 8, 9, 30, tzinfo=timezone.utc)
CLICKED = datetime(2026, 11, 10, 16, 47, 11, tzinfo=timezone.utc)
WINDOW = timedelta(seconds=grants.DEFAULT_TTL_SECONDS)
CLICKER = "user:owner"


def _pending(subject_type: str, subject_id: str = "343") -> ApprovalGrant:
    return ApprovalGrant(
        id=87, workspace_id=uuid4(), subject_type=subject_type, subject_id=subject_id, kind=KIND_APPROVAL,
        status=GrantStatus.PENDING.value, requested_at=ASKED, expires_at=ASKED + WINDOW,
    )


def test_a_yes_to_an_old_board_ticket_request_authorises_it_for_a_full_window():
    grant = grants.grant_grant(_pending(SUBJECT_BOARD_TASK), granted_by=CLICKER, now=CLICKED)
    assert grant.status == GrantStatus.GRANTED.value and grant.granted_at == CLICKED
    assert grant.expires_at == CLICKED + WINDOW
    assert grants.is_authorising(grant, now=CLICKED + timedelta(seconds=5))  # the gate's look, 5 s later


def test_a_tool_call_card_keeps_its_expiry_when_approved():
    asked_until = ASKED + WINDOW
    grant = grants.grant_grant(_pending(SUBJECT_TOOL_CALL, "platform_send:ab12"), granted_by=CLICKER, now=CLICKED)
    assert grant.expires_at == asked_until
    assert not grants.is_authorising(grant, now=CLICKED)


def test_the_approvals_api_writes_the_same_window():
    from api.approval_grants import _cas_grant

    written = {}

    class _Query:
        def filter(self, *conds):
            return self

        def update(self, values, synchronize_session=False):
            written.update(values)
            return 1

    db = SimpleNamespace(query=lambda model: _Query())
    grant = _pending(SUBJECT_BOARD_TASK)
    assert _cas_grant(db, grant, actor=CLICKER, now=CLICKED) is True
    assert written == {"status": GrantStatus.GRANTED.value, "granted_at": CLICKED, "granted_by": CLICKER,
                       "expires_at": CLICKED + WINDOW}
    assert grant.expires_at == CLICKED + WINDOW and grants.is_authorising(grant, now=CLICKED)

    written.clear()
    card = _pending(SUBJECT_TOOL_CALL, "platform_send:ab12")
    assert _cas_grant(db, card, actor=CLICKER, now=CLICKED) is True
    assert "expires_at" not in written and card.expires_at == ASKED + WINDOW


def test_run_now_on_a_ticket_whose_request_lapsed_authorises_it(monkeypatch):
    from services import board_consent

    lapsed = _pending(SUBJECT_BOARD_TASK)
    monkeypatch.setattr(grants, "find_active_grant", lambda db, ws, **kw: None)
    monkeypatch.setattr(grants, "find_pending_grant", lambda db, ws, **kw: lapsed)

    class _DB:
        def commit(self):
            pass

    out = board_consent.record_operator_consent(_DB(), workspace_id="ws", task_id=343, agent_id=15,
                                                actor=CLICKER, why=board_consent.WHY_RUN_NOW)
    assert out == board_consent.GRANTED
    assert grants.is_authorising(lapsed), "Run Now's yes on a lapsed request authorised nothing"
