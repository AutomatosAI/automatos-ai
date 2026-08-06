"""The instruction is the approval — human-directed admin turns skip the ask.

Gerard, 2026-08-06, after 12 approval cards for one "delete the agents":
the confirmation gate exists to stop the AGENT deciding to do something
destructive on its own — not to make a workspace admin repeat an instruction
they just gave in chat. The gate now lets a confirmation-gated action run
when the call is human-directed:

    interactive chat lane      (server-threaded conversation_id — heartbeat/
                                cadence/board/mission lanes never carry one)
  AND driving user resolves to owner/admin in workspace_members
                               (fresh DB lookup keyed by the server-threaded
                                clerk principal — nothing model-writable)

Everything else keeps the ask, and every failure falls closed to the ask.
The execution is stamped ``human_directed`` — a third, distinct audit
marker beside ``autonomous`` (dial-skip) and ``approved_via_grant_id``
(card-approved grant). Attribution must be honest: instructed is neither
autonomous nor card-approved.

Pure unit tests — role resolver monkeypatched at its module seam; a source
pin holds the gate wiring. No DB, no app boot.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from unittest.mock import MagicMock

import pytest

import modules.tools.discovery.platform_executor as pe
from modules.tools.execution.telemetry import _build_router_decision

_WS = "28a228aa-dd63-46c7-baac-d29a0eb67283"
_CHAT_CTX = {"conversation_id": "conv-1", "user_id": "user_3CzR92xxx"}


# ---------------------------------------------------------------------------
# _human_directed_admin truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,expected", [
    ("owner", True),
    ("admin", True),
    ("Admin", True),      # role strings normalise
    ("editor", False),
    ("member", False),
    ("viewer", False),
    (None, False),        # no membership row
])
def test_role_gates_the_skip(monkeypatch, role, expected):
    monkeypatch.setattr(pe, "_workspace_role_for_clerk", lambda db, ws, uid: (
        role.strip().lower() if isinstance(role, str) else role
    ))
    assert pe._human_directed_admin(MagicMock(), _WS, dict(_CHAT_CTX)) is expected


def test_agent_lane_never_skips_and_never_looks_up(monkeypatch):
    """Heartbeat/cadence/board/mission lanes carry no conversation_id — the
    role lookup must not even run (an admin's id in an agent-lane context is
    not an instruction)."""
    def _boom(db, ws, uid):
        raise AssertionError("role lookup must not run for agent-lane calls")

    monkeypatch.setattr(pe, "_workspace_role_for_clerk", _boom)
    assert pe._human_directed_admin(MagicMock(), _WS, {"user_id": "user_x"}) is False
    assert pe._human_directed_admin(MagicMock(), _WS, {"board_task_id": 9}) is False
    assert pe._human_directed_admin(MagicMock(), _WS, None) is False


def test_missing_or_non_string_principal_fails_closed(monkeypatch):
    monkeypatch.setattr(pe, "_workspace_role_for_clerk", lambda *a: "owner")
    assert pe._human_directed_admin(MagicMock(), _WS, {"conversation_id": "c"}) is False
    assert pe._human_directed_admin(
        MagicMock(), _WS, {"conversation_id": "c", "user_id": 47}
    ) is False


def test_role_resolver_error_fails_closed():
    """Any DB failure in the resolver returns None → the ask stands."""
    db = MagicMock()
    db.query.side_effect = RuntimeError("db down")
    assert pe._workspace_role_for_clerk(db, _WS, "user_x") is None


# ---------------------------------------------------------------------------
# Audit attribution
# ---------------------------------------------------------------------------

def test_router_decision_carries_human_directed():
    d = _build_router_decision({}, human_directed=True)
    assert d == {"human_directed": True}


def test_markers_stay_distinct():
    d = _build_router_decision(
        {}, autonomous=True, approved_via_grant_id=7, human_directed=True
    )
    assert d["autonomous"] is True
    assert d["approved_via_grant_id"] == 7
    assert d["human_directed"] is True
    assert _build_router_decision({}) is None  # no markers → no row noise


# ---------------------------------------------------------------------------
# Gate wiring pin — the skip must guard the consume/ask branch
# ---------------------------------------------------------------------------

def test_gate_wiring_source_pin():
    src = (Path(pe.__file__)).read_text(encoding="utf-8")

    # The skip is computed from the caller context…
    assert "_human_directed_admin(self.db, self.workspace_id, caller_context)" in src

    # …and the consume/ask branch is entered only when NOT human-directed.
    gate = re.search(
        r"human_directed = bool\(.*?consume_tool_grant", src, re.S
    )
    assert gate is not None, "human_directed must be decided before the consume/ask branch"
    assert "and not human_directed" in gate.group(0), (
        "the requires_confirmation branch must be guarded by `not human_directed`"
    )

    # The execution is stamped for the audit trail.
    assert '{**result, "human_directed": True}' in src
