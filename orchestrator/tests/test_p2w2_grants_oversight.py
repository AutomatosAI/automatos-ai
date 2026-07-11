"""PRD-196 S1 (P2-15, governance I.1) — the approvals list carries Art.14 oversight.

The inbox card must show *why* a human is in the loop without inventing a
rationale client-side. The grant-list endpoint enriches every row with the pure
``oversight_for_risk`` mapping (tier + rationale + requires_approval), fail-safe
to human-in-the-loop when the grant's risk tier is unknown/absent.

Pure: fake ctx + a MagicMock session whose query chain yields fake grants; no
DB, no network (the PRD-185 S12 / S2 test shape).
"""

from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import asyncio  # noqa: E402
import uuid  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import approval_grants as grants_api  # noqa: E402
from modules.policy.ai_act import OversightTier  # noqa: E402
from modules.policy.policy_document import RISK_DESTRUCTIVE, RISK_READ  # noqa: E402


class _Grant:
    """A minimal ApprovalGrant-shaped fake with the real ``to_dict`` keys."""

    def __init__(self, grant_id: int, risk_tier, subject_id="42"):
        self.id = grant_id
        self.risk_tier = risk_tier
        self.subject_id = subject_id
        self.subject_type = "board_task"
        self.status = "pending"

    def to_dict(self):
        return {
            "id": self.id,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "risk_tier": self.risk_tier,
            "status": self.status,
        }


def _ctx(workspace_id=None):
    user = SimpleNamespace(id="u", email=None, role="user", system_role="user", clerk_user_id="c")
    return SimpleNamespace(user=user, workspace_id=workspace_id or uuid.uuid4())


def _db_returning(rows):
    db = MagicMock()
    chain = db.query.return_value.filter.return_value
    chain.order_by.return_value.limit.return_value.all.return_value = rows
    # ``.filter().filter()`` (status filter) must also terminate on the chain.
    chain.filter.return_value.order_by.return_value.limit.return_value.all.return_value = rows
    return db


# ---------------------------------------------------------------------------
# The pure enrichment mapping
# ---------------------------------------------------------------------------

def test_grant_with_oversight_maps_known_risk():
    g = _Grant(1, RISK_DESTRUCTIVE)
    out = grants_api.grant_with_oversight(g)
    assert out["id"] == 1
    assert out["oversight"]["tier"] == OversightTier.HUMAN_IN_THE_LOOP.value
    assert out["oversight"]["requires_approval"] is True
    assert "approve" in out["oversight"]["rationale"].lower()


def test_grant_with_oversight_read_is_monitor():
    out = grants_api.grant_with_oversight(_Grant(2, RISK_READ))
    assert out["oversight"]["tier"] == OversightTier.MONITOR.value
    assert out["oversight"]["requires_approval"] is False


def test_grant_with_oversight_unknown_falls_safe_to_human_in_the_loop():
    # An unknown / absent risk tier is treated as needing human approval — the
    # card never silently lowers oversight (ai_act fallback).
    for bad in (None, "", "high", "not-a-risk-class"):
        out = grants_api.grant_with_oversight(_Grant(3, bad))
        assert out["oversight"]["tier"] == OversightTier.HUMAN_IN_THE_LOOP.value
        assert out["oversight"]["requires_approval"] is True


# ---------------------------------------------------------------------------
# The list endpoint carries oversight for every grant
# ---------------------------------------------------------------------------

def test_list_grants_carries_oversight():
    rows = [_Grant(10, RISK_DESTRUCTIVE), _Grant(11, RISK_READ), _Grant(12, None)]
    resp = asyncio.run(grants_api.list_grants(status=None, ctx=_ctx(), db=_db_returning(rows)))

    grants = resp["grants"]
    assert len(grants) == 3
    assert all("oversight" in g for g in grants), "every grant carries an oversight block"
    by_id = {g["id"]: g for g in grants}
    assert by_id[10]["oversight"]["tier"] == OversightTier.HUMAN_IN_THE_LOOP.value
    assert by_id[11]["oversight"]["tier"] == OversightTier.MONITOR.value
    # unknown risk still gets a fail-safe oversight block, never omitted
    assert by_id[12]["oversight"]["tier"] == OversightTier.HUMAN_IN_THE_LOOP.value
    assert by_id[10]["oversight"]["rationale"]
