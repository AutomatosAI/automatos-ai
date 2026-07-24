"""PRD-181 S1 — audit-log completeness (also EU-AI-Act Art.12).

Every policy verdict — allow, ask, and deny — must write an ``AuditLog`` row,
tenant-scoped, carrying actor + tool + verdict + reason. The policy *bus* is the
single write point: an audit handler attaches to it (the seam bus.py:18 was built
for) and the gate chokepoint fires the bus for every verdict.

Stdlib-only: the ``AuditService`` DB write is exercised against a fake session
that records the rows staged, so no Postgres is needed. The bus + handler are
pure Python.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.policy.audit_handler import (  # noqa: E402
    audit_policy_verdict,
    make_audit_handler,
)
from modules.policy.bus import EventContext, PolicyBus  # noqa: E402
from modules.policy.types import Event, PolicyError, Verdict  # noqa: E402


# ---------------------------------------------------------------------------
# A fake sync DB session that records staged AuditLog rows (no Postgres).
# ---------------------------------------------------------------------------

class _FakeSession:
    def __init__(self) -> None:
        self.added: List[Any] = []
        self.committed = False

    def add(self, obj: Any) -> None:
        self.added.append(obj)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:  # pragma: no cover - error path only
        self.committed = False


def _ctx(**kw: Any) -> EventContext:
    base: Dict[str, Any] = dict(
        workspace_id=kw.pop("workspace_id", uuid4()),
        agent_id=kw.pop("agent_id", 7),
        tool_name=kw.pop("tool_name", "platform_delete_agent"),
        tool_input=kw.pop("tool_input", {"id": 7}),
        caller_context=kw.pop("caller_context", {"user_id": 42, "system_role": "user"}),
    )
    ctx = EventContext(**base)
    ctx.data.update(kw)
    return ctx


# ---------------------------------------------------------------------------
# test_audit_completeness — allow, ask, deny each write a row with the fields.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "verdict,expected",
    [
        (Verdict.allow("posture=balanced routes read to auto"), "allow"),
        (
            Verdict.ask(
                PolicyError(
                    code="approval_required",
                    message_for_model="needs human approval; NOT executed",
                    remediation="approve in the queue",
                    retryable=True,
                )
            ),
            "ask",
        ),
        (
            Verdict.deny(
                PolicyError(
                    code="super_admin_required",
                    message_for_model="restricted to super admin",
                    remediation="cannot be taken by an agent",
                    retryable=False,
                )
            ),
            "deny",
        ),
    ],
)
def test_audit_completeness(verdict, expected):
    """Each verdict disposition writes exactly one AuditLog row carrying
    tenant, actor, tool, verdict, and reason."""
    db = _FakeSession()
    ws = uuid4()
    ctx = _ctx(workspace_id=ws, verdict=verdict, risk="destructive")

    row = audit_policy_verdict(db, Event.PRE_TOOL_USE, ctx)

    assert row is not None, "a verdict must produce an audit row"
    assert len(db.added) == 1, "exactly one row per verdict (no double-logging)"
    assert db.committed is True

    # tenant
    assert str(row.workspace_id) == str(ws)
    # actor — the human user when present
    assert row.user_id == 42
    # tool
    assert row.resource_type == "tool"
    assert row.resource_name == "platform_delete_agent"
    # verdict + reason live in the structured details
    assert row.details["verdict"] == expected
    assert row.details["reason"] == verdict.reason
    assert row.action == f"policy:{expected}"


def test_audit_records_the_risk_tier_and_error_code():
    """The row carries the risk tier and (for deny/ask) the policy error code —
    the Art.12 record the rest of the wave reads."""
    db = _FakeSession()
    v = Verdict.deny(PolicyError(code="budget_exceeded", message_for_model="over budget"))
    ctx = _ctx(verdict=v, risk="external_side_effect")

    row = audit_policy_verdict(db, Event.PRE_TOOL_USE, ctx)

    assert row.details["risk"] == "external_side_effect"
    assert row.details["error_code"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# test_audit_is_per_tenant — rows carry workspace_id.
# ---------------------------------------------------------------------------

def test_audit_is_per_tenant():
    """Two different workspaces write rows tagged with their own workspace_id."""
    ws_a, ws_b = uuid4(), uuid4()
    db = _FakeSession()

    audit_policy_verdict(db, Event.PRE_TOOL_USE, _ctx(workspace_id=ws_a, verdict=Verdict.allow("a")))
    audit_policy_verdict(db, Event.PRE_TOOL_USE, _ctx(workspace_id=ws_b, verdict=Verdict.allow("b")))

    tenants = {str(r.workspace_id) for r in db.added}
    assert tenants == {str(ws_a), str(ws_b)}


# ---------------------------------------------------------------------------
# System / agent actor — no human user_id (the Art.12 hard case).
# The old handlers_members pattern SKIPPED these; S1 must NOT — an agent tool
# call is exactly what Art.12 needs recorded.
# ---------------------------------------------------------------------------

def test_agent_actor_without_user_is_still_audited():
    """A tool call with no human principal (agent factory / heartbeat) is still
    audited — user_id NULL, actor_type='agent', agent id preserved."""
    db = _FakeSession()
    ctx = _ctx(caller_context=None, agent_id=99, verdict=Verdict.allow("agent read"))

    row = audit_policy_verdict(db, Event.PRE_TOOL_USE, ctx)

    assert row is not None, "an agent/system actor MUST be audited (Art.12)"
    assert row.user_id is None, "no human principal ⇒ user_id NULL, not skipped"
    assert row.details["actor_type"] == "agent"
    assert row.details["agent_id"] == 99


def test_system_actor_type_when_no_agent_and_no_user():
    db = _FakeSession()
    ctx = _ctx(caller_context=None, agent_id=0, verdict=Verdict.allow("system"))
    row = audit_policy_verdict(db, Event.PRE_TOOL_USE, ctx)
    assert row.details["actor_type"] == "system"


# ---------------------------------------------------------------------------
# Bus integration — the handler attaches to the bus and fires on the event.
# ---------------------------------------------------------------------------

def test_handler_attaches_to_bus_and_fires():
    """The audit handler registered on the bus writes a row when the bus fires,
    and returns no verdict (audit is a side-effect, never a policy opinion)."""
    db = _FakeSession()
    bus = PolicyBus()
    bus.register(Event.PRE_TOOL_USE, make_audit_handler(lambda: db))

    verdict = bus.fire(Event.PRE_TOOL_USE, _ctx(verdict=Verdict.allow("ok")))

    assert len(db.added) == 1, "the bus handler wrote the audit row"
    # audit must not change the merged verdict — it returns None (no opinion)
    assert verdict.decision.value == "allow"


def test_handler_never_raises_into_the_bus():
    """A broken DB must not take the tool loop down — the handler swallows and
    returns None so the bus treats it as no-opinion."""
    class _Boom:
        def add(self, obj): raise RuntimeError("db down")
        def commit(self): raise RuntimeError("db down")
        def rollback(self): pass

    bus = PolicyBus()
    bus.register(Event.PRE_TOOL_USE, make_audit_handler(lambda: _Boom()))
    # must not raise
    verdict = bus.fire(Event.PRE_TOOL_USE, _ctx(verdict=Verdict.allow("ok")))
    assert verdict.decision.value == "allow"


def test_no_verdict_in_context_is_a_noop():
    """A fired event that carries no verdict (a non-policy event) writes nothing."""
    db = _FakeSession()
    ctx = _ctx()
    ctx.data.pop("verdict", None)
    row = audit_policy_verdict(db, Event.PRE_TOOL_USE, ctx)
    assert row is None
    assert db.added == []
