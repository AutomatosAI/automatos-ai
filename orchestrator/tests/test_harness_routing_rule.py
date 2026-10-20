"""PRD-142 Wave 4 (W4-S6): platform_create_routing_rule.

The routing-rule handler inserts a workspace-scoped row into the routing_rules
table (read by the UniversalRouter at Tier 2a) and fails closed on a rule with no
target or no matcher. The HARNESS apply branch that mapped routing_rule_add (and
power_mode_*) prescriptions onto platform actions went with HARNESS's direct apply
helper (PRD-263 US-048), and with it this file's harness_service import and its
apscheduler stub. HARNESS's change-kind registry (US-049, a later PR) brings the
kinds it may propose, with their own tests. Dummy POSTGRES_* lets the config
import chain load without a DB.
"""
import asyncio
import os
from uuid import UUID

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")

from modules.tools.discovery.handlers_routing import create_routing_rule  # noqa: E402

_WS = UUID("00000000-0000-0000-0000-000000000001")


def _run(coro):
    return asyncio.run(coro)


class _FakeDB:
    """Captures the inserted RoutingRule; simulates a PK on refresh."""

    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 123

    def rollback(self):
        pass


# --- the handler -----------------------------------------------------------

@pytest.fixture
def _targets_are_ours(monkeypatch):
    """F149's tenancy check has its own tests; the fake DB here can't answer it."""
    monkeypatch.setattr("core.security.workspace_scope.routing_target_error", lambda *args: None)


def test_create_routing_rule_writes_row_workspace_scoped(_targets_are_ours):
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {
        "source_channel": "telegram", "target_agent_id": 7, "priority": 5,
    }))
    assert res["success"] is True
    assert db.committed is True
    assert len(db.added) == 1
    rule = db.added[0]
    assert rule.workspace_id == _WS  # scoped from the executor context, never params
    assert rule.source_channel == "telegram"
    assert rule.target_agent_id == 7
    assert rule.priority == 5
    assert rule.is_active is True


def test_create_routing_rule_requires_a_target():
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {"source_channel": "telegram"}))
    assert res["success"] is False
    assert db.added == []  # nothing written on a bad rule


def test_create_routing_rule_requires_a_matcher():
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {"target_agent_id": 7}))
    assert res["success"] is False
    assert db.added == []


def test_create_routing_rule_coerces_bad_priority(_targets_are_ours):
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {
        "source_pattern": "invoice", "target_agent_id": 1, "priority": "not-a-number",
    }))
    assert res["success"] is True
    assert db.added[0].priority == 0  # defaults rather than raising
