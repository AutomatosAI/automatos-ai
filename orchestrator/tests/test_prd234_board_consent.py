"""PRD-234: a human's Run Now / drag to In Progress records the board-task grant."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.services.approval_grants as grants  # noqa: E402
from services import board_consent  # noqa: E402


class _DB:
    def __init__(self):
        self.commits = 0
        self.rollbacks = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _wire(monkeypatch, *, active=None, pending=None):
    created = []
    granted = []
    monkeypatch.setattr(grants, "find_active_grant", lambda db, ws, **kw: active)
    monkeypatch.setattr(grants, "find_pending_grant", lambda db, ws, **kw: pending)

    def _create(db, ws, **kw):
        g = SimpleNamespace(id=99, status="pending", **kw)
        created.append(g)
        return g

    def _grant(g, *, granted_by, now=None):
        g.status = "granted"
        g.granted_by = granted_by
        granted.append(g)
        return g

    monkeypatch.setattr(grants, "create_grant", _create)
    monkeypatch.setattr(grants, "grant_grant", _grant)
    return created, granted


def test_an_active_grant_means_nothing_to_do(monkeypatch):
    created, granted = _wire(monkeypatch, active=SimpleNamespace(id=1))
    db = _DB()
    assert board_consent.record_operator_consent(db, workspace_id="ws", task_id=71, agent_id=15,
                                                 actor="user:2", why=board_consent.WHY_RUN_NOW) == "active"
    assert not created and not granted and db.commits == 0


def test_a_pending_grant_is_approved_by_the_operator(monkeypatch):
    pending = SimpleNamespace(id=11, status="pending")
    created, granted = _wire(monkeypatch, pending=pending)
    db = _DB()
    out = board_consent.record_operator_consent(db, workspace_id="ws", task_id=71, agent_id=15,
                                                actor="user:2", why=board_consent.WHY_MOVED_TO_IN_PROGRESS)
    assert out == "granted" and pending.status == "granted" and pending.granted_by == "user:2"
    assert not created and db.commits == 1


def test_no_grant_yet_creates_one_already_granted(monkeypatch):
    created, granted = _wire(monkeypatch)
    db = _DB()
    out = board_consent.record_operator_consent(db, workspace_id="ws", task_id=72, agent_id=15,
                                                actor="user:2", why=board_consent.WHY_RUN_NOW)
    assert out == "created" and len(created) == 1
    assert created[0].subject_type == "board_task" and created[0].subject_id == "72"
    assert created[0].reason == board_consent.WHY_RUN_NOW and created[0].status == "granted"
    assert db.commits == 1


def test_consent_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(grants, "find_active_grant", _boom)
    db = _DB()
    assert board_consent.record_operator_consent(db, workspace_id="ws", task_id=1, agent_id=None,
                                                 actor="user:2", why="x") == "error"
    assert db.rollbacks == 1


def test_actor_ref_matches_the_approvals_api_shape():
    assert board_consent.actor_ref(SimpleNamespace(user_id=2)) == "user:2"
    assert board_consent.actor_ref(SimpleNamespace(user_id=None, internal_user_id=7)) == "user:7"
    assert board_consent.actor_ref(SimpleNamespace()) == "user:unknown"
