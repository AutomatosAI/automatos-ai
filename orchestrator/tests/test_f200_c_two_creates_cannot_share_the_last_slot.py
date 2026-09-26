"""F200 (review MEDIUM) — two creates cannot share a workspace's last agent slot.

The limit check counted the workspace's agents without a lock, so two creates at
limit-1 (a double click, two admins, a package install beside a manual create)
both passed and the workspace ended over its plan. The count now runs under a
per-workspace advisory lock taken without waiting: the second create is told the
workspace is busy and to try again, and a retry after the first lands meets the
limit. It never waits, so a create whose transaction spans awaits cannot freeze
the event loop (F105).
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace as NS

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the agent-limit race tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


def _agent(session, ws, name):
    session.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration, owner_type) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json), 'workspace')"), {"n": name, "w": ws})


@pytest.fixture
def one_slot_left(engine, new_session):
    from core.models.workspaces import Workspace
    from services.agent_quota import plan_agent_limit

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name, plan) VALUES (CAST(:id AS uuid), 'f200c', 'basic')"), {"id": ws})
    _, limit = plan_agent_limit(s.get(Workspace, uuid.UUID(ws)))
    for n in range(limit - 1):
        _agent(s, ws, f"Barista {n}")
    s.commit()
    yield NS(ws=uuid.UUID(ws), limit=limit, new=new_session)
    s = new_session.sweep()
    s.execute(text("DELETE FROM agents WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_two_creates_at_the_last_slot_cannot_both_pass(one_slot_left):
    from services.agent_quota import agent_limit_refusal

    first, second = one_slot_left.new(), one_slot_left.new()

    assert agent_limit_refusal(first, one_slot_left.ws) is None          # the first create holds the count
    raced = agent_limit_refusal(second, one_slot_left.ws)
    assert raced is not None and raced.get("busy") is True               # before: None, and both created
    assert raced["http_status"] == 409 and "try again in a moment" in raced["message"]

    _agent(first, str(one_slot_left.ws), "Roaster")                      # the first create lands
    first.commit()
    second.rollback()
    retry = agent_limit_refusal(second, one_slot_left.ws)

    assert retry is not None and retry.get("over_quota") is True
    assert f"includes {one_slot_left.limit} agents and this workspace has {one_slot_left.limit}" in retry["message"]


def test_one_transaction_can_check_again(one_slot_left):
    """A package's several clones run in one transaction: the lock is re-entrant."""
    from services.agent_quota import agent_limit_refusal

    installing = one_slot_left.new()
    assert agent_limit_refusal(installing, one_slot_left.ws) is None
    assert agent_limit_refusal(installing, one_slot_left.ws) is None


def test_the_count_lock_is_never_released_as_a_read():
    """Review HIGH: a session that only ran the lock looked read-only to F105's
    release, which rolls it back before a model round and so dropped the lock."""
    from core.database.read_release import is_plain_read

    assert is_plain_read("SELECT pg_try_advisory_xact_lock(hashtext('agent-limit:c1'))") is False   # before: True
    assert is_plain_read("SELECT pg_advisory_xact_lock(1)") is False
    assert is_plain_read("SELECT count(*) FROM agents") is True
