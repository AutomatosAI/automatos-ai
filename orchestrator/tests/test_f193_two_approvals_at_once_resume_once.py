"""F193 (a) — two approvals at once resume the call once.

grant_approval checked the status it had read, then set GRANTED in memory, so a
double click (or two admins) that both read PENDING both approved and both
resumed the stored call. The flip is now a compare-and-set: only the approval
that turns PENDING into GRANTED resumes it; the other is told it was already
decided and nothing runs again.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, text


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the approval race tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def pending(engine, new_session):
    from modules.tools.execution.tool_grants import issue_tool_grant

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f193')"), {"id": ws})
    grant = issue_tool_grant(s, ws, action="platform_delete_playbook", params={"playbook_id": 103},
                             permission_level="destructive", description="Delete a playbook",
                             caller_context={"mission_id": "m-1"})
    s.commit()
    yield ws, grant.id
    s = new_session.sweep()
    s.execute(text("DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_two_approvals_at_once_resume_the_call_once(pending, new_session, monkeypatch):
    import api.approval_grants as ag

    resumed = []

    async def _requeue(db, grant):
        resumed.append(grant.id)

    monkeypatch.setattr(ag, "_requeue_subject", _requeue)
    monkeypatch.setattr(ag, "_audit", lambda *a, **k: None)
    ws, grant_id = pending
    ctx = NS(workspace_id=UUID(ws), user=NS(id=7, clerk_user_id=None, email="owner@cafe.test"))
    first, second = new_session(), new_session()
    seen = ag._load_grant(second, ctx, grant_id)          # the second click read it while it was pending
    assert seen.status == "pending"                       # (held, so its session keeps this read)

    asyncio.run(ag.grant_approval(grant_id, ctx=ctx, db=first))
    with pytest.raises(HTTPException) as refused:
        asyncio.run(ag.grant_approval(grant_id, ctx=ctx, db=second))

    assert refused.value.status_code == 422 and "already decided" in refused.value.detail
    assert resumed == [grant_id]
