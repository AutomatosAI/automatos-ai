"""F207 — "decisions needed" is never silently empty.

Refresh 5's boot log: "KPI decisions-needed failed: column escalation_level does
not exist". The reports query read and sorted by agent_reports.escalation_level,
which nothing writes and c1's table no longer has; the endpoint caught the error
and answered "0 items", so the Command Centre showed no report awaiting the
owner's approval. Its missions half matched state_type 'BLOCKED' against the
stored 'blocked', so no blocked mission ever showed either. The query no longer
reads the column, matches the stored value, and a failure says so instead of
answering "nothing needs you".
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from uuid import UUID

import pytest
from sqlalchemy import create_engine, text

# agent_reports as c1 has it (the columns this query reads; no escalation_level).
# A DB built from the models has no agent_reports at all, so the test builds it.
C1_AGENT_REPORTS = """
    CREATE TABLE agent_reports (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL,
        title TEXT, summary TEXT, status TEXT, agent_name TEXT,
        requires_approval BOOLEAN DEFAULT FALSE,
        acknowledged_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ DEFAULT now()
    )
"""


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the decisions-needed tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def waiting(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    built = not s.execute(text("SELECT to_regclass('agent_reports') IS NOT NULL")).scalar()
    if built:
        s.execute(text(C1_AGENT_REPORTS))
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f207')"), {"id": ws})
    s.execute(text(
        "INSERT INTO agent_reports (workspace_id, title, summary, status, agent_name, requires_approval) "
        "VALUES (CAST(:w AS uuid), 'Wholesale price list', 'New prices for the cafés', 'pending', 'Numbers', TRUE)"),
        {"w": ws})
    s.execute(text(
        "INSERT INTO orchestration_runs (workspace_id, goal, created_by, state, state_type) "
        "VALUES (CAST(:w AS uuid), 'Launch the autumn blend', 'owner', 'awaiting_approval', 'blocked')"), {"w": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM orchestration_runs WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DROP TABLE agent_reports") if built else
              text("DELETE FROM agent_reports WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_a_report_awaiting_approval_and_a_blocked_mission_both_show(waiting, new_session):
    from api.kpi_api import get_decisions_needed

    out = asyncio.run(get_decisions_needed(limit=10, ctx=NS(workspace_id=UUID(waiting)), db=new_session()))

    assert [item["kind"] for item in out["items"]] == ["report", "mission"]     # night: []
    assert out["items"][0]["title"] == "Wholesale price list" and out["items"][0]["escalation_level"] is None
    assert "error" not in out


def test_a_list_that_could_not_be_loaded_says_so():
    from api.kpi_api import get_decisions_needed

    class _Down:
        def execute(self, *a, **k):
            raise RuntimeError("connection reset")

    out = asyncio.run(get_decisions_needed(limit=10, ctx=NS(workspace_id=uuid.uuid4()), db=_Down()))

    assert out["items"] == [] and out["error"]                                  # night: a silent 0
    assert "connection reset" not in out["error"]
