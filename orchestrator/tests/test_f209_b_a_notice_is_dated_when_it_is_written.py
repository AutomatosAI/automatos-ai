"""F209 (b) — a notification's time is when it was written.

notifications.created_at defaulted to now(), which in Postgres is the
TRANSACTION's start. A board run's session opens its transaction as the run
begins and writes its task_complete at the end, so the notice was dated at the
run's start: it sorted before things that happened earlier in the bell and read
"N min ago" from when the run began (#483/#484 "completed" 0–2 s after they were
created). The writer now stamps clock_timestamp(), the moment of the insert.
"""
from __future__ import annotations

import uuid

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
        pytest.skip(f"the notification-time tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


# The table as alembic/versions/prd128_notifications.py makes it (a DB built from
# the models has none), with the default this finding is about.
PRD128_NOTIFICATIONS = """
    CREATE TABLE notifications (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        user_id INTEGER, event_type VARCHAR(50) NOT NULL, title VARCHAR(255) NOT NULL,
        message TEXT, link_type VARCHAR(30), link_id TEXT, agent_id INTEGER, agent_name VARCHAR(100),
        status VARCHAR(20) NOT NULL DEFAULT 'ok', read_at TIMESTAMPTZ, dismissed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""


@pytest.fixture
def workspace(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    built = not s.execute(text("SELECT to_regclass('notifications') IS NOT NULL")).scalar()
    if built:
        s.execute(text(PRD128_NOTIFICATIONS))
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f209b')"), {"id": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DROP TABLE notifications") if built else
              text("DELETE FROM notifications WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def test_a_notice_written_late_in_a_long_transaction_is_dated_when_it_was_written(workspace, new_session):
    from core.services.notification_dispatcher import NotificationDispatcher

    run = new_session()
    began = run.execute(text("SELECT now()")).scalar()               # the run's transaction starts
    run.execute(text("SELECT pg_sleep(1.2)"))                        # ...the run works...
    NotificationDispatcher(run, workspace)._insert_in_app(
        user_id=None, event_type="task_complete", title="Task: Weekly numbers", message="Revenue: £4,210",
        link_type="task", link_id="51", agent_id=None, agent_name=None, status="ok")
    written = run.execute(text("SELECT created_at FROM notifications WHERE workspace_id = CAST(:w AS uuid)"),
                          {"w": workspace}).scalar()
    run.commit()

    assert (written - began).total_seconds() >= 1.0                  # before: 0, the transaction's start
