"""PRD-196 S5 (P2-15, governance J.9) — audit retention with a hard Art.12 floor.

- PURE: the 180-day floor binds at read — a configured 30-day window computes a
  >= 180-day cutoff (a config can never dip under the legal minimum).
- DB-backed (real Postgres, skips with none up): the sweep deletes only rows past
  the cutoff and writes ONE ``audit:retention_sweep`` summary row per affected
  workspace (system actor), leaving recent rows untouched. Workspaces seeded first.
"""
from __future__ import annotations

import os

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import uuid  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

import pytest  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from services.audit_retention import (  # noqa: E402
    AUDIT_RETENTION_FLOOR_DAYS,
    compute_cutoff,
    effective_retention_days,
    sweep_expired_audit_logs,
)


# ---------------------------------------------------------------------------
# PURE — the floor binds
# ---------------------------------------------------------------------------

def test_effective_retention_clamps_to_floor():
    assert AUDIT_RETENTION_FLOOR_DAYS == 180
    assert effective_retention_days(30) == 180, "below-floor config is clamped up"
    assert effective_retention_days(179) == 180
    assert effective_retention_days(365) == 365
    assert effective_retention_days(500) == 500


def test_cutoff_respects_floor():
    now = datetime(2026, 7, 11, tzinfo=timezone.utc)
    # A 30-day config still yields a cutoff at least 180 days back.
    cutoff = compute_cutoff(now, 30)
    assert (now - cutoff) >= timedelta(days=AUDIT_RETENTION_FLOOR_DAYS)
    # A generous window is honoured exactly.
    assert compute_cutoff(now, 365) == now - timedelta(days=365)


# ---------------------------------------------------------------------------
# DB-backed — sweep deletes only expired + writes the summary row
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from sqlalchemy import create_engine, text
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"audit retention suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(engine, new_session):
    from sqlalchemy import text

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
        {"id": ws, "n": "gov-retention"},
    )
    s.commit()
    s.close()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM audit_logs WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _seed(session, ws, action, created_at):
    from sqlalchemy import text

    session.execute(
        text(
            "INSERT INTO audit_logs (workspace_id, actor_type, action, created_at) "
            "VALUES (CAST(:w AS uuid), 'system', :a, :ts)"
        ),
        {"w": ws, "a": action, "ts": created_at},
    )


def test_sweep_deletes_only_expired_and_writes_summary(new_session, workspace):
    from sqlalchemy import text

    now = datetime.now(timezone.utc)
    s = new_session()
    # 2 well past the 365-day default cutoff, 2 recent.
    _seed(s, workspace, "policy:deny", now - timedelta(days=400))
    _seed(s, workspace, "gdpr:export", now - timedelta(days=500))
    _seed(s, workspace, "policy:allow", now - timedelta(days=5))
    _seed(s, workspace, "approval_grant:granted", now - timedelta(days=1))
    s.commit()

    result = sweep_expired_audit_logs(s, now=now)
    assert result["total_deleted"] == 2, "only the two >365-day rows are deleted"
    assert result["workspaces_affected"] == 1

    # recent rows survive
    survivors = s.execute(
        text(
            "SELECT action FROM audit_logs WHERE workspace_id = CAST(:w AS uuid) "
            "AND action <> 'audit:retention_sweep'"
        ),
        {"w": workspace},
    ).fetchall()
    actions = {r[0] for r in survivors}
    assert actions == {"policy:allow", "approval_grant:granted"}

    # exactly one system summary row, carrying the deleted count
    summary = s.execute(
        text(
            "SELECT actor_type, details FROM audit_logs "
            "WHERE workspace_id = CAST(:w AS uuid) AND action = 'audit:retention_sweep'"
        ),
        {"w": workspace},
    ).fetchall()
    assert len(summary) == 1
    assert summary[0][0] == "system"
    assert summary[0][1]["rows_deleted"] == 2
    s.close()
