"""PRD-196 S3 (P2-15, governance I.2/I.5) — audit-log read view + honest status.

Two layers:
- PURE (no DB): datetime-filter parsing (bad value → 422) and the resilient
  retention-status reader (honest before/after S5 lands).
- DB-backed (real Postgres, skips cleanly with none up, the test.yml pattern):
  the read is fail-closed to ``ctx.workspace_id`` (another workspace's rows never
  return, whatever the params), the action-prefix/actor/date filters compose, and
  ``/status`` reports the flag honestly (OFF is OFF).

Workspaces are seeded FIRST (the PRD-158 FK lesson) before any audit/grant row.
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
from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import governance as gov  # noqa: E402


def _ctx(ws):
    user = SimpleNamespace(id="u", system_role="user", clerk_user_id="c")
    return SimpleNamespace(user=user, workspace_id=ws)


# ---------------------------------------------------------------------------
# PURE — datetime filter parsing + resilient retention reader
# ---------------------------------------------------------------------------

def test_parse_dt_accepts_iso_and_z():
    assert gov._parse_dt(None, "since") is None
    assert gov._parse_dt("2026-07-11T10:00:00Z", "since").year == 2026
    assert gov._parse_dt("2026-07-11T10:00:00+00:00", "until").month == 7


def test_parse_dt_rejects_garbage_as_422():
    with pytest.raises(HTTPException) as ei:
        gov._parse_dt("not-a-date", "since")
    assert ei.value.status_code == 422


def test_retention_status_is_honest_when_reader_present():
    # After S5 the reader exists — status carries a floor-enforced number.
    st = gov._retention_status()
    assert set(st) == {"retention_days", "floor_days", "configured"}
    if st["configured"]:
        assert st["retention_days"] >= st["floor_days"] >= 180


# ---------------------------------------------------------------------------
# DB-backed — fail-closed tenancy, filter composition, honest status
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
        pytest.skip(f"governance audit suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def two_workspaces(engine, new_session):
    from sqlalchemy import text

    ws_a, ws_b = str(uuid.uuid4()), str(uuid.uuid4())
    s = new_session()
    for ws, name in ((ws_a, "gov-a"), (ws_b, "gov-b")):
        s.execute(
            text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
            {"id": ws, "n": name},
        )
    s.commit()
    s.close()
    yield ws_a, ws_b
    s = new_session()
    for ws in (ws_a, ws_b):
        s.execute(text("DELETE FROM audit_logs WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM approval_grants WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
        s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def _seed_audit(session, ws, *, action, actor_type="system", created_at=None):
    from sqlalchemy import text

    session.execute(
        text(
            "INSERT INTO audit_logs (workspace_id, actor_type, action, created_at) "
            "VALUES (CAST(:w AS uuid), :a, :act, :ts)"
        ),
        {"w": ws, "a": actor_type, "act": action, "ts": created_at or datetime.now(timezone.utc)},
    )


def test_audit_log_scoped_to_ctx_workspace(new_session, two_workspaces):
    ws_a, ws_b = two_workspaces
    s = new_session()
    _seed_audit(s, ws_a, action="policy:deny")
    _seed_audit(s, ws_a, action="gdpr:export")
    _seed_audit(s, ws_b, action="policy:deny")
    s.commit()

    resp = asyncio.run(gov.get_audit_log(ctx=_ctx(ws_a), db=s))
    actions = [r["action"] for r in resp["rows"]]
    assert resp["total"] == 2, "only this workspace's rows are counted"
    assert set(actions) == {"policy:deny", "gdpr:export"}
    # ws_b's row never appears in ws_a's read — fail-closed tenancy.
    s.close()


def test_audit_log_filters_compose(new_session, two_workspaces):
    ws_a, _ = two_workspaces
    s = new_session()
    old = datetime.now(timezone.utc) - timedelta(days=10)
    _seed_audit(s, ws_a, action="policy:deny", actor_type="system")
    _seed_audit(s, ws_a, action="policy:allow", actor_type="agent")
    _seed_audit(s, ws_a, action="gdpr:export", actor_type="user")
    _seed_audit(s, ws_a, action="policy:deny", actor_type="system", created_at=old)
    s.commit()

    # action_prefix
    policy = asyncio.run(gov.get_audit_log(ctx=_ctx(ws_a), db=s, action_prefix="policy:"))
    assert all(r["action"].startswith("policy:") for r in policy["rows"])
    assert policy["total"] == 3

    # actor_type
    sys_rows = asyncio.run(gov.get_audit_log(ctx=_ctx(ws_a), db=s, actor_type="system"))
    assert all(r["actor_type"] == "system" for r in sys_rows["rows"])

    # since window drops the 10-day-old row
    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    recent = asyncio.run(gov.get_audit_log(ctx=_ctx(ws_a), db=s, action_prefix="policy:", since=since))
    assert recent["total"] == 2, "the 10-day-old policy row is filtered out by since"
    s.close()


def test_governance_status_reports_flag_honestly(new_session, two_workspaces, monkeypatch):
    ws_a, _ = two_workspaces
    s = new_session()
    _seed_audit(s, ws_a, action="policy:deny")
    s.commit()

    monkeypatch.setattr(gov, "policy_plane_enabled", lambda: False)
    off = asyncio.run(gov.get_status(ctx=_ctx(ws_a), db=s))
    assert off["policy_plane"]["enforcing"] is False, "an OFF plane reads OFF, loudly"
    assert off["audit"]["policy_verdicts"]["total"] == 1
    assert "by_status" in off["grants"]

    monkeypatch.setattr(gov, "policy_plane_enabled", lambda: True)
    on = asyncio.run(gov.get_status(ctx=_ctx(ws_a), db=s))
    assert on["policy_plane"]["enforcing"] is True
    s.close()
