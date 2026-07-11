"""PRD-196 S4 (P2-15, governance I.3) — policy posture + budget editors.

- PURE: ``set_budget`` validates only the documented keys (negative ceilings,
  wrong window, bool-as-number) BEFORE any DB write — a ValueError the API turns
  into 422.
- DB-backed (real Postgres, skips with none up): PUT policy rejects an invalid
  posture (422, nothing persisted) and round-trips a valid one; PUT budget
  rejects unknown keys and lands a valid budget under ``plan_limits.budget``
  exactly (a JSONB sub-key, not a new column). Workspaces seeded first.
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

import pytest  # noqa: E402
from fastapi import HTTPException  # noqa: E402

import tests.conftest as _conftest  # noqa: E402

_conftest._restore_real_app_modules()

from api import governance as gov  # noqa: E402
from modules.policy.budget import set_budget  # noqa: E402


def _ctx(ws):
    user = SimpleNamespace(id="u", system_role="user", clerk_user_id="c")
    return SimpleNamespace(user=user, workspace_id=ws)


# ---------------------------------------------------------------------------
# PURE — set_budget validation fires before any DB access
# ---------------------------------------------------------------------------

def test_set_budget_rejects_bad_values_before_db():
    db = MagicMock()  # must never be queried on a validation failure
    for kwargs in (
        {"max_cost_usd": -1},
        {"max_cost_usd": True},           # bool is not a number
        {"max_total_tokens": -5},
        {"max_total_tokens": 1.5},        # not an int
        {"window": "week"},               # not day|month|all
    ):
        with pytest.raises(ValueError):
            set_budget(db, uuid.uuid4(), **kwargs)
    db.query.assert_not_called()


# ---------------------------------------------------------------------------
# DB-backed — policy + budget round-trip / rejection
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
        pytest.skip(f"governance policy/budget suite needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def new_session(engine):
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


@pytest.fixture
def workspace(engine, new_session):
    from sqlalchemy import text

    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(
        text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), :n) ON CONFLICT (id) DO NOTHING"),
        {"id": ws, "n": "gov-policy"},
    )
    s.commit()
    s.close()
    yield ws
    s = new_session()
    s.execute(text("DELETE FROM audit_logs WHERE workspace_id = CAST(:id AS uuid)"), {"id": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": ws})
    s.commit()
    s.close()


def test_put_policy_rejects_invalid_posture(new_session, workspace):
    s = new_session()
    with pytest.raises(HTTPException) as ei:
        asyncio.run(gov.put_policy(body={"posture": "reckless"}, ctx=_ctx(workspace), db=s))
    assert ei.value.status_code == 422
    # nothing persisted — GET still reports the Balanced default
    got = asyncio.run(gov.get_policy(ctx=_ctx(workspace), db=s))
    assert got["posture"] == "balanced"
    s.close()


def test_put_policy_roundtrip(new_session, workspace):
    s = new_session()
    asyncio.run(
        gov.put_policy(
            body={
                "posture": "strict",
                "agents_inherit_admin": True,
                "route_overrides": {"external_side_effect": "auto"},
            },
            ctx=_ctx(workspace),
            db=s,
        )
    )
    got = asyncio.run(gov.get_policy(ctx=_ctx(workspace), db=s))
    assert got["posture"] == "strict"
    assert got["agents_inherit_admin"] is True
    assert got["route_overrides"] == {"external_side_effect": "auto"}
    s.close()


def test_put_budget_validates_keys_and_lands_under_plan_limits(new_session, workspace):
    from sqlalchemy import text

    s = new_session()
    # unknown key → 422
    with pytest.raises(HTTPException) as ei:
        asyncio.run(gov.put_budget(body={"max_cost_usd": 10, "bogus": 1}, ctx=_ctx(workspace), db=s))
    assert ei.value.status_code == 422

    # negative ceiling → 422 (set_budget's ValueError)
    with pytest.raises(HTTPException) as ei2:
        asyncio.run(gov.put_budget(body={"max_cost_usd": -3}, ctx=_ctx(workspace), db=s))
    assert ei2.value.status_code == 422

    # valid budget lands under plan_limits.budget exactly
    asyncio.run(
        gov.put_budget(
            body={"max_cost_usd": 25.0, "max_total_tokens": 5_000_000, "window": "day"},
            ctx=_ctx(workspace),
            db=s,
        )
    )
    row = s.execute(
        text("SELECT plan_limits FROM workspaces WHERE id = CAST(:id AS uuid)"),
        {"id": workspace},
    ).fetchone()
    budget = (row[0] or {}).get("budget")
    assert budget == {"max_cost_usd": 25.0, "max_total_tokens": 5_000_000, "window": "day"}
    s.close()
