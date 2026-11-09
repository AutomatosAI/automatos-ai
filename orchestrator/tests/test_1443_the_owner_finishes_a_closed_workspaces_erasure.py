"""CodeRabbit on #1443: the owner can finish an erasure that answered 503.

The erase closes the workspace before it erases anything, and answers 503 when a step
failed, so the caller asks again. Every request lane refuses a closed workspace, so on
hosted the owner's retry was a 400 and the erasure could never finish. Now the erase
route, and only it, lets the workspace's owner reach their own closed workspace; anyone
else is refused as before. The real lanes on the real database; Clerk stubbed.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fastapi import HTTPException


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


class _Clerk:
    """A Clerk whose only good token is ``good:<clerk user id>``."""

    def verify_token(self, token):
        return {"sub": token.split(":", 1)[1]} if token.startswith("good:") else None

    def extract_user_info(self, claims):
        return {"clerk_user_id": claims["sub"], "email": f"{claims['sub']}@example.test", "system_role": "user"}


@pytest.fixture
def clerk(monkeypatch):
    from core.auth import hybrid, workspace_eraser

    monkeypatch.setattr(workspace_eraser, "get_clerk_auth", _Clerk)
    monkeypatch.setattr(hybrid, "get_clerk_auth", _Clerk)


@pytest.fixture
def seeded(new_session):
    """``(seed, sweep)``: seed a user and the workspace they own; sweep removes them all."""
    from sqlalchemy import text

    users, workspaces = [], []

    def seed(*, closed: bool):
        tag = uuid4().hex[:10]
        ws = str(uuid4())
        s = new_session()
        user = s.execute(text("INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id"),
                         {"u": f"erase-{tag}", "e": f"erase-{tag}@example.test", "c": f"user_{tag}"}).scalar()
        s.execute(text("INSERT INTO workspaces (id, name, owner_id, deleted_at, is_active) VALUES (CAST(:id AS uuid), "
                       "'#1443 erase', :owner, CASE WHEN CAST(:closed AS boolean) THEN now() END, NOT :closed)"),
                  {"id": ws, "owner": user, "closed": closed})
        s.commit()
        s.close()
        users.append(user)
        workspaces.append(ws)
        return f"user_{tag}", ws

    yield seed
    s = new_session.sweep()
    s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ws AS uuid[]))"), {"ws": workspaces})
    s.execute(text("DELETE FROM users WHERE id = ANY(:u)"), {"u": users})
    s.commit()
    s.close()


def _request(ws: str, token: str | None):
    from starlette.requests import Request

    headers = [(b"x-workspace-id", ws.encode())]
    if token:
        headers.append((b"authorization", f"Bearer {token}".encode()))
    return Request({"type": "http", "method": "POST", "path": "/api/v1/gdpr/erase", "headers": headers,
                    "query_string": b""})


def _eraser(new_session, ws: str, token: str | None):
    from core.auth.workspace_eraser import require_workspace_eraser

    db = new_session()
    try:
        return asyncio.run(require_workspace_eraser(_request(ws, token), db))
    finally:
        db.close()


def test_the_owner_reaches_their_closed_workspace_to_finish_its_erasure(new_session, clerk, seeded):
    owner, ws = seeded(closed=True)

    ctx = _eraser(new_session, ws, f"good:{owner}")

    assert str(ctx.workspace_id) == ws and ctx.auth_type == "clerk" and ctx.user.clerk_user_id == owner


def test_the_usual_lanes_still_refuse_the_closed_workspace(new_session, clerk, seeded):
    from core.auth.hybrid import get_request_context_hybrid

    owner, ws = seeded(closed=True)

    with pytest.raises(HTTPException) as refused:
        asyncio.run(get_request_context_hybrid(_request(ws, f"good:{owner}")))
    assert refused.value.status_code == 400


def test_someone_else_is_refused_as_before(new_session, clerk, seeded):
    _, ws = seeded(closed=True)
    stranger, _ = seeded(closed=False)

    with pytest.raises(HTTPException) as refused:
        _eraser(new_session, ws, f"good:{stranger}")
    assert refused.value.status_code == 400 and refused.value.detail == "Invalid workspace_id"


def test_a_token_that_does_not_verify_is_a_401(new_session, clerk, seeded):
    _, ws = seeded(closed=True)

    with pytest.raises(HTTPException) as refused:
        _eraser(new_session, ws, "forged")
    assert refused.value.status_code == 401


def test_an_open_workspace_goes_through_the_usual_admin_check(new_session, clerk, seeded):
    owner, ws = seeded(closed=False)
    _, other_ws = seeded(closed=False)

    assert str(_eraser(new_session, ws, f"good:{owner}").workspace_id) == ws
    # The owner of one workspace naming another they have no part in falls back to their own,
    # exactly as require_workspace_admin's lane does; the closed-workspace path plays no part.
    assert str(_eraser(new_session, other_ws, f"good:{owner}").workspace_id) == ws


def test_only_the_erase_route_takes_the_exception():
    from api import gdpr
    from core.auth.workspace_admin import require_workspace_admin
    from core.auth.workspace_eraser import require_workspace_eraser

    def dependency_of(path):
        route = next(r for r in gdpr.router.routes if r.path == path)
        return next(d.call for d in route.dependant.dependencies if d.name == "ctx")

    assert dependency_of("/api/v1/gdpr/erase") is require_workspace_eraser
    assert dependency_of("/api/v1/gdpr/export") is require_workspace_admin
    assert dependency_of("/api/v1/gdpr/erase-subject") is require_workspace_admin
