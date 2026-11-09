"""#1431: purging a workspace deletes its owner's account only when they have nothing left.

``purge_workspace_sync`` deleted the owner's Clerk sign-in unconditionally, then their
``users`` row when no OTHER workspace had them as ``owner_id``. On the local edition the
operator's main workspace often has no ``owner_id``, so purging a test workspace the
operator owned tried to delete the operator (a foreign key happened to stop it, and the
whole purge rolled back while ``purge_test_workspace.py`` still printed ``purged: true``
and exited 0). On hosted, membership was never counted, and the sign-in went regardless.

Now the account goes only when the owner neither owns nor belongs to any other
workspace, never on the local edition, and the script says when a purge failed. The real
purge on the real database; S3 and Clerk stubbed.
"""
from __future__ import annotations

import json
from uuid import uuid4

import pytest


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    yield eng
    eng.dispose()


@pytest.fixture
def purge(monkeypatch):
    """The purge with S3 off and Clerk recorded; yields the Clerk ids it deleted."""
    from services import workspace_purge

    deleted = []

    async def delete_clerk_user(clerk_user_id):
        deleted.append(clerk_user_id)
        return True

    monkeypatch.setattr(workspace_purge, "_build_s3_client", lambda: None)
    monkeypatch.setattr(workspace_purge, "_delete_clerk_user", delete_clerk_user)
    return workspace_purge.purge_workspace_sync, deleted


@pytest.fixture
def edition(monkeypatch):
    from config import config

    return lambda name: monkeypatch.setattr(config, "AUTH_EDITION", name)


def _run(new_session, sql, **params):
    from sqlalchemy import text

    s = new_session()
    try:
        result = s.execute(text(sql), params)
        row = result.fetchone() if result.returns_rows else None
        s.commit()
        return row
    finally:
        s.close()


def _user(new_session) -> int:
    tag = uuid4().hex[:10]
    return _run(new_session, "INSERT INTO users (username, email, clerk_user_id) VALUES (:u, :e, :c) RETURNING id",
                u=f"purge-{tag}", e=f"purge-{tag}@example.test", c=f"user_{tag}").id


def _workspace(new_session, owner, *, gone: bool) -> str:
    ws = str(uuid4())
    _run(new_session, "INSERT INTO workspaces (id, name, owner_id, deleted_at) VALUES (CAST(:id AS uuid), "
         "'#1431 purge', :owner, CASE WHEN CAST(:gone AS boolean) THEN now() END)", id=ws, owner=owner, gone=gone)
    return ws


def _member(new_session, user, ws):
    _run(new_session, "INSERT INTO workspace_members (workspace_id, user_id, role) VALUES (CAST(:ws AS uuid), :u, "
         "'member')", ws=ws, u=user)


def _user_exists(new_session, user) -> bool:
    return _run(new_session, "SELECT count(*) AS n FROM users WHERE id = :u", u=user).n == 1


def _cleanup(new_session, users, workspaces):
    from sqlalchemy import text

    s = new_session.sweep()
    s.execute(text("DELETE FROM workspace_members WHERE workspace_id = ANY(CAST(:ws AS uuid[]))"), {"ws": workspaces})
    s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ws AS uuid[]))"), {"ws": workspaces})
    s.execute(text("DELETE FROM users WHERE id = ANY(:u)"), {"u": users})
    s.commit()
    s.close()


def test_on_the_local_edition_the_owner_is_never_deleted(new_session, purge, edition):
    run, clerk_deleted = purge
    edition("local")
    owner = _user(new_session)
    gone = _workspace(new_session, owner, gone=True)
    try:
        result = run(gone)
        assert result.skipped_reason is None and result.workspace_row_deleted
        assert _user_exists(new_session, owner) and "users" not in result.rows_deleted
        assert clerk_deleted == []
    finally:
        _cleanup(new_session, [owner], [gone])


@pytest.mark.parametrize("left", ["owns another workspace", "belongs to another workspace"])
def test_an_owner_with_somewhere_left_keeps_their_account(new_session, purge, edition, left):
    run, clerk_deleted = purge
    edition("saas")
    owner = _user(new_session)
    gone = _workspace(new_session, owner, gone=True)
    other = _workspace(new_session, owner if left.startswith("owns") else None, gone=False)
    if left.startswith("belongs"):
        _member(new_session, owner, other)
    try:
        result = run(gone)
        assert result.skipped_reason is None and result.workspace_row_deleted
        assert _user_exists(new_session, owner) and "users" not in result.rows_deleted
        assert clerk_deleted == []                       # the sign-in stays too
    finally:
        _cleanup(new_session, [owner], [gone, other])


def test_an_owner_with_nothing_left_is_deleted_with_their_sign_in(new_session, purge, edition):
    run, clerk_deleted = purge
    edition("saas")
    owner = _user(new_session)
    clerk_id = _run(new_session, "SELECT clerk_user_id FROM users WHERE id = :u", u=owner).clerk_user_id
    gone = _workspace(new_session, owner, gone=True)
    try:
        result = run(gone)
        assert result.skipped_reason is None and result.rows_deleted.get("users") == 1
        assert not _user_exists(new_session, owner)
        assert clerk_deleted == [clerk_id]
    finally:
        _cleanup(new_session, [owner], [gone])


def test_the_script_says_when_a_purge_failed(monkeypatch, capsys, new_session):
    """A purge that stopped rolled back: the script says purged false and exits 1."""
    from scripts import purge_test_workspace as script
    from services import workspace_purge

    ws = str(uuid4())
    _run(new_session, "INSERT INTO workspaces (id, name, settings) VALUES (CAST(:id AS uuid), '#1431 script', "
         "CAST(:s AS jsonb))", id=ws, s=json.dumps({"managed_by": script.MANAGED_BY, "purpose": "sim:1431"}))
    monkeypatch.setattr(workspace_purge, "purge_workspace_sync",
                        lambda workspace_id: workspace_purge.PurgeResult(workspace_id=str(workspace_id),
                                                                         skipped_reason="exception: boom"))
    try:
        assert script.main(["--workspace-id", ws, "--expect-purpose", "sim:1431"]) == script.EXIT_ERROR
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["purged"] is False
    finally:
        _cleanup(new_session, [], [ws])
