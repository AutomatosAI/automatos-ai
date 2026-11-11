"""PRD-274 Wave 1 follow-up: a key's service user goes with its workspace, and stays while the key is revoked.

A server key's chat turns run as a non-login ``users`` row (``sdk_api_keys.service_user_id``, decision 12). Users
aren't workspace rows, so a purge left it behind once its key and conversations were gone. Now the purge deletes
the service users of the workspace's keys, revoked ones too, after the workspace's rows; one still referenced
elsewhere is kept and logged, and the purge goes on. Revoking a key keeps its service user, because deleting it
would delete the key's conversations, its end users' among them. The real purge on the real database; S3, the
memory stores and Clerk are stubbed, as the other purge tests do.
"""
from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import text

from core.auth.scopes import AGENTS_CHAT


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    from sqlalchemy import create_engine

    from core.database.database import get_database_url

    eng = create_engine(get_database_url(), pool_pre_ping=True)
    with eng.begin() as conn:  # a test database built before the column
        conn.execute(text("ALTER TABLE sdk_api_keys ADD COLUMN IF NOT EXISTS service_user_id INTEGER "
                          "REFERENCES users(id) ON DELETE SET NULL"))
    yield eng
    eng.dispose()


@pytest.fixture
def purge(monkeypatch):
    from services import gdpr_service, workspace_purge

    async def no_clerk(clerk_user_id):
        return True

    monkeypatch.setattr(workspace_purge, "_build_s3_client", lambda: None)
    monkeypatch.setattr(gdpr_service, "_erase_durable_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(gdpr_service, "_erase_field_memory", lambda ws, subject_id=None: 0)
    monkeypatch.setattr(workspace_purge, "_delete_clerk_user", no_clerk)
    return workspace_purge.purge_workspace_sync


def _run(new_session, sql, **params):
    s = new_session()
    try:
        result = s.execute(text(sql), params)
        rows = result.fetchall() if result.returns_rows else None
        s.commit()
        return rows
    finally:
        s.close()


def _workspace(new_session, owner) -> str:
    ws = str(uuid.uuid4())
    _run(new_session, "INSERT INTO workspaces (id, name, owner_id) VALUES (CAST(:id AS uuid), 'w3-service-users', :o)",
         id=ws, o=owner)
    return ws


def _owner(new_session) -> int:
    tag = uuid.uuid4().hex[:10]
    return _run(new_session, "INSERT INTO users (username, email) VALUES (:u, :e) RETURNING id",
                u=f"owner-{tag}", e=f"owner-{tag}@example.test")[0][0]


def _service_user(new_session, ws):
    """A server key of ``ws`` and the service user its first chat turn makes; (key id, users.id)."""
    from api.chat import get_user_id
    from core.services.api_key_service import ApiKeyService

    s = new_session()
    try:
        made = ApiKeyService.create_api_key(s, uuid.UUID(ws), "acme support", key_type="server", permissions=[AGENTS_CHAT])
        ctx = SimpleNamespace(workspace_id=uuid.UUID(ws), auth_type="sdk_key", api_key_id=made["id"],
                              user=SimpleNamespace(id=f"sdk:{made['id']}", email=None, clerk_user_id=None))
        return made["id"], get_user_id(s, ctx)
    finally:
        s.close()


def _chat(new_session, ws, user_id) -> str:
    return _run(new_session, "INSERT INTO chats (id, user_id, workspace_id, title, visibility) VALUES "
                "(gen_random_uuid(), :u, CAST(:w AS uuid), 'a conversation', 'private') RETURNING id::text",
                u=user_id, w=ws)[0][0]


def _exists(new_session, table, column, value) -> bool:
    return bool(_run(new_session, f"SELECT 1 FROM {table} WHERE {column} = :v", v=value))  # noqa: S608


@pytest.fixture
def made(new_session):
    workspaces, users = [], []
    yield workspaces, users
    s = new_session.sweep()
    s.execute(text("DELETE FROM workspace_members WHERE workspace_id = ANY(CAST(:ws AS uuid[]))"), {"ws": workspaces})
    for table in ("chats", "sdk_api_keys"):
        s.execute(text(f"DELETE FROM {table} WHERE workspace_id = ANY(CAST(:ws AS uuid[]))"), {"ws": workspaces})  # noqa: S608
    s.execute(text("DELETE FROM workspaces WHERE id = ANY(CAST(:ws AS uuid[]))"), {"ws": workspaces})
    s.execute(text("DELETE FROM users WHERE id = ANY(:u)"), {"u": users})
    s.commit()
    s.close()


def test_revoking_a_key_keeps_its_service_user_and_its_conversations(new_session, made):
    from core.services.api_key_service import ApiKeyService

    workspaces, users = made
    owner = _owner(new_session)
    ws = _workspace(new_session, owner)
    key_id, service_user = _service_user(new_session, ws)
    workspaces.append(ws)
    users.extend([owner, service_user])
    chat = _chat(new_session, ws, service_user)
    s = new_session()
    try:
        assert ApiKeyService.revoke_api_key(s, uuid.UUID(str(key_id)), uuid.UUID(ws)) is True
    finally:
        s.close()
    assert _exists(new_session, "users", "id", service_user) and _exists(new_session, "chats", "id", chat)


def test_a_purge_deletes_its_keys_service_users_and_keeps_one_still_referenced(new_session, made, purge):
    workspaces, users = made
    owner = _owner(new_session)
    ws, elsewhere = _workspace(new_session, owner), _workspace(new_session, owner)
    workspaces.extend([ws, elsewhere])
    (_, gone), (_, referenced) = _service_user(new_session, ws), _service_user(new_session, ws)
    users.extend([owner, gone, referenced])
    chat = _chat(new_session, ws, gone)
    _run(new_session, "INSERT INTO workspace_members (workspace_id, user_id, role) VALUES (CAST(:w AS uuid), :u, "
         "'member')", w=elsewhere, u=referenced)  # a reference purge does not own: the user must stay
    _run(new_session, "UPDATE workspaces SET deleted_at = now() WHERE id = CAST(:w AS uuid)", w=ws)
    result = purge(uuid.UUID(ws))
    assert result.skipped_reason is None and result.workspace_row_deleted
    assert not _exists(new_session, "users", "id", gone) and not _exists(new_session, "chats", "id", chat)
    assert _exists(new_session, "users", "id", referenced)
    assert result.rows_deleted["users (key service users)"] == 1
