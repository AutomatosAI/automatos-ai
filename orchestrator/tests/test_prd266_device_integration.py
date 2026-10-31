"""PRD-266 US-004b — a paired phone, end to end: its real ``Bearer amd_…`` on the real routes.

Nothing stands in for the request context: each call carries a token minted the way pairing
mints it (``core.auth.pairing.mint_token``) through the REAL ``get_request_context_hybrid``
(the bearer registry's decorator, then ``core/auth/device.py``) against Postgres. The phone's
person is an ``admin`` member of the workspace (not the operator, so no ``super_admin``
bypass), and every resolver on the way has to know ``device``:

* ``POST /api/chat`` (``require_workspace_permission("agents:execute")``): the turn is the
  person's, in the bound workspace; only the reply stream is patched;
* ``GET /api/v1/tasks`` (``require_task_context``): the board is the bound workspace's;
* ``POST /api/v1/approval-grants/{id}/answer`` (``require_workspace_answerer``): the question is
  answered (the resume and the chat confirmation are patched; the answer's write is real);
* ``GET /api/themes/files/{digest}/{path}`` (``services/themes/membership.py``): an opted-in
  private theme of the bound workspace loads; the same person's theme in another workspace
  does not reach the phone.

A viewer's phone is refused the chat and the answer (403), as a viewer is, and still reads the
board.

PRD-261's theme routes, as the phone calls them, run on these fixtures in
``test_prd266_device_theme_routes.py`` (M-04):

* ``GET /api/themes/mobile/{install_id}`` (``api/themes_mobile.py``, membership as above): the
  bound workspace's opted-in install as data, every file URL it names loading on the same bearer;
  the same person's install in another workspace is the route's one 404;
* ``GET /api/themes/active`` and ``PUT /api/themes/choice`` (``members:read``): the bound
  workspace's answer, then the phone's own choice of the install;
* ``GET /api/themes/looks/{theme_id}`` (``members:read``): the person's looks in the bound
  workspace only.

The theme is the hello-clip package with a ``mobile`` block (light tokens, a sans face, clips),
so the phone's answer names a font, a clip and a still.
"""
from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import text

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from config import config  # noqa: E402
from core.auth import device, hybrid  # noqa: E402
from core.auth.bearer_registry import DEVICE_TOKEN_PREFIX  # noqa: E402
from core.auth.pairing import mint_token  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.approval_grants import KIND_QUESTION, ApprovalGrant  # noqa: E402
from core.models.mobile_devices import MobileDevice  # noqa: E402

T0 = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)
TOKEN_BYTES = 32
REPLY = '0:"Morning. Two tickets wait in Review."\n'
ENTRY = "dist/entry.js"
# theme.json's mobile block (M-01, the corpus's "light tokens only" with a font and clips added).
MOBILE_FONT = "assets/fonts/mobile/inter.ttf"
MOBILE_LIGHT = {"--background": "0 0% 100%", "--foreground": "#14181c"}
MOBILE = {"tokens": {"light": MOBILE_LIGHT}, "fonts": {"sans": MOBILE_FONT}, "mascot": "clips"}


class _BorrowedSession:
    """The test's session, lent to the device lane: ``close()`` is a no-op."""

    def __init__(self, session):
        self._session = session

    def __getattr__(self, name):
        return getattr(self._session, name)

    def close(self):
        return None


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def db_session(test_engine):
    """The suite's transactional session, joined by SAVEPOINT (overrides conftest's here).

    The routes commit, and an error path may roll back; here that reaches the last savepoint
    only, and every seed commits (``_settle``), so no route can take the seed with it.
    """
    from sqlalchemy.orm import Session

    connection = test_engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    yield session
    session.close()
    transaction.rollback()
    connection.close()


def _settle(db) -> None:
    """Release the seed's savepoint: a route's rollback never reaches it."""
    db.commit()


@pytest.fixture
def local(monkeypatch, db_session):
    """The local edition, the real device lane on the test's session, a frozen clock."""
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(device, "SessionLocal", lambda: _BorrowedSession(db_session))
    monkeypatch.setattr(device, "_now", lambda: T0)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return config


@pytest.fixture
def ws(seed_workspace, db_session) -> uuid.UUID:
    workspace_id = uuid.UUID(seed_workspace())
    _settle(db_session)
    return workspace_id


@pytest.fixture
def theme_files():
    """The hello-clip package with ``MOBILE`` (one sfnt face as its sans), and its stored files (in memory)."""
    from services.themes.package import read_theme_package
    from tests.fixtures.themes import build
    from tests.test_prd261_mobile_package import MOBILE_ENGINES, _face

    mascot = {"states": [build.CLIP_STATE]}
    manifest = build.changed_manifest({"engines": MOBILE_ENGINES, "mascot": mascot, "mobile": MOBILE})
    data = build.build_package("hello-clip", replace={build.MANIFEST: manifest, MOBILE_FONT: _face()})
    return SimpleNamespace(package=read_theme_package(data), stored={})


@pytest.fixture
def phone(db_session, local, theme_files, monkeypatch):
    """A client on the routers a phone calls, with NO stand-in for the request context.

    Stored theme files are read from memory: the files and mobile routes' reader dependency,
    and the looks store's reader (``services/themes/looks.py``).
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import services.themes.looks as looks_store
    from api import approval_grants, board_tasks, chat, themes_files, themes_install, themes_looks, themes_mobile

    def _stored(prefix, path):
        return theme_files.stored.get((prefix, path))

    app = FastAPI()
    for module in (chat, board_tasks, approval_grants, themes_files, themes_mobile, themes_install, themes_looks):
        app.include_router(module.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[themes_files.get_theme_file_reader] = lambda: _stored
    monkeypatch.setattr(looks_store, "read_stored_file", _stored)
    return TestClient(app)


def _person(db, workspace_id, role) -> int:
    """A users row (no Clerk identity, not the operator), a ``role`` member of ``workspace_id``."""
    tag = uuid.uuid4().hex[:12]
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active) VALUES (:u, :e, true) RETURNING id"),
        {"u": f"prd266-phone-{tag}", "e": f"prd266-phone-{tag}@t.test"},
    ).scalar()
    _join(db, workspace_id, pk, role)
    return pk


def _join(db, workspace_id, user_pk, role) -> None:
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :uid, :role, TRUE)"),
               {"ws": str(workspace_id), "uid": user_pk, "role": role})
    _settle(db)


def _paired_phone(db, workspace_id, user_pk) -> dict:
    """An active paired row for the person, and its token's ``Authorization`` header."""
    token, digest = mint_token(DEVICE_TOKEN_PREFIX, TOKEN_BYTES)
    db.add(MobileDevice(workspace_id=workspace_id, user_id=user_pk, kind="paired", status="active",
                        install_id=f"prd266-phone-{uuid.uuid4().hex[:12]}", platform="ios",
                        token_hash=digest, paired_at=T0, events=[]))
    _settle(db)
    return {"Authorization": f"Bearer {token}"}


def _phone_of(db, workspace_id, role) -> tuple:
    user_pk = _person(db, workspace_id, role)
    return user_pk, _paired_phone(db, workspace_id, user_pk)


# --------------------------------------------------------------------------- chat


@pytest.fixture
def turns(monkeypatch):
    """The chat route's reply stream, patched: Auto answers with ``REPLY``; each turn is kept."""
    from api import chat

    seen = []

    async def _auto_answers(db, ctx, request, message_text, history_length, before=""):
        return SimpleNamespace(agent_id=None, assessment=None)

    async def _the_reply(turn):
        seen.append(turn)
        yield REPLY

    async def _inline(*, chat_id, produce, on_complete=None):
        async for chunk in produce():
            yield chunk

    monkeypatch.setattr(chat, "_turn_lane", _auto_answers)
    monkeypatch.setattr(chat, "_produce_turn", _the_reply)
    monkeypatch.setattr(chat, "run_detached_turn", _inline)
    return seen


def _ask(phone, headers):
    return phone.post("/api/chat", json={"message": {"role": "user", "parts": [
        {"type": "text", "text": "What needs me this morning?"}]}}, headers=headers)


def test_a_paired_phone_chats_as_its_person_in_its_workspace(phone, db_session, ws, turns):
    user_pk, headers = _phone_of(db_session, ws, "admin")

    response = _ask(phone, headers)

    assert response.status_code == 200, response.text
    assert REPLY in response.text
    (turn,) = turns
    assert (turn.user_id, turn.workspace_id, turn.is_super_admin) == (user_pk, ws, False)
    chat_row = db_session.execute(text("SELECT user_id, workspace_id FROM chats WHERE id = CAST(:id AS uuid)"),
                                  {"id": turn.chat_id}).one()
    assert (chat_row.user_id, str(chat_row.workspace_id)) == (user_pk, str(ws))


# --------------------------------------------------------------------------- the board


def _ticket(db, workspace_id, title) -> None:
    db.execute(text("INSERT INTO board_tasks (workspace_id, title, status, priority) "
                    "VALUES (CAST(:ws AS uuid), :title, 'inbox', 'medium')"),
               {"ws": str(workspace_id), "title": title})
    _settle(db)


def test_a_paired_phone_reads_its_workspaces_board(phone, db_session, ws, seed_workspace):
    _, headers = _phone_of(db_session, ws, "admin")
    elsewhere = uuid.UUID(seed_workspace())
    _ticket(db_session, ws, "prd266 ours: chase the invoice")
    _ticket(db_session, elsewhere, "prd266 theirs: never shown")

    response = phone.get("/api/v1/tasks", headers=headers)

    assert response.status_code == 200, response.text
    titles = {task["title"] for task in response.json()["tasks"]}
    assert "prd266 ours: chase the invoice" in titles
    assert "prd266 theirs: never shown" not in titles


# --------------------------------------------------------------------------- answering a question


@pytest.fixture
def resumes(monkeypatch):
    """The answer's resume and chat confirmation, patched; the answer's own write is real."""
    from api import approval_grants

    resumed = []

    async def _requeue(db, grant):
        resumed.append(grant.id)
        return False

    monkeypatch.setattr(approval_grants, "_requeue_subject", _requeue)
    monkeypatch.setattr(approval_grants, "_confirm_answer_into_chat", lambda db, grant, *, resumed: None)
    monkeypatch.setattr(approval_grants, "_audit", lambda db, ctx, action, grant: None)
    return resumed


def _question(db, workspace_id) -> int:
    grant = ApprovalGrant(workspace_id=workspace_id, subject_type="tool_call", subject_id="prd266-call",
                          kind=KIND_QUESTION, status="pending", question_md="Which colour for the deck?",
                          options=["red", "blue"])
    db.add(grant)
    _settle(db)
    return grant.id


def test_a_paired_phone_answers_a_question(phone, db_session, ws, resumes):
    _, headers = _phone_of(db_session, ws, "admin")
    grant_id = _question(db_session, ws)

    response = phone.post(f"/api/v1/approval-grants/{grant_id}/answer", json={"option": "blue"}, headers=headers)

    assert response.status_code == 200, response.text
    assert resumes == [grant_id]
    db_session.expire_all()
    answered = db_session.get(ApprovalGrant, grant_id)
    assert (answered.status, answered.answer_text) == ("granted", "blue")


# --------------------------------------------------------------------------- theme files


def _private_theme(db, theme_files, workspace_id, user_pk) -> uuid.UUID:
    """The package uploaded to ``workspace_id``, installed there and opted in to by ``user_pk``: the install's id."""
    from core.models.themes import ThemeInstallation, ThemeOptIn, ThemePackage

    package = theme_files.package
    prefix = f"themes/_workspace/{workspace_id}/{package.manifest['id']}/{package.manifest['version']}/"
    row = ThemePackage(theme_id=package.manifest["id"], version=package.manifest["version"], owner_type="workspace",
                       workspace_id=workspace_id, digest=package.digest, manifest=package.manifest,
                       file_manifest=package.file_manifest, storage_prefix=prefix, status="pending", reviewed=False)
    db.add(row)
    db.flush()
    install = ThemeInstallation(workspace_id=workspace_id, theme_id=row.theme_id, package_id=row.id)
    db.add(install)
    db.add(ThemeOptIn(workspace_id=workspace_id, user_id=user_pk, package_id=row.id))
    db.flush()
    install_id = install.id
    _settle(db)
    theme_files.stored.update({(prefix, path): item.data for path, item in package.files.items()})
    return install_id


def _theme_file(phone, theme_files, headers):
    return phone.get(f"/api/themes/files/{theme_files.package.digest}/{ENTRY}", headers=headers)


def test_a_paired_phone_loads_its_workspaces_theme_files(phone, db_session, ws, theme_files):
    user_pk, headers = _phone_of(db_session, ws, "admin")
    _private_theme(db_session, theme_files, ws, user_pk)

    response = _theme_file(phone, theme_files, headers)

    assert response.status_code == 200, response.text
    assert response.content == theme_files.package.files[ENTRY].data


def test_a_paired_phone_never_loads_another_workspaces_theme_files(phone, db_session, ws, theme_files, seed_workspace):
    user_pk, headers = _phone_of(db_session, ws, "admin")
    elsewhere = uuid.UUID(seed_workspace())
    _join(db_session, elsewhere, user_pk, "owner")
    _private_theme(db_session, theme_files, elsewhere, user_pk)

    assert _theme_file(phone, theme_files, headers).status_code == 404


# --------------------------------------------------------------------------- a viewer's phone


def test_a_viewers_phone_is_refused_what_a_viewer_is_and_still_reads_the_board(
        phone, db_session, ws, turns, resumes):
    _, headers = _phone_of(db_session, ws, "viewer")
    grant_id = _question(db_session, ws)

    assert _ask(phone, headers).status_code == 403
    assert phone.post(f"/api/v1/approval-grants/{grant_id}/answer", json={"option": "red"},
                      headers=headers).status_code == 403
    assert phone.get("/api/v1/tasks", headers=headers).status_code == 200
    assert turns == [] and resumes == []
