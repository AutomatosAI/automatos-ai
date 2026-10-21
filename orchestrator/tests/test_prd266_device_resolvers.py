"""PRD-266 US-004b — every resolver knows ``device``: a paired phone acts as its bound person.

The device lane (``core/auth/device.py``) binds ``auth_type="device"``, the row's workspace and
a ``UserContext`` whose ``id`` is the EMAIL, with no ``clerk_user_id`` and the integer PK on
``raw_claims`` (``source: mobile_device``). Every context here is built by the lane's own
``_user_context``, so it has exactly that shape. One resolver at a time, against Postgres
wherever the resolver reads it:

* ``core/auth/workspace_permission.py``: a device is judged by its person's role in its
  workspace, read by ``users.id`` (the member row, then the owner fallback); a viewer-bound
  device is refused where a viewer is (403 through the gate); a person with no role there is
  refused; the operator's device is ``super_admin`` and bypasses without a query; the
  ``users.id`` read is the device lane's alone (a Clerk context without a Clerk id still has
  no role).
* ``core/auth/workspace_admin.py``: an owner's or admin's device passes, an editor's or
  viewer's does not, nor another workspace's admin; a context without a Clerk identity is read
  by ``users.id`` only on the device path (an API key still never queries).
* ``services/themes/membership.py``: a device's membership is exactly its bound workspace,
  never another the person belongs to and never every workspace (the operator's phone too).
* ``core/auth/step_up.py``: a device passes in the local edition (the edition gate; no code
  change) and, like every non-Clerk context, is refused on saas.
* ``api/notifications.py``: its user resolver IS ``resolve_user_pk`` (its own copy is gone), so
  a device resolves to its person, and its unread count is its person's.
* ``modules/voice/caller.py``: the anonymous local operator and a device resolve through the
  claim rule; an unknown claim source is never trusted and nothing falls back to anyone.
* The claim rule itself (``core.auth.principal.claimed_user_pk``, security review): a Clerk
  token whose payload carries a lane's ``source`` and someone else's ``user_id`` never names
  that person, in any of the resolvers above.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
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
from core.auth import device, step_up  # noqa: E402
from core.auth import workspace_admin as admin  # noqa: E402
from core.auth import workspace_permission as permission  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import claimed_user_pk, resolve_user_pk  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.policy.roles import workspace_has_permission  # noqa: E402
from modules.voice.caller import resolve_caller_int_id  # noqa: E402
from services.themes import membership  # noqa: E402

EXECUTE = "agents:execute"
READ = "agents:read"


class _NoQueries:
    """A session that fails the test if anything is read: a decision that needs no database."""

    def execute(self, *args, **kwargs):
        raise AssertionError("this decision must not query the database")

    def query(self, *args, **kwargs):
        raise AssertionError("this decision must not query the database")


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def ws(seed_workspace) -> uuid.UUID:
    return uuid.UUID(seed_workspace())


def _person(db, clerk_user_id=None) -> tuple:
    """A users row: ``(pk, email)``."""
    tag = uuid.uuid4().hex[:12]
    email = f"prd266-resolvers-{tag}@t.test"
    pk = db.execute(
        text("INSERT INTO users (username, email, is_active, clerk_user_id) VALUES (:u, :e, true, :c) RETURNING id"),
        {"u": f"prd266-resolvers-{tag}", "e": email, "c": clerk_user_id},
    ).scalar()
    return pk, email


def _member(db, workspace_id, user_pk, role, active=True) -> None:
    db.execute(
        text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
             "VALUES (CAST(:ws AS uuid), :uid, :role, :active)"),
        {"ws": str(workspace_id), "uid": user_pk, "role": role, "active": active},
    )
    db.flush()


def _owns(db, workspace_id, user_pk) -> None:
    db.execute(text("UPDATE workspaces SET owner_id = :uid WHERE id = CAST(:ws AS uuid)"),
               {"uid": user_pk, "ws": str(workspace_id)})
    db.flush()


def _device_ctx(workspace_id, person) -> RequestContext:
    """A paired device's context exactly as the lane binds it (``device._user_context``)."""
    user = device._user_context(SimpleNamespace(id=uuid.uuid4()), person)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type=device.DEVICE_AUTH_TYPE)


@pytest.fixture
def operator(db_session, monkeypatch):
    """The local operator's users row, ``config.LOCAL_OPERATOR_EMAIL`` pointed at it."""
    person = _person(db_session)
    monkeypatch.setattr(config, "LOCAL_OPERATOR_EMAIL", person[1])
    return person


def _gate_client(db, ctx, permission_name) -> TestClient:
    app = FastAPI()

    @app.get("/gated", dependencies=[Depends(permission.require_workspace_permission(permission_name))])
    def gated():
        return {"ok": True}

    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[get_db] = lambda: db
    return TestClient(app)


def test_the_device_context_here_is_the_lanes_shape(ws, db_session):
    person = _person(db_session)
    ctx = _device_ctx(ws, person)

    assert (ctx.auth_type, ctx.user.id, ctx.user.clerk_user_id, ctx.user.system_role) == \
        ("device", person[1], None, "user")
    assert ctx.user.raw_claims["source"] == "mobile_device" and ctx.user.raw_claims["user_id"] == person[0]


# --------------------------------------------------------------------------- workspace_permission


@pytest.mark.parametrize("role", ["owner", "admin", "editor", "viewer"])
def test_a_device_is_judged_by_its_persons_role_in_its_workspace(ws, db_session, role):
    person = _person(db_session)
    _member(db_session, ws, person[0], role)
    ctx = _device_ctx(ws, person)

    assert permission.resolve_workspace_role(db_session, ctx) == role
    for wanted in (EXECUTE, READ, "workspace:manage", "members:change_role"):
        assert permission.workspace_permission_granted(db_session, ctx, wanted) is \
            workspace_has_permission(role, wanted), (role, wanted)


def test_a_viewer_bound_device_is_refused_where_a_viewer_is(ws, db_session):
    viewer, editor = _person(db_session), _person(db_session)
    _member(db_session, ws, viewer[0], "viewer")
    _member(db_session, ws, editor[0], "editor")

    refused = _gate_client(db_session, _device_ctx(ws, viewer), EXECUTE).get("/gated")
    allowed = _gate_client(db_session, _device_ctx(ws, editor), EXECUTE).get("/gated")
    reads = _gate_client(db_session, _device_ctx(ws, viewer), READ).get("/gated")

    assert refused.status_code == 403 and refused.json()["detail"] == f"Permission denied: {EXECUTE}"
    assert allowed.status_code == 200
    assert reads.status_code == 200


def test_the_owner_fallback_is_read_by_users_id(ws, db_session):
    person = _person(db_session)
    _owns(db_session, ws, person[0])

    ctx = _device_ctx(ws, person)

    assert permission.resolve_workspace_role(db_session, ctx) == "owner"
    assert permission.workspace_permission_granted(db_session, ctx, "workspace:manage") is True


def test_a_device_whose_person_has_no_role_there_is_refused(ws, db_session, seed_workspace):
    person = _person(db_session)
    elsewhere = uuid.UUID(seed_workspace())
    _member(db_session, elsewhere, person[0], "owner")
    _member(db_session, ws, person[0], "admin", active=False)

    ctx = _device_ctx(ws, person)

    assert permission.resolve_workspace_role(db_session, ctx) is None
    assert permission.workspace_permission_granted(db_session, ctx, READ) is False


def test_the_operators_device_bypasses_without_a_query(ws, operator):
    ctx = _device_ctx(ws, operator)

    assert ctx.user.system_role == "super_admin"
    assert permission.workspace_permission_granted(_NoQueries(), ctx, "workspace:delete") is True


def test_the_users_id_read_is_the_device_lanes_alone(ws, db_session):
    """A Clerk context with the same claims but no Clerk id still has no role (unchanged)."""
    person = _person(db_session)
    _member(db_session, ws, person[0], "owner")
    lane_user = _device_ctx(ws, person).user

    clerk = RequestContext(workspace_id=ws, user=lane_user, auth_type="clerk")
    api_key = RequestContext(workspace_id=ws, user=lane_user, auth_type="api_key")

    assert permission.resolve_workspace_role(db_session, clerk) is None
    assert permission.workspace_permission_granted(db_session, api_key, READ) is False


# --------------------------------------------------------------------------- workspace_admin


@pytest.mark.parametrize("role, is_admin", [("owner", True), ("admin", True), ("editor", False), ("viewer", False)])
def test_a_devices_person_is_judged_workspace_admin_by_users_id(ws, db_session, role, is_admin):
    person = _person(db_session)
    _member(db_session, ws, person[0], role)

    assert admin.may_see_own_workspace_health(db_session, _device_ctx(ws, person)) is is_admin


def test_a_device_whose_person_owns_the_workspace_is_its_admin(ws, db_session):
    person = _person(db_session)
    _owns(db_session, ws, person[0])

    assert admin.may_see_own_workspace_health(db_session, _device_ctx(ws, person)) is True


def test_another_workspaces_admin_is_not_this_ones(ws, db_session, seed_workspace):
    person = _person(db_session)
    _member(db_session, uuid.UUID(seed_workspace()), person[0], "admin")

    assert admin.may_see_own_workspace_health(db_session, _device_ctx(ws, person)) is False


def test_require_workspace_admin_lets_an_admins_device_through_and_refuses_an_editors(ws, db_session):
    owner_person, editor_person = _person(db_session), _person(db_session)
    _member(db_session, ws, owner_person[0], "admin")
    _member(db_session, ws, editor_person[0], "editor")
    allowed = _device_ctx(ws, owner_person)

    assert asyncio.run(admin.require_workspace_admin(ctx=allowed, db=db_session)) is allowed
    with pytest.raises(HTTPException) as refused:
        asyncio.run(admin.require_workspace_admin(ctx=_device_ctx(ws, editor_person), db=db_session))
    assert refused.value.status_code == 403


def test_a_context_without_a_clerk_id_is_read_by_users_id_only_on_the_device_path(ws, db_session):
    person = _person(db_session)
    lane_user = _device_ctx(ws, person).user
    api_key = RequestContext(workspace_id=ws, user=lane_user, auth_type="api_key")

    assert admin.may_see_own_workspace_health(_NoQueries(), api_key) is False


def test_the_operators_device_sees_its_workspace_as_the_operator_does(ws, operator):
    assert admin.may_see_own_workspace_health(_NoQueries(), _device_ctx(ws, operator)) is True


# --------------------------------------------------------------------------- theme membership


def test_device_is_a_person_auth_type_named_as_the_lane_names_it():
    assert membership.DEVICE_AUTH == device.DEVICE_AUTH_TYPE
    assert device.DEVICE_AUTH_TYPE in membership.PERSON_AUTH_TYPES


def test_a_devices_membership_is_its_bound_workspace_only(ws, db_session, seed_workspace):
    clerk_id = f"user_prd266{uuid.uuid4().hex[:12]}"
    person = _person(db_session, clerk_user_id=clerk_id)
    other = uuid.UUID(seed_workspace())
    _member(db_session, ws, person[0], "viewer")
    _member(db_session, other, person[0], "owner")
    signed_in = RequestContext(workspace_id=ws, auth_type="clerk",
                               user=UserContext(id=clerk_id, email=person[1], clerk_user_id=clerk_id))

    phone = membership.caller_membership(db_session, _device_ctx(ws, person))

    assert phone == membership.Membership(user_pk=person[0], workspace_ids=frozenset({ws}), every_workspace=False)
    assert phone.includes(ws) and not phone.includes(other)
    assert membership.caller_membership(db_session, signed_in).workspace_ids == frozenset({ws, other})


def test_the_operators_phone_reads_theme_files_of_its_bound_workspace_only(ws, operator, monkeypatch, seed_workspace):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    other = uuid.UUID(seed_workspace())
    at_the_desk = RequestContext(workspace_id=ws, auth_type="anonymous", user=UserContext(
        id=operator[1], email=operator[1], system_role="super_admin",
        raw_claims={"source": "local_operator", "user_id": operator[0]}))

    phone = membership.caller_membership(_NoQueries(), _device_ctx(ws, operator))

    assert (phone.user_pk, phone.workspace_ids, phone.every_workspace) == (operator[0], frozenset({ws}), False)
    assert not phone.includes(other)
    assert membership.caller_membership(_NoQueries(), at_the_desk).every_workspace is True


# --------------------------------------------------------------------------- step-up


def test_a_device_passes_step_up_in_the_local_edition(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    ctx = _device_ctx(uuid.uuid4(), (7, "prd266-step-up@t.test"))

    assert step_up.check_step_up(ctx, 5, time.time()) is None
    assert asyncio.run(step_up.require_step_up(5)(ctx=ctx)) is ctx


def test_a_device_context_is_refused_step_up_on_saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    ctx = _device_ctx(uuid.uuid4(), (7, "prd266-step-up@t.test"))

    with pytest.raises(HTTPException) as refused:
        step_up.check_step_up(ctx, 5, time.time())

    assert refused.value.status_code == 403
    assert refused.value.detail["code"] == step_up.STEP_UP_UNAVAILABLE_CODE


# --------------------------------------------------------------------------- notifications


def test_the_notifications_resolver_is_the_one_resolver():
    from api import notifications

    assert notifications._resolve_user_id is resolve_user_pk
    assert "def _resolve_user_id(" not in Path(notifications.__file__).read_text(encoding="utf-8")


def test_notifications_resolve_a_device_to_its_person(ws, db_session):
    from api import notifications

    person = _person(db_session)

    assert notifications._resolve_user_id(_NoQueries(), _device_ctx(ws, person)) == person[0]


# The table as alembic/versions/prd128_notifications.py makes it: a test DB built from the
# models (scripts/init_test_db.py) has none. Postgres DDL is transactional, so db_session's
# rollback drops it again; on a migrated DB, IF NOT EXISTS leaves the real one alone.
PRD128_NOTIFICATIONS = """
    CREATE TABLE IF NOT EXISTS notifications (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        event_type VARCHAR(50) NOT NULL, title VARCHAR(255) NOT NULL,
        message TEXT, link_type VARCHAR(30), link_id TEXT, agent_id INTEGER, agent_name VARCHAR(100),
        status VARCHAR(20) NOT NULL DEFAULT 'ok', read_at TIMESTAMPTZ, dismissed_at TIMESTAMPTZ,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    )
"""


@pytest.fixture
def notifications_table(db_session) -> None:
    db_session.execute(text(PRD128_NOTIFICATIONS))
    db_session.flush()


def _notify(db, workspace_id, user_pk, title) -> None:
    db.execute(
        text("INSERT INTO notifications (workspace_id, user_id, event_type, title) "
             "VALUES (CAST(:ws AS uuid), :uid, 'task_complete', :title)"),
        {"ws": str(workspace_id), "uid": user_pk, "title": title},
    )
    db.flush()


def test_a_devices_unread_count_is_its_persons_in_its_workspace(ws, db_session, seed_workspace,
                                                                notifications_table):
    from api import notifications

    person, someone_else = _person(db_session), _person(db_session)
    _notify(db_session, ws, person[0], "yours")
    _notify(db_session, ws, None, "everyone's")
    _notify(db_session, ws, someone_else[0], "theirs")
    _notify(db_session, uuid.UUID(seed_workspace()), person[0], "yours, in another workspace")
    app = FastAPI()
    app.include_router(notifications.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: _device_ctx(ws, person)
    app.dependency_overrides[get_db] = lambda: db_session

    response = TestClient(app).get("/api/notifications/unread-count")

    assert response.status_code == 200, response.text
    assert response.json() == {"success": True, "count": 2}


# --------------------------------------------------------------------------- the voice caller


def test_the_voice_caller_resolves_the_anonymous_local_operator():
    ctx = RequestContext(workspace_id=uuid.uuid4(), auth_type="anonymous", user=UserContext(
        id="operator@t.test", email="operator@t.test", system_role="super_admin",
        raw_claims={"source": "local_operator", "user_id": 42}))

    assert resolve_caller_int_id(_NoQueries(), ctx) == 42


def test_the_voice_caller_resolves_a_device():
    ctx = _device_ctx(uuid.uuid4(), (43, "prd266-voice@t.test"))

    assert resolve_caller_int_id(_NoQueries(), ctx) == 43


def test_the_voice_caller_never_trusts_another_claim_source_nor_falls_back(db_session):
    unknown = f"user_prd266{uuid.uuid4().hex[:12]}"
    clerk = RequestContext(workspace_id=uuid.uuid4(), auth_type="clerk", user=UserContext(
        id=unknown, email="prd266-voice-clerk@t.test", clerk_user_id=unknown,
        raw_claims={"source": "clerk", "user_id": 1, "sub": unknown}))
    nobody = RequestContext(workspace_id=uuid.uuid4(), auth_type="anonymous", user=UserContext())

    assert resolve_caller_int_id(db_session, clerk) is None
    assert resolve_caller_int_id(_NoQueries(), nobody) is None


# --------------------------------------------------------------------------- the claim rule


def _forged(clerk_id, email, claims) -> RequestContext:
    """A Clerk-lane context whose verified token carries a lane's claim of its own."""
    user = UserContext(id=clerk_id, email=email, clerk_user_id=clerk_id, raw_claims={"sub": clerk_id, **claims})
    return RequestContext(workspace_id=uuid.uuid4(), user=user, auth_type="clerk")


@pytest.mark.parametrize("source", ["mobile_device", "local_operator"])
def test_a_clerk_token_carrying_a_lanes_claim_never_names_another_person(db_session, source):
    from api import notifications

    clerk_id = f"user_prd266{uuid.uuid4().hex[:12]}"
    person, victim = _person(db_session, clerk_user_id=clerk_id), _person(db_session)
    ctx = _forged(clerk_id, person[1], {"source": source, "user_id": victim[0]})

    assert claimed_user_pk(ctx) is None
    assert resolve_user_pk(db_session, ctx) == person[0]
    assert notifications._resolve_user_id(db_session, ctx) == person[0]
    assert resolve_caller_int_id(db_session, ctx) == person[0]
    assert membership.caller_membership(db_session, ctx).user_pk == person[0]
