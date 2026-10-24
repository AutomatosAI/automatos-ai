"""PRD-266 US-019b (§4.3 "Web", K15) — whose phone it is, for an owner or admin.

``GET /api/v1/mobile/devices`` (``api/mobile_devices.py`` over ``services/mobile/devices.py``):

* A workspace owner or admin (and the local operator) sees the workspace's devices; each row
  that is not the caller's own carries ``owner_label``: its person's name, else their email.
  The caller's own rows carry none.
* A member sees only their own devices (W1's rule), and none of them carries a label.
* A label is added to ``to_dict()``, never in place of it: no secret appears beside it.

The route runs on Postgres (CI) inside the test's rolled-back transaction; the request
context stands in for the auth lane, and the real role resolver reads ``workspace_members``.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy import text

DEVICES = "/api/v1/mobile/devices"
NAMED = "Ana Lopez"


@pytest.fixture
def ws(seed_workspace) -> uuid.UUID:
    return uuid.UUID(seed_workspace())


@pytest.fixture
def client_for(db_session):
    """A client on the device router whose every request is ``ctx``."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import mobile_devices
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    def _make(ctx):
        app = FastAPI()
        app.include_router(mobile_devices.router)
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
        return TestClient(app)

    return _make


def _person(db, name=None) -> tuple:
    """A users row signed in with Clerk: ``(pk, email, clerk_user_id)``."""
    tag = uuid.uuid4().hex[:12]
    email, clerk_id = f"prd266-owner-{tag}@t.test", f"user_prd266owner{tag}"
    pk = db.execute(
        text("INSERT INTO users (username, email, name, is_active, clerk_user_id) "
             "VALUES (:u, :e, :n, true, :c) RETURNING id"),
        {"u": f"prd266-owner-{tag}", "e": email, "n": name, "c": clerk_id},
    ).scalar()
    return pk, email, clerk_id


def _member(db, workspace_id, role, name=None) -> tuple:
    """A Clerk caller who is a ``role`` of ``workspace_id``: ``(ctx, pk, email)``."""
    from core.auth.dependencies import RequestContext, UserContext

    pk, email, clerk_id = _person(db, name)
    db.execute(text("INSERT INTO workspace_members (workspace_id, user_id, role, is_active) "
                    "VALUES (CAST(:ws AS uuid), :user, :role, TRUE)"),
               {"ws": str(workspace_id), "user": pk, "role": role})
    user = UserContext(id=clerk_id, email=email, clerk_user_id=clerk_id)
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="clerk"), pk, email


def _operator(db, workspace_id) -> tuple:
    """The anonymous local operator (hybrid's local lane): super_admin, its PK on raw_claims."""
    from core.auth.dependencies import RequestContext, UserContext

    pk, email, _ = _person(db)
    user = UserContext(id=email, email=email, system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": pk})
    return RequestContext(workspace_id=workspace_id, user=user, auth_type="anonymous"), pk


def _phone(db, workspace_id, user_pk, name="iPhone 16") -> str:
    """An active clerk phone of ``user_pk`` in ``workspace_id``; its id."""
    from core.models.mobile_devices import MobileDevice

    row = MobileDevice(workspace_id=workspace_id, user_id=user_pk, kind="clerk", status="active",
                       install_id=f"prd266-owner-{uuid.uuid4().hex[:12]}", name=name, platform="ios",
                       events=[])
    db.add(row)
    db.flush()
    return str(row.id)


def _listed(response) -> dict:
    """``{device id: row}`` from a 200 GET /devices."""
    assert response.status_code == 200, response.text
    return {device["id"]: device for device in response.json()["devices"]}


def _workspace(db, ws) -> dict:
    """An admin, a named editor and an editor with no name, each with a phone."""
    admin, admin_pk, _ = _member(db, ws, "admin")
    named, named_pk, _ = _member(db, ws, "editor", name=NAMED)
    unnamed, unnamed_pk, unnamed_email = _member(db, ws, "editor")
    return {
        "admin": admin, "named": named, "unnamed": unnamed, "unnamed_email": unnamed_email,
        "admin_phone": _phone(db, ws, admin_pk, name="Pixel 9"),
        "named_phone": _phone(db, ws, named_pk),
        "unnamed_phone": _phone(db, ws, unnamed_pk, name="Galaxy S25"),
    }


def test_an_admin_sees_whose_each_phone_is_and_no_label_on_their_own(db_session, ws, client_for):
    seeded = _workspace(db_session, ws)

    listed = _listed(client_for(seeded["admin"]).get(DEVICES))

    assert set(listed) == {seeded["admin_phone"], seeded["named_phone"], seeded["unnamed_phone"]}
    assert listed[seeded["named_phone"]]["owner_label"] == NAMED
    assert listed[seeded["unnamed_phone"]]["owner_label"] == seeded["unnamed_email"]
    assert listed[seeded["admin_phone"]].get("owner_label") is None
    assert listed[seeded["named_phone"]]["name"] == "iPhone 16"


def test_a_member_sees_only_their_own_phones_without_a_label(db_session, ws, client_for):
    seeded = _workspace(db_session, ws)

    listed = _listed(client_for(seeded["named"]).get(DEVICES))

    assert set(listed) == {seeded["named_phone"]}
    assert all(device.get("owner_label") is None for device in listed.values())


def test_the_local_operator_sees_everyone_elses_label(db_session, ws, client_for):
    seeded = _workspace(db_session, ws)
    operator, operator_pk = _operator(db_session, ws)
    own = _phone(db_session, ws, operator_pk)

    listed = _listed(client_for(operator).get(DEVICES))

    assert listed[own].get("owner_label") is None
    assert listed[seeded["named_phone"]]["owner_label"] == NAMED
    assert listed[seeded["admin_phone"]]["owner_label"].startswith("prd266-owner-")


def test_a_label_names_no_one_outside_the_workspace_and_rides_beside_no_secret(
        db_session, ws, seed_workspace, client_for):
    from core.models.mobile_devices import SECRET_COLUMNS

    seeded = _workspace(db_session, ws)
    elsewhere = uuid.UUID(seed_workspace())
    _, stranger_pk, _ = _member(db_session, elsewhere, "editor", name="Somebody Else")
    stranger_phone = _phone(db_session, elsewhere, stranger_pk)

    listed = _listed(client_for(seeded["admin"]).get(DEVICES))

    assert stranger_phone not in listed
    assert "Somebody Else" not in {device.get("owner_label") for device in listed.values()}
    assert not any(set(SECRET_COLUMNS) & set(device) for device in listed.values())
    assert not any("expo_push_token" in device for device in listed.values())


def test_the_service_builds_the_views_and_the_router_only_calls_it():
    import inspect

    from api import mobile_devices
    from services.mobile import devices

    assert "to_dict()" in inspect.getsource(devices._view)
    assert "devices.list_devices(db, caller)" in inspect.getsource(mobile_devices.list_devices)
    assert "to_dict" not in inspect.getsource(mobile_devices.list_devices)
