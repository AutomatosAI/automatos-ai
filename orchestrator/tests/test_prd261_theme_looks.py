"""PRD-261 A-09a — ``GET/PUT /api/themes/looks/{themeId}``: looks per user, workspace and theme.

The router is mounted on a test app (PA-5: B2-wire registers it in ``main.py``) over SQLite,
with the real workspace-permission gate and the real ``resolve_user_pk``. Pinned here:

* a round trip, in the SaaS (Clerk) lane and in the local edition's anonymous owner lane;
* another member's looks are invisible, and another workspace's theme is 404 (never 403);
* an invalid look is 400 naming the field (``looks[0].weight``, ``active.name``);
* a newer version that drops a knob, or a choice, carries the look over with the default;
  a look that can't be carried is left out and the active one falls back to ``defaultLook``;
* builtin Play (``automatos/play``) is checked against the committed schema with no install;
* an install pointing at another workspace's private package is 404; a stored schema whose
  bytes don't match the file manifest is 503 with a plain message.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import sqlalchemy as sa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import api.themes_looks as looks_api  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
import services.themes.looks as looks_service  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemeInstallation, ThemeLook, ThemePackage  # noqa: E402
from services.themes.contract import HTML_CLASS_OWNERS  # noqa: E402
from services.themes.digest import sha256_hex  # noqa: E402

WS_A = uuid.UUID("00000000-0000-0000-0000-0000000000a1")
WS_B = uuid.UUID("00000000-0000-0000-0000-0000000000b2")
# (users.id, clerk id, workspace)
ANN, BEN, CAT = (1, "user_ann", WS_A), (2, "user_ben", WS_A), (3, "user_cat", WS_B)
INK = "acme/ink"
URL = f"/api/themes/looks/{INK}"
PLAY_URL = "/api/themes/looks/automatos/play"

INK_V1 = {
    "schema": "automatos-look-schema/v1",
    "look": {"schema": "acme-ink-look/v1"},
    "knobs": {
        "paper": {"kind": "choice", "choices": ["white", "cream"], "default": "white"},
        "weight": {"kind": "number", "min": 1, "max": 9, "integer": True, "default": 4},
    },
}
# v2 drops "weight", drops the "cream" paper and adds "ink".
INK_V2 = {
    "schema": "automatos-look-schema/v1",
    "look": {"schema": "acme-ink-look/v1"},
    "knobs": {
        "paper": {"kind": "choice", "choices": ["white", "grey"], "default": "white"},
        "ink": {"kind": "choice", "choices": ["black", "blue"], "default": "black"},
    },
}
MINE = {"schema": "acme-ink-look/v1", "id": "mine", "name": "Mine", "sub": "", "paper": "cream", "weight": 7}
NIGHT = {**MINE, "id": "night", "name": "Night", "paper": "white", "weight": 2}
EMBER = {
    "schema": "automatos-interface-look/v1", "interface": "play", "id": "my-ember", "name": "My Ember", "sub": "",
    "ground": "graphite", "accent": "molten", "font": "serif", "shell": "bronze", "light": "mint", "glow": 24,
}


@pytest.fixture
def db():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id TEXT, email TEXT)"))
        conn.execute(sa.text("CREATE TABLE workspaces (id TEXT PRIMARY KEY, owner_id INTEGER)"))
        conn.execute(sa.text(
            "CREATE TABLE workspace_members (workspace_id TEXT, user_id INTEGER, role TEXT, is_active BOOLEAN)"
        ))
        for pk, clerk, workspace in (ANN, BEN, CAT):
            conn.execute(sa.text("INSERT INTO users VALUES (:pk, :clerk, :email)"),
                         {"pk": pk, "clerk": clerk, "email": f"{clerk}@example.test"})
            conn.execute(sa.text("INSERT INTO workspace_members VALUES (:ws, :pk, 'viewer', 1)"),
                         {"ws": str(workspace), "pk": pk})
    tables = [ThemePackage.__table__, ThemeInstallation.__table__, ThemeLook.__table__]
    ThemePackage.metadata.create_all(engine, tables=tables)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def storage(monkeypatch):
    """The package files a test stored, by (storage prefix, path); a fresh schema cache."""
    files = {}
    monkeypatch.setattr(looks_service, "read_package_file", lambda package, path: files[(package.storage_prefix, path)])
    monkeypatch.setattr(looks_service, "_schema_cache", looks_service._SchemaCache(8))
    return files


def _package(db, storage, schema, version="1.0.0", owner=WS_A, stored_bytes=None) -> ThemePackage:
    raw = json.dumps(schema).encode("utf-8")
    prefix = f"themes/_workspace/{owner}/{INK}/{version}/"
    package = ThemePackage(
        theme_id=INK, version=version, owner_type="workspace", workspace_id=owner, digest=sha256_hex(raw + version.encode()),
        manifest={"id": INK, "look": "plain"}, storage_prefix=prefix,
        file_manifest={"looks/schema.json": {"type": "application/json", "size": len(raw), "sha256": sha256_hex(raw)}},
    )
    storage[(prefix, "looks/schema.json")] = raw if stored_bytes is None else stored_bytes
    db.add(package)
    db.commit()
    return package


def _install(db, package, workspace=WS_A) -> ThemeInstallation:
    install = ThemeInstallation(workspace_id=workspace, theme_id=INK, package_id=package.id)
    db.add(install)
    db.commit()
    return install


def _saas(person) -> RequestContext:
    _pk, clerk, workspace = person
    user = UserContext(id=clerk, email=f"{clerk}@example.test", clerk_user_id=clerk)
    return RequestContext(workspace_id=workspace, user=user, auth_type="clerk")


def _local(pk=ANN[0], workspace=WS_A) -> RequestContext:
    claims = {"source": "local_operator", "user_id": pk} if pk is not None else None
    user = UserContext(id="operator@local", email="operator@local", system_role="super_admin", raw_claims=claims)
    return RequestContext(workspace_id=workspace, user=user, auth_type="anonymous")


def _client(db, ctx) -> TestClient:
    app = FastAPI()
    app.include_router(looks_api.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx

    def _db():
        yield db

    app.dependency_overrides[get_db] = _db
    return TestClient(app)


def _put(db, ctx, looks, active, url=URL):
    return _client(db, ctx).put(url, json={"looks": looks, "active": active})


@pytest.fixture
def ink(db, storage):
    return _install(db, _package(db, storage, INK_V1))


def test_a_saved_look_round_trips(db, ink):
    saved = _put(db, _saas(ANN), [MINE, NIGHT], MINE)
    assert saved.status_code == 200, saved.text
    body = _client(db, _saas(ANN)).get(URL).json()
    assert body == {"themeId": INK, "looks": [MINE, NIGHT], "active": MINE, "defaultLook": "plain", "dropped": 0}
    assert _put(db, _saas(ANN), [NIGHT], None).status_code == 200
    assert _client(db, _saas(ANN)).get(URL).json()["looks"] == [NIGHT]
    assert db.query(ThemeLook).count() == 1


def test_the_local_editions_owner_round_trips(db, ink):
    assert _put(db, _local(), [MINE], MINE).status_code == 200
    body = _client(db, _local()).get(URL).json()
    assert (body["looks"], body["active"]) == ([MINE], MINE)
    assert db.query(ThemeLook).one().user_id == ANN[0]


def test_another_members_looks_are_invisible(db, ink):
    assert _put(db, _saas(ANN), [MINE], MINE).status_code == 200
    ben = _client(db, _saas(BEN)).get(URL)
    assert ben.status_code == 200
    assert (ben.json()["looks"], ben.json()["active"]) == ([], None)
    assert _put(db, _saas(BEN), [NIGHT], None).status_code == 200
    assert _client(db, _saas(ANN)).get(URL).json()["looks"] == [MINE]
    assert {row.user_id for row in db.query(ThemeLook)} == {ANN[0], BEN[0]}


def test_another_workspaces_theme_is_404_not_403(db, ink):
    assert _put(db, _saas(ANN), [MINE], MINE).status_code == 200
    cat = _client(db, _saas(CAT))
    for response in (cat.get(URL), cat.put(URL, json={"looks": [], "active": None})):
        assert response.status_code == 404
        assert response.json() == {"detail": looks_service.NOT_INSTALLED}
    assert _client(db, _saas(ANN)).get(URL).json()["looks"] == [MINE]


def test_an_install_naming_another_workspaces_private_package_is_404(db, storage):
    _install(db, _package(db, storage, INK_V1, owner=WS_B))
    assert _client(db, _saas(ANN)).get(URL).status_code == 404


def test_an_unknown_or_malformed_theme_id_is_404(db, ink):
    client = _client(db, _saas(ANN))
    assert client.get("/api/themes/looks/acme/other").status_code == 404
    assert client.get("/api/themes/looks/not-a-theme-id").status_code == 404


def test_an_invalid_look_is_400_naming_the_field(db, ink):
    response = _put(db, _saas(ANN), [{**MINE, "weight": 12}], {**MINE, "name": ""})
    assert response.status_code == 400
    body = response.json()
    assert isinstance(body["detail"], str) and body["detail"]
    fields = {(error["code"], error["path"]) for error in body["errors"]}
    assert ("LOOK_BAD_VALUE", "looks[0].weight") in fields
    assert ("LOOK_BAD_TEXT", "active.name") in fields
    assert db.query(ThemeLook).count() == 0


def test_an_unknown_knob_a_duplicate_id_and_too_many_looks_are_400(db, ink):
    unknown = _put(db, _saas(ANN), [{**MINE, "colour": "red"}], None).json()["errors"]
    assert ("LOOK_UNKNOWN_FIELD", "looks[0].colour") in {(e["code"], e["path"]) for e in unknown}
    duplicate = _put(db, _saas(ANN), [MINE, MINE], None).json()["errors"]
    assert ("LOOK_DUPLICATE_ID", "looks[1].id") in {(e["code"], e["path"]) for e in duplicate}
    many = [{**MINE, "id": f"look-{n}"} for n in range(looks_service.MAX_SAVED_LOOKS + 1)]
    assert _put(db, _saas(ANN), many, None).json()["errors"][0]["code"] == "TOO_MANY_LOOKS"


def test_a_body_without_both_fields_is_refused_before_any_write(db, ink):
    client = _client(db, _saas(ANN))
    assert client.put(URL, json={"active": None}).status_code == 422
    assert client.put(URL, json={"looks": [], "active": None, "extra": 1}).status_code == 422
    assert db.query(ThemeLook).count() == 0


def test_a_newer_version_carries_looks_over_with_the_defaults(db, storage, ink):
    assert _put(db, _saas(ANN), [MINE, NIGHT], NIGHT).status_code == 200
    ink.package_id, ink.previous_package_id = _package(db, storage, INK_V2, version="2.0.0").id, ink.package_id
    db.commit()
    body = _client(db, _saas(ANN)).get(URL).json()
    carried_mine = {"schema": "acme-ink-look/v1", "id": "mine", "name": "Mine", "sub": "", "paper": "white", "ink": "black"}
    assert body["looks"] == [carried_mine, {**carried_mine, "id": "night", "name": "Night"}]
    assert body["active"] == {**carried_mine, "id": "night", "name": "Night"}
    assert body["dropped"] == 0
    # The row keeps what was saved: a rollback reads the v1 looks unchanged.
    ink.package_id = ink.previous_package_id
    db.commit()
    assert _client(db, _saas(ANN)).get(URL).json()["looks"] == [MINE, NIGHT]


def test_a_look_that_cant_be_carried_is_left_out_and_active_falls_back(db, ink):
    broken = {**MINE, "id": "bad id with spaces"}
    db.add(ThemeLook(workspace_id=WS_A, user_id=ANN[0], theme_id=INK, looks=[MINE, broken, "junk"], active=broken))
    db.commit()
    body = _client(db, _saas(ANN)).get(URL).json()
    # Two looks are lost ("broken" and "junk"); the active one is "broken" again, so it counts once.
    assert (body["looks"], body["active"], body["dropped"], body["defaultLook"]) == ([MINE], None, 2, "plain")


def test_an_active_look_that_isnt_saved_counts_as_its_own_loss(db, ink):
    lost = {**NIGHT, "id": "lost one"}
    db.add(ThemeLook(workspace_id=WS_A, user_id=ANN[0], theme_id=INK, looks=[MINE], active=lost))
    db.commit()
    body = _client(db, _saas(ANN)).get(URL).json()
    assert (body["looks"], body["active"], body["dropped"]) == ([MINE], None, 1)


def test_a_stored_schema_that_doesnt_match_its_manifest_is_503(db, storage):
    _install(db, _package(db, storage, INK_V1, stored_bytes=b'{"tampered": true}'))
    response = _client(db, _saas(ANN)).get(URL)
    assert response.status_code == 503
    assert response.json() == {"detail": looks_service.SCHEMA_UNAVAILABLE}


def test_builtin_play_uses_the_committed_schema_without_an_install(db, storage):
    assert looks_service.BUILTIN_PLAY_THEME_ID == HTML_CLASS_OWNERS["play"]
    assert list(looks_service.builtin_play_schema().knobs) == ["ground", "accent", "font", "shell", "light", "glow"]
    empty = _client(db, _saas(ANN)).get(PLAY_URL).json()
    assert empty == {"themeId": "automatos/play", "looks": [], "active": None, "defaultLook": "ember", "dropped": 0}
    assert _put(db, _saas(ANN), [EMBER], EMBER, url=PLAY_URL).status_code == 200
    assert _client(db, _saas(ANN)).get(PLAY_URL).json()["active"] == EMBER
    refused = _put(db, _saas(ANN), [{**EMBER, "ground": "neon"}], None, url=PLAY_URL)
    assert refused.status_code == 400
    assert refused.json()["errors"][0]["path"] == "looks[0].ground"


def test_a_caller_with_no_person_is_refused_plainly(db, ink):
    response = _client(db, _local(pk=None)).get(URL)
    assert response.status_code == 403
    assert response.json() == {"detail": looks_api.NO_PERSON}


def test_both_routes_are_plain_def_and_gated_for_members():
    routes = [route for route in looks_api.router.routes if route.path == "/api/themes/looks/{theme_id:path}"]
    assert sorted(method for route in routes for method in route.methods) == ["GET", "PUT"]
    for route in routes:
        assert not inspect.iscoroutinefunction(route.endpoint)
        marks = [getattr(dep.call, PERMISSION_MARKER_ATTR, None) for dep in route.dependant.dependencies]
        assert "members:read" in marks
