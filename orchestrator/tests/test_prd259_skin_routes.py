"""PRD-259 S-003 — the private-skin routes: POST /api/skins/import, GET /api/skins/installed,
GET /api/skins/{skin_id}/assets/{asset_path:path}.

The routes run against SQLite (the two tables, built by ``create_all``) and a fake S3
client patched into ``services/skins/storage.py``. Two workspaces, A and B, prove the
tenant isolation: B never sees A's private skin, and gets 404 (never 403) on its files.
Callers are the local edition's anonymous principal, which the workspace gate treats
as the owner; the gate itself is asserted on the route.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config  # noqa: E402
from core.models.marketplace_skins import MarketplaceSkin, SkinInstallation  # noqa: E402
from services.skins import storage  # noqa: E402
from tests.fixtures.skin_packages import PNG, base_skin, full_package, package, woff2  # noqa: E402

FONT_PATH = "assets/fonts/brand.woff2"
FONT_SKIN = dict(fonts={"sans": FONT_PATH, "serif": "newsreader", "mono": "geist-mono"})
ITEM_KEYS = {"id", "slug", "name", "version", "base", "tones", "tokens", "fonts", "shape", "mascot", "preview", "ownerType"}


class _FakeS3:
    """Objects by key in one bucket; a missing key raises like boto3 would."""

    def __init__(self, fail_writes: bool = False):
        self.objects = {}
        self.fail_writes = fail_writes

    def put_object(self, Bucket, Key, Body, ContentType):  # noqa: N803 (boto3's names)
        if self.fail_writes:
            raise ConnectionError("storage is down")
        self.objects[Key] = (Body, ContentType)

    def get_object(self, Bucket, Key):  # noqa: N803
        import io

        return {"Body": io.BytesIO(self.objects[Key][0])}

    def delete_object(self, Bucket, Key):  # noqa: N803
        self.objects.pop(Key, None)


@pytest.fixture
def env(monkeypatch):
    import api.marketplace_skins as skins_api

    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    MarketplaceSkin.metadata.create_all(engine, tables=[MarketplaceSkin.__table__, SkinInstallation.__table__])
    fake = _FakeS3()
    monkeypatch.setattr(storage, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    monkeypatch.setattr(skins_api, "resolve_user_pk", lambda db, ctx: None)
    yield type("Env", (), {"sessions": sessionmaker(bind=engine), "s3": fake})
    engine.dispose()


def _client(env, workspace_id):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.marketplace_skins import router
    from core.auth.dependencies import RequestContext, UserContext
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(router)

    def _db():
        session = env.sessions()
        try:
            yield session
        finally:
            session.close()

    def _ctx():
        return RequestContext(
            workspace_id=workspace_id,
            user=UserContext(id="local-operator", email="local@example.com", system_role="user"),
            auth_type="anonymous",
        )

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_request_context_hybrid] = _ctx
    return TestClient(app)


def _import(client, data: bytes):
    return client.post("/api/skins/import", files={"file": ("brand.autoskin", data, "application/zip")})


def _rows(env, model):
    session = env.sessions()
    try:
        return session.query(model).all()
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def test_import_stores_the_files_privately_and_installs_the_skin(env):
    workspace_a = uuid4()
    response = _import(_client(env, workspace_a), full_package())

    assert response.status_code == 200, response.text
    item = response.json()
    assert set(item) == ITEM_KEYS
    assert item["slug"] == "studio-own" and item["version"] == "1.0.0" and item["ownerType"] == "workspace"
    assert item["base"] == "studio" and item["tones"] == ["light", "dark"]
    assert item["fonts"]["sans"] == FONT_PATH and item["preview"] == "assets/preview.png"

    prefix = f"skins/_workspace/{workspace_a}/studio-own/1.0.0/"
    assert env.s3.objects[prefix + FONT_PATH] == (woff2(), "font/woff2")
    assert all(key.startswith(prefix) for key in env.s3.objects)
    assert len(env.s3.objects) == 2 + 36  # the font, the preview and the 36 mascot clips

    (row,) = _rows(env, MarketplaceSkin)
    assert (row.owner_type, row.status, row.workspace_id, row.asset_prefix) == ("workspace", "approved", workspace_a, prefix)
    assert str(row.id) == item["id"]
    assert row.skin["manifest"][FONT_PATH]["type"] == "font/woff2"
    (installation,) = _rows(env, SkinInstallation)
    assert (installation.workspace_id, installation.skin_id, installation.version) == (workspace_a, row.id, "1.0.0")
    assert installation.uninstalled_at is None


def test_a_refused_package_is_400_with_the_codes_and_stores_nothing(env):
    response = _import(_client(env, uuid4()), package(extra=[("../x", b"x"), ("assets/skin.css", b"*{}")]))

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert [error["code"] for error in detail["errors"]] == ["UNSAFE_PATH", "NOT_ALLOWED"]
    assert all(error["message"] for error in detail["errors"])
    assert env.s3.objects == {} and _rows(env, MarketplaceSkin) == []


def test_an_invalid_skin_json_is_400_with_the_validators_codes(env):
    skin = base_skin()
    skin["tokens"]["light"]["foreground"] = skin["tokens"]["light"]["background"]
    response = _import(_client(env, uuid4()), package(skin))
    assert response.status_code == 400
    assert "CONTRAST" in [error["code"] for error in response.json()["detail"]["errors"]]


def test_a_package_over_the_cap_is_413(env, monkeypatch):
    monkeypatch.setattr(config, "SKIN_MAX_PACKAGE_MB", 1)
    response = _import(_client(env, uuid4()), b"PK\x03\x04" + b"\x00" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert [error["code"] for error in response.json()["detail"]["errors"]] == ["PACKAGE_TOO_LARGE"]
    assert _rows(env, MarketplaceSkin) == []


def test_a_storage_failure_is_503_and_writes_no_row(env):
    env.s3.fail_writes = True
    response = _import(_client(env, uuid4()), package())
    assert response.status_code == 503
    assert _rows(env, MarketplaceSkin) == [] and _rows(env, SkinInstallation) == []


def test_a_failed_write_on_reimport_keeps_the_previous_version(env):
    workspace_a = uuid4()
    client = _client(env, workspace_a)
    _import(client, package())
    before = dict(env.s3.objects)
    env.s3.fail_writes = True

    assert _import(client, package(base_skin(version="1.1.0"))).status_code == 503
    (row,) = _rows(env, MarketplaceSkin)
    assert row.version == "1.0.0"
    (installation,) = _rows(env, SkinInstallation)
    assert installation.version == "1.0.0"
    assert env.s3.objects == before


def test_reimporting_the_same_slug_replaces_its_content_and_version(env):
    workspace_a = uuid4()
    client = _client(env, workspace_a)
    first = _import(client, package(base_skin(**FONT_SKIN), {"assets/preview.png": PNG, FONT_PATH: woff2()})).json()
    second = _import(client, package(base_skin(version="1.1.0", name="Studio, again"))).json()

    assert second["id"] == first["id"] and second["version"] == "1.1.0" and second["name"] == "Studio, again"
    (row,) = _rows(env, MarketplaceSkin)
    assert row.version == "1.1.0" and set(row.skin["manifest"]) == {"assets/preview.png"}
    (installation,) = _rows(env, SkinInstallation)
    assert installation.version == "1.1.0" and installation.uninstalled_at is None
    new_prefix = f"skins/_workspace/{workspace_a}/studio-own/1.1.0/"
    assert set(env.s3.objects) == {new_prefix + "assets/preview.png"}  # the 1.0.0 files are gone


def test_reimporting_the_same_version_removes_the_files_it_dropped(env):
    workspace_a = uuid4()
    client = _client(env, workspace_a)
    _import(client, package(base_skin(**FONT_SKIN), {"assets/preview.png": PNG, FONT_PATH: woff2()}))
    _import(client, package())

    prefix = f"skins/_workspace/{workspace_a}/studio-own/1.0.0/"
    assert set(env.s3.objects) == {prefix + "assets/preview.png"}
    assert len(_rows(env, MarketplaceSkin)) == 1


def test_two_workspaces_importing_one_slug_never_share_a_row_or_a_file(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    id_a = _import(_client(env, workspace_a), package()).json()["id"]
    id_b = _import(_client(env, workspace_b), package()).json()["id"]

    assert id_a != id_b
    assert {row.workspace_id for row in _rows(env, MarketplaceSkin)} == {workspace_a, workspace_b}
    assert {key.split("/")[2] for key in env.s3.objects} == {str(workspace_a), str(workspace_b)}


# ---------------------------------------------------------------------------
# Installed
# ---------------------------------------------------------------------------


def test_installed_lists_this_workspaces_skins_only(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    imported = _import(_client(env, workspace_a), package()).json()

    listed = _client(env, workspace_a).get("/api/skins/installed")
    assert listed.status_code == 200
    assert listed.json() == [imported]
    assert _client(env, workspace_b).get("/api/skins/installed").json() == []


def test_a_forged_installation_never_shows_another_workspaces_private_skin(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    skin_id = _import(_client(env, workspace_a), package()).json()["id"]
    session = env.sessions()
    row = session.query(MarketplaceSkin).one()
    session.add(SkinInstallation(workspace_id=workspace_b, skin_id=row.id, version=row.version))
    session.commit()
    session.close()

    client_b = _client(env, workspace_b)
    assert client_b.get("/api/skins/installed").json() == []
    assert client_b.get(f"/api/skins/{skin_id}/assets/assets/preview.png").status_code == 404


def test_a_marketplace_skin_is_visible_only_where_it_is_installed(env):
    workspace_b, workspace_c = uuid4(), uuid4()
    session = env.sessions()
    skin = MarketplaceSkin(
        slug="ink", name="Ink", version="1.0.0", owner_type="marketplace", status="approved",
        skin={**base_skin(slug="ink"), "manifest": {"assets/preview.png": {"type": "image/png", "size": len(PNG), "sha256": "x"}}},
        asset_prefix="skins/ink/1.0.0/",
    )
    session.add(skin)
    session.flush()
    session.add(SkinInstallation(workspace_id=workspace_b, skin_id=skin.id, version="1.0.0"))
    session.commit()
    skin_id = str(skin.id)
    session.close()
    env.s3.objects["skins/ink/1.0.0/assets/preview.png"] = (PNG, "image/png")

    assert [item["ownerType"] for item in _client(env, workspace_b).get("/api/skins/installed").json()] == ["marketplace"]
    assert _client(env, workspace_b).get(f"/api/skins/{skin_id}/assets/assets/preview.png").status_code == 200
    assert _client(env, workspace_c).get(f"/api/skins/{skin_id}/assets/assets/preview.png").status_code == 404
    assert _client(env, workspace_c).get("/api/skins/installed").json() == []


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


def test_an_asset_is_served_with_its_checked_type_and_the_three_headers(env):
    workspace_a = uuid4()
    client = _client(env, workspace_a)
    skin_id = _import(client, full_package()).json()["id"]

    response = client.get(f"/api/skins/{skin_id}/assets/{FONT_PATH}?v=1.0.0")
    assert response.status_code == 200
    assert response.content == woff2()
    assert response.headers["content-type"] == "font/woff2"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == f"private, max-age={config.SKIN_ASSET_CACHE_SECONDS}, immutable"
    assert client.get(f"/api/skins/{skin_id}/assets/assets/mascot/idle.webm").headers["content-type"] == "video/webm"


def test_another_workspace_gets_404_on_a_private_skins_assets(env):
    workspace_a, workspace_b = uuid4(), uuid4()
    skin_id = _import(_client(env, workspace_a), full_package()).json()["id"]

    response = _client(env, workspace_b).get(f"/api/skins/{skin_id}/assets/{FONT_PATH}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Skin asset not found."


@pytest.mark.parametrize("path", [
    "assets/fonts/other.woff2", "skin.json", "assets/../skin.json", "assets/preview.png/", "ASSETS/preview.png", "",
])
def test_a_path_outside_the_manifest_is_404(env, path):
    workspace_a = uuid4()
    client = _client(env, workspace_a)
    skin_id = _import(client, package()).json()["id"]
    assert client.get(f"/api/skins/{skin_id}/assets/{path}").status_code == 404


@pytest.mark.parametrize("skin_id", ["not-a-uuid", "00000000-0000-0000-0000-000000000000"])
def test_an_unknown_skin_is_404(env, skin_id):
    assert _client(env, uuid4()).get(f"/api/skins/{skin_id}/assets/assets/preview.png").status_code == 404


def test_a_file_missing_from_storage_is_404(env):
    workspace_a = uuid4()
    client = _client(env, workspace_a)
    skin_id = _import(client, package()).json()["id"]
    env.s3.objects.clear()
    assert client.get(f"/api/skins/{skin_id}/assets/assets/preview.png").status_code == 404


# ---------------------------------------------------------------------------
# The routes' shape
# ---------------------------------------------------------------------------


def _route(path: str, method: str):
    from api.marketplace_skins import router

    return next(r for r in router.routes if r.path == path and method in r.methods)


def test_import_is_gated_by_the_workspace_upload_permission():
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    route = _route("/api/skins/import", "POST")
    gates = [getattr(dep.dependency, PERMISSION_MARKER_ATTR, None) for dep in route.dependencies]
    assert "workspace:manage" in gates


def test_every_route_is_a_plain_def():
    for path, method in (
        ("/api/skins/import", "POST"), ("/api/skins/installed", "GET"),
        ("/api/skins/{skin_id}/assets/{asset_path:path}", "GET"),
    ):
        assert not inspect.iscoroutinefunction(_route(path, method).endpoint), path


_PROBE = r"""
import json
from main import _body_limit_for, app
from services.skins.package import package_max_bytes

routes = sorted(
    [method, route.path] for route in app.routes if getattr(route, "path", "").startswith("/api/skins")
    for method in sorted(set(getattr(route, "methods", None) or []) - {"HEAD", "OPTIONS"})
)
print("SKINS=" + json.dumps({
    "limit": _body_limit_for("/api/skins/import"), "package": package_max_bytes(), "routes": routes,
}))
"""


def test_the_real_app_serves_the_routes_and_lets_a_full_package_through():
    """main.py's transport cap would 413 a valid 15 MB package at the 10 MB default (S-003)."""
    import json
    import os
    import subprocess

    env = dict(os.environ)
    env.update({
        "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test", "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "59432", "POSTGRES_DB": "test", "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
    })
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=str(ORCH_ROOT), env=env, capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(out for out in proc.stdout.splitlines() if out.startswith("SKINS="))
    probe = json.loads(line.split("=", 1)[1])

    assert probe["limit"] == probe["package"] + 512 * 1024 == 15 * 1024 * 1024 + 512 * 1024
    assert probe["routes"] == sorted([
        ["POST", "/api/skins/import"], ["GET", "/api/skins/installed"], ["GET", "/api/skins/schema"],
        ["GET", "/api/skins/{skin_id}/assets/{asset_path:path}"],
    ])


def test_the_router_file_stays_under_400_lines():
    assert len((ORCH_ROOT / "api" / "marketplace_skins.py").read_text(encoding="utf-8").splitlines()) < 400
