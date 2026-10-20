"""PRD-261 M-04 — ``GET /api/themes/mobile/{install_id}``: the member's theme as data for the phone.

The mobile and files routers are mounted on a test app over SQLite (the theme tables, and the three
tables the membership query reads); stored files live in memory behind the files route's reader and
the looks store's reader. Package rows are built here, each ``theme.json`` passing ``validate_manifest``:
a ``fromLooks`` block with two mobile fonts and ``clips`` for two states, and a ``tokens`` block with
``stills``. Pinned: the proofs of plan §2.6 M-04 (the answer of §2.2.11), each test named for one.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

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

import core.models  # noqa: E402,F401  (registers the tables the theme FKs name)
import services.themes.looks as looks_service  # noqa: E402
from api import themes_files, themes_mobile  # noqa: E402
from config import config  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.workspace_permission import PERMISSION_MARKER_ATTR  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemeInstallation, ThemeLook, ThemeOptIn, ThemePackage, ThemeRevocation  # noqa: E402
from services.themes import serve  # noqa: E402
from services.themes.digest import digest_of_manifest, file_manifest  # noqa: E402
from services.themes.manifest import validate_manifest  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

ROUTE = "/api/themes/mobile/{install_id}"
ENTRY, STYLE, PREPAINT_PATH, SCHEMA_PATH = "dist/entry.js", "dist/theme.css", "boot/prepaint.json", "looks/schema.json"
SANS, MONO = "assets/fonts/mobile/harbour-sans.ttf", "assets/fonts/mobile/harbour-mono.otf"
STATES = ("idle", "wave")
LOOK_FORMAT = "automatos-e2e-hello-look/v1"
LOOK_SCHEMA = {
    "schema": "automatos-look-schema/v1",
    "look": {"schema": LOOK_FORMAT},
    "knobs": {
        "ground": {"kind": "choice", "label": "Ground", "choices": ["paper", "ink"], "default": "paper"},
        "accent": {"kind": "choice", "label": "Accent", "choices": ["sea", "ember"], "default": "sea"},
    },
}
PREPAINT = {
    "knobs": {
        "ground": {
            "paper": {"--background": "0 0% 100%", "--foreground": "222 47% 11%", "--hello-glow": "0px"},
            "ink": {"--background": "222 47% 8%", "--foreground": "#F4F4F5"},
        },
        "accent": {"sea": {"--primary": "#1d6fd8", "--ring": "#1d6fd8"}, "ember": {"--primary": "16 85% 55%"}},
    },
    "pairs": [{"a": "ground", "b": "accent", "table": {"ink": {"ember": {"--primary-foreground": "0 0% 100%"}}}}],
    "numeric": [],
    "defaults": {"ground": "paper", "accent": "sea"},
    "tone": {"knob": "ground", "values": {"paper": "light", "ink": "dark"}},
}
# The defaults (paper, sea) painted: Studio tokens only, as the table writes them.
DEFAULT_LIGHT = {"--background": "0 0% 100%", "--foreground": "222 47% 11%", "--primary": "#1d6fd8", "--ring": "#1d6fd8"}
FROM_LOOKS = {"fromLooks": True, "fonts": {"sans": SANS, "mono": MONO}, "mascot": "clips"}
LIGHT_ONLY = {"--background": "0 0% 100%", "--foreground": "#14181c", "--primary": "221 83% 53%"}
TOKENS = {"tokens": {"light": LIGHT_ONLY}, "mascot": "stills"}
ALICE, BOB, CAROL, DAN, OPERATOR = 1, 2, 3, 4, 9
HOME, TEAM, ELSEWHERE = (uuid.UUID(int=n) for n in (0xA1, 0xA2, 0xA3))
RAW_TABLES = (
    "CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id VARCHAR, email VARCHAR)",
    "CREATE TABLE workspaces (id VARCHAR PRIMARY KEY, owner_id INTEGER, paused_at TIMESTAMP, deleted_at TIMESTAMP)",
    "CREATE TABLE workspace_members (workspace_id VARCHAR, user_id INTEGER, role VARCHAR, is_active BOOLEAN)",
)
THEME_TABLES = [model.__table__ for model in (ThemePackage, ThemeInstallation, ThemeOptIn, ThemeRevocation, ThemeLook)]


def _seed_people(db) -> None:
    """Alice owns HOME and is a member of TEAM; Carol owns TEAM and ELSEWHERE; Dan is in TEAM; Bob nowhere."""
    for pk, name in ((ALICE, "alice"), (BOB, "bob"), (CAROL, "carol"), (DAN, "dan"), (OPERATOR, "operator")):
        db.execute(sa.text("INSERT INTO users VALUES (:id, :cid, :email)"),
                   {"id": pk, "cid": f"user_{name}", "email": f"{name}@example.com"})
    for ws, owner in ((HOME, ALICE), (TEAM, CAROL), (ELSEWHERE, CAROL)):
        db.execute(sa.text("INSERT INTO workspaces VALUES (:id, :owner, NULL, NULL)"), {"id": str(ws), "owner": owner})
    for user in (ALICE, DAN):
        db.execute(sa.text("INSERT INTO workspace_members VALUES (:ws, :uid, 'viewer', :active)"),
                   {"ws": str(TEAM), "uid": user, "active": True})


@pytest.fixture
def world(monkeypatch):
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        for ddl in RAW_TABLES:
            conn.execute(sa.text(ddl))
    ThemePackage.metadata.create_all(engine, tables=THEME_TABLES)
    db = sessionmaker(bind=engine)()
    _seed_people(db)
    db.commit()
    stored = {}
    monkeypatch.setattr(looks_service, "read_stored_file", lambda prefix, path: stored.get((prefix, path)))
    monkeypatch.setattr(looks_service, "_schema_cache", looks_service._SchemaCache(8))
    try:
        yield SimpleNamespace(db=db, stored=stored, reader=lambda prefix, path: stored.get((prefix, path)))
    finally:
        db.close()
        engine.dispose()


def _manifest(mobile):
    extra = {"engines": {"automatos": "^1.1.0", "sdk": "^1.1.0"}, "mascot": {"states": list(STATES)}}
    manifest = {**build.hello_manifest(), **extra, **({"mobile": mobile} if mobile is not None else {})}
    assert validate_manifest(manifest).errors == []
    return manifest


def _files(manifest, prepaint):
    clips = {f"assets/mascot/{state}.mp4": (f"clip {state}".encode(), "video/mp4") for state in STATES}
    stills = {f"assets/mascot/{state}.png": (b"\x89PNG still " + state.encode(), "image/png") for state in STATES}
    return {
        "theme.json": (build.manifest_bytes(manifest), "application/json"),
        ENTRY: (b"export default {};\n", "text/javascript"),
        STYLE: (b".hello { color: red; }\n", "text/css"),
        PREPAINT_PATH: (json.dumps(prepaint).encode("utf-8"), "application/json"),
        SCHEMA_PATH: (json.dumps(LOOK_SCHEMA).encode("utf-8"), "application/json"),
        SANS: (b"\x00\x01\x00\x00 sans", "font/ttf"),
        MONO: (b"OTTO mono", "font/otf"),
        **clips, **stills,
    }


def _package(world, mobile=FROM_LOOKS, owner="marketplace", workspace=None, reviewed=True, prepaint=PREPAINT):
    manifest = _manifest(mobile)
    files = _files(manifest, prepaint)
    listing = file_manifest(files)
    digest = digest_of_manifest(listing)
    prefix = f"themes/{owner}/{digest[:16]}/"
    row = ThemePackage(
        theme_id=manifest["id"], version=manifest["version"], owner_type=owner, workspace_id=workspace,
        digest=digest, manifest=manifest, file_manifest=listing, storage_prefix=prefix,
        status="approved" if owner == "marketplace" else "pending", reviewed=reviewed,
    )
    world.db.add(row)
    world.db.flush()
    world.stored.update({(prefix, path): data for path, (data, _type) in files.items()})
    return row


def _install(world, package, workspace=TEAM):
    install = ThemeInstallation(workspace_id=workspace, theme_id=package.theme_id, package_id=package.id)
    world.db.add(install)
    world.db.commit()
    return install


def _look(look_id, ground, accent):
    return {"schema": LOOK_FORMAT, "id": look_id, "name": look_id.title(), "sub": "", "ground": ground, "accent": accent}


def _store_looks(world, user, active, workspace=TEAM):
    theme_id = build.hello_manifest()["id"]
    world.db.add(ThemeLook(workspace_id=workspace, user_id=user, theme_id=theme_id, looks=[_look("day", "paper", "sea")], active=active))
    world.db.commit()


def _person(name, workspace=HOME):
    user = UserContext(id=f"user_{name}", email=f"{name}@example.com", clerk_user_id=f"user_{name}")
    return RequestContext(workspace_id=workspace, user=user, auth_type="clerk")


def _client(world, ctx):
    app = FastAPI()
    app.include_router(themes_files.router)
    app.include_router(themes_mobile.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    app.dependency_overrides[themes_files.get_theme_file_reader] = lambda: world.reader

    def _db():
        yield world.db

    app.dependency_overrides[get_db] = _db
    return TestClient(app, raise_server_exceptions=False)


def _mobile(world, ctx, install_id, headers=None):
    return _client(world, ctx).get(f"/api/themes/mobile/{install_id}", headers=headers or {})


def _url(package, path):
    return serve.module_url(package.digest, path)


def _urls(node):
    """Every string in the answer that names a theme file."""
    if isinstance(node, dict):
        return [url for value in node.values() for url in _urls(value)]
    if isinstance(node, list):
        return [url for value in node for url in _urls(value)]
    return [node] if isinstance(node, str) and node.startswith(serve.FILES_ROUTE) else []


def test_from_looks_answers_the_callers_stored_active_look_under_its_tone(world):
    package = _package(world)
    install = _install(world, package)
    _store_looks(world, ALICE, _look("night", "ink", "ember"))
    resp = _mobile(world, _person("alice"), install.id)
    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "private, no-store"
    body = resp.json()
    assert {key: body[key] for key in ("schema", "installId", "themeId", "version")} == {
        "schema": 1, "installId": str(install.id), "themeId": package.theme_id, "version": package.version,
    }
    mobile = body["mobile"]
    assert (mobile["source"], mobile["tone"]) == ("fromLooks", "dark")
    assert mobile["tokens"] == {"light": None, "dark": {
        "--background": "222 47% 8%", "--foreground": "#F4F4F5", "--primary": "16 85% 55%", "--primary-foreground": "0 0% 100%",
    }}


def test_with_no_stored_look_the_tables_defaults_paint_and_each_member_gets_their_own(world):
    install = _install(world, _package(world))
    _store_looks(world, ALICE, _look("night", "ink", "ember"))
    _store_looks(world, CAROL, _look("dawn", "paper", "ember"))
    dan = _mobile(world, _person("dan"), install.id).json()["mobile"]
    assert (dan["tone"], dan["tokens"]) == ("light", {"light": DEFAULT_LIGHT, "dark": None})
    carol = _mobile(world, _person("carol"), install.id).json()["mobile"]
    paper_ember = {"--background": "0 0% 100%", "--foreground": "222 47% 11%", "--primary": "16 85% 55%"}
    assert carol["tokens"] == {"light": paper_ember, "dark": None}
    assert _mobile(world, _person("alice"), install.id).json()["mobile"]["tone"] == "dark"


def test_an_unreadable_look_store_paints_the_defaults_and_an_unreadable_table_answers_null(world, caplog):
    package = _package(world)
    install = _install(world, package)
    _store_looks(world, ALICE, _look("night", "ink", "ember"))
    world.stored[(package.storage_prefix, SCHEMA_PATH)] = b'{"tampered": true}'
    mobile = _mobile(world, _person("alice"), install.id).json()["mobile"]
    assert mobile["tokens"] == {"light": DEFAULT_LIGHT, "dark": None}
    assert "the phone gets the default look" in caplog.text
    world.stored[(package.storage_prefix, PREPAINT_PATH)] = b"{}"
    assert _mobile(world, _person("alice"), install.id).json()["mobile"] is None
    assert "refused as tampered" in caplog.text


def test_a_tokens_block_comes_back_as_written_with_no_tone(world):
    package = _package(world, mobile=TOKENS)
    install = _install(world, package)
    mobile = _mobile(world, _person("alice"), install.id).json()["mobile"]
    assert (mobile["source"], mobile["tone"], mobile["tokens"]) == ("tokens", None, {"light": LIGHT_ONLY, "dark": None})
    assert mobile["fonts"] == {}
    assert mobile["mascot"] == {"mode": "stills", "states": [
        {"name": state, "clip": None, "still": _url(package, f"assets/mascot/{state}.png")} for state in STATES
    ]}


def test_every_file_url_is_module_url_served_to_the_same_caller_and_never_code(world):
    package = _package(world)
    install = _install(world, package)
    client = _client(world, _person("alice"))
    mobile = client.get(f"/api/themes/mobile/{install.id}").json()["mobile"]
    assert mobile["fonts"] == {"sans": _url(package, SANS), "mono": _url(package, MONO)}
    assert mobile["mascot"] == {"mode": "clips", "states": [
        {"name": state, "clip": _url(package, f"assets/mascot/{state}.mp4"), "still": _url(package, f"assets/mascot/{state}.png")}
        for state in STATES
    ]}
    urls = _urls(mobile)
    assert len(urls) == 2 + 2 * len(STATES)
    for found in urls:
        assert not found.endswith((".js", ".css")), found
        served = client.get(found)
        assert served.status_code == 200, found
        assert served.content == world.stored[(package.storage_prefix, found.split(f"{package.digest}/", 1)[1])]


def test_no_block_private_code_without_the_opt_in_and_a_revoked_package_answer_mobile_null(world):
    plain = _install(world, _package(world, mobile=None), workspace=HOME)
    assert _mobile(world, _person("alice"), plain.id).json()["mobile"] is None
    private = _package(world, mobile=TOKENS, owner="workspace", workspace=TEAM, reviewed=False)
    install = _install(world, private)
    body = _mobile(world, _person("alice"), install.id).json()
    assert (body["mobile"], body["installId"]) == (None, str(install.id))
    world.db.add(ThemeOptIn(workspace_id=TEAM, user_id=ALICE, package_id=private.id))
    world.db.commit()
    assert _mobile(world, _person("alice"), install.id).json()["mobile"]["source"] == "tokens"
    world.db.add(ThemeRevocation(digest=private.digest, level="disable", reason="withdrawn"))
    world.db.commit()
    assert _mobile(world, _person("alice"), install.id).json()["mobile"] is None


def test_unknown_malformed_and_foreign_install_ids_answer_one_identical_404(world):
    foreign = _install(world, _package(world), workspace=ELSEWHERE)
    answers = [_mobile(world, _person("alice"), install_id) for install_id in (uuid.uuid4(), "not-a-uuid", foreign.id)]
    assert {resp.status_code for resp in answers} == {404}
    assert len({resp.text for resp in answers}) == 1
    assert {resp.headers["cache-control"] for resp in answers} == {"private, no-store"}
    assert _mobile(world, _person("carol"), foreign.id).status_code == 200


def test_an_api_key_caller_and_someone_from_no_workspace_get_404(world):
    install = _install(world, _package(world))
    key = RequestContext(workspace_id=TEAM, user=UserContext(system_role="admin"), auth_type="api_key")
    assert _mobile(world, key, install.id).status_code == 404
    assert _mobile(world, _person("bob", workspace=TEAM), install.id).status_code == 404


def test_the_workspace_header_changes_nothing(world):
    install = _install(world, _package(world))
    _store_looks(world, ALICE, _look("night", "ink", "ember"))
    answers = [
        _mobile(world, _person("alice", workspace=workspace), install.id, headers={"X-Workspace-ID": str(workspace)}).json()
        for workspace in (HOME, TEAM, ELSEWHERE)
    ]
    assert answers[0] == answers[1] == answers[2]
    assert answers[0]["mobile"]["tone"] == "dark"


def test_the_local_operator_reads_any_workspaces_install(world, monkeypatch):
    install = _install(world, _package(world), workspace=ELSEWHERE)
    user = UserContext(id="operator@example.com", email="operator@example.com", system_role="super_admin",
                       raw_claims={"source": "local_operator", "user_id": OPERATOR})
    operator = RequestContext(workspace_id=HOME, user=user, auth_type="anonymous")
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    assert _mobile(world, operator, install.id).json()["mobile"]["tokens"] == {"light": DEFAULT_LIGHT, "dark": None}
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    assert _mobile(world, operator, install.id).status_code == 404


def test_a_storage_failure_is_a_503(world):
    install = _install(world, _package(world))

    def broken(_prefix, _path):
        raise serve.ThemeStorageUnavailable("down")

    world.reader = broken
    resp = _mobile(world, _person("alice"), install.id)
    assert resp.status_code == 503
    assert resp.headers["cache-control"] == "private, no-store"


def test_the_route_is_a_plain_def_with_no_workspace_gate():
    (route,) = themes_mobile.router.routes
    assert (route.path, route.methods) == (ROUTE, {"GET"})
    assert not inspect.iscoroutinefunction(themes_mobile.get_mobile_theme)
    calls = [sub.call for sub in route.dependant.dependencies]
    assert not any(getattr(call, PERMISSION_MARKER_ATTR, None) for call in calls)


def test_the_route_and_its_service_never_read_ctx_workspace_id():
    for relative in ("api/themes_mobile.py", "services/themes/mobile.py"):
        source = (_ORCH / relative).read_text(encoding="utf-8")
        reads = [node.lineno for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Attribute) and node.attr == "workspace_id"
                 and isinstance(node.value, ast.Name) and node.value.id == "ctx"]
        assert reads == [], f"{relative} reads ctx.workspace_id at {reads}"
        assert 'getattr(ctx, "workspace_id"' not in source, relative


def test_main_mounts_the_router_once_and_the_manifest_lists_the_route_once():
    tree = ast.parse((_ORCH / "main.py").read_text(encoding="utf-8"))
    mounted = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and getattr(node.func, "attr", None) == "include_router"
               and node.args and getattr(node.args[0], "id", None) == "mobile_theme_router"]
    assert len(mounted) == 1
    manifest = json.loads((_ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert manifest["route_count"] == len(manifest["routes"])
    assert [r for r in manifest["routes"] if r["path"] == ROUTE] == [{"method": "GET", "path": ROUTE}]


_PROBE = """
import json
from fastapi.routing import APIRoute, iter_route_contexts
import main
served = [method + " " + route.path for route in iter_route_contexts(main.app.routes)
          if isinstance(route.original_route, APIRoute) and route.path.startswith("/api/themes/mobile")
          for method in sorted(route.methods - {"HEAD", "OPTIONS"})]
print(json.dumps(served))
"""


def test_the_app_serves_the_route_exactly_once():
    env = {**os.environ, "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test"}
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(_ORCH), env=env, capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert json.loads(proc.stdout.strip().splitlines()[-1]) == [f"GET {ROUTE}"]
