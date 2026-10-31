"""PRD-271 C03 — the phone's answer carries the theme's layout (``services/themes/mobile.py`` §4.4).

``GET /api/themes/mobile/{install_id}`` keeps ``schema`` 1 and every field of the Studio skin, and adds
``mobile.layout``: ``{"contract", "tone", "slots"}`` with every slot of the contract, or null. The world
is ``test_prd261_mobile_endpoint.py``'s: the mobile router over SQLite (the theme tables and the three
tables the membership query reads), stored files in memory behind the route's reader and the looks
store's. Each package's ``theme.json`` passes ``validate_manifest``; a package built straight into a row
can hold defaults the reader would refuse, which is how the null cases are reached.
"""
from __future__ import annotations

import json
import os
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
from api import themes_mobile  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import ThemeInstallation, ThemeLook, ThemeOptIn, ThemePackage, ThemeRevocation  # noqa: E402
from services.themes import serve  # noqa: E402
from services.themes.digest import digest_of_manifest, file_manifest  # noqa: E402
from services.themes.manifest import validate_manifest  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

PREPAINT_PATH, SCHEMA_PATH = "boot/prepaint.json", "looks/schema.json"
LOOK_FORMAT = "automatos-e2e-hello-look/v1"
LOOK_SCHEMA = {
    "schema": "automatos-look-schema/v1",
    "look": {"schema": LOOK_FORMAT},
    "knobs": {
        "ground": {"kind": "choice", "label": "Ground", "choices": ["paper", "ink"], "default": "paper"},
        "accent": {"kind": "choice", "label": "Accent", "choices": ["sea", "ember"], "default": "sea"},
        "corners": {"kind": "choice", "label": "Corners", "choices": ["16", "28"], "default": "28"},
    },
}
# A table that paints the skin's Studio tokens and the layout's own variables: --hello-* in any form.
PREPAINT = {
    "knobs": {
        "ground": {
            "paper": {"--background": "0 0% 100%", "--foreground": "222 47% 11%", "--hello-bg": "#14181C"},
            "ink": {"--background": "222 47% 8%", "--foreground": "240 4.76% 95.88%", "--hello-bg": "#030507"},
        },
        "accent": {
            "sea": {"--hello-accent": "#3FE0FF", "--hello-ink": "#021116"},
            "ember": {"--hello-accent": "#FF8A3D", "--hello-ink": "#1A0A02"},
        },
        "corners": {"16": {}, "28": {}},
    },
    "pairs": [],
    "numeric": [],
    "defaults": {"ground": "paper", "accent": "sea", "corners": "28"},
    "tone": {"knob": "ground", "values": {"paper": "light", "ink": "dark"}},
}
TOKENS = {"tokens": {"light": {"--background": "0 0% 100%", "--foreground": "#14181c"}}}
HUD_VALUES = {"bg": "#030507", "accent": "#3FE0FF", "accentInk": "#021116"}
HUD_DEFAULTS = {"grid": 14, "scan": "on", "glow": 22, "auto": "on"}
FROM_LOOK = {"bg": {"look": "--hello-bg"}, "accent": {"look": "--hello-accent"}, "accentInk": {"look": "--hello-ink"},
             "grid": {"look": "corners"}}
ALICE, DAN = 1, 4
TEAM = uuid.UUID(int=0xA2)
RAW_TABLES = (
    "CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id VARCHAR, email VARCHAR)",
    "CREATE TABLE workspaces (id VARCHAR PRIMARY KEY, owner_id INTEGER, paused_at TIMESTAMP, deleted_at TIMESTAMP)",
    "CREATE TABLE workspace_members (workspace_id VARCHAR, user_id INTEGER, role VARCHAR, is_active BOOLEAN)",
)
THEME_TABLES = [model.__table__ for model in (ThemePackage, ThemeInstallation, ThemeOptIn, ThemeRevocation, ThemeLook)]


@pytest.fixture
def world(monkeypatch):
    """Alice and Dan are members of TEAM, which Carol (3) owns."""
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        for ddl in RAW_TABLES:
            conn.execute(sa.text(ddl))
        for pk, name in ((ALICE, "alice"), (3, "carol"), (DAN, "dan")):
            conn.execute(sa.text("INSERT INTO users VALUES (:id, :cid, :email)"),
                         {"id": pk, "cid": f"user_{name}", "email": f"{name}@example.com"})
        conn.execute(sa.text("INSERT INTO workspaces VALUES (:id, 3, NULL, NULL)"), {"id": str(TEAM)})
        for user in (ALICE, DAN):
            conn.execute(sa.text("INSERT INTO workspace_members VALUES (:ws, :uid, 'viewer', 1)"), {"ws": str(TEAM), "uid": user})
    ThemePackage.metadata.create_all(engine, tables=THEME_TABLES)
    db = sessionmaker(bind=engine)()
    stored = {}
    monkeypatch.setattr(looks_service, "read_stored_file", lambda prefix, path: stored.get((prefix, path)))
    monkeypatch.setattr(looks_service, "_schema_cache", looks_service.BoundedCache(8))
    try:
        yield SimpleNamespace(db=db, stored=stored, reader=lambda prefix, path: stored.get((prefix, path)))
    finally:
        db.close()
        engine.dispose()


def _manifest(mobile):
    manifest = {**build.hello_manifest(), "engines": {"automatos": "^1.2.0", "sdk": "^1.1.0"}, "mobile": mobile}
    assert validate_manifest(manifest).errors == []
    return manifest


def _package(world, mobile, prepaint=PREPAINT, owner="marketplace", reviewed=True):
    manifest = _manifest(mobile)
    files = {
        "theme.json": (build.manifest_bytes(manifest), "application/json"),
        "dist/entry.js": (b"export default {};\n", "text/javascript"),
        PREPAINT_PATH: (json.dumps(prepaint).encode("utf-8"), "application/json"),
        SCHEMA_PATH: (json.dumps(LOOK_SCHEMA).encode("utf-8"), "application/json"),
    }
    listing = file_manifest(files)
    digest = digest_of_manifest(listing)
    prefix = f"themes/{owner}/{digest[:16]}/"
    row = ThemePackage(
        theme_id=manifest["id"], version=manifest["version"], owner_type=owner, workspace_id=None if owner == "marketplace" else TEAM,
        digest=digest, manifest=manifest, file_manifest=listing, storage_prefix=prefix,
        status="approved" if owner == "marketplace" else "pending", reviewed=reviewed,
    )
    world.db.add(row)
    world.db.flush()
    world.stored.update({(prefix, path): data for path, (data, _type) in files.items()})
    install = ThemeInstallation(workspace_id=TEAM, theme_id=row.theme_id, package_id=row.id)
    world.db.add(install)
    world.db.commit()
    return row, install


def _store_look(world, user, ground, accent, corners="28"):
    look = {"schema": LOOK_FORMAT, "id": "mine", "name": "Mine", "sub": "", "ground": ground, "accent": accent, "corners": corners}
    world.db.add(ThemeLook(workspace_id=TEAM, user_id=user, theme_id=build.hello_manifest()["id"], looks=[look], active=look))
    world.db.commit()


def _person(user_pk, name):
    return RequestContext(workspace_id=TEAM, user=UserContext(id=f"user_{name}", email=f"{name}@example.com",
                                                              clerk_user_id=f"user_{name}"), auth_type="clerk")


def _phone(user_pk, name):
    """A paired phone (PRD-266): the device lane's person, bound to the device's workspace."""
    user = UserContext(id=f"{name}@example.com", email=f"{name}@example.com",
                       raw_claims={"source": "mobile_device", "device_id": "d", "user_id": user_pk})
    return RequestContext(workspace_id=TEAM, user=user, auth_type="device")


def _answer(world, ctx, install):
    app = FastAPI()
    app.include_router(themes_mobile.router)
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx

    def _db():
        yield world.db

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[themes_mobile.get_theme_file_reader] = lambda: world.reader
    resp = TestClient(app, raise_server_exceptions=False).get(f"/api/themes/mobile/{install.id}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_fixed_slots_come_back_with_every_default_the_contracts_tone_and_schema_1(world):
    _row, install = _package(world, {**TOKENS, "layout": "hud@1", "slots": HUD_VALUES})
    body = _answer(world, _person(ALICE, "alice"), install)
    assert body["schema"] == 1
    mobile = body["mobile"]
    assert (mobile["source"], mobile["tone"], mobile["tokens"]["light"]) == ("tokens", None, TOKENS["tokens"]["light"])
    assert mobile["layout"] == {"contract": "hud@1", "tone": "dark", "slots": {**HUD_VALUES, **HUD_DEFAULTS}}


def test_a_block_with_no_layout_answers_layout_null(world):
    _row, install = _package(world, TOKENS)
    assert _answer(world, _person(ALICE, "alice"), install)["mobile"]["layout"] is None


def test_look_slots_follow_the_members_saved_look_and_a_member_with_none_gets_the_defaults(world):
    _row, install = _package(world, {"fromLooks": True, "layout": "hud@1", "slots": FROM_LOOK})
    _store_look(world, ALICE, "ink", "ember", corners="16")
    alice = _answer(world, _person(ALICE, "alice"), install)["mobile"]
    assert alice["tone"] == "dark"  # the skin and the slots paint the same look
    assert alice["layout"]["slots"] == {**HUD_DEFAULTS, "bg": "#030507", "accent": "#FF8A3D", "accentInk": "#1A0A02", "grid": 16}
    dan = _answer(world, _person(DAN, "dan"), install)["mobile"]
    assert dan["layout"]["slots"] == {**HUD_DEFAULTS, "bg": "#14181C", "accent": "#3FE0FF", "accentInk": "#021116", "grid": 28}


def test_a_paired_phone_gets_what_the_member_gets_in_the_browser(world):
    _row, install = _package(world, {"fromLooks": True, "layout": "hud@1", "slots": FROM_LOOK})
    _store_look(world, ALICE, "ink", "ember", corners="16")
    assert _answer(world, _phone(ALICE, "alice"), install) == _answer(world, _person(ALICE, "alice"), install)


def test_a_look_that_cant_fill_a_slot_paints_the_defaults_whole(world):
    # ember's accent is a length here, which no colour slot takes: the whole default look is painted instead.
    accents = {**PREPAINT["knobs"]["accent"], "ember": {"--hello-accent": "12px", "--hello-ink": "#1A0A02"}}
    table = {**PREPAINT, "knobs": {**PREPAINT["knobs"], "accent": accents}}
    _row, install = _package(world, {**TOKENS, "layout": "hud@1", "slots": FROM_LOOK}, prepaint=table)
    _store_look(world, ALICE, "ink", "ember", corners="16")
    slots = _answer(world, _person(ALICE, "alice"), install)["mobile"]["layout"]["slots"]
    assert slots == {**HUD_DEFAULTS, "bg": "#14181C", "accent": "#3FE0FF", "accentInk": "#021116", "grid": 28}


def test_defaults_that_cant_fill_a_slot_answer_layout_null_and_still_serve_the_skin(world, caplog):
    grounds = {name: {key: value for key, value in option.items() if key != "--hello-bg"}
               for name, option in PREPAINT["knobs"]["ground"].items()}
    table = {**PREPAINT, "knobs": {**PREPAINT["knobs"], "ground": grounds}}
    _row, install = _package(world, {"fromLooks": True, "layout": "hud@1", "slots": FROM_LOOK}, prepaint=table)
    mobile = _answer(world, _person(ALICE, "alice"), install)["mobile"]
    assert mobile["layout"] is None and mobile["source"] == "fromLooks" and mobile["tokens"]["light"]
    assert "hud@1 defaults fill no bg" in caplog.text


def test_a_table_refused_on_read_answers_layout_null_under_a_tokens_skin(world, caplog):
    row, install = _package(world, {**TOKENS, "layout": "hud@1", "slots": FROM_LOOK})
    world.stored[(row.storage_prefix, PREPAINT_PATH)] = b"{}"
    mobile = _answer(world, _person(ALICE, "alice"), install)["mobile"]
    assert (mobile["source"], mobile["layout"]) == ("tokens", None)
    assert "slots come from looks it can't paint" in caplog.text


def test_storage_failing_under_a_tokens_skin_costs_the_layout_never_the_skin(world, caplog):
    _row, install = _package(world, {**TOKENS, "layout": "hud@1", "slots": FROM_LOOK})

    def broken(_prefix, _path):
        raise serve.ThemeStorageUnavailable("down")

    world.reader = broken
    mobile = _answer(world, _person(ALICE, "alice"), install)["mobile"]
    assert (mobile["source"], mobile["tokens"]["light"], mobile["layout"]) == ("tokens", TOKENS["tokens"]["light"], None)
    assert "can't be read from storage: the phone gets its skin, no layout" in caplog.text


def test_revoked_or_not_opted_in_answers_mobile_null_layout_and_all(world):
    row, install = _package(world, {**TOKENS, "layout": "hud@1", "slots": HUD_VALUES}, owner="workspace", reviewed=False)
    assert _answer(world, _person(ALICE, "alice"), install)["mobile"] is None
    world.db.add(ThemeOptIn(workspace_id=TEAM, user_id=ALICE, package_id=row.id))
    world.db.commit()
    assert _answer(world, _person(ALICE, "alice"), install)["mobile"]["layout"]["contract"] == "hud@1"
    world.db.add(ThemeRevocation(digest=row.digest, level="disable", reason="withdrawn"))
    world.db.commit()
    assert _answer(world, _person(ALICE, "alice"), install)["mobile"] is None


def test_the_answer_carries_no_url_or_code_in_the_slots(world):
    _row, install = _package(world, {"fromLooks": True, "layout": "hud@1", "slots": FROM_LOOK})
    slots = _answer(world, _person(ALICE, "alice"), install)["mobile"]["layout"]["slots"]
    texts = [value for value in slots.values() if isinstance(value, str)]
    assert texts and not any("(" in value and not value.startswith("rgba(") for value in texts)
    assert not any(marker in value for value in texts for marker in ("url", "var", "/"))
