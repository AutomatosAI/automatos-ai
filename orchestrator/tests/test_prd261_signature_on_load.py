"""PRD-261 C-05c — a stored signature that stops checking out disables its version on load, with no write.

The real routers (``api/themes_files.py``: files and boot; ``api/themes_install.py``: active,
choice, default, update and rollback; ``api/marketplace_themes.py``: the listings) run on a test
app over SQLite (``create_all``), with the real hello package (``tests/fixtures/themes/build.py``)
read by the real reader and stored in memory, in both editions (SaaS: a Clerk member who signed in
a minute ago; local: the anonymous operator). The keys file is a temporary one holding RFC 8032
public keys (``signing-vector.json``), whose published seeds sign here. Pins (plan §2.6 C-05c, QB-28):

* a signature from a key the keys file doesn't hold, bytes no key signed, a file list that no
  longer adds up, or a key revoked in the keys file (or, in the local edition only, by the hosted
  feed: ``theme_revocation_sync.revoked_key_ids``) disables the version: ``active`` answers the base
  with ``revoked {installId, level, reason}``, the boot is the base boot, files are 404, the listing
  hides it, no default, update or rollback moves to it, nothing is stored; a new signature lifts it;
* an unsigned version reads no key and verifies nothing; a retired key's signature still runs; a
  keys file that doesn't load disables nothing (logged); a stored ``disable`` keeps its reason;
* 50 file requests for one digest verify once; a new keys file or revoked-key list verifies again;
* the transient disable is never in the session, even across a route that commits.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

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

import api.marketplace_themes as listings_api  # noqa: E402
import api.themes_install as install_api  # noqa: E402
import core.auth.workspace_permission as permission_mod  # noqa: E402
import core.models  # noqa: E402,F401  (registers every table the FKs name)
from api import themes_files  # noqa: E402
from config import config  # noqa: E402
from core.auth import step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.principal import LOCAL_OPERATOR_CLAIM_SOURCE  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.models.themes import (  # noqa: E402
    THEME_REVOCATION_SYNC_ROW_ID,
    ThemeChoice,
    ThemeInstallation,
    ThemeLook,
    ThemeOptIn,
    ThemePackage,
    ThemePublisher,
    ThemeRevocation,
    ThemeRevocationSync,
    ThemeWorkspaceSettings,
)
from services.themes import revocations, signature_state, signature_verify, signing  # noqa: E402
from services.themes.defaults import WITHDRAWN  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from services.themes.signature_state import SIGNATURE_DISABLE_REASON  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

THEME, LISTING, ENTRY, README = "automatos-e2e/hello", "automatos-e2e.hello", "dist/entry.js", "README.md"
OWNER_PK, CLERK = 7, "clerk-7"
WS = uuid.UUID(int=0xC05C)
VECTOR = json.loads((_ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
SIGNER, STRANGER = VECTOR["ed25519"]["cases"][:2]
NO_ONES_SIGNATURE = base64.b64encode(bytes(range(64))).decode("ascii")  # 64 bytes no key signed
TABLES = [model.__table__ for model in (
    ThemePackage, ThemeInstallation, ThemeChoice, ThemeLook, ThemeOptIn, ThemeWorkspaceSettings, ThemeRevocation,
    ThemeRevocationSync, ThemePublisher,
)]
RAW_TABLES = (
    "CREATE TABLE users (id INTEGER PRIMARY KEY, clerk_user_id VARCHAR, email VARCHAR)",
    "CREATE TABLE workspaces (id VARCHAR PRIMARY KEY, owner_id INTEGER, paused_at TIMESTAMP, deleted_at TIMESTAMP)",
    "CREATE TABLE workspace_members (workspace_id VARCHAR, user_id INTEGER, role VARCHAR, is_active BOOLEAN)",
)


def _ctx(edition):
    """The local edition's anonymous operator, or a SaaS Clerk member who signed in a minute ago."""
    if edition == "local":
        user = UserContext(id="operator@local", email="operator@local", system_role="super_admin",
                           raw_claims={"source": LOCAL_OPERATOR_CLAIM_SOURCE, "user_id": OWNER_PK})
        return RequestContext(workspace_id=WS, user=user, auth_type="anonymous")
    claims = {"sub": CLERK, "iat": int(time.time()), "fva": [1, -1]}
    user = UserContext(id=CLERK, clerk_user_id=CLERK, email="owner@example.com", system_role="user", raw_claims=claims)
    return RequestContext(workspace_id=WS, user=user, auth_type="clerk")


def _keys(world, *held):
    """Rewrite the keys file to hold each ``(case, status)`` of ``held``."""
    keys = [{"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": "2026-10-20"}
            for case, status in held]
    world.keys_file.write_text(json.dumps({"schema": 1, "keys": keys}), encoding="utf-8")


def _engine():
    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    with engine.begin() as conn:
        for ddl in RAW_TABLES:
            conn.execute(sa.text(ddl))
        conn.execute(sa.text("INSERT INTO users VALUES (:id, :clerk, 'owner@example.com')"),
                     {"id": OWNER_PK, "clerk": CLERK})
        conn.execute(sa.text("INSERT INTO workspaces VALUES (:id, :owner, NULL, NULL)"),
                     {"id": str(WS), "owner": OWNER_PK})
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    return engine


def _client(world):
    app = FastAPI()
    for router in (themes_files.router, install_api.router, listings_api.router):
        app.include_router(router)
    step_up.install_step_up_handler(app)
    app.dependency_overrides[get_request_context_hybrid] = lambda: world.ctx
    app.dependency_overrides[get_db] = lambda: world.db
    app.dependency_overrides[themes_files.get_theme_file_reader] = lambda: world.reader
    return TestClient(app)


@pytest.fixture(params=("saas", "local"))
def world(request, monkeypatch, tmp_path):
    engine = _engine()
    stored = {}
    state = SimpleNamespace(
        edition=request.param, ctx=_ctx(request.param), db=sessionmaker(bind=engine)(), stored=stored,
        keys_file=tmp_path / "signing-keys.json", reader=lambda prefix, path: stored.get((prefix, path)),
    )
    monkeypatch.setattr(config, "AUTH_EDITION", request.param)
    monkeypatch.setattr(config, "THEMES_MARKETPLACE", "on", raising=False)
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: "owner")
    monkeypatch.setattr(signature_verify, "KEYS_PATH", state.keys_file)
    _keys(state, (SIGNER, "active"))
    signature_state.clear_cache()
    state.client = _client(state)
    try:
        yield state
    finally:
        signature_state.clear_cache()
        state.db.close()
        engine.dispose()


def _sign(world, row, case=SIGNER):
    """Store the RFC seed's signature of the version's canonical text, as the paste route stores it."""
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(case["seed"]))
    text = signing.canonical_text(signing.signing_payload(row)).encode("ascii")
    _store(world, row, signature=base64.b64encode(seed.sign(text)).decode("ascii"))


def _store(world, row, **fields):
    for name, value in fields.items():
        setattr(row, name, value)
    world.db.commit()


def _row(world, version="1.0.0", signer=SIGNER):
    """An approved, reviewed Marketplace version of the hello theme, its files stored, signed by ``signer``."""
    data = build.build_package(replace={build.MANIFEST: build.changed_manifest({"version": version})})
    read = read_theme_package(data)
    prefix = f"themes/{THEME}/{version}/"
    row = ThemePackage(
        theme_id=THEME, version=version, owner_type="marketplace", digest=read.digest, manifest=read.manifest,
        file_manifest=read.file_manifest, storage_prefix=prefix, status="approved", reviewed=True,
    )
    world.db.add(row)
    world.db.commit()
    world.stored.update({(prefix, path): item.data for path, item in read.files.items()})
    if signer is not None:
        _sign(world, row, signer)
    return row


def _feed_revokes(world, *key_ids):
    """The single sync row as the local pull leaves it, with ``key_ids`` revoked by the hosted feed."""
    world.db.merge(ThemeRevocationSync(id=THEME_REVOCATION_SYNC_ROW_ID, revoked_key_ids=list(key_ids)))
    world.db.commit()


def _new_file_list(row):
    """The row's README listed with another hash: its file list no longer adds up to its digest."""
    return {**row.file_manifest, README: {**row.file_manifest[README], "sha256": "99" * 32}}


BREAKS = {
    "signed by a key not in the keys file": lambda world, row: _sign(world, row, STRANGER),
    "bytes no key signed": lambda world, row: _store(world, row, signature=NO_ONES_SIGNATURE),
    "a file list that no longer adds up": lambda world, row: _store(world, row, file_manifest=_new_file_list(row)),
    "the signing key revoked in the keys file": lambda world, row: _keys(world, (SIGNER, "revoked")),
}


def _ok(resp, status=200):
    assert resp.status_code == status, resp.text
    return resp.json()


def _install_and_choose(world, row):
    install = ThemeInstallation(workspace_id=WS, theme_id=THEME, package_id=row.id)
    world.db.add(install)
    world.db.commit()
    _ok(world.client.put("/api/themes/choice", json={"choice": str(install.id)}))
    return install


def _file(world, row, path=ENTRY):
    return world.client.get(f"/api/themes/files/{row.digest}/{path}")


def _loads(world, row, install):
    """What a member gets on load: active, boot, a file, and whether the listing shows the theme."""
    listed = {item["listingId"] for item in _ok(world.client.get("/api/marketplace/themes"))["listings"]}
    return SimpleNamespace(
        active=_ok(world.client.get("/api/themes/active")),
        boot=_ok(world.client.get(f"/api/themes/boot/{install.id}")),
        file=_file(world, row).status_code,
        listed=LISTING in listed,
        detail=world.client.get(f"/api/marketplace/themes/{LISTING}").status_code,
    )


def _assert_runs(world, row, install):
    loaded = _loads(world, row, install)
    assert loaded.active["choice"] == str(install.id), loaded.active
    assert loaded.active["theme"]["revoked"] is None and "revoked" not in loaded.active
    assert loaded.boot["entry"]["themeId"] == THEME
    assert (loaded.file, loaded.listed, loaded.detail) == (200, True, 200)


def _assert_disabled(world, row, install):
    loaded = _loads(world, row, install)
    assert (loaded.active["choice"], loaded.active["theme"]) == ("studio", None), loaded.active  # the theme's base
    notice = {"installId": str(install.id), "level": "disable", "reason": SIGNATURE_DISABLE_REASON}
    assert loaded.active["revoked"] == notice
    assert loaded.boot["entry"] is None and loaded.boot["addClasses"] == ["studio"]  # the base boot
    assert "hello" in loaded.boot["removeClasses"]
    assert (loaded.file, loaded.listed, loaded.detail) == (404, False, 404)
    _refused_as_withdrawn(world.client.put("/api/themes/default", json={"installId": str(install.id)}))


def _refused_as_withdrawn(resp):
    assert resp.status_code == 409 and resp.json()["detail"] == WITHDRAWN, resp.text


def _nothing_stored(world):
    assert world.db.query(ThemeRevocation).count() == 0


@pytest.mark.parametrize("why", sorted(BREAKS))
def test_a_signature_that_stops_checking_out_disables_the_version_on_load(world, why):
    row = _row(world)
    install = _install_and_choose(world, row)
    _assert_runs(world, row, install)
    BREAKS[why](world, row)
    _assert_disabled(world, row, install)
    _nothing_stored(world)


def test_re_signing_with_an_active_key_lifts_the_disable(world):
    row = _row(world)
    install = _install_and_choose(world, row)
    _keys(world, (SIGNER, "revoked"), (STRANGER, "active"))
    _assert_disabled(world, row, install)
    _sign(world, row, STRANGER)
    _assert_runs(world, row, install)
    _nothing_stored(world)


def test_a_key_the_hosted_feed_revoked_disables_the_version_in_the_local_edition(world):
    row = _row(world)
    install = _install_and_choose(world, row)
    _feed_revokes(world, SIGNER["keyId"])
    if world.edition == "local":
        _assert_disabled(world, row, install)
    else:  # SaaS publishes the feed and never pulls it: its revoked keys are its own keys file
        _assert_runs(world, row, install)
    _feed_revokes(world, STRANGER["keyId"])  # the feed now revokes another key only
    _assert_runs(world, row, install)
    _nothing_stored(world)


def _switch(world, install, to=None):
    """Update the install to ``to``; roll it back without one."""
    if to is None:
        return world.client.post(f"/api/themes/installs/{install.id}/rollback")
    return world.client.post(f"/api/themes/installs/{install.id}/update", json={"packageId": str(to.id)})


def _listed_versions(world):
    return set(_ok(world.client.get(f"/api/marketplace/themes/{LISTING}"))["versions"])


def test_update_and_rollback_to_a_version_whose_signature_fails_are_refused_until_it_is_re_signed(world):
    first = _row(world, "1.0.0")
    second = _row(world, "1.1.0", STRANGER)  # a key the keys file doesn't hold
    install = _install_and_choose(world, first)
    assert _listed_versions(world) == {"1.0.0"}
    _refused_as_withdrawn(_switch(world, install, second))
    _sign(world, second)
    assert _listed_versions(world) == {"1.0.0", "1.1.0"}
    assert _ok(_switch(world, install, second))["version"] == "1.1.0"

    _sign(world, first, STRANGER)  # now the rollback target's signature fails
    _refused_as_withdrawn(_switch(world, install))
    _sign(world, first)
    assert _ok(_switch(world, install))["version"] == "1.0.0"
    _nothing_stored(world)


def _count_verifies(monkeypatch):
    """Every ``signature_verify.verify`` call, by the payload's digest."""
    calls, real = [], signature_verify.verify

    def counting(payload, signature, keys=None):
        calls.append(payload["digest"])
        return real(payload, signature, keys)

    monkeypatch.setattr(signature_verify, "verify", counting)
    return calls


def test_an_unsigned_version_reads_no_key_and_verifies_nothing(world, monkeypatch):
    calls = _count_verifies(monkeypatch)
    world.keys_file.unlink()  # no keys file at all: an unsigned version never needs one
    _feed_revokes(world, SIGNER["keyId"], STRANGER["keyId"])
    row = _row(world, signer=None)
    install = _install_and_choose(world, row)
    _assert_runs(world, row, install)
    assert calls == []


def test_a_retired_keys_signature_runs_and_a_keys_file_that_does_not_load_disables_nothing(world, caplog):
    row = _row(world)
    install = _install_and_choose(world, row)
    _keys(world, (SIGNER, "retired"), (STRANGER, "active"))
    _assert_runs(world, row, install)
    world.keys_file.write_text("{not json", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger=signature_state.__name__):
        _assert_runs(world, row, install)
    logged = [record.getMessage() for record in caplog.records if record.name == signature_state.__name__]
    assert any("doesn't load" in message for message in logged), logged


def test_a_stored_disable_keeps_its_reason_and_a_stored_warn_gives_way_to_the_signatures(world):
    row = _row(world, signer=STRANGER)
    install = _install_and_choose(world, row)
    world.db.add(ThemeRevocation(digest=row.digest, level="warn", reason="Known to flicker."))
    world.db.commit()
    assert _ok(world.client.get("/api/themes/active"))["revoked"]["reason"] == SIGNATURE_DISABLE_REASON
    world.db.add(ThemeRevocation(theme_id=THEME, level="disable", reason="Withdrawn by its publisher."))
    world.db.commit()
    revoked = _ok(world.client.get("/api/themes/active"))["revoked"]
    assert revoked == {"installId": str(install.id), "level": "disable", "reason": "Withdrawn by its publisher."}


def test_fifty_file_requests_verify_once_and_a_new_key_set_verifies_again(world, monkeypatch):
    row = _row(world)
    calls = _count_verifies(monkeypatch)
    for _ in range(50):
        assert _file(world, row).status_code == 200
    assert calls == [row.digest]
    _feed_revokes(world, STRANGER["keyId"])  # a new revoked-key list: read (and so checked again) locally only
    assert _file(world, row).status_code == 200
    feed_checks = 1 if world.edition == "local" else 0
    assert len(calls) == 1 + feed_checks
    _keys(world, (SIGNER, "active"), (STRANGER, "active"))  # a new keys file
    assert _file(world, row).status_code == 200 and len(calls) == 2 + feed_checks
    _keys(world, (SIGNER, "revoked"))
    assert _file(world, row).status_code == 404 and len(calls) == 3 + feed_checks
    assert _file(world, row).status_code == 404 and len(calls) == 3 + feed_checks  # the same key set: cached


def test_the_transient_disable_is_never_in_the_session_even_across_a_commit(world):
    row = _row(world, signer=STRANGER)
    first = revocations.revocation_of(world.db, row)
    assert (first.level, first.reason, first.digest) == ("disable", SIGNATURE_DISABLE_REASON, row.digest)
    assert sa.inspect(first).transient and first not in world.db and not world.db.new
    install = _install_and_choose(world, row)  # PUT /api/themes/choice commits, then answers active
    assert _ok(world.client.get("/api/themes/active"))["revoked"]["installId"] == str(install.id)
    assert revocations.revocation_of(world.db, row) is not first  # built again on every load
    assert not world.db.new
    _nothing_stored(world)
