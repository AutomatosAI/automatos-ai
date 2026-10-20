"""PRD-261 C-05d — ``POST /api/themes/import`` with an optional ``signature`` part (a ``.sig.json``).

The import and install routers run on a test app over SQLite with the fake S3 of
``test_prd261_theme_import.py`` (its helpers are reused, never its tests). The signing keys file
points at the RFC 8032 public keys of the shared vector (``tests/fixtures/themes/signing-vector.json``);
the RFC's published seeds, held in memory, make the signatures. Pins (plan §2.6 C-05d):

* a verified envelope is stored on the row and ``GET /api/themes/installed`` shows
  ``signature {state: verified, keyId}``; the same bytes uploaded again with it gain it;
* an envelope for other bytes, another id or version is 400 ``SIGNATURE_MISMATCH``; an inactive or
  unknown key, another key named, a malformed envelope are 400 ``SIGNATURE_BAD``; over 4 KB is 413;
  a keys file that does not load is 503; each stores no row and puts no file;
* a non-admin may upload ``automatos/...`` with a verified envelope (201); without one the 403 stands;
* QB-24's answer is one named constant; without the part the upload is what it was; the import
  route's body limit leaves room for the envelope.
"""
from __future__ import annotations

import ast
import base64
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

import sqlalchemy as sa  # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.models  # noqa: E402,F401  (registers every table the FKs name)
from config import config  # noqa: E402
from core.models.themes import (  # noqa: E402
    ThemeChoice, ThemeInstallation, ThemeOptIn, ThemePackage, ThemeRevocation, ThemeRevocationSync,
    ThemeWorkspaceSettings,
)
from services.themes import records, signed_upload, storage  # noqa: E402
from services.themes import signature_verify as sv  # noqa: E402
from services.themes.package import THEME_IMPORT_ROUTE, read_theme_package  # noqa: E402
from tests.test_prd261_theme_import import (  # noqa: E402  (its helpers; none of its tests)
    HELLO_ID, MEMBERS_DDL, SWITCH, WORKSPACES_DDL, _FakeS3, _hello, _local_ctx, _members, _rows, _saas, _upload,
)

VECTOR = json.loads((ORCH / "tests" / "fixtures" / "themes" / "signing-vector.json").read_text(encoding="utf-8"))
SIGNER, OTHER = VECTOR["ed25519"]["cases"][0], VECTOR["ed25519"]["cases"][1]
ADDED = "2026-10-21"
TABLES = [model.__table__ for model in (
    ThemePackage, ThemeInstallation, ThemeWorkspaceSettings, ThemeChoice, ThemeOptIn, ThemeRevocation,
    ThemeRevocationSync,
)]
INSTALLED = "/api/themes/installed"
PART_FRAMING = 1024  # a multipart part's boundary and headers, well over what a client sends
PACKAGE_FIELDS = {"id", "themeId", "version", "name", "digest", "ownerType", "reviewed", "scanVerdict", "scanReport"}


def _key_entry(case, status="active"):
    return {"keyId": case["keyId"], "spki": case["spki"], "status": status, "added": ADDED}


@pytest.fixture
def env(monkeypatch, tmp_path):
    import api.themes_import as themes_import
    import api.themes_install as themes_install

    engine = sa.create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    ThemePackage.metadata.create_all(engine, tables=TABLES)
    with engine.begin() as conn:
        conn.execute(sa.text(MEMBERS_DDL))
        conn.execute(sa.text(WORKSPACES_DDL))
    fake = _FakeS3()
    monkeypatch.setattr(storage, "get_s3_client", lambda: fake)
    monkeypatch.setattr(storage, "ensure_bucket", lambda bucket: None)
    for module in (themes_import, themes_install):
        monkeypatch.setattr(module, "resolve_user_pk", lambda db, ctx: None)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, SWITCH, "auto", raising=False)
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")  # the review stage never calls the API here

    def keys(*entries, text=None):
        path = tmp_path / "signing-keys.json"
        path.write_text(text if text is not None else json.dumps({"schema": 1, "keys": list(entries)}), encoding="utf-8")
        monkeypatch.setattr(sv, "KEYS_PATH", path)

    keys(_key_entry(SIGNER))
    yield SimpleNamespace(sessions=sessionmaker(bind=engine), s3=fake, keys=keys)
    engine.dispose()


def _client(env, ctx):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.themes_import import router as import_router
    from api.themes_install import router as install_router
    from core.auth.hybrid import get_request_context_hybrid
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(import_router)
    app.include_router(install_router)

    def _db():
        session = env.sessions()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_request_context_hybrid] = lambda: ctx
    return TestClient(app)


def _envelope(data: bytes, case=SIGNER, **changes):
    """The kit's envelope for these package bytes, signed by ``case``'s RFC seed, with ``changes``."""
    checked = read_theme_package(data)
    hashes = {path: row["sha256"] for path, row in checked.file_manifest.items()}
    payload = sv.payload_for(checked.manifest["id"], checked.manifest["version"], hashes)
    seed = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(case["seed"]))
    signature = base64.b64encode(seed.sign(sv.canonical_bytes(payload))).decode("ascii")
    envelope = {
        "form": signed_upload.ENVELOPE_FORM, "algorithm": "ed25519", "keyId": case["keyId"],
        "id": payload["id"], "version": payload["version"], "digest": payload["digest"], "signature": signature,
    }
    return {**envelope, **changes}


def _upload_signed(client, data: bytes, envelope=None, raw: bytes = None):
    body = raw if raw is not None else json.dumps(envelope).encode("utf-8")
    return client.post(THEME_IMPORT_ROUTE, files={
        "file": ("hello.autotheme", data, "application/zip"), "signature": ("hello.sig.json", body, "application/json"),
    })


def _refused(response, status, code, message=None):
    assert response.status_code == status, response.text
    (error,) = response.json()["detail"]["errors"]
    assert (error["code"], error["path"]) == (code, signed_upload.SIGNATURE_PART)
    assert message is None or error["message"] == message


def _nothing_stored(env):
    assert _rows(env, ThemePackage) == [] and _rows(env, ThemeInstallation) == [] and env.s3.objects == {}


def _installed(client):
    return {row["version"]: row for row in client.get(INSTALLED).json()["installs"]}


# --- a verified envelope --------------------------------------------------------------------


def test_a_verified_envelope_is_stored_and_the_installed_list_shows_its_key(env):
    data = _hello()
    envelope = _envelope(data)
    client = _client(env, _local_ctx(uuid4()))
    response = _upload_signed(client, data, envelope)

    assert response.status_code == 201, response.text
    (row,) = _rows(env, ThemePackage)
    assert row.signature == envelope["signature"] and row.reviewed is False and row.owner_type == "workspace"
    assert set(env.s3.objects) == {row.storage_prefix + path for path in row.file_manifest}
    assert _installed(client)["1.0.0"]["signature"] == {"state": "verified", "keyId": SIGNER["keyId"]}


def test_an_unsigned_install_shows_unsigned(env):
    client = _client(env, _local_ctx(uuid4()))
    assert _upload(client, _hello()).status_code == 201
    assert _installed(client)["1.0.0"]["signature"] == {"state": "unsigned", "keyId": None}


def test_the_same_bytes_again_with_their_envelope_gain_the_signature(env):
    data = _hello()
    client = _client(env, _local_ctx(uuid4()))
    assert _upload(client, data).status_code == 201
    again = _upload_signed(client, data, _envelope(data))

    assert again.status_code == 200 and again.json()["duplicate"] is True
    (row,) = _rows(env, ThemePackage)
    assert row.signature == _envelope(data)["signature"] and len(_rows(env, ThemeInstallation)) == 1
    assert _installed(client)["1.0.0"]["signature"]["state"] == "verified"


# --- refusals: nothing stored ---------------------------------------------------------------


def test_an_envelope_for_other_bytes_is_400_mismatch(env):
    other = _hello(name="Hello, other")
    response = _upload_signed(_client(env, _local_ctx(uuid4())), _hello(), _envelope(other))
    _refused(response, 400, signed_upload.SIGNATURE_MISMATCH,
             signed_upload.FOR_ANOTHER_PACKAGE.format(field="package digest"))
    _nothing_stored(env)


@pytest.mark.parametrize("field, value, called", [("id", "acme/other", "theme id"), ("version", "1.0.1", "version")])
def test_an_envelope_naming_another_id_or_version_is_400_mismatch(env, field, value, called):
    data = _hello()
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, _envelope(data, **{field: value}))
    _refused(response, 400, signed_upload.SIGNATURE_MISMATCH, signed_upload.FOR_ANOTHER_PACKAGE.format(field=called))
    assert value not in response.text  # nothing from the file is echoed
    _nothing_stored(env)


@pytest.mark.parametrize("status", ["retired", "revoked"])
def test_a_signature_by_an_inactive_key_is_400_bad(env, status):
    env.keys(_key_entry(SIGNER, status=status))
    data = _hello()
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, _envelope(data))
    _refused(response, 400, signed_upload.SIGNATURE_BAD, signed_upload.INACTIVE_KEY)
    _nothing_stored(env)


def test_a_signature_no_key_in_the_file_verifies_is_400_bad(env):
    data = _hello()
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, _envelope(data, case=OTHER))
    _refused(response, 400, signed_upload.SIGNATURE_BAD, signed_upload.NOT_VERIFIED)
    _nothing_stored(env)


def test_an_envelope_naming_another_key_than_the_signer_is_400_bad(env):
    env.keys(_key_entry(SIGNER), _key_entry(OTHER))
    data = _hello()
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, _envelope(data, keyId=OTHER["keyId"]))
    _refused(response, 400, signed_upload.SIGNATURE_BAD, signed_upload.OTHER_KEY)
    _nothing_stored(env)


def _malformed(envelope):
    without_key = {key: value for key, value in envelope.items() if key != "keyId"}
    return {
        "not json": b"{form", "not an object": b"[]", "empty": b"",
        "a missing key": json.dumps(without_key).encode(), "an extra key": json.dumps({**envelope, "note": "x"}).encode(),
        "another form": json.dumps({**envelope, "form": "automatos-theme-signature/v2"}).encode(),
        "another algorithm": json.dumps({**envelope, "algorithm": "rsa"}).encode(),
        "a bad keyId": json.dumps({**envelope, "keyId": "ed25519:XYZ"}).encode(),
        "a short signature": json.dumps({**envelope, "signature": "QUJD"}).encode(),
        "a signature with a newline": json.dumps({**envelope, "signature": envelope["signature"] + "\n"}).encode(),
    }


@pytest.mark.parametrize("name", sorted(_malformed({"keyId": "", "signature": ""})))
def test_a_malformed_envelope_is_400_bad(env, name):
    data = _hello()
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, raw=_malformed(_envelope(data))[name])
    _refused(response, 400, signed_upload.SIGNATURE_BAD)
    _nothing_stored(env)


def test_an_envelope_over_4_kb_is_413(env):
    data = _hello()
    padded = json.dumps(_envelope(data)).encode("utf-8") + b" " * signed_upload.ENVELOPE_MAX_BYTES
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, raw=padded)
    _refused(response, 413, signed_upload.SIGNATURE_TOO_LARGE, signed_upload.TOO_LARGE)
    _nothing_stored(env)


def test_a_keys_file_that_does_not_load_is_503_and_stores_nothing(env):
    from api.themes_import import SIGNATURES_UNAVAILABLE

    env.keys(text="{schema")
    data = _hello()
    response = _upload_signed(_client(env, _local_ctx(uuid4())), data, _envelope(data))
    assert response.status_code == 503 and response.json()["detail"] == SIGNATURES_UNAVAILABLE
    _nothing_stored(env)


# --- the reserved publisher -----------------------------------------------------------------


def test_a_non_admin_may_upload_under_automatos_only_with_a_verified_envelope(env, monkeypatch):
    workspace = uuid4()
    _members(env, workspace, 1)
    data = _hello(id="automatos/play")
    client = _saas(env, monkeypatch, workspace)

    refused = _upload(client, data)
    assert refused.status_code == 403 and '"automatos" publisher' in refused.json()["detail"]
    assert _rows(env, ThemePackage) == []
    response = _upload_signed(client, data, _envelope(data))
    assert response.status_code == 201, response.text
    (row,) = _rows(env, ThemePackage)
    assert (row.theme_id, row.signature) == ("automatos/play", _envelope(data)["signature"])


def test_the_reserved_rule_reads_the_verified_signature():
    assert records.reserved_publisher_refusal("automatos/play", None) is not None
    assert records.reserved_publisher_refusal("automatos/play", None, signed=True) is None


# --- QB-24, the unchanged path and the body limit ---------------------------------------------


def test_qb24_is_one_named_constant_set_to_the_plans_answer():
    # QB-24 is open (plan §5): the planned answer, a verified private upload counts as reviewed.
    assert records.SIGNED_UPLOAD_COUNTS_AS_REVIEWED is True


def test_without_the_part_the_upload_answers_what_it_did(env):
    from api.themes_import import import_theme, router

    (route,) = [r for r in router.routes if getattr(r, "path", None) == THEME_IMPORT_ROUTE]
    assert {param.name for param in route.dependant.body_params} == {"file", signed_upload.SIGNATURE_PART}
    parameters = inspect.signature(import_theme).parameters
    assert parameters[signed_upload.SIGNATURE_PART].default.default is None  # File(None): optional
    body = _upload(_client(env, _local_ctx(uuid4())), _hello()).json()
    assert set(body) == {"package", "install", "status", "duplicate"} and set(body["package"]) == PACKAGE_FIELDS
    (row,) = _rows(env, ThemePackage)
    assert row.signature is None and HELLO_ID == row.theme_id


def _constant(node):
    """The value of a constant arithmetic expression (``512 * 1024``) in main.py's source."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    assert isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)), ast.dump(node)
    left, right = _constant(node.left), _constant(node.right)
    return left + right if isinstance(node.op, ast.Add) else left * right


def test_the_import_body_limit_leaves_room_for_the_envelope():
    # test_prd261_b2_wire pins the live limit at the package cap plus this room.
    tree = ast.parse((ORCH / "main.py").read_text(encoding="utf-8"))
    (limits,) = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                 and any(getattr(target, "id", None) == "PATH_BODY_LIMITS" for target in node.targets)]
    (entry,) = [value for key, value in zip(limits.keys, limits.values) if getattr(key, "id", None) == "THEME_IMPORT_ROUTE"]
    assert isinstance(entry, ast.BinOp) and isinstance(entry.op, ast.Add)
    assert ast.unparse(entry.left) == "theme_cap_bytes('package')"
    assert _constant(entry.right) >= signed_upload.ENVELOPE_MAX_BYTES + PART_FRAMING


# --- the view the installed list shows ------------------------------------------------------


def _stored(data, signature):
    checked = read_theme_package(data)
    return ThemePackage(theme_id=checked.manifest["id"], version=checked.manifest["version"], digest=checked.digest,
                        file_manifest=checked.file_manifest, signature=signature)


def test_the_view_reads_a_retired_key_as_invalid_and_a_broken_keys_file_fails_closed(env):
    data = _hello()
    package = _stored(data, _envelope(data)["signature"])
    assert signed_upload.signature_view(package) == {"state": "verified", "keyId": SIGNER["keyId"]}
    env.keys(_key_entry(SIGNER, status="retired"))
    assert signed_upload.signature_view(package) == {"state": "invalid", "keyId": SIGNER["keyId"]}
    env.keys(text="{schema")
    assert signed_upload.signature_view(package) == {"state": "invalid", "keyId": None}
    assert signed_upload.signature_view(_stored(data, None)) == {"state": "unsigned", "keyId": None}
