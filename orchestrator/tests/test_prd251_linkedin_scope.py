"""PRD-251 S0.4a — the LinkedIn image workaround is workspace-scoped.

Before: ``_load_linkedin_credentials()`` took the FIRST active LinkedIn
credential in the whole database and cached it — with its access token — in
process globals, so every workspace posted images as one organisation.

Pinned here, against in-memory SQLite copies of ``credential_types`` /
``credentials`` (the real query runs), a stubbed decrypting store and a mocked
LinkedIn (``httpx.MockTransport`` records every request):

* two workspaces with two credentials each post with their OWN token and
  organisation URN — even though another workspace's credential is the first
  active row in the table;
* a workspace with no credential of its own gets a clear error and NO request
  reaches LinkedIn (or anywhere);
* workspace A's cached token is never used for workspace B;
* no process-wide single credential or token cache remains, and all three
  callers pass the workspace;
* P251-RVW-4: in one process with warm caches, a credential that another worker
  deactivates or deletes stops the next post before any LinkedIn request. A
  replaced one posts with the new token and organisation, and an access token
  edited in place is used from the next post on. An old credential's refreshed
  token is never paired with its replacement. The change is made through a
  separate connection, with no in-process hook: the loader re-reads the
  credential and calls clear_workspace_cache.
* P251-RVW-7: an image URL goes through the platform's one SSRF decision
  (``core/security/web_access.py``, its resolver seam stubbed here). A URL
  that resolves into a blocked range (127.1, 0x7f000001, localhost, 10.x,
  100.64.x) is refused before any request is sent, and the result carries the
  reason. A public URL is fetched from the address that was checked, whatever
  WEB_ACCESS is set to, and a redirect is not followed. ``_is_safe_url`` is
  gone.
"""
from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import socket
import sys
import time
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

import httpx  # noqa: E402
import sqlalchemy as sa  # noqa: E402
from sqlalchemy.dialects.postgresql import JSONB  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import core.composio.linkedin_image_workaround as lw  # noqa: E402
import core.credentials.service as credential_service  # noqa: E402
import core.database.database as database_mod  # noqa: E402
import core.security.web_access as web_access  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.models.credentials import Credential, CredentialType  # noqa: E402
from core.models.system_settings import SystemSetting  # noqa: E402

WS_A = uuid.uuid4()
WS_B = uuid.uuid4()
WS_NONE = uuid.uuid4()
WS_INACTIVE_ONLY = uuid.uuid4()
ORG_A = "urn:li:organization:1001"
ORG_B = "urn:li:organization:2002"

# credential id → what the store decrypts it to
SECRETS = {
    1: {"access_token": "token-B", "organization_urn": ORG_B, "client_id": "b", "client_secret": "b"},
    2: {"access_token": "token-A-retired", "organization_urn": "urn:li:organization:9999"},
    3: {"access_token": "token-A", "organization_urn": ORG_A, "client_id": "a", "client_secret": "a"},
    4: {"api_key": "sk-other-type"},
    5: {"access_token": "token-inactive", "organization_urn": "urn:li:organization:7777"},
}
# (id, workspace, credential type id, active). Workspace B's credential is the
# FIRST active LinkedIn row — exactly what the old loader picked for everyone.
CREDENTIAL_ROWS = [
    (1, WS_B, 1, True),
    (2, WS_A, 1, False),
    (3, WS_A, 1, True),
    (4, WS_B, 2, True),
    (5, WS_INACTIVE_ONLY, 1, False),
]
CDN = "cdn.example.com"
CDN_IP = "93.184.216.34"
IMAGE_POST = {"text": "Three weeks to Web Summit", "images": [f"https://{CDN}/countdown.png"]}

# What the system resolver answers for each image host. The resolver seam
# (web_access._getaddrinfo) is stubbed: no DNS in a unit test. The two numeric
# shorthands resolve to loopback, as the system resolver does.
RESOLVES_TO = {
    CDN: CDN_IP,
    "127.1": "127.0.0.1",
    "0x7f000001": "127.0.0.1",
    "localhost": "127.0.0.1",
    "assets.internal.test": "10.0.0.5",
    "edge.cgnat.test": "100.64.0.1",
}


def _resolve(host, port, proto=None):
    ip = RESOLVES_TO.get(host)
    if ip is None:
        raise socket.gaierror(socket.EAI_NONAME, f"unknown host {host}")
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _portable(col_type):
    if isinstance(col_type, JSONB):
        return sa.JSON()
    if isinstance(col_type, sa.Uuid) or type(col_type).__name__.upper() == "UUID":
        return sa.Uuid()
    return col_type


def _sqlite_copy(table, metadata):
    return sa.Table(
        table.name,
        metadata,
        *[sa.Column(c.name, _portable(c.type), primary_key=c.primary_key) for c in table.columns],
    )


class _FakeStore:
    """Stands in for the decrypting CredentialStore (no encryption keys in tests)."""

    def __init__(self, db):
        self.db = db

    def get_decrypted_credential(self, credential_id, **_kwargs):
        return dict(SECRETS[credential_id])


class _LinkedIn:
    """A mocked api.linkedin.com (plus the image CDN) that records every request.

    An image fetch connects to the address web_access checked, so the CDN is
    recognised by its Host header, not by the URL's host (the pinned IP)."""

    def __init__(self):
        self.requests = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.headers.get("host") == CDN:
            if request.url.path == "/moved.png":
                return httpx.Response(302, headers={"location": "http://assets.internal.test/secret.png"})
            return httpx.Response(200, content=b"\x89PNG-bytes", headers={"content-type": "image/png"})
        if "initializeUpload" in str(request.url):
            owner = json.loads(request.content)["initializeUploadRequest"]["owner"]
            return httpx.Response(
                200,
                json={"value": {"uploadUrl": "https://upload.linkedin.test/put", "image": f"urn:li:image:{owner}"}},
            )
        if request.method == "PUT":
            return httpx.Response(201)
        if request.url.path == "/rest/posts":
            return httpx.Response(201, headers={"x-restli-id": "urn:li:share:1"})
        return httpx.Response(404)

    def linkedin(self):
        return [r for r in self.requests if r.headers.get("host") != CDN]


_CACHES = (lw._creds_by_workspace, lw._token_by_workspace, lw._lock_by_workspace, lw._version_by_workspace)


@pytest.fixture
def env(monkeypatch):
    engine = sa.create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    copies = sa.MetaData()
    _sqlite_copy(CredentialType.__table__, copies)
    _sqlite_copy(Credential.__table__, copies)
    copies.create_all(engine)
    # Every real database has system_settings. The smoke route reads the Composio
    # deny list first (no row here → nothing denied); without the table that read
    # cannot complete, and the deny list fails closed (P251-RVW-1).
    SystemSetting.__table__.create(bind=engine)
    with engine.begin() as conn:
        for type_id, name in ((1, lw.CREDENTIAL_TYPE_NAME), (2, "openAiApi")):
            conn.execute(
                sa.text(
                    "INSERT INTO credential_types (id, name, display_name, schema_definition, is_active) "
                    "VALUES (:id, :name, :name, '[]', 1)"
                ),
                {"id": type_id, "name": name},
            )
        for cred_id, ws_id, type_id, active in CREDENTIAL_ROWS:
            conn.execute(
                sa.text(
                    "INSERT INTO credentials (id, name, workspace_id, credential_type_id, encrypted_data, is_active) "
                    "VALUES (:id, :name, :ws, :type_id, 'ciphertext', :active)"
                ),
                {"id": cred_id, "name": f"cred-{cred_id}", "ws": ws_id.hex, "type_id": type_id, "active": active},
            )

    monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    monkeypatch.setattr(credential_service, "CredentialStore", _FakeStore)
    monkeypatch.setattr(web_access, "_getaddrinfo", _resolve)
    monkeypatch.setattr(web_access.config, "WEB_ACCESS_DENY", ())
    for cache in _CACHES:
        cache.clear()

    linkedin = _LinkedIn()
    real_async_client = httpx.AsyncClient

    def mocked_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(linkedin.handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(lw.httpx, "AsyncClient", mocked_client)
    try:
        yield SimpleNamespace(engine=engine, linkedin=linkedin)
    finally:
        for cache in _CACHES:
            cache.clear()
        engine.dispose()


def _post_as(workspace_id, params=None):
    return asyncio.run(
        lw.execute_linkedin_image_post(
            params=dict(params or IMAGE_POST),
            workspace_id=workspace_id,
            entity_id="entity-ignored",
            composio_client=None,
        )
    )


def _bodies(requests, path_part):
    return [json.loads(r.content) for r in requests if path_part in str(r.url)]


# ---------------------------------------------------------------------------
# Each workspace posts with its own credential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "workspace_id, token, org",
    [(WS_A, "token-A", ORG_A), (WS_B, "token-B", ORG_B)],
)
def test_each_workspace_posts_with_its_own_credential_and_organisation(env, workspace_id, token, org):
    result = _post_as(workspace_id)

    assert result["success"] is True, result
    sent = env.linkedin.linkedin()
    assert sent, "nothing reached LinkedIn"
    assert {r.headers["authorization"] for r in sent} == {f"Bearer {token}"}
    assert [b["initializeUploadRequest"]["owner"] for b in _bodies(sent, "initializeUpload")] == [org]
    assert [b["author"] for b in _bodies(sent, "/rest/posts")] == [org]


def test_both_workspaces_in_one_process_never_cross(env):
    assert _post_as(WS_A)["success"] and _post_as(WS_B)["success"]

    sent = env.linkedin.linkedin()
    authors = [b["author"] for b in _bodies(sent, "/rest/posts")]
    assert authors == [ORG_A, ORG_B]
    by_token = {}
    for request in sent:
        by_token.setdefault(request.headers["authorization"], set()).add(
            json.loads(request.content).get("author") if request.url.path == "/rest/posts" else None
        )
    assert by_token["Bearer token-A"] - {None} == {ORG_A}
    assert by_token["Bearer token-B"] - {None} == {ORG_B}


def test_the_loader_filters_by_workspace_and_takes_the_active_credential(env):
    assert lw._load_linkedin_credentials(WS_A)["access_token"] == "token-A"  # not B's first row, not the retired one
    assert lw._load_linkedin_credentials(str(WS_B))["organization_urn"] == ORG_B  # str ids work too
    with pytest.raises(lw.LinkedInCredentialError):
        lw._load_linkedin_credentials(WS_INACTIVE_ONLY)


# ---------------------------------------------------------------------------
# No credential of its own → a clear error and no request anywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workspace_id", [WS_NONE, WS_INACTIVE_ONLY, None])
def test_a_workspace_without_a_credential_gets_an_error_and_no_request(env, workspace_id):
    result = _post_as(workspace_id)

    assert result["success"] is False
    assert "LinkedIn" in result["error"]
    assert env.linkedin.requests == []


def test_an_explicit_author_cannot_borrow_another_workspaces_token(env):
    result = _post_as(WS_NONE, {**IMAGE_POST, "author": ORG_A})
    assert result["success"] is False
    assert env.linkedin.requests == []


# ---------------------------------------------------------------------------
# Token caches are per workspace
# ---------------------------------------------------------------------------


def test_workspace_a_cached_token_is_never_used_for_workspace_b(env):
    lw._token_by_workspace[str(WS_A)] = ("cached-token-A", time.time() + 3600)

    async def _tokens():
        async with httpx.AsyncClient() as http:
            return await lw._get_access_token(http, WS_B), await lw._get_access_token(http, WS_A)

    token_b, token_a = asyncio.run(_tokens())
    assert token_b == "token-B"
    assert token_a == "cached-token-A"

    _post_as(WS_B)
    assert {r.headers["authorization"] for r in env.linkedin.linkedin()} == {"Bearer token-B"}
    assert lw._token_by_workspace[str(WS_A)][0] == "cached-token-A"
    assert lw._token_by_workspace[str(WS_B)][0] == "token-B"


def test_credential_cache_is_per_workspace(env):
    lw._load_linkedin_credentials(WS_A)
    lw._load_linkedin_credentials(WS_B)
    assert lw._creds_by_workspace[str(WS_A)]["organization_urn"] == ORG_A
    assert lw._creds_by_workspace[str(WS_B)]["organization_urn"] == ORG_B
    lw.clear_workspace_cache(WS_A)
    assert str(WS_A) not in lw._creds_by_workspace and str(WS_B) in lw._creds_by_workspace


# ---------------------------------------------------------------------------
# No process-wide single cache; all three callers pass the workspace
# ---------------------------------------------------------------------------


def test_no_module_level_single_credential_or_token_remains():
    tree = ast.parse(Path(lw.__file__).read_text(encoding="utf-8"))
    module_names = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        module_names.update(t.id for t in targets if isinstance(t, ast.Name))
    assert not module_names & {"_cached_creds", "_cached_token", "_cached_token_expires", "_token_lock"}
    for name in ("_creds_by_workspace", "_token_by_workspace", "_lock_by_workspace"):
        assert isinstance(getattr(lw, name), dict)
    assert "global " not in Path(lw.__file__).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "relpath, needle",
    [
        ("core/composio/tool_executor.py", r"execute_linkedin_image_post\(\s*params=params,\s*workspace_id=workspace_id,"),
        ("api/recipe_executor.py", r"execute_linkedin_image_post\(\s*params=tool_args,\s*workspace_id=workspace_id,"),
        ("api/composio.py", r"_load_linkedin_credentials\(ctx\.workspace_id\)"),
        ("api/composio.py", r"_get_access_token\(http, ctx\.workspace_id\)"),
    ],
)
def test_every_caller_passes_the_workspace(relpath, needle):
    assert re.search(needle, (_ORCH / relpath).read_text(encoding="utf-8")), f"{relpath}: {needle}"


def test_the_removal_checklist_names_all_three_callers():
    doc = lw.__doc__
    for caller in ("tool_executor.py", "recipe_executor.py", "api/composio.py"):
        assert caller in doc


def test_the_smoke_route_is_admin_gated_and_uses_the_callers_workspace(env):
    import api.composio as composio_api
    from core.auth.workspace_permission import PERMISSION_MARKER_ATTR

    route = next(r for r in composio_api.router.routes if r.path.endswith("/linkedin/test-upload-init"))
    markers = [getattr(d.call, PERMISSION_MARKER_ATTR, None) for d in route.dependant.dependencies]
    assert "workspace:manage" in markers

    ctx = RequestContext(
        workspace_id=WS_NONE, user=UserContext(id="admin", system_role="user"), auth_type="clerk"
    )
    result = asyncio.run(composio_api.test_linkedin_upload_init(ctx=ctx))
    assert result["ok"] is False and "LinkedIn" in result["error"]
    assert env.linkedin.requests == []


# ---------------------------------------------------------------------------
# P251-RVW-4: a changed credential never keeps posting from a warm process
# ---------------------------------------------------------------------------

ORG_A_NEW = "urn:li:organization:3003"
REPLACEMENT = {"access_token": "token-A-new", "organization_urn": ORG_A_NEW, "client_id": "a2", "client_secret": "a2"}


def _another_worker(env, *statements):
    """Change the credential rows the way another worker would: on its own
    connection and transaction, with no in-process hook."""
    with env.engine.begin() as conn:
        for statement in statements:
            conn.execute(sa.text(statement))


def _warm(env, workspace_id=WS_A):
    """One successful post, so the workspace's credential and token are cached."""
    assert _post_as(workspace_id)["success"] is True
    assert str(workspace_id) in lw._creds_by_workspace and str(workspace_id) in lw._token_by_workspace
    env.linkedin.requests.clear()


def _replace_workspace_a_credential(env, monkeypatch):
    monkeypatch.setitem(SECRETS, 6, dict(REPLACEMENT))
    _another_worker(
        env,
        "UPDATE credentials SET is_active = 0 WHERE id = 3",
        "INSERT INTO credentials (id, name, workspace_id, credential_type_id, encrypted_data, is_active) "
        f"VALUES (6, 'cred-6', '{WS_A.hex}', 1, 'ciphertext-6', 1)",
    )


@pytest.mark.parametrize("change", [
    "UPDATE credentials SET is_active = 0 WHERE id = 3",
    "DELETE FROM credentials WHERE id = 3",
])
def test_a_deactivated_or_deleted_credential_stops_the_next_post_with_no_linkedin_request(env, change):
    _warm(env)
    _another_worker(env, change)

    result = _post_as(WS_A)

    assert result["success"] is False
    assert "no active LinkedIn Community Management credential" in result["error"]
    assert env.linkedin.requests == []
    assert str(WS_A) not in lw._creds_by_workspace and str(WS_A) not in lw._token_by_workspace


def test_a_replaced_credential_posts_with_its_own_token_and_organisation(env, monkeypatch):
    _warm(env)
    _replace_workspace_a_credential(env, monkeypatch)

    result = _post_as(WS_A)

    assert result["success"] is True, result
    sent = env.linkedin.linkedin()
    assert {r.headers["authorization"] for r in sent} == {"Bearer token-A-new"}
    assert [b["initializeUploadRequest"]["owner"] for b in _bodies(sent, "initializeUpload")] == [ORG_A_NEW]
    assert [b["author"] for b in _bodies(sent, "/rest/posts")] == [ORG_A_NEW]


def test_an_access_token_edited_in_place_is_used_by_the_next_post(env, monkeypatch):
    _warm(env)
    monkeypatch.setitem(SECRETS, 3, {**SECRETS[3], "access_token": "token-A-rotated"})
    # What CredentialStore.update_credential writes: re-encrypted data and a new updated_at.
    _another_worker(
        env, "UPDATE credentials SET encrypted_data = 'ciphertext-rotated', updated_at = CURRENT_TIMESTAMP WHERE id = 3",
    )

    result = _post_as(WS_A)

    assert result["success"] is True, result
    sent = env.linkedin.linkedin()
    assert {r.headers["authorization"] for r in sent} == {"Bearer token-A-rotated"}
    assert [b["author"] for b in _bodies(sent, "/rest/posts")] == [ORG_A]


def test_an_old_credentials_refreshed_token_is_never_paired_with_its_replacement(env, monkeypatch):
    _warm(env)
    # What a token refresh leaves behind: the refreshed token cached and the credential copy dropped.
    lw._token_by_workspace[str(WS_A)] = ("token-A-refreshed", time.time() + 3600)
    lw._creds_by_workspace.pop(str(WS_A))
    _replace_workspace_a_credential(env, monkeypatch)

    result = _post_as(WS_A)

    assert result["success"] is True, result
    sent = env.linkedin.linkedin()
    assert {r.headers["authorization"] for r in sent} == {"Bearer token-A-new"}
    assert [b["author"] for b in _bodies(sent, "/rest/posts")] == [ORG_A_NEW]


def test_an_unchanged_credential_keeps_its_cached_token(env):
    """Only a changed credential drops the cache, not every post."""
    _warm(env)
    lw._token_by_workspace[str(WS_A)] = ("token-A-cached", time.time() + 3600)

    assert _post_as(WS_A)["success"] is True

    assert {r.headers["authorization"] for r in env.linkedin.linkedin()} == {"Bearer token-A-cached"}


def test_the_loader_calls_clear_workspace_cache():
    tree = ast.parse(Path(lw.__file__).read_text(encoding="utf-8"))
    loader = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_load_linkedin_credentials"
    )
    called = {
        node.func.id for node in ast.walk(loader)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "clear_workspace_cache" in called


# ---------------------------------------------------------------------------
# P251-RVW-7: an image URL goes through the one SSRF decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url, host",
    [
        ("http://127.1/", "127.1"),
        ("http://0x7f000001/", "0x7f000001"),
        ("http://localhost/", "localhost"),
        ("http://assets.internal.test/countdown.png", "assets.internal.test"),  # → 10.0.0.5
        ("http://edge.cgnat.test/countdown.png", "edge.cgnat.test"),  # → 100.64.0.1
    ],
)
def test_an_image_url_into_a_blocked_range_is_refused_before_any_request(env, url, host):
    result = _post_as(WS_A, {**IMAGE_POST, "images": [url]})

    assert result["success"] is False
    assert f"'{host}' is not reachable from agents: resolves into a blocked range" in result["error"]
    assert env.linkedin.requests == []


def test_a_refused_url_beside_a_public_one_is_reported_and_never_fetched(env):
    params = {**IMAGE_POST, "images": ["http://assets.internal.test/secret.png", *IMAGE_POST["images"]]}

    result = _post_as(WS_A, params)

    assert result["success"] is True, result
    assert (result["data"]["images_uploaded"], result["data"]["images_failed"]) == (1, 1)
    failure = result["data"]["failures"][0]
    assert failure["image"] == "image[0]"
    assert "'assets.internal.test' is not reachable from agents" in failure["reason"]
    assert all(r.headers.get("host") != "assets.internal.test" for r in env.linkedin.requests)
    assert all(r.url.host != RESOLVES_TO["assets.internal.test"] for r in env.linkedin.requests)


def test_a_public_image_url_is_fetched_from_the_address_that_was_checked(env):
    assert _post_as(WS_A)["success"] is True

    fetched = [r for r in env.linkedin.requests if r.headers.get("host") == CDN]
    assert len(fetched) == 1
    assert fetched[0].url.host == CDN_IP
    assert fetched[0].extensions["sni_hostname"] == CDN


@pytest.mark.parametrize("web_access_on", [True, False])
def test_a_public_image_url_posts_whatever_web_access_is_set_to(env, monkeypatch, web_access_on):
    monkeypatch.setattr(web_access.config, "WEB_ACCESS", web_access_on)

    result = _post_as(WS_A)

    assert result["success"] is True, result
    assert [b["author"] for b in _bodies(env.linkedin.linkedin(), "/rest/posts")] == [ORG_A]


def test_a_redirect_from_an_image_url_is_not_followed(env):
    result = _post_as(WS_A, {**IMAGE_POST, "images": [f"https://{CDN}/moved.png"]})

    assert result["success"] is False
    assert "HTTP 302" in result["error"]
    assert [r.url.path for r in env.linkedin.requests] == ["/moved.png"]


def test_the_url_fetch_uses_the_one_ssrf_decision_and_no_copy_of_it():
    source = Path(lw.__file__).read_text(encoding="utf-8")
    assert not hasattr(lw, "_is_safe_url")
    assert "ip_address" not in source and "is_global" not in source
    assert lw.resolve_outbound_async is web_access.resolve_outbound_async
    assert lw.build_pinned_request is web_access.build_pinned_request
    assert re.search(r"resolve_outbound_async\(\s*url,\s*enforce_switch=False\s*\)", source)
    assert re.search(r"build_pinned_request\(\s*http,\s*\"GET\",\s*url,\s*target\s*\)", source)
