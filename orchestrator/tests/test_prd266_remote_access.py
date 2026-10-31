"""PRD-266 US-005a (§4.3) — remote access: the boot check and the anonymous lane.

* ``services.mobile.remote_access.is_loopback``: 127.0.0.0/8, ``::1`` and ``localhost`` are
  loopback; ``0.0.0.0``, ``::`` and every other address are not.
* The boot decision for every mode × publish-address pair: ``off`` refuses a published
  install with a message naming the fix, ``paired`` starts, ``open`` starts and warns at
  every boot; an unknown mode refuses; saas is never checked. ``config.validate_auth_edition``
  enforces it (a lazy import; the function stays under 50 code lines).
* ``paired`` closes the anonymous lane in the ``core/auth/bearer_registry.py`` seam: a local
  request with no credential at all is ``401 {"code": "device_pairing_required"}`` and never
  reaches hybrid; the API key still works, ``POST /api/v1/mobile/pair`` is still reachable,
  and ``POST /pairing-codes`` needs a credential (the first code comes from the console).
* ``off`` and ``open`` keep today's anonymous lane, and saas is untouched by the mode.

DB-free: hybrid's session and workspace reads are stubbed, ``/pair`` runs on a session whose
every query finds nothing, and the rate limiter is a stub.
"""
from __future__ import annotations

import ast
import inspect
import logging
import os
import sys
import textwrap
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

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
from core.auth import bearer_registry, hybrid  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from services.mobile import remote_access  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000266a5")
TEST_API_KEY = "prd266-remote-access-test-key"
PAIRING_REQUIRED = "device_pairing_required"
LOOPBACK = "127.0.0.1"
EVERYWHERE = "0.0.0.0"
PAIR = "/api/v1/mobile/pair"
PAIRING_CODES = "/api/v1/mobile/pairing-codes"
MAX_CODE_LINES = 50


def _addresses(api: str, web: str) -> dict:
    return {"API_PUBLISH_ADDRESS": api, "WEB_PUBLISH_ADDRESS": web}


# --------------------------------------------------------------------------- is_loopback


@pytest.mark.parametrize("address", [
    "127.0.0.1", "127.1.2.3", "127.255.255.254", "::1", "[::1]", "localhost", "LocalHost", " 127.0.0.1 ",
])
def test_loopback_addresses(address):
    assert remote_access.is_loopback(address) is True


@pytest.mark.parametrize("address", [
    "0.0.0.0", "::", "[::]", "192.168.1.10", "10.0.0.5", "172.17.0.1", "203.0.113.9", "128.0.0.1",
    "fe80::1", "2001:db8::1", "example.com", "localhost.example.com", "", "   ", None,
])
def test_every_other_address_reaches_beyond_this_machine(address):
    assert remote_access.is_loopback(address) is False


# --------------------------------------------------------------------------- the boot decision

PAIRS = [
    (LOOPBACK, LOOPBACK, False),
    (EVERYWHERE, LOOPBACK, True),
    (LOOPBACK, EVERYWHERE, True),
    (EVERYWHERE, "::", True),
    ("localhost", "::1", False),
    ("192.168.1.10", "192.168.1.10", True),
]


@pytest.mark.parametrize("api,web,published", PAIRS)
def test_off_refuses_a_published_install_and_names_the_fix(api, web, published):
    decision = remote_access.boot_decision("local", "off", _addresses(api, web))

    assert decision.warning is None
    if not published:
        assert decision.refusal is None
        return
    for fix in ("LOCAL_REMOTE_ACCESS=paired", "LOCAL_REMOTE_ACCESS=open", "127.0.0.1", "mint_pairing_code"):
        assert fix in decision.refusal
    exposed = [name for name, value in _addresses(api, web).items() if not remote_access.is_loopback(value)]
    assert all(f"{name}=" in decision.refusal for name in exposed)


@pytest.mark.parametrize("api,web,published", PAIRS)
def test_paired_starts_whatever_the_addresses(api, web, published):
    assert remote_access.boot_decision("local", "paired", _addresses(api, web)) == remote_access.BootDecision()


@pytest.mark.parametrize("api,web,published", PAIRS)
def test_open_starts_and_warns_only_while_published(api, web, published):
    decision = remote_access.boot_decision("local", "open", _addresses(api, web))

    assert decision.refusal is None
    assert (decision.warning is not None) is published
    if published:
        assert "LOCAL_REMOTE_ACCESS=paired" in decision.warning


@pytest.mark.parametrize("mode", ["banana", "", "PAIRED ", "on", "true"])
def test_an_unknown_mode_refuses_even_on_loopback(mode):
    decision = remote_access.boot_decision("local", mode, _addresses(LOOPBACK, LOOPBACK))

    assert decision.refusal and "LOCAL_REMOTE_ACCESS" in decision.refusal


@pytest.mark.parametrize("mode", ["off", "paired", "open", "banana"])
@pytest.mark.parametrize("api,web,published", PAIRS)
def test_saas_is_never_checked(mode, api, web, published):
    assert remote_access.boot_decision("saas", mode, _addresses(api, web)) == remote_access.BootDecision()


def test_open_warns_at_every_boot(caplog):
    caplog.set_level(logging.WARNING, logger=remote_access.__name__)

    for _boot in range(2):
        remote_access.enforce_boot_decision("local", "open", _addresses(EVERYWHERE, LOOPBACK))

    warnings = [r for r in caplog.records if r.name == remote_access.__name__ and r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert all("API_PUBLISH_ADDRESS=0.0.0.0" in r.getMessage() for r in warnings)


def test_a_refused_boot_is_a_runtime_error():
    with pytest.raises(RuntimeError) as refused:
        remote_access.enforce_boot_decision("local", "off", _addresses(LOOPBACK, EVERYWHERE))

    assert isinstance(refused.value, remote_access.RemoteAccessRefused)
    assert "WEB_PUBLISH_ADDRESS=0.0.0.0" in str(refused.value)


# --------------------------------------------------------------------------- config.validate_auth_edition


@pytest.fixture
def boot(monkeypatch):
    """The live config as a local boot sees it, with nothing else that could refuse."""
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLI_RUNTIME_ENABLED", False)
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(WS))
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    monkeypatch.setattr(config, "API_PUBLISH_ADDRESS", LOOPBACK)
    monkeypatch.setattr(config, "WEB_PUBLISH_ADDRESS", LOOPBACK)
    return config


def test_the_publish_settings_default_to_loopback_and_are_named_in_the_surface():
    import json

    surface = json.loads((_ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))
    for name in ("API_PUBLISH_ADDRESS", "WEB_PUBLISH_ADDRESS", "LOCAL_REMOTE_ACCESS"):
        assert name in surface["settings"]
    assert config.API_PUBLISH_ADDRESS == LOOPBACK
    assert config.WEB_PUBLISH_ADDRESS == LOOPBACK
    assert config.LOCAL_REMOTE_ACCESS == "off"


def test_boot_refuses_off_with_a_published_api(boot, monkeypatch):
    monkeypatch.setattr(boot, "API_PUBLISH_ADDRESS", EVERYWHERE)

    with pytest.raises(RuntimeError) as refused:
        boot.validate_auth_edition()

    assert "LOCAL_REMOTE_ACCESS=paired" in str(refused.value)


def test_boot_starts_paired_with_both_published(boot, monkeypatch):
    monkeypatch.setattr(boot, "LOCAL_REMOTE_ACCESS", "paired")
    monkeypatch.setattr(boot, "API_PUBLISH_ADDRESS", EVERYWHERE)
    monkeypatch.setattr(boot, "WEB_PUBLISH_ADDRESS", EVERYWHERE)

    boot.validate_auth_edition()  # must not raise


def test_boot_starts_open_and_warns(boot, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, logger=remote_access.__name__)
    monkeypatch.setattr(boot, "LOCAL_REMOTE_ACCESS", "open")
    monkeypatch.setattr(boot, "WEB_PUBLISH_ADDRESS", EVERYWHERE)

    boot.validate_auth_edition()

    assert any("LOCAL_REMOTE_ACCESS=open" in r.getMessage() for r in caplog.records)


def test_boot_refuses_an_unknown_mode(boot, monkeypatch):
    monkeypatch.setattr(boot, "LOCAL_REMOTE_ACCESS", "banana")

    with pytest.raises(RuntimeError):
        boot.validate_auth_edition()


def test_a_loopback_install_boots_as_today(boot):
    boot.validate_auth_edition()  # off + loopback: must not raise


def test_saas_boot_never_reads_the_mode(boot, monkeypatch):
    monkeypatch.setattr(boot, "AUTH_EDITION", "saas")
    monkeypatch.setattr(boot, "CLERK_JWKS_URL", "https://clerk.prd266.test/.well-known/jwks.json")
    monkeypatch.setattr(boot, "CLERK_SECRET_KEY", "not-a-key")
    monkeypatch.setattr(boot, "LOCAL_REMOTE_ACCESS", "banana")
    monkeypatch.setattr(boot, "API_PUBLISH_ADDRESS", EVERYWHERE)

    boot.validate_auth_edition()  # must not raise


def _code_lines(function) -> int:
    """Code lines of a function as CI's shape gate counts them: no docstring, comment or blank."""
    source = textwrap.dedent(inspect.getsource(function))
    node = ast.parse(source).body[0]
    doc = node.body[0] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else None
    skip = set(range(doc.lineno, doc.end_lineno + 1)) if doc else set()
    lines = source.splitlines()
    return sum(1 for number, line in enumerate(lines, 1)
               if number not in skip and line.strip() and not line.strip().startswith("#"))


def test_validate_auth_edition_imports_remote_access_lazily_and_stays_short():
    tree = ast.parse((_ORCH / "config.py").read_text(encoding="utf-8"))
    top_level = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("services" in (getattr(n, "module", "") or "") for n in top_level)
    source = inspect.getsource(type(config).validate_auth_edition)
    assert "from services.mobile import remote_access" in source
    assert "remote_access.enforce_boot_decision(" in source
    assert _code_lines(type(config).validate_auth_edition) < MAX_CODE_LINES


# --------------------------------------------------------------------------- the anonymous lane


def _request(headers=None):
    return SimpleNamespace(method="GET", headers=dict(headers or {}), query_params={}, state=SimpleNamespace())


def _recording():
    calls = []

    async def wrapped(request):
        calls.append(request)
        return "today's lane"

    return bearer_registry.resolve_registered_bearers(wrapped), calls


@pytest.fixture
def paired(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")
    return config


NO_CREDENTIAL = [
    {},
    {"authorization": "Bearer "},  # an empty bearer: hybrid reads it as none
    {"authorization": "Basic x"},  # a scheme no lane reads
    {"x-api-key": "   "},  # a blank key
    {"x-workspace-id": str(WS)},  # naming a workspace is not a credential
]


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", NO_CREDENTIAL)
async def test_paired_refuses_a_request_with_no_credential_before_hybrid(paired, headers):
    decorated, calls = _recording()

    with pytest.raises(HTTPException) as refused:
        await decorated(_request(headers))

    assert refused.value.status_code == 401
    assert refused.value.detail["code"] == PAIRING_REQUIRED
    assert refused.value.detail["message"]
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {"x-api-key": TEST_API_KEY},
    {"X-Api-Key": TEST_API_KEY},
    {"authorization": f"ApiKey {TEST_API_KEY}"},
    {"authorization": "Bearer eyJhbGciOi.jwt.sig"},  # a Clerk JWT still goes to the Clerk lane
])
async def test_paired_passes_every_credential_to_its_lane_unchanged(paired, headers):
    decorated, calls = _recording()
    request = _request(headers)

    assert await decorated(request) == "today's lane"
    assert calls == [request]


@pytest.mark.asyncio
@pytest.mark.parametrize("edition,mode", [("local", "off"), ("local", "open"), ("saas", "paired"), ("saas", "off")])
async def test_off_open_and_saas_keep_todays_lane(monkeypatch, edition, mode):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)
    decorated, calls = _recording()
    request = _request()

    assert await decorated(request) == "today's lane"
    assert calls == [request]


def test_the_api_key_reader_moved_to_the_registry_and_hybrid_reads_it_there():
    assert hybrid._get_api_key is bearer_registry.api_key
    assert "def _get_api_key" not in inspect.getsource(hybrid)
    assert bearer_registry.api_key(_request({"x-api-key": " k "})) == "k"
    assert bearer_registry.api_key(_request({"authorization": "ApiKey k"})) == "k"
    assert bearer_registry.api_key(_request({"authorization": "Bearer k"})) is None
    assert bearer_registry.api_key(_request()) is None


# --------------------------------------------------------------------------- through FastAPI


class _NoRowSession:
    """A session whose every query finds nothing: a wrong pairing code, without a database."""

    def query(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def with_for_update(self, *args, **kwargs):
        return self

    def first(self):
        return None

    def commit(self):
        return None

    def rollback(self):
        return None

    def close(self):
        return None


@pytest.fixture
def app(monkeypatch):
    """``/whoami`` behind the real decorated dependency, beside the mobile router."""
    from api import mobile_devices
    from core.database.database import get_db

    monkeypatch.setattr(config, "ORCHESTRATOR_API_KEY", TEST_API_KEY)
    monkeypatch.setattr(config, "DEFAULT_WORKSPACE_ID", str(WS))
    monkeypatch.setattr(hybrid, "SessionLocal", lambda: _NoRowSession())
    monkeypatch.setattr(hybrid, "_workspace_exists", lambda db, workspace_id: True)
    monkeypatch.setattr(hybrid, "_assert_workspace_usable", lambda db, workspace_id, is_admin: None)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    monkeypatch.setattr(hybrid, "_local_operator_user_context",
                        lambda db: UserContext(id="operator@t.test", email="operator@t.test", system_role="super_admin"))
    monkeypatch.setattr(mobile_devices, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 3, 2, 600)))

    application = FastAPI()

    @application.get("/whoami")
    def whoami(ctx: RequestContext = Depends(hybrid.get_request_context_hybrid)):
        return {"auth_type": ctx.auth_type, "system_role": ctx.user.system_role}

    application.include_router(mobile_devices.router)
    application.dependency_overrides[get_db] = lambda: _NoRowSession()
    return TestClient(application)


def _pair_body() -> dict:
    return {"code": "ABCD-EFGH", "platform": "web", "install_id": "prd266-remote-access-browser"}


def test_paired_no_credential_is_401_device_pairing_required(app, paired):
    response = app.get("/whoami", headers={"X-Workspace-ID": str(WS)})

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == PAIRING_REQUIRED


def test_paired_an_api_key_still_works(app, paired):
    response = app.get("/whoami", headers={"X-API-Key": TEST_API_KEY})

    assert response.status_code == 200, response.text
    assert response.json()["auth_type"] == "api_key"


def test_paired_pair_is_still_reachable_without_a_credential(app, paired):
    response = app.post(PAIR, json=_pair_body())

    assert response.status_code == 401
    assert response.json() == {"detail": {"code": "invalid_pairing_code"}}


def test_paired_minting_a_code_needs_a_credential(app, paired):
    response = app.post(PAIRING_CODES)

    assert response.status_code == 401
    assert response.json()["detail"]["code"] == PAIRING_REQUIRED


@pytest.mark.parametrize("mode", ["off", "open"])
def test_off_and_open_keep_the_anonymous_operator(app, monkeypatch, mode):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "REQUIRE_AUTH", False)  # the local edition's posture
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)

    response = app.get("/whoami", headers={"X-Workspace-ID": str(WS)})

    assert response.status_code == 200, response.text
    assert response.json() == {"auth_type": "anonymous", "system_role": "super_admin"}
