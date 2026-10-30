"""PRD-266 US-001 (§4.1) — the discovery document: ``GET /.well-known/automatos.json``.

* Both editions answer ``automatos-discovery/v1`` with every §4.1 section and nothing else.
* saas: ``auth.modes ["clerk"]``, the publishable key and the Clerk frontend API decoded from
  it (``pk_test_`` / ``pk_live_`` + base64 of ``"<host>$"``); a key that is not Clerk's is a
  500 naming the setting, never the value. ``pairing.enabled`` is false.
* local: ``auth.modes ["device"]``, ``clerk`` null, ``pairing.enabled`` only in paired mode.
  No ``"oidc"`` mode in either edition (§4.11).
* ``push.expo`` (W2 US-010c, K13): true on saas; on a local install the operator's stored
  answer, none here (``test_prd266_push_consent.py`` stores one).
* K15: a local install still on the hosted ``PUBLIC_API_HOST`` answers with the request's own
  base URL, never ``api.automatos.app``; ``web_url`` is ``FRONTEND_URL``.
* Reachable with no credential while paired mode has closed the anonymous lane; a bearer or a
  workspace header sent along changes nothing.
* No secret: every config-surface setting named like a secret is given a distinct value and
  none of them appears in the body (the publishable key is the one exception).
* ``capabilities`` is ``CAPABILITY_ROUTES``'s names once every one that needs setup has it
  (``tts`` needs ``VOICE_TTS_URL``: ``test_voice_tts_speech.py``; ``suggestion_push`` needs
  ``SUGGESTION_PUSH_ENABLED``: ``test_prd268w2_suggestion_push_rules.py``), and the live app (a clean subprocess,
  the ``test_prd261_c1_wire.py`` probe) serves every mapped route, and serves the discovery
  route once, from ``api.well_known``, with no request-context dependency.
* The route is a plain ``def``, registered by ``RouterSpec``, listed in the route manifest, and
  its module names the public-route exception.

DB-free: the route reads config, and the local Expo answer through a stand-in holding none.
"""
from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

# Blessed preamble: dummy POSTGRES_* satisfies the config import chain when the suite's
# own environment does not set it (CI does, and these defaults never override it).
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

ORCH = Path(__file__).resolve().parents[1]
if str(ORCH) not in sys.path:
    sys.path.insert(0, str(ORCH))

from api import well_known  # noqa: E402
from config import HOSTED_PUBLIC_API_HOST, config  # noqa: E402
from core.auth import hybrid  # noqa: E402
from core.auth.dependencies import RequestContext  # noqa: E402
from services.mobile import discovery  # noqa: E402
from services.push import consent  # noqa: E402

DISCOVERY = "/.well-known/automatos.json"
SCHEMA = "automatos-discovery/v1"
SECTIONS = {"schema", "name", "edition", "api_url", "web_url", "auth", "pairing", "push", "capabilities"}
FRONTEND = "https://studio.prd266.test"
CLERK_HOST = "clerk.prd266.test"
PUBLISHABLE_KEY = "pk_live_" + base64.b64encode(f"{CLERK_HOST}$".encode()).decode()
# The names §4.1 gives capabilities; a build lists the ones it implements.
KNOWN_CAPABILITIES = {"push", "spoken_mode", "voice_live", "mobile_theme", "report_content", "digest", "digest_now",
                      "tts", "brief", "suggestions", "readiness", "suggestion_push"}
SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PRIVATE", "DSN")
PUBLIC_KEY_SETTING = "CLERK_PUBLISHABLE_KEY"


def _clerk_key(prefix: str, decoded: str, *, padded: bool = True) -> str:
    encoded = base64.b64encode(decoded.encode()).decode()
    return prefix + (encoded if padded else encoded.rstrip("="))


@pytest.fixture
def client() -> TestClient:
    application = FastAPI()
    application.include_router(well_known.router)
    return TestClient(application)


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "CLERK_PUBLISHABLE_KEY", PUBLISHABLE_KEY)
    monkeypatch.setattr(config, "FRONTEND_URL", FRONTEND)
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    return config


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "CLERK_PUBLISHABLE_KEY", None)
    monkeypatch.setattr(config, "FRONTEND_URL", "http://localhost:3000")
    monkeypatch.setattr(config, "PUBLIC_API_HOST", HOSTED_PUBLIC_API_HOST)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    # The operator has not answered "Send notifications through Expo?" (no settings row).
    monkeypatch.setattr(consent, "get_system_setting", lambda category, key, default=None: default)
    return config


def _document(client: TestClient, **kwargs) -> dict:
    response = client.get(DISCOVERY, **kwargs)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- the two editions


def test_saas_is_schema_v1_and_signs_in_with_clerk(client, saas):
    body = _document(client)

    assert set(body) == SECTIONS
    assert body["schema"] == SCHEMA and body["name"] == "Automatos" and body["edition"] == "saas"
    assert body["api_url"] == f"https://{HOSTED_PUBLIC_API_HOST}"
    assert body["web_url"] == FRONTEND
    assert body["auth"] == {"modes": ["clerk"],
                            "clerk": {"publishable_key": PUBLISHABLE_KEY, "frontend_api": CLERK_HOST}}
    assert body["pairing"] == {"enabled": False}
    assert body["push"] == {"expo": True}


def test_saas_never_offers_pairing_whatever_the_remote_access_mode(client, saas, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")

    assert _document(client)["pairing"] == {"enabled": False}


@pytest.mark.parametrize("mode, enabled", [("off", False), ("open", False), ("paired", True)])
def test_local_signs_in_with_a_device_and_pairs_only_in_paired_mode(client, local, monkeypatch, mode, enabled):
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)

    body = _document(client)

    assert set(body) == SECTIONS
    assert body["schema"] == SCHEMA and body["edition"] == "local"
    assert body["auth"] == {"modes": ["device"], "clerk": None}
    assert body["pairing"] == {"enabled": enabled}
    assert body["push"] == {"expo": False}


def test_a_local_install_never_advertises_a_clerk_key(client, local, monkeypatch):
    monkeypatch.setattr(config, "CLERK_PUBLISHABLE_KEY", PUBLISHABLE_KEY)

    response = client.get(DISCOVERY)

    assert response.json()["auth"]["clerk"] is None
    assert PUBLISHABLE_KEY not in response.text and CLERK_HOST not in response.text


@pytest.mark.parametrize("edition", ["saas", "local"])
@pytest.mark.parametrize("mode", ["off", "open", "paired"])
def test_no_oidc_mode_is_ever_emitted(client, saas, monkeypatch, edition, mode):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)

    body = _document(client)

    assert "oidc" not in body["auth"]["modes"] and "oidc" not in json.dumps(body)


# --------------------------------------------------------------------------- the Clerk frontend API


@pytest.mark.parametrize("prefix", ["pk_test_", "pk_live_"])
@pytest.mark.parametrize("host", ["clerk.automatos.app", "a.clerk.accounts.dev", "ab.example", "abc.example"])
@pytest.mark.parametrize("padded", [True, False])
def test_the_frontend_api_is_decoded_from_the_publishable_key(prefix, host, padded):
    assert discovery.clerk_frontend_api(_clerk_key(prefix, f"{host}$", padded=padded)) == host


@pytest.mark.parametrize("key", [
    None,
    "",
    "   ",
    _clerk_key("sk_live_", f"{CLERK_HOST}$"),  # a secret key's prefix
    _clerk_key("pk_live_", CLERK_HOST),  # no terminating "$"
    _clerk_key("pk_live_", "$"),  # no host
    "pk_live_!!not-base64!!",
    "pk_live_" + base64.b64encode(b"\xff\xfe$").decode(),  # not text
])
def test_anything_but_a_clerk_publishable_key_decodes_to_nothing(key):
    assert discovery.clerk_frontend_api(key) is None


@pytest.mark.parametrize("key", [None, "", "sk_live_not-a-publishable-key", _clerk_key("pk_live_", "no-dollar")])
def test_saas_without_a_clerk_publishable_key_is_a_500_naming_the_setting(client, saas, monkeypatch, caplog, key):
    monkeypatch.setattr(config, "CLERK_PUBLISHABLE_KEY", key)

    with caplog.at_level(logging.ERROR, logger=well_known.__name__):
        response = client.get(DISCOVERY)

    assert response.status_code == 500
    assert "NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY" in response.json()["detail"]
    if key:
        assert key not in response.text and key not in caplog.text
    assert any(r.levelno == logging.ERROR and r.name == well_known.__name__ for r in caplog.records)


# --------------------------------------------------------------------------- the two addresses (K15)


def test_a_local_install_on_the_hosted_default_answers_with_its_own_base_url(client, local):
    body = _document(client)

    assert body["api_url"] == "http://testserver"
    assert body["web_url"] == "http://localhost:3000"
    assert HOSTED_PUBLIC_API_HOST not in json.dumps(body)


def test_a_local_install_that_named_its_host_answers_with_it(client, local, monkeypatch):
    monkeypatch.setattr(config, "PUBLIC_API_HOST", "automatos.example.org")

    assert _document(client)["api_url"] == "https://automatos.example.org"


def test_a_misconfigured_frontend_url_is_a_500_naming_the_setting(client, saas, monkeypatch):
    monkeypatch.setattr(config, "FRONTEND_URL", "automatos://open")

    response = client.get(DISCOVERY)

    assert response.status_code == 500
    assert "FRONTEND_URL" in response.json()["detail"]
    assert "automatos://open" not in response.text


# --------------------------------------------------------------------------- public, even in paired mode


def test_paired_mode_closes_the_anonymous_lane_but_not_discovery(client, local, monkeypatch):
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "paired")

    @client.app.get("/whoami")
    def whoami(ctx: RequestContext = Depends(hybrid.get_request_context_hybrid)):
        return {"auth_type": ctx.auth_type}

    refused = client.get("/whoami")
    answered = client.get(DISCOVERY)

    assert refused.status_code == 401
    assert refused.json()["detail"]["code"] == "device_pairing_required"
    assert answered.status_code == 200 and answered.json()["pairing"] == {"enabled": True}


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer amd_not-a-real-device-token"},
    {"Authorization": "Bearer not-a-jwt"},
    {"X-Workspace-ID": str(uuid.uuid4())},
    {"X-API-Key": "not-the-key"},
])
def test_a_credential_sent_along_changes_nothing(client, saas, headers):
    assert _document(client, headers=headers) == _document(client)


# --------------------------------------------------------------------------- no secret


def _secret_settings() -> list:
    names = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    return [name for name in names if any(marker in name for marker in SECRET_MARKERS) and name != PUBLIC_KEY_SETTING]


@pytest.fixture
def planted_secrets(monkeypatch) -> dict:
    """A distinct value on every config-surface setting named like a secret."""
    planted = {}
    for name in _secret_settings():
        value = f"prd266-planted-{name.lower()}-{uuid.uuid4().hex}"
        if isinstance(inspect.getattr_static(config, name, None), property):
            monkeypatch.setattr(type(config), name, property(lambda _self, planted_value=value: planted_value))
        else:
            monkeypatch.setattr(config, name, value, raising=False)
        planted[name] = value
    return planted


@pytest.mark.parametrize("edition, mode", [("saas", "off"), ("local", "off"), ("local", "paired")])
def test_no_secret_setting_appears_in_the_body(client, saas, planted_secrets, monkeypatch, edition, mode):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", mode)

    response = client.get(DISCOVERY)

    assert response.status_code == 200, response.text
    assert {"CLERK_SECRET_KEY", "ORCHESTRATOR_API_KEY", "POSTGRES_PASSWORD", "GITHUB_APP_PRIVATE_KEY"} <= set(
        planted_secrets)
    leaked = sorted(name for name, value in planted_secrets.items() if value in response.text)
    assert leaked == []
    assert (PUBLISHABLE_KEY in response.text) is (edition == "saas")


# --------------------------------------------------------------------------- capabilities


@pytest.mark.parametrize("edition", ["saas", "local"])
def test_capabilities_are_the_mapped_names_and_only_known_ones(client, saas, monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    # Every capability that needs setup has it: Auto's server voice is configured, and the
    # suggestion push is switched on.
    monkeypatch.setattr(config, "VOICE_TTS_URL", "http://voice-tts:8100")
    monkeypatch.setattr(config, "SUGGESTION_PUSH_ENABLED", True)

    capabilities = _document(client)["capabilities"]

    assert capabilities == sorted(discovery.CAPABILITY_ROUTES)
    assert set(capabilities) <= KNOWN_CAPABILITIES
    assert "push" in capabilities  # W2 US-009: phones register for push through the Expo sender


@pytest.mark.parametrize("edition", ["saas", "local"])
def test_mobile_theme_is_listed_on_the_mobile_theme_route(client, saas, monkeypatch, edition):
    """PRD-261 M-04: the app offers installed themes on the phone only when this build lists them."""
    monkeypatch.setattr(config, "AUTH_EDITION", edition)

    assert "mobile_theme" in _document(client)["capabilities"]
    assert discovery.CAPABILITY_ROUTES["mobile_theme"] == ("GET", "/api/themes/mobile/{install_id}")


_PROBE = """
import json
from fastapi.routing import APIRoute, iter_route_contexts
from core.auth.hybrid import get_request_context_hybrid
from services.mobile.discovery import CAPABILITY_ROUTES
import main

def calls(dependant):
    out = []
    for sub in dependant.dependencies:
        out += [sub.call] + calls(sub)
    return out

# dump_routes' keys: one per HTTP method; a method-less route (a WebSocket) under "".
served = {}
for route in iter_route_contexts(main.app.routes):
    path = getattr(route, "path", None)
    endpoint = getattr(route, "endpoint", None)
    if not path or endpoint is None:
        continue
    handler = getattr(endpoint, "__module__", "?") + "." + getattr(endpoint, "__name__", "?")
    methods = set(getattr(route, "methods", None) or {""}) - {"HEAD", "OPTIONS"}
    for method in sorted(methods):
        entry = {"handler": handler, "context": None}
        if isinstance(route.original_route, APIRoute):
            entry["context"] = get_request_context_hybrid in calls(route.dependant)
        served.setdefault(method + " " + path, []).append(entry)
print(json.dumps({"served": served,
                  "capabilities": {name: list(key) for name, key in CAPABILITY_ROUTES.items()}}))
"""


@pytest.fixture(scope="module")
def live_app() -> dict:
    env = dict(os.environ)
    env.update({
        "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test", "POSTGRES_HOST": "127.0.0.1",
        "POSTGRES_PORT": "59432", "POSTGRES_DB": "test",
        "DATABASE_URL": "postgresql://test:test@127.0.0.1:59432/test",
    })
    proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=str(ORCH), env=env,
                          capture_output=True, text=True, timeout=240)
    assert proc.returncode == 0, f"probe failed (rc={proc.returncode})\nSTDERR:\n{proc.stderr[-3000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_the_live_app_serves_discovery_once_from_well_known_with_no_context(live_app):
    assert live_app["served"][f"GET {DISCOVERY}"] == [
        {"handler": "api.well_known.discovery_document", "context": False}]


def test_the_live_app_serves_every_capabilitys_route(live_app):
    assert live_app["capabilities"] == {name: list(key) for name, key in discovery.CAPABILITY_ROUTES.items()}
    missing = sorted(f"{name}: {method} {path}" for name, (method, path) in live_app["capabilities"].items()
                     if f"{method} {path}" not in live_app["served"])
    assert missing == []


# --------------------------------------------------------------------------- the route


def test_the_route_is_a_plain_def_registered_and_listed():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert not inspect.iscoroutinefunction(well_known.discovery_document)
    assert list(inspect.signature(well_known.discovery_document).parameters) == ["request"]
    assert RouterSpec("api.well_known") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    assert {"method": "GET", "path": DISCOVERY} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


def test_the_module_names_its_public_route_exception():
    doc = well_known.__doc__

    assert "PUBLIC" in doc and "AGENTS.md exception 1" in doc and "#1253" in doc
    assert discovery.DISCOVERY_SCHEMA == SCHEMA
