"""PRD-277 US-402 and US-408, without a database: the two routes, the registry, the boot and the log.

* **Routes.** ``GET /api/v1/auth/oidc/start`` and ``/callback`` are the router's only routes, are
  in ``route-manifest.json`` and in ``MANIFEST_ROUTERS``, and answer 404 before anything else in
  ``local`` and ``saas``. They are public (no request-context dependency), plain ``def``, and
  limited per client IP before any database work: over the limit is 429 with ``Retry-After``.
  The callback path is the exact address sign-in registers at the provider.
  They are GET routes, so the boundary sweep (which classifies changing methods only) has no
  entry for them; this file pins what it would: public by design, rate limited.
* **US-408.** Registration closes at start-up: after it, registering is ``RegistrationClosed``;
  registering one name twice is ``AlreadyRegistered``; asking for a method closes it too.
  ``safe_return_path`` keeps only a path on the install.
* **Boot.** ``finish_sign_in_start_up`` closes registration in every edition, adds no stage in
  ``local`` and ``saas``, and in enterprise reports the provider check as the ``oidc_provider``
  stage, failed or not, without stopping the API.
* **The access log** never holds a sign-in callback's query (its code and state).
* ``hybrid.py`` gains no branch for any of it.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from fastapi import FastAPI  # noqa: E402
from fastapi.routing import APIRoute  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import auth_oidc  # noqa: E402
from config import config  # noqa: E402
from core.auth import oidc_http, oidc_provider, sign_in_flow, sign_in_start_up  # noqa: E402
from core.auth.oidc_sign_in import CALLBACK_PATH, OidcSignIn, callback_address  # noqa: E402
from core.auth.sign_in_methods import (  # noqa: E402
    AlreadyRegistered,
    RegistrationClosed,
    SignInRegistry,
    safe_return_path,
)
from core.database.database import get_db  # noqa: E402
from core.models.bootstrap import BootstrapReport, BootstrapStage  # noqa: E402
from core.monitoring.path_digests import DigestPathFilter  # noqa: E402
from core.security.rate_limiter import DEFAULT_LIMITS  # noqa: E402

ORCH = Path(__file__).resolve().parents[1]
START = ("GET", "/api/v1/auth/oidc/start")
CALLBACK = ("GET", "/api/v1/auth/oidc/callback")
INSTALL = "https://automatos.example.test"


def _routes() -> dict:
    return {(method, route.path): route for route in auth_oidc.router.routes if isinstance(route, APIRoute)
            for method in route.methods}


def _calls(dependant) -> list:
    found = [dependant.call]
    for sub in dependant.dependencies:
        found.extend(_calls(sub))
    return found


def _app(fail_db: bool = True) -> TestClient:
    app = FastAPI()
    app.include_router(auth_oidc.router)
    if fail_db:
        app.dependency_overrides[get_db] = lambda: pytest.fail("no database work may happen here")
    return TestClient(app, base_url=INSTALL)


# --------------------------------------------------------------------------- routes


def test_the_router_declares_exactly_start_and_callback_at_the_registered_address():
    assert set(_routes()) == {START, CALLBACK}
    assert CALLBACK[1] == CALLBACK_PATH


def test_the_routes_are_in_both_manifests():
    from router_manifest import MANIFEST_ROUTERS, RouterSpec

    assert RouterSpec("api.auth_oidc") in MANIFEST_ROUTERS
    manifest = json.loads((ORCH / "reports" / "route-manifest.json").read_text(encoding="utf-8"))
    listed = [(entry["method"], entry["path"]) for entry in manifest["routes"]]
    assert {START, CALLBACK} <= set(listed)
    assert manifest["route_count"] == len(listed) == len(set(listed))
    assert listed == sorted(listed, key=lambda key: (key[1], key[0]))


def test_the_routes_are_public_plain_def_and_limited_per_ip_before_the_database():
    from core.auth.hybrid import get_request_context_hybrid

    limits = {START: auth_oidc.limit_starts_per_ip, CALLBACK: auth_oidc.limit_callbacks_per_ip}
    for key, route in _routes().items():
        calls = _calls(route.dependant)
        assert get_request_context_hybrid not in calls, key
        assert not inspect.iscoroutinefunction(route.endpoint), key
        assert calls.index(limits[key]) < calls.index(get_db), key
        assert calls.index(auth_oidc._signs_in_with_oidc) < calls.index(limits[key]), key


def test_each_route_has_its_own_per_ip_bucket():
    assert DEFAULT_LIMITS["oidc_start_ip"] == (300, 600) and DEFAULT_LIMITS["oidc_callback_ip"] == (300, 600)
    assert auth_oidc._START_LIMIT.bucket_prefix != auth_oidc._CALLBACK_LIMIT.bucket_prefix


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_both_routes_are_404_before_anything_where_sign_in_is_not_oidc(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(auth_oidc, "_rate_store", SimpleNamespace(check=lambda *a, **k: pytest.fail("counted")))
    client = _app()

    for _method, path in (START, CALLBACK):
        answer = client.get(path, params={"code": "x", "state": "y"}, follow_redirects=False)
        assert (answer.status_code, answer.json()) == (404, {"detail": "Not Found"}), path


@pytest.mark.parametrize(("key", "operation"), [(START, "oidc_start_ip"), (CALLBACK, "oidc_callback_ip")])
def test_over_the_limit_is_429_with_retry_after_and_no_database_work(monkeypatch, key, operation):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    asked = []

    def refuse(bucket, limit, window, label):
        asked.append((bucket, limit, window))
        return False, limit, 0, 42

    monkeypatch.setattr(auth_oidc, "_rate_store", SimpleNamespace(check=refuse))
    answer = _app().get(key[1], follow_redirects=False)

    assert answer.status_code == 429 and answer.headers["retry-after"] == "42"
    assert "code" not in answer.text and [(limit, window) for _b, limit, window in asked] == [DEFAULT_LIMITS[operation]]


def test_a_refused_start_goes_to_the_sign_in_page_with_its_code_only(monkeypatch):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(auth_oidc, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 300, 299, 600)))

    def unavailable(db, return_to):
        raise sign_in_flow.SignInRefusal(sign_in_flow.UNAVAILABLE)

    monkeypatch.setattr(sign_in_flow, "start_sign_in", unavailable)
    app = FastAPI()
    app.include_router(auth_oidc.router)
    app.dependency_overrides[get_db] = lambda: None
    answer = TestClient(app, base_url=INSTALL).get(START[1], follow_redirects=False)

    assert answer.status_code == 303
    assert answer.headers["location"] == "/sign-in?error=identity_provider_unavailable"
    assert answer.headers["cache-control"] == "no-store" and answer.headers["referrer-policy"] == "no-referrer"
    assert auth_oidc.SIGN_IN_COOKIE not in answer.headers.get("set-cookie", "")


def test_the_redirect_address_is_the_install_origin_and_the_callback_path(monkeypatch):
    monkeypatch.setattr(config, "FRONTEND_URL", "HTTPS://Automatos.Example.Test:443/app/")
    assert callback_address() == "https://automatos.example.test" + CALLBACK_PATH


# --------------------------------------------------------------------------- US-408 registry


def test_registering_twice_or_after_start_up_is_an_error():
    registry = SignInRegistry({"oidc": OidcSignIn()})
    registry.register_hook("first", lambda db, signed_in: None)
    registry.register_hook("second", lambda db, signed_in: None)

    with pytest.raises(AlreadyRegistered):
        registry.register_method("oidc", OidcSignIn())
    with pytest.raises(AlreadyRegistered):
        registry.register_hook("first", lambda db, signed_in: None)
    assert [name for name, _hook in registry.hooks()] == ["first", "second"]
    registry.close()
    with pytest.raises(RegistrationClosed):
        registry.register_hook("late", lambda db, signed_in: None)
    with pytest.raises(RegistrationClosed):
        registry.register_method("late", OidcSignIn())


def test_the_first_sign_in_closes_registration_too():
    registry = SignInRegistry({"oidc": OidcSignIn()})
    assert isinstance(registry.method("oidc"), OidcSignIn) and registry.closed
    with pytest.raises(RegistrationClosed):
        registry.register_hook("late", lambda db, signed_in: None)


@pytest.mark.parametrize(("value", "kept"), [
    ("/projects?tab=1#part", "/projects?tab=1"), ("/", "/"), ("//evil.example.test", "/"),
    ("/\\evil.example.test", "/"), ("https://evil.example.test/x", "/"), ("javascript:alert(1)", "/"),
    ("projects", "/"), ("/a b", "/"), ("/a\nb", "/"), ("/" + "a" * 1100, "/"), (None, "/"), (7, "/"),
    ("/%2F%2Fevil.example.test", "/%2F%2Fevil.example.test"),
])
def test_only_a_path_on_the_install_is_ever_a_return_address(value, kept):
    assert safe_return_path(value) == kept


# --------------------------------------------------------------------------- boot


def _boot(monkeypatch, edition, check):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    registry = SignInRegistry({"oidc": OidcSignIn()})
    monkeypatch.setattr(sign_in_start_up, "REGISTRY", registry)
    monkeypatch.setattr(oidc_provider, "check_provider", check)
    report = BootstrapReport()
    asyncio.run(sign_in_start_up.finish_sign_in_start_up(report))
    return report, registry


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_the_boot_adds_no_stage_outside_enterprise_and_closes_registration(monkeypatch, edition):
    report, registry = _boot(monkeypatch, edition, lambda: pytest.fail("the provider was read"))
    assert report.stages == [] and registry.closed


def test_in_enterprise_an_unreachable_provider_is_a_failed_stage_and_the_boot_goes_on(monkeypatch):
    def unreachable():
        raise oidc_http.ProviderUnreachable(oidc_http.UNREACHABLE_MESSAGE)

    report, registry = _boot(monkeypatch, "enterprise", unreachable)

    (stage,) = report.stages
    assert (stage.stage, stage.status) == (BootstrapStage.OIDC_PROVIDER, "failed") and registry.closed
    assert stage.error == oidc_http.UNREACHABLE_MESSAGE


def test_in_enterprise_a_readable_provider_is_a_successful_stage(monkeypatch):
    report, _registry = _boot(monkeypatch, "enterprise", lambda: None)
    assert [(stage.stage, stage.status) for stage in report.stages] == [(BootstrapStage.OIDC_PROVIDER, "success")]


def test_the_background_boot_runs_the_sign_in_start_up():
    import main

    assert "await finish_sign_in_start_up(report)" in inspect.getsource(main._boot_in_background)


# --------------------------------------------------------------------------- the access log and hybrid


def test_the_access_log_never_holds_a_sign_in_callbacks_query():
    code, state = "prd277-" + "w4-code-marker", "prd277-" + "w4-state-marker"
    record = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                               ("203.0.113.7:5000", "GET", f"{CALLBACK_PATH}?code={code}&state={state}", "1.1", 303),
                               None)
    other = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                              ("203.0.113.7:5000", "GET", "/api/v1/things?page=2", "1.1", 200), None)

    assert DigestPathFilter().filter(record) is True and DigestPathFilter().filter(other) is True

    line = record.getMessage()
    assert code not in line and state not in line and f"{CALLBACK_PATH}?{{redacted}}" in line
    assert "/api/v1/things?page=2" in other.getMessage()


def test_hybrid_gains_no_branch_for_sign_in():
    source = (ORCH / "core" / "auth" / "hybrid.py").read_text(encoding="utf-8")
    assert "oidc" not in source.lower() and "sign_in_flow" not in source
