"""PRD-277 US-603 without a database: what a phone's start may ask for, where its answer goes, and the exchange route.

* **The start's parameters** (``core.auth.phone_sign_in``): none named is a browser's start; any named means all
  five, once each: an ``app_link`` KEY of the fixed ``APP_LINKS`` (never an address), an S256 challenge of 43
  base64url characters (``plain`` and a missing method refused), the device's install id and ``ios`` or
  ``android``. A malformed one is 400 ``invalid_phone_start`` before the provider is asked or anything is stored,
  and so is a phone's start with ``step_up``. A refusal of a phone's start goes to its app link.
* **The answer** is the fixed app link with ``code`` or ``error`` and nothing else.
* **The exchange route** (``POST /api/v1/mobile/sign-in/exchange``) is public (no request context), plain
  ``def``, 404 before anything else where no device signs in on the phone (``local``, ``saas``), and limited
  per client IP with ``/pair``'s own ``mobile_pair_ip`` bucket before the database. Every refusal, a code or
  verifier of any length included, is the one 401 ``invalid_sign_in_code`` with ``no-store``; a validation
  error never echoes what was sent.
* **Which devices are taken** (``core.auth.device.token_kinds``) and **what each edition change revokes**
  (``edition_change.KINDS_REVOKED``): pairing off stops ``paired`` devices only; another edition stops both.
* **The code** is at least 128 bits and lives about a minute; S256(verifier) is compared with the stored challenge.
"""
from __future__ import annotations

import inspect
import os
from types import MappingProxyType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_oidc  # noqa: E402
from api import mobile_devices as devices_api  # noqa: E402
from config import config  # noqa: E402
from core.auth import device, sign_in_flow  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.oidc_pending import code_challenge  # noqa: E402
from core.auth.phone_sign_in import (  # noqa: E402
    APP_LINKS,
    InvalidPhoneStart,
    PhoneStart,
    app_answer,
    phone_start_from,
)
from core.database.database import get_db  # noqa: E402
from core.security.rate_limiter import DEFAULT_LIMITS  # noqa: E402
from services.mobile import device_audit, edition_change, sign_in_codes  # noqa: E402
from services.push import fanout  # noqa: E402

INSTALL = "https://automatos.example.test"
START = "/api/v1/auth/oidc/start"
EXCHANGE = "/api/v1/mobile/sign-in/exchange"
VERIFIER = "v" * 43 + "-._~"
CHALLENGE = code_challenge(VERIFIER)
GOOD = {"app_link": "go", "code_challenge": CHALLENGE, "code_challenge_method": "S256",
        "install_id": "prd277-w6b-phone", "platform": "ios"}


def _sent(**changes) -> dict:
    """``GOOD`` with ``changes`` (a value of None drops the name), each as the tuple of values sent."""
    merged = {**GOOD, **changes}
    return {name: (value,) if isinstance(value, str) else value for name, value in merged.items() if value is not None}


def _enterprise(monkeypatch, pairing="true") -> None:
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "ENTERPRISE_DEVICE_PAIRING", pairing)


# --------------------------------------------------------------------------- the start's parameters


def test_a_start_naming_no_phone_parameter_is_a_browsers():
    assert phone_start_from({}) is None
    assert phone_start_from({"return_to": ("/projects",)}) is None


def test_a_well_formed_phone_start_is_read_and_hides_its_challenge_and_device_from_repr():
    phone = phone_start_from(_sent())
    assert phone == PhoneStart(app_link="go", code_challenge=CHALLENGE, install_id="prd277-w6b-phone", platform="ios")
    assert CHALLENGE not in repr(phone) and "prd277-w6b-phone" not in repr(phone)


@pytest.mark.parametrize("changes", [
    {"app_link": None}, {"code_challenge": None}, {"code_challenge_method": None}, {"install_id": None},
    {"platform": None},
    {"app_link": ("go", "go")}, {"code_challenge": (CHALLENGE, CHALLENGE)},
    {"app_link": "automatos://go/signed-in"}, {"app_link": "https://attacker.example/catch"}, {"app_link": "GO"},
    {"code_challenge_method": "plain"}, {"code_challenge_method": "s256"},
    {"code_challenge": CHALLENGE[:42]}, {"code_challenge": CHALLENGE + "A"}, {"code_challenge": "+" * 43},
    {"install_id": "short"}, {"install_id": "has space in it"}, {"platform": "web"}, {"platform": ""},
], ids=lambda changes: ",".join(f"{k}={v!r}"[:40] for k, v in changes.items()))
def test_a_phone_start_with_any_parameter_missing_repeated_or_malformed_is_refused(changes):
    with pytest.raises(InvalidPhoneStart) as refused:
        phone_start_from(_sent(**changes))
    assert CHALLENGE not in str(refused.value)


def test_the_app_links_are_a_fixed_set_naming_automatos_go_only():
    assert isinstance(APP_LINKS, MappingProxyType)
    assert dict(APP_LINKS) == {"go": "automatos://go/signed-in"}
    with pytest.raises(TypeError):
        APP_LINKS["evil"] = "https://attacker.example"  # type: ignore[index]


def test_the_answer_is_the_app_link_with_the_code_or_the_error_and_nothing_else():
    phone = phone_start_from(_sent())
    for answer, expected in ((app_answer(phone, code="c0de"), {"code": ["c0de"]}),
                             (app_answer(phone, error="sign_in_refused"), {"error": ["sign_in_refused"]})):
        parts = urlsplit(answer)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == APP_LINKS["go"]
        assert parse_qs(parts.query) == expected


# --------------------------------------------------------------------------- the start route


def _start_client(monkeypatch, start_sign_in=None) -> TestClient:
    _enterprise(monkeypatch)
    monkeypatch.setattr(auth_oidc, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 300, 299, 600)))
    monkeypatch.setattr(sign_in_flow, "start_sign_in",
                        start_sign_in or (lambda *a, **k: pytest.fail("nothing may start")))
    monkeypatch.setattr(sign_in_flow, "start_step_up", lambda *a, **k: pytest.fail("nothing may start"))
    app = FastAPI()
    app.include_router(auth_oidc.router)
    app.dependency_overrides[get_db] = lambda: None
    return TestClient(app, base_url=INSTALL)


@pytest.mark.parametrize("params", [
    {**GOOD, "code_challenge_method": "plain"}, {**GOOD, "app_link": "elsewhere"}, {"app_link": "go"},
    {**GOOD, "step_up": "1"},
], ids=["plain", "unknown app link", "one parameter only", "with step_up"])
def test_a_malformed_phone_start_is_400_before_the_provider_is_asked(monkeypatch, params):
    answer = _start_client(monkeypatch).get(START, params=params, follow_redirects=False)

    assert (answer.status_code, answer.json()) == (400, {"detail": {"code": auth_oidc.INVALID_PHONE_START}})
    assert answer.headers["cache-control"] == "no-store"
    assert auth_oidc.SIGN_IN_COOKIE not in answer.headers.get("set-cookie", "")


def test_a_phone_start_is_handed_on_and_its_refusal_goes_to_its_app_link(monkeypatch):
    seen = []

    def unavailable(db, return_to, client_address, phone=None):
        seen.append(phone)
        raise sign_in_flow.SignInRefusal(sign_in_flow.UNAVAILABLE, phone=phone)

    answer = _start_client(monkeypatch, unavailable).get(START, params=GOOD, follow_redirects=False)

    assert seen == [phone_start_from(_sent())]
    assert answer.status_code == 303
    assert answer.headers["location"] == f"{APP_LINKS['go']}?error={sign_in_flow.UNAVAILABLE}"


def test_a_browsers_start_is_handed_no_phone(monkeypatch):
    seen = []

    def unavailable(db, return_to, client_address, **kwargs):
        seen.append(kwargs)
        raise sign_in_flow.SignInRefusal(sign_in_flow.UNAVAILABLE)

    answer = _start_client(monkeypatch, unavailable).get(START, follow_redirects=False)

    assert seen == [{}] and answer.headers["location"] == "/sign-in?error=identity_provider_unavailable"


# --------------------------------------------------------------------------- the exchange route


def _exchange_route() -> APIRoute:
    return next(route for route in devices_api.router.routes
                if isinstance(route, APIRoute) and route.path == "/api/v1/mobile/sign-in/exchange")


def _calls(dependant) -> list:
    found = [dependant.call]
    for sub in dependant.dependencies:
        found.extend(_calls(sub))
    return found


def test_the_exchange_is_public_plain_def_and_limited_with_pairs_bucket_before_the_database():
    route = _exchange_route()
    calls = _calls(route.dependant)

    assert route.methods == {"POST"} and not inspect.iscoroutinefunction(route.endpoint)
    assert get_request_context_hybrid not in calls
    assert calls.index(devices_api._require_phone_sign_in) < calls.index(devices_api.limit_pair_attempts_per_ip)
    assert calls.index(devices_api.limit_pair_attempts_per_ip) < calls.index(get_db)
    assert devices_api._PAIR_LIMIT.operation == "mobile_pair_ip" and "mobile_pair_ip" in DEFAULT_LIMITS


class _Db:
    """A session the exchange commits or rolls back; it is never queried in these tests."""

    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1

    def rollback(self):
        return None


def _exchange_client(monkeypatch, db=None) -> TestClient:
    monkeypatch.setattr(devices_api, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 30, 29, 600)))
    monkeypatch.setattr(edition_change, "_status", edition_change.NOT_RUN)
    app = FastAPI()
    app.include_router(devices_api.router)
    app.dependency_overrides[get_db] = lambda: db or _Db()
    return TestClient(app, base_url=INSTALL)


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_the_exchange_is_404_where_no_phone_signs_in(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    client = _exchange_client(monkeypatch)
    monkeypatch.setattr(devices_api, "_rate_store", SimpleNamespace(check=lambda *a, **k: pytest.fail("counted")))

    answer = client.post(EXCHANGE, json={"code": "x", "code_verifier": VERIFIER})

    assert answer.status_code == 404


@pytest.mark.parametrize("pairing", ["true", "false"])
@pytest.mark.parametrize("body", [
    {"code": "not-a-code", "code_verifier": VERIFIER},
    {"code": "c" * 5000, "code_verifier": VERIFIER},
    {"code": "c" * 43, "code_verifier": "v" * 5000},
    {"code": "", "code_verifier": ""},
], ids=["unknown", "long code", "long verifier", "empty"])
def test_every_refused_exchange_is_one_401_whatever_was_sent(monkeypatch, pairing, body):
    _enterprise(monkeypatch, pairing)
    monkeypatch.setattr(sign_in_codes, "_claimed", lambda db, code: None)

    answer = _exchange_client(monkeypatch).post(EXCHANGE, json=body)

    assert (answer.status_code, answer.json()) == (401, {"detail": {"code": devices_api.INVALID_SIGN_IN_CODE}})
    assert answer.headers["cache-control"] == "no-store"


def test_a_long_code_is_refused_before_any_lookup():
    db = SimpleNamespace(query=lambda *a: pytest.fail("no lookup for a code that is not one"))
    assert sign_in_codes.exchange_code(db, "c" * (sign_in_codes.SIGN_IN_CODE_MAX_LENGTH + 1), VERIFIER,
                                       (None, None)) is None


# --------------------------------------------------------------------------- the code and its verifier


def test_the_code_is_at_least_128_bits_and_lives_about_a_minute():
    assert sign_in_codes.SIGN_IN_CODE_BYTES * 8 >= 128
    assert sign_in_codes.SIGN_IN_CODE_TTL_SECONDS == 60


@pytest.mark.parametrize("verifier, fits", [
    (VERIFIER, True), (VERIFIER + "x", False), ("w" * 43, False), ("é" * 43, False),
    ("v" * (sign_in_codes.CODE_VERIFIER_MAX_LENGTH + 1), False), ("", False),
])
def test_only_the_verifier_whose_s256_is_the_stored_challenge_fits(verifier, fits):
    assert sign_in_codes._verifier_fits(SimpleNamespace(sign_in_challenge=CHALLENGE), verifier) is fits


def test_a_row_without_a_challenge_fits_no_verifier():
    assert sign_in_codes._verifier_fits(SimpleNamespace(sign_in_challenge=None), VERIFIER) is False


# --------------------------------------------------------------------------- which devices, and what is revoked


@pytest.mark.parametrize("edition, pairing, kinds", [
    ("local", "true", ("paired",)), ("saas", "true", ()),
    ("enterprise", "true", ("paired", "oidc")), ("enterprise", "false", ("oidc",)),
])
def test_the_device_lane_and_push_take_the_kinds_the_edition_takes(monkeypatch, edition, pairing, kinds):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "ENTERPRISE_DEVICE_PAIRING", pairing)
    assert device.token_kinds() == kinds
    assert fanout._edition_kinds() == (kinds or ("clerk",))


def test_pairing_off_revokes_paired_devices_only_and_another_edition_revokes_both():
    assert edition_change.KINDS_REVOKED == {
        device_audit.REVOKED_EDITION_CHANGED: ("paired", "oidc"),
        device_audit.REVOKED_PAIRING_SWITCHED_OFF: ("paired",),
    }
