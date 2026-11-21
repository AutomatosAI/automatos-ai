"""PRD-277 US-603 without a database: what a phone's start may ask for, where its answer goes, and the exchange route.

* **The start's parameters** (``core.auth.phone_sign_in``): none named is a browser's start; any named means all
  five, once each: an ``app_link`` KEY of the fixed ``APP_LINKS`` (never an address), an S256 challenge of 43
  base64url characters (``plain`` and a missing method refused), the device's install id and ``ios`` or
  ``android``. A malformed one is 400 ``invalid_phone_start`` before the provider is asked or anything is stored,
  and so is a phone's start with ``step_up``. A refusal of a phone's start goes to its app link.
* **The answer** is the fixed app link with ``code`` or ``error`` and nothing else. A refusal once the person
  is known that only a phone meets (``PHONE_REFUSALS``) goes to the confirmation page, never the app, and is
  audited with itself as the reason (``session_audit.PHONE_REFUSAL_REASONS``).
* **The confirmation routes** (``api/phone_sign_in.py``) are exactly ``GET /waiting``, ``POST /continue`` and
  ``POST /cancel`` under ``/api/v1/auth/oidc/phone``: public (no request context), plain ``def``, 404 where the
  edition does not sign in with OIDC, limited per IP in their own bucket (``oidc_phone_confirm_ip``), and the
  two POSTs refuse a cross-site request (US-302's check) before the limit and the database. The cookie is
  ``__Host-``, Secure, HttpOnly, SameSite=Lax, ``Path=/``, five minutes.
* **The exchange route** (``POST /api/v1/mobile/sign-in/exchange``) is public (no request context), plain
  ``def``, 404 before anything else where no device signs in on the phone (``local``, ``saas``), and limited
  per client IP in a bucket of its own (``phone_sign_in_exchange_ip``, not a ``mobile_pair`` one) before the
  database. Every refusal, a code or verifier of any length included, is the one 401 ``invalid_sign_in_code``
  with ``no-store``; a validation error never echoes what was sent.
* **The device cap** is counted under the person's own advisory lock, taken before the count.
* **Which devices are taken** (``core.auth.device.token_kinds``) and **what each edition change revokes**
  (``edition_change.KINDS_REVOKED``): pairing off stops ``paired`` devices only; another edition stops both.
* **The code** is at least 128 bits and lives about a minute; S256(verifier) is compared with the stored challenge.
"""
from __future__ import annotations

import inspect
import os
import uuid
from types import MappingProxyType, SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from fastapi import FastAPI, Response
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from api import auth_oidc  # noqa: E402
from api import mobile_devices as devices_api  # noqa: E402
from api import phone_sign_in as phone_routes  # noqa: E402
from config import config  # noqa: E402
from core.auth import device, session_audit, sign_in_flow  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.auth.oidc_pending import code_challenge  # noqa: E402
from core.auth.phone_sign_in import (  # noqa: E402
    APP_LINKS,
    APP_NAMES,
    CONFIRM_COOKIE,
    CONFIRM_PAGE,
    PHONE_REFUSALS,
    InvalidPhoneStart,
    PhoneStart,
    app_answer,
    phone_start_from,
    set_confirm_cookie,
)
from core.database.database import get_db  # noqa: E402
from core.security.rate_limiter import DEFAULT_LIMITS  # noqa: E402
from services.mobile import device_audit, devices, edition_change, sign_in_codes, sign_in_confirmation  # noqa: E402
from services.push import fanout  # noqa: E402

INSTALL = "https://automatos.example.test"
START = "/api/v1/auth/oidc/start"
EXCHANGE = "/api/v1/mobile/sign-in/exchange"
PHONE = "/api/v1/auth/oidc/phone"
COOKIE = {"cookie": f"{CONFIRM_COOKIE}={'t' * 43}"}
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


def test_the_app_links_and_their_names_are_a_fixed_set_naming_automatos_go_only():
    assert isinstance(APP_LINKS, MappingProxyType) and isinstance(APP_NAMES, MappingProxyType)
    assert dict(APP_LINKS) == {"go": "automatos://go/signed-in"} and dict(APP_NAMES) == {"go": "Automatos Go"}
    with pytest.raises(TypeError):
        APP_LINKS["evil"] = "https://attacker.example"  # type: ignore[index]


def test_the_answer_is_the_app_link_with_the_code_or_the_error_and_nothing_else():
    for answer, expected in ((app_answer("go", code="c0de"), {"code": ["c0de"]}),
                             (app_answer("go", error="sign_in_refused"), {"error": ["sign_in_refused"]})):
        parts = urlsplit(answer)
        assert f"{parts.scheme}://{parts.netloc}{parts.path}" == APP_LINKS["go"]
        assert parse_qs(parts.query) == expected


def test_a_phones_refusal_once_its_person_is_known_goes_to_the_page_and_any_other_to_the_app():
    phone = phone_start_from(_sent())
    for code in PHONE_REFUSALS:
        assert auth_oidc._refused(code, phone).headers["location"] == f"{CONFIRM_PAGE}?error={code}"
    assert auth_oidc._refused(sign_in_flow.REFUSED, phone).headers["location"] == \
        f"{APP_LINKS['go']}?error={sign_in_flow.REFUSED}"
    assert auth_oidc._refused(PHONE_REFUSALS[0]).headers["location"] == f"/sign-in?error={PHONE_REFUSALS[0]}"


def test_each_phone_refusal_is_audited_with_itself_as_the_reason():
    assert session_audit.PHONE_REFUSAL_REASONS == PHONE_REFUSALS
    assert set(PHONE_REFUSALS) <= set(session_audit.REFUSAL_REASONS)


def test_a_phones_sign_in_redirects_to_the_page_with_its_cookie_and_no_session():
    done = sign_in_flow.SignInDone(session=None, return_to="/projects", phone_confirmation="t" * 43)

    answer = auth_oidc._signed_in(done)

    assert (answer.status_code, answer.headers["location"]) == (303, CONFIRM_PAGE)
    assert answer.headers["cache-control"] == "no-store" and "t" * 43 not in repr(done)
    (cookie,) = answer.headers.getlist("set-cookie")
    assert cookie.startswith(f"{CONFIRM_COOKIE}={'t' * 43};")
    assert config.SESSION_COOKIE_NAME not in cookie


def test_the_confirmation_cookie_is_host_only_secure_http_only_lax_and_lives_five_minutes():
    response = Response()
    set_confirm_cookie(response, "token")
    parts = {part.strip() for part in response.headers["set-cookie"].split(";")}
    assert CONFIRM_COOKIE.startswith("__Host-")
    assert {f"{CONFIRM_COOKIE}=token", "HttpOnly", "Secure", "Path=/", "SameSite=lax", "Max-Age=300"} <= parts
    assert not any(part.lower().startswith("domain=") for part in parts)


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


def test_the_exchange_is_public_plain_def_and_limited_in_its_own_bucket_before_the_database():
    route = _exchange_route()
    calls = _calls(route.dependant)

    assert route.methods == {"POST"} and not inspect.iscoroutinefunction(route.endpoint)
    assert get_request_context_hybrid not in calls and devices_api.limit_pair_attempts_per_ip not in calls
    assert calls.index(devices_api._require_phone_sign_in) < calls.index(devices_api.limit_sign_in_exchanges_per_ip)
    assert calls.index(devices_api.limit_sign_in_exchanges_per_ip) < calls.index(get_db)
    operation = devices_api._EXCHANGE_LIMIT.operation
    assert operation == "phone_sign_in_exchange_ip" and not operation.startswith("mobile_pair")
    assert DEFAULT_LIMITS[operation] == (300, 600)
    assert devices_api._EXCHANGE_LIMIT.bucket_prefix != devices_api._PAIR_LIMIT.bucket_prefix


# --------------------------------------------------------------------------- the confirmation routes


def _phone_routes() -> dict:
    return {(method, route.path): route for route in phone_routes.router.routes if isinstance(route, APIRoute)
            for method in route.methods}


def test_the_confirmation_routes_are_three_public_plain_defs_limited_before_the_database():
    routes = _phone_routes()

    assert set(routes) == {("GET", f"{PHONE}/waiting"), ("POST", f"{PHONE}/continue"), ("POST", f"{PHONE}/cancel")}
    assert DEFAULT_LIMITS["oidc_phone_confirm_ip"] == (600, 600)
    assert auth_oidc._CONFIRM_LIMIT.operation == "oidc_phone_confirm_ip"
    for (method, _path), route in routes.items():
        calls = _calls(route.dependant)
        assert not inspect.iscoroutinefunction(route.endpoint) and get_request_context_hybrid not in calls
        expected = [auth_oidc._signs_in_with_oidc, *([phone_routes.sent_from_this_install] if method == "POST" else []),
                    auth_oidc.limit_phone_confirmations_per_ip, get_db]
        assert [call for call in calls if call in expected] == expected


def _phone_client(monkeypatch, edition="enterprise") -> TestClient:
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "FRONTEND_URL", INSTALL)
    monkeypatch.setattr(auth_oidc, "_rate_store", SimpleNamespace(check=lambda *a, **k: (True, 600, 599, 600)))
    for name in ("waiting", "confirm", "cancel"):
        monkeypatch.setattr(sign_in_confirmation, name, lambda *a, **k: pytest.fail("nothing may be read"))
    app = FastAPI()
    app.include_router(phone_routes.router)
    app.dependency_overrides[get_db] = lambda: _Db()
    return TestClient(app, base_url=INSTALL)


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_the_confirmation_routes_are_404_where_the_edition_does_not_sign_in_with_oidc(monkeypatch, edition):
    client = _phone_client(monkeypatch, edition)
    for answer in (client.get(f"{PHONE}/waiting"), client.post(f"{PHONE}/continue"), client.post(f"{PHONE}/cancel")):
        assert answer.status_code == 404


@pytest.mark.parametrize("path", ["continue", "cancel"])
@pytest.mark.parametrize("headers", [
    {"origin": "https://attacker.example", "x-automatos-csrf": "1"},
    {"origin": "https://attacker.example"},
    {"referer": "https://attacker.example/page"},
    {"origin": INSTALL},
    {"origin": "null", "x-automatos-csrf": "1"},
    {},
], ids=["other origin with header", "cross-site form", "other referer", "no header", "opaque origin", "nothing"])
def test_a_post_from_anywhere_but_the_installs_own_page_is_refused_before_anything_is_read(monkeypatch, path,
                                                                                         headers):
    answer = _phone_client(monkeypatch).post(f"{PHONE}/{path}", headers={**headers, **COOKIE})

    assert (answer.status_code, answer.json()["detail"]["code"]) == (403, "cross_site_request_refused")


def test_continue_answers_the_app_link_with_the_code_and_clears_the_cookie(monkeypatch):
    client = _phone_client(monkeypatch)
    confirmed = sign_in_confirmation.Confirmed(code="c0de", app_link="go", user_pk=7, workspace_id=uuid.uuid4(),
                                               platform="ios")
    written = []
    monkeypatch.setattr(sign_in_confirmation, "confirm", lambda db, token: confirmed if token == "t" * 43 else None)
    monkeypatch.setattr(session_audit, "record_phone_sign_in", lambda db, **row: written.append(row))

    answer = client.post(f"{PHONE}/continue", headers={"origin": INSTALL, "x-automatos-csrf": "1", **COOKIE})

    assert answer.status_code == 200 and answer.json() == {"app_link": f"{APP_LINKS['go']}?code=c0de"}
    assert answer.headers["cache-control"] == "no-store" and answer.headers["referrer-policy"] == "no-referrer"
    cleared = answer.headers["set-cookie"]
    assert cleared.startswith(f"{CONFIRM_COOKIE}=") and "Max-Age=0" in cleared and "c0de" not in cleared
    assert [row["user_id"] for row in written] == [7] and "c0de" not in str(written)


def test_the_page_read_answers_four_fields_and_nothing_waiting_is_404(monkeypatch):
    client = _phone_client(monkeypatch)
    shown = sign_in_confirmation.Waiting(person="a@example.test", app_link="go", platform="android",
                                         removed_before=True)
    monkeypatch.setattr(sign_in_confirmation, "waiting", lambda db, token: shown if token else None)

    found = client.get(f"{PHONE}/waiting", headers=COOKIE)
    nothing = client.get(f"{PHONE}/waiting")

    assert found.json() == {"person": "a@example.test", "app": "Automatos Go", "platform": "android",
                            "removed_before": True}
    assert (nothing.status_code, nothing.json()) == (404, {"detail": {"code": phone_routes.NOTHING_WAITING}})


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


def test_the_device_cap_is_counted_under_the_persons_lock_taken_first(monkeypatch):
    seen = []
    db = SimpleNamespace(execute=lambda statement, params: seen.append(("lock", str(statement), dict(params))))
    monkeypatch.setattr(sign_in_codes, "_signs_in_again", lambda *a: seen.append(("again",)) or False)
    monkeypatch.setattr(devices, "assert_room_for_a_device", lambda *a: seen.append(("count",)))

    sign_in_codes.hold_room_for_a_device(db, uuid.uuid4(), 42, "phone-1")

    assert [step[0] for step in seen] == ["lock", "again", "count"]
    assert "pg_advisory_xact_lock" in seen[0][1]
    assert seen[0][2] == {"space": sign_in_codes.DEVICE_CAP_LOCK_SPACE, "user_pk": 42}


def test_a_confirmation_token_is_never_looked_up_as_a_code():
    looked_up = []
    db = SimpleNamespace(query=lambda *a: looked_up.append(a) or pytest.fail("a token is not a code"))
    assert sign_in_confirmation._awaiting(db, "t" * (sign_in_confirmation.CONFIRM_TOKEN_MAX_LENGTH + 1),
                                          lock=True) is None
    assert sign_in_confirmation.CONFIRM_TOKEN_BYTES * 8 >= 128
    assert "sign_in_confirm_hash" not in inspect.getsource(sign_in_codes)


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
