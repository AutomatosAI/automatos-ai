"""PRD-277 Wave 3 (US-301, US-302): the session lane's seams, without a database.

* The registry: ``BEARER_RESOLVERS`` carries an ``ams_`` entry, registered only where the edition
  issues sessions. In ``local`` and ``saas`` an ``ams_`` bearer reaches the wrapped function
  exactly as any unregistered bearer did, and a session cookie is never read; ``hybrid.py`` gains
  no branch.
* Where the token comes from: the ``Authorization`` header wins; the cookie is read only when
  there is no ``Authorization`` header and no API key.
* Cross-site request forgery (US-302): a cross-site form post is refused before any lookup, a
  same-origin fetch with the header is allowed, and a header-authenticated call is never checked.
  A safe method is not checked; a same-origin call without the header, a call with no origin and
  an opaque origin are refused; a ``Referer`` stands in for a missing ``Origin``.
* The cookie: ``__Host-`` name, ``Secure``, ``HttpOnly``, ``SameSite=Lax``, ``Path=/``, no
  ``Domain``, and a ``Max-Age`` of what is left of the session.
* Start-up: the session settings are checked where sessions are issued, and nowhere else.
* Step-up in enterprise goes through one seam, which reads the session's ``auth_time`` (US-406).
* The 401 the web app recognises names its reason and never the token.
"""
from __future__ import annotations

import inspect
import json
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.responses import Response

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from config import config  # noqa: E402
from core import edition_start_up, editions  # noqa: E402
from core.auth import bearer_registry, hybrid, session_cookie, session_lane, sessions, step_up  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.principal import claimed_user_pk  # noqa: E402

ORCH = Path(__file__).resolve().parents[1]
INSTALL = "https://automatos.example.test"
ELSEWHERE = "https://elsewhere.example.test"
WS = uuid.UUID("00000000-0000-0000-0000-0000000277c3")
CSRF_VALUE = "1"


def _token() -> str:
    """A stand-in session token, built at run time so no literal looks like one."""
    return "ams" + "_" + secrets.token_urlsafe(32)


def _request(method="GET", headers=None):
    return SimpleNamespace(method=method, headers=dict(headers or {}), query_params={}, state=SimpleNamespace())


def _cookie(token: str) -> dict:
    return {"cookie": f"other=1; {config.SESSION_COOKIE_NAME}={token}"}


def _same_origin() -> dict:
    return {"origin": INSTALL, config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE}


class _Lane:
    """Stands in for the session lane's database work: records each token it is asked about."""

    def __init__(self):
        self.asked = []

    def __call__(self, request, token):
        self.asked.append(token)
        user = UserContext(id="p@example.test", email="p@example.test",
                           raw_claims={"source": "auth_session", "user_id": 7, "auth_time": 1.0})
        return RequestContext(workspace_id=WS, user=user, auth_type=session_lane.SESSION_AUTH_TYPE)


@pytest.fixture
def lane(monkeypatch):
    spy = _Lane()
    monkeypatch.setattr(session_lane, "_resolve_in_own_session", spy)
    monkeypatch.setattr(hybrid, "_enrich_log_context", lambda ctx: None)
    return spy


@pytest.fixture
def enterprise(monkeypatch, lane):
    monkeypatch.setattr(config, "AUTH_EDITION", "enterprise")
    monkeypatch.setattr(config, "REQUIRE_AUTH", True)
    monkeypatch.setattr(config, "FRONTEND_URL", INSTALL)
    return lane


def _recording():
    calls = []

    async def wrapped(request):
        calls.append(request)
        return "today's lane"

    return bearer_registry.resolve_registered_bearers(wrapped), calls


# --------------------------------------------------------------------------- the registry


def test_the_registry_carries_the_session_entry_and_hybrid_no_branch():
    assert bearer_registry.BEARER_RESOLVERS["ams_"] is session_lane.resolve_session_bearer
    assert bearer_registry.BEARER_RESOLVERS["amd_"].__module__ == "core.auth.device"
    source = inspect.getsource(hybrid)
    assert "ams_" not in source and "session_lane" not in source and "cookie" not in source.lower()


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_outside_enterprise_no_lane_claims_an_ams_token(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    assert bearer_registry.resolver_for(_token()) is None
    assert bearer_registry.resolver_for("amd_" + "x") is bearer_registry.BEARER_RESOLVERS["amd_"]


def test_in_enterprise_the_session_lane_claims_an_ams_token(enterprise):
    assert bearer_registry.resolver_for(_token()) is session_lane.resolve_session_bearer


@pytest.mark.asyncio
@pytest.mark.parametrize("edition", ["local", "saas"])
async def test_outside_enterprise_an_ams_bearer_and_a_cookie_reach_todays_lane(monkeypatch, lane, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(config, "LOCAL_REMOTE_ACCESS", "off")
    decorated, calls = _recording()
    token = _token()
    requests = [_request(headers={"authorization": "Bearer " + token}),
                _request("POST", headers={**_cookie(token), "origin": ELSEWHERE})]

    for request in requests:
        assert await decorated(request) == "today's lane"

    assert calls == requests and lane.asked == []
    assert session_cookie.cookie_token(requests[1]) is None


@pytest.mark.asyncio
async def test_in_enterprise_a_header_token_is_the_session_lanes(enterprise):
    decorated, calls = _recording()
    token = _token()

    ctx = await decorated(_request(headers={"authorization": "Bearer " + token}))

    assert ctx.auth_type == "session" and enterprise.asked == [token] and calls == []


@pytest.mark.asyncio
async def test_the_header_wins_over_the_cookie(enterprise):
    decorated, _ = _recording()
    in_header, in_cookie = _token(), _token()

    await decorated(_request(headers={"authorization": "Bearer " + in_header, **_cookie(in_cookie)}))

    assert enterprise.asked == [in_header]


@pytest.mark.asyncio
@pytest.mark.parametrize("other", [{"authorization": "Basic x"}, {"x-api-key": "k" + "ey"},
                                   {"authorization": "Bearer " + ".".join(("h", "c", "s"))}])
async def test_the_cookie_is_not_read_beside_any_other_credential(enterprise, other):
    decorated, calls = _recording()

    assert await decorated(_request(headers={**other, **_cookie(_token())})) == "today's lane"

    assert enterprise.asked == [] and len(calls) == 1


@pytest.mark.asyncio
async def test_no_cookie_and_no_credential_still_reaches_todays_lane(enterprise):
    decorated, calls = _recording()

    assert await decorated(_request(headers={"cookie": "other=1"})) == "today's lane"
    assert enterprise.asked == [] and len(calls) == 1


# --------------------------------------------------------------------------- US-302


@pytest.mark.asyncio
async def test_a_cross_site_form_post_is_refused_before_any_lookup(enterprise):
    decorated, calls = _recording()
    form_post = _request("POST", headers={**_cookie(_token()), "origin": ELSEWHERE,
                                          "content-type": "application/x-www-form-urlencoded"})

    with pytest.raises(HTTPException) as refused:
        await decorated(form_post)

    assert refused.value.status_code == 403
    assert refused.value.detail["code"] == session_cookie.CROSS_SITE_REFUSED
    assert enterprise.asked == [] and calls == []


@pytest.mark.asyncio
async def test_a_same_origin_fetch_with_the_header_is_allowed(enterprise):
    decorated, _ = _recording()
    token = _token()

    ctx = await decorated(_request("POST", headers={**_cookie(token), **_same_origin()}))

    assert ctx.auth_type == "session" and enterprise.asked == [token]


@pytest.mark.asyncio
async def test_a_header_authenticated_call_is_never_checked(enterprise):
    decorated, _ = _recording()
    token = _token()

    ctx = await decorated(_request("DELETE", headers={"authorization": "Bearer " + token, "origin": ELSEWHERE}))

    assert ctx.auth_type == "session" and enterprise.asked == [token]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
async def test_a_safe_method_from_anywhere_is_not_checked(enterprise, method):
    decorated, _ = _recording()

    ctx = await decorated(_request(method, headers={**_cookie(_token()), "origin": ELSEWHERE}))

    assert ctx.auth_type == "session"


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [
    {"origin": INSTALL},                                              # no header
    {config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE},                 # no origin at all
    {"origin": "null", config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE},  # an opaque origin
    {"origin": ELSEWHERE, config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE},
    {"origin": INSTALL + ":8443", config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE},  # another port
    {"origin": "http://automatos.example.test", config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE},
    {"origin": INSTALL, config.SESSION_CSRF_HEADER.lower(): "   "},  # a blank header
])
async def test_every_other_unsafe_cookie_request_is_refused(enterprise, headers):
    decorated, _ = _recording()

    with pytest.raises(HTTPException) as refused:
        await decorated(_request("PATCH", headers={**_cookie(_token()), **headers}))

    assert refused.value.status_code == 403 and enterprise.asked == []


@pytest.mark.asyncio
async def test_a_referer_stands_in_for_a_missing_origin(enterprise):
    decorated, _ = _recording()
    headers = {**_cookie(_token()), "referer": INSTALL + "/settings?tab=team",
               config.SESSION_CSRF_HEADER.lower(): CSRF_VALUE}

    ctx = await decorated(_request("PUT", headers=headers))

    assert ctx.auth_type == "session"


def test_the_install_origin_ignores_a_default_port_and_case(monkeypatch):
    monkeypatch.setattr(config, "FRONTEND_URL", "HTTPS://Automatos.Example.Test:443/app")
    assert session_cookie.install_origin() == INSTALL
    assert session_cookie.request_origin(_request(headers={"origin": INSTALL})) == INSTALL


def test_the_csrf_header_is_not_one_the_cors_settings_let_another_origin_send():
    main_source = (ORCH / "main.py").read_text(encoding="utf-8")
    allowed = main_source.split("allow_headers=[", 1)[1].split("]", 1)[0]
    assert config.SESSION_CSRF_HEADER.lower() not in allowed.lower()


# --------------------------------------------------------------------------- the cookie


def test_the_cookie_is_host_only_secure_http_only_and_lax():
    response = Response()
    now = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)
    token = _token()

    session_cookie.set_session_cookie(response, token, now + timedelta(hours=12), now=now)

    header = response.headers["set-cookie"]
    parts = [part.strip().lower() for part in header.split(";")]
    assert header.startswith(f"__Host-automatos_session={token}")
    assert {"secure", "httponly", "path=/", "samesite=lax", "max-age=43200"} <= set(parts)
    assert not any(part.startswith("domain") for part in parts)


def test_clearing_the_cookie_keeps_its_attributes():
    response = Response()

    session_cookie.clear_session_cookie(response)

    parts = [part.strip().lower() for part in response.headers["set-cookie"].split(";")]
    assert parts[0].startswith("__host-automatos_session=") and "max-age=0" in parts
    assert {"secure", "httponly", "path=/", "samesite=lax"} <= set(parts)


# --------------------------------------------------------------------------- settings


def test_the_session_settings_have_the_prds_defaults_and_are_in_the_surface():
    assert (config.SESSION_IDLE_MINUTES, config.SESSION_ABSOLUTE_HOURS) == (120, 12)  # D-15
    assert config.SESSION_COOKIE_NAME == "__Host-automatos_session"
    assert config.SESSION_CSRF_HEADER == "X-Automatos-CSRF"
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert set(edition_start_up.SESSION_SETTINGS) <= set(surface)
    assert surface == sorted(surface)


def _settings(**overrides):
    values = {name: getattr(config, name) for name in edition_start_up.SESSION_SETTINGS}
    return {**values, **overrides}


def test_start_up_accepts_the_defaults_where_sessions_are_issued():
    assert edition_start_up.check_session_settings(editions.capabilities_for("enterprise"), _settings()) is None


@pytest.mark.parametrize("name, value", [
    ("SESSION_COOKIE_NAME", "automatos_session"),
    ("SESSION_COOKIE_NAME", "__Secure-automatos_session"),
    ("SESSION_COOKIE_NAME", "__Host-a b"),
    ("SESSION_CSRF_HEADER", ""),
    ("SESSION_CSRF_HEADER", "X Automatos"),
    ("SESSION_IDLE_MINUTES", 0),
    ("SESSION_ABSOLUTE_HOURS", -1),
    ("SESSION_IDLE_MINUTES", 12 * 60 + 1),
    ("SESSION_IDLE_MINUTES", 4),            # below the last-seen interval: an active person cut off
    ("SESSION_IDLE_MINUTES", ""),           # config.py hands a value that is not a number on as text
    ("SESSION_ABSOLUTE_HOURS", "twelve"),
])
def test_start_up_refuses_a_bad_session_setting_naming_it(name, value):
    with pytest.raises(edition_start_up.EditionNotReady) as refused:
        edition_start_up.check_session_settings(editions.capabilities_for("enterprise"), _settings(**{name: value}))

    assert name in str(refused.value)


def test_the_idle_floor_is_the_last_seen_interval_and_is_named():
    from core.auth.device import LAST_SEEN_WRITE_INTERVAL

    assert timedelta(minutes=edition_start_up.SESSION_IDLE_MINUTES_MIN) == LAST_SEEN_WRITE_INTERVAL
    with pytest.raises(edition_start_up.EditionNotReady) as refused:
        edition_start_up.check_session_settings(editions.capabilities_for("enterprise"),
                                                _settings(SESSION_IDLE_MINUTES=4))
    assert "SESSION_IDLE_MINUTES" in str(refused.value) and "at least 5" in str(refused.value)
    floor = _settings(SESSION_IDLE_MINUTES=edition_start_up.SESSION_IDLE_MINUTES_MIN)
    assert edition_start_up.check_session_settings(editions.capabilities_for("enterprise"), floor) is None


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_start_up_reads_no_session_setting_elsewhere(edition):
    assert edition_start_up.check_session_settings(editions.capabilities_for(edition), {}) is None


# --------------------------------------------------------------------------- the context


def test_a_session_context_names_its_person_by_users_id():
    ctx = _Lane()(None, _token())

    assert claimed_user_pk(ctx) == 7
    assert session_lane.session_auth_time(ctx) == 1.0


@pytest.mark.parametrize("ctx", [
    SimpleNamespace(auth_type="device", user=SimpleNamespace(raw_claims={"auth_time": 1.0})),
    SimpleNamespace(auth_type="session", user=SimpleNamespace(raw_claims={"source": "mobile_device",
                                                                          "auth_time": 1.0})),
    SimpleNamespace(auth_type="session", user=SimpleNamespace(raw_claims={"source": "auth_session"})),
    SimpleNamespace(auth_type="session", user=SimpleNamespace(raw_claims=None)),
])
def test_only_the_session_lanes_context_has_an_auth_time(ctx):
    assert session_lane.session_auth_time(ctx) is None


def test_the_refusal_names_its_reason_and_never_the_token(caplog):
    with caplog.at_level(logging.WARNING, logger="core.auth.session_lane"):
        refused = session_lane.session_ended("expired")

    assert refused.status_code == 401
    assert refused.detail["code"] == "session_ended" and refused.detail["reason"] == "expired"
    assert "expired" in caplog.text and "ams_" not in caplog.text


# --------------------------------------------------------------------------- step-up


def test_step_up_in_enterprise_goes_through_one_seam_that_reads_the_sessions_auth_time(enterprise, monkeypatch):
    """PRD-277 US-406 filled the seam: a session whose person signed in at 1.0 passes for ten minutes."""
    ctx = _Lane()(None, _token())
    assert step_up.check_step_up(ctx, 10, 2.0) is None
    with pytest.raises(step_up.StepUpRequired) as stale:
        step_up.check_step_up(ctx, 10, 1.0 + 10 * 60)
    assert stale.value.content["code"] == step_up.STEP_UP_CODE
    assert stale.value.content["step_up"] == {"start": step_up.SIGN_IN_AGAIN_START, "max_age_minutes": 10}

    seen = []
    monkeypatch.setattr(step_up, "confirm_by_signing_in_again", lambda *args: seen.append(args))
    assert step_up.check_step_up(ctx, 10, 2.0) is None
    assert seen == [(ctx, 10, 2.0)]


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_the_seam_is_never_reached_elsewhere(monkeypatch, edition):
    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    monkeypatch.setattr(step_up, "confirm_by_signing_in_again", lambda *args: pytest.fail("reached"))
    fresh = SimpleNamespace(auth_type="clerk", user=SimpleNamespace(raw_claims={"fva": [0, -1], "iat": 2.0}))
    assert step_up.check_step_up(fresh, 10, 2.0) is None


# --------------------------------------------------------------------------- a dead cookie is cleared


def _cookie_app():
    from fastapi import Depends, FastAPI

    app = FastAPI()
    session_cookie.install_cookie_session_ended_handler(app)

    @app.api_route("/probe", methods=["GET", "POST"])
    def probe(ctx=Depends(hybrid.get_request_context_hybrid)):
        return {"auth_type": ctx.auth_type}

    return app


def _ending(reason):
    def resolve(request, token):
        raise session_lane.session_ended(reason)

    return resolve


def _cleared(answer) -> bool:
    cookie = answer.headers.get("set-cookie", "")
    return cookie.startswith(config.SESSION_COOKIE_NAME + "=") and "max-age=0" in cookie.lower()


@pytest.mark.parametrize("reason", ["expired", "revoked", "unknown"])
def test_an_ended_session_in_the_cookie_is_401_and_clears_the_cookie(enterprise, monkeypatch, reason):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(session_lane, "_resolve_in_own_session", _ending(reason))

    answer = TestClient(_cookie_app()).get("/probe", headers=_cookie(_token()))

    assert answer.status_code == 401 and _cleared(answer)
    assert answer.json() == {"detail": {"code": "session_ended", "reason": reason,
                                        "message": session_lane.SESSION_ENDED_MESSAGE}}
    parts = {part.strip().lower() for part in answer.headers["set-cookie"].split(";")}
    assert {"secure", "httponly", "path=/", "samesite=lax"} <= parts


def _ended_by(revoked_reason):
    def resolve(request, token):
        raise session_lane._refused(SimpleNamespace(revoked_reason=revoked_reason), "revoked")

    return resolve


def test_a_session_a_step_up_replaced_is_the_same_401_and_keeps_the_cookie(enterprise, monkeypatch):
    """The browser's cookie already holds the successor: a request that left earlier must not clear it."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(session_lane, "_resolve_in_own_session", _ended_by(sessions.REVOKED_STEP_UP))

    answer = TestClient(_cookie_app()).get("/probe", headers=_cookie(_token()))

    assert answer.status_code == 401 and "set-cookie" not in answer.headers
    assert answer.json() == {"detail": {"code": "session_ended", "reason": "revoked",
                                        "message": session_lane.SESSION_ENDED_MESSAGE}}


@pytest.mark.parametrize("revoked_reason", [sessions.REVOKED_SIGN_OUT, sessions.REVOKED_BY_ADMIN,
                                            sessions.REVOKED_DEACTIVATED])
def test_a_session_any_other_end_revoked_still_clears_the_cookie(enterprise, monkeypatch, revoked_reason):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(session_lane, "_resolve_in_own_session", _ended_by(revoked_reason))

    answer = TestClient(_cookie_app()).get("/probe", headers=_cookie(_token()))

    assert answer.status_code == 401 and _cleared(answer)
    assert answer.json()["detail"] == {"code": "session_ended", "reason": "revoked",
                                       "message": session_lane.SESSION_ENDED_MESSAGE}


def test_only_the_step_ups_end_is_answered_without_clearing():
    replaced = session_lane._refused(SimpleNamespace(revoked_reason=sessions.REVOKED_STEP_UP), "revoked")
    others = [session_lane._refused(None, "unknown"),
              session_lane._refused(SimpleNamespace(revoked_reason=None), "expired"),
              session_lane._refused(SimpleNamespace(revoked_reason=sessions.REVOKED_BY_ADMIN), "revoked")]

    assert isinstance(replaced, session_lane.SessionReplaced) and replaced.status_code == 401
    assert not any(isinstance(refused, session_lane.SessionReplaced) for refused in others)
    assert all(refused.detail["code"] == "session_ended" for refused in [replaced, *others])


def test_an_ended_session_in_a_header_is_answered_as_before(enterprise, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(session_lane, "_resolve_in_own_session", _ending("expired"))

    answer = TestClient(_cookie_app()).get("/probe", headers={"authorization": "Bearer " + _token()})

    assert answer.status_code == 401 and "set-cookie" not in answer.headers
    assert answer.json()["detail"]["code"] == "session_ended"


def test_any_other_refusal_on_the_cookie_path_keeps_the_cookie(enterprise, monkeypatch):
    from fastapi.testclient import TestClient

    def not_in_workspace(request, token):
        raise HTTPException(status_code=403, detail={"code": session_lane.WORKSPACE_REQUIRED, "message": "m"})

    monkeypatch.setattr(session_lane, "_resolve_in_own_session", not_in_workspace)
    client = TestClient(_cookie_app())

    refused = client.get("/probe", headers=_cookie(_token()))
    cross_site = client.post("/probe", headers={**_cookie(_token()), "origin": ELSEWHERE})

    assert refused.status_code == 403 and "set-cookie" not in refused.headers
    assert cross_site.status_code == 403 and "set-cookie" not in cross_site.headers
    assert cross_site.json()["detail"]["code"] == session_cookie.CROSS_SITE_REFUSED


def test_the_app_installs_the_cookie_handler_once():
    main_source = (ORCH / "main.py").read_text(encoding="utf-8")
    assert main_source.count("install_cookie_session_ended_handler(app)") == 1
    assert issubclass(session_cookie.CookieSessionEnded, HTTPException)
