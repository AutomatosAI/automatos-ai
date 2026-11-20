"""PRD-277 US-103: ``enterprise`` is a known edition that refuses to start half-configured.

* ``AUTH_EDITION=enterprise`` is accepted (trimmed and lower-cased like the others); an
  unknown value still becomes ``saas``.
* ``REQUIRE_AUTH`` is true in enterprise whatever the environment says.
* Start-up aborts, naming each missing setting and never a value, when the issuer, client id,
  client secret or the install's public address (``FRONTEND_URL``, set in the environment, not
  its localhost default) is unset or blank.
* Last, while ``OIDC_SIGN_IN_IN_THIS_BUILD`` is false, start-up aborts with "not available in
  this build". Flipping that one switch is all that stands between a configured enterprise
  install and a boot, which is what Wave 4 removes.
* ``local`` and ``saas`` never read the OIDC settings.

* A request made after that reload, through the real request-context dependency, sees
  ``REQUIRE_AUTH`` forced on: no credential is 401.

Pure and DB-free: ``config.py`` is reloaded under a controlled environment, as
``test_prd175_auth_edition.py`` does, and its module is restored after every test. The
request-time tests stub the database session.
"""
from __future__ import annotations

import importlib
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core import edition_start_up  # noqa: E402
# Imported before any reload, so each holds the process's own config singleton (restored after).
from core.auth import bearer_registry, hybrid  # noqa: E402
from core.edition_start_up import NOT_AVAILABLE, OIDC_START_UP_SETTINGS, EditionNotReady  # noqa: E402
from core.editions import capabilities_for  # noqa: E402

ENV_KEYS = (
    "AUTH_EDITION", "REQUIRE_AUTH", "OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "FRONTEND_URL",
    "CLI_RUNTIME_ENABLED", "DEFAULT_WORKSPACE_ID", "CLERK_JWKS_URL", "CLERK_SECRET_KEY",
    "LOCAL_REMOTE_ACCESS", "API_PUBLISH_ADDRESS", "WEB_PUBLISH_ADDRESS", "CLIENT_IP_HEADER",
)
# Distinct values, so a test can prove that none of them ever reaches a message. A value that
# stands in for a secret is built from parts, so no literal looks like a credential.
OIDC_ENV = {
    "OIDC_ISSUER": "https://idp.example.test/realms/prd277-issuer-marker",
    "OIDC_CLIENT_ID": "prd277-client-id-marker",
    "OIDC_CLIENT_SECRET": "prd277" + "-" + "client" + "-" + "marker",
    "FRONTEND_URL": "https://automatos.example.test",
    # Wave 4: where sign-in is OIDC, the address it limits and audits by must come from the proxy.
    "CLIENT_IP_HEADER": "X-Real-IP",
}
STAND_IN_CLERK_SECRET = "prd277" + "-" + "stand-in"
LOCAL_WS = "00000000-0000-0000-0000-0000000277aa"
ENTERPRISE_WS = "00000000-0000-0000-0000-0000000277ab"


@pytest.fixture(autouse=True)
def _restore_config_module():
    """Put ``config``'s attributes back after a reload, so the fresh singleton never leaks."""
    import config as config_mod

    saved = dict(config_mod.__dict__)
    try:
        yield
    finally:
        config_mod.__dict__.clear()
        config_mod.__dict__.update(saved)


def _fresh_config(**env):
    """Reload config.py with only ``env`` set among ``ENV_KEYS`` and return its singleton."""
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update({key: value for key, value in env.items() if value is not None})
    try:
        import config as config_mod

        importlib.reload(config_mod)
        return config_mod.config
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _enterprise(**overrides):
    """An enterprise config with every OIDC setting set, less or plus ``overrides``."""
    return _fresh_config(AUTH_EDITION="enterprise", **{**OIDC_ENV, **overrides})


def _refusal(cfg) -> str:
    with pytest.raises(EditionNotReady) as refused:
        cfg.validate_auth_edition()
    return str(refused.value)


def _assert_no_value_in(message: str) -> None:
    for value in OIDC_ENV.values():
        assert value not in message


# --------------------------------------------------------------------------- the flag


@pytest.mark.parametrize("raw", ["enterprise", " Enterprise ", "ENTERPRISE"])
def test_enterprise_is_accepted(raw):
    assert _fresh_config(AUTH_EDITION=raw).AUTH_EDITION == "enterprise"


@pytest.mark.parametrize("raw", ["enterprize", "oidc", "banana"])
def test_an_unknown_value_still_becomes_saas(raw):
    cfg = _fresh_config(AUTH_EDITION=raw)
    assert cfg.AUTH_EDITION == "saas"
    assert cfg.REQUIRE_AUTH is True


@pytest.mark.parametrize("require_auth", [None, "false", "0", "no", "FALSE", "true"])
def test_require_auth_is_on_in_enterprise_whatever_the_environment_says(require_auth):
    assert _fresh_config(AUTH_EDITION="enterprise", REQUIRE_AUTH=require_auth).REQUIRE_AUTH is True


def test_local_and_saas_keep_their_require_auth():
    assert _fresh_config(AUTH_EDITION="local", REQUIRE_AUTH="true").REQUIRE_AUTH is False
    assert _fresh_config(AUTH_EDITION="saas", REQUIRE_AUTH="false").REQUIRE_AUTH is False


def test_the_oidc_settings_are_read_trimmed_and_empty_by_default():
    cfg = _fresh_config(OIDC_ISSUER="  https://idp.example.test  ")
    assert cfg.OIDC_ISSUER == "https://idp.example.test"
    assert cfg.OIDC_CLIENT_ID == "" and cfg.OIDC_CLIENT_SECRET == ""


# --------------------------------------------------------------------------- the missing settings


def test_the_four_settings_are_the_issuer_client_id_secret_and_public_address():
    assert OIDC_START_UP_SETTINGS == ("OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "FRONTEND_URL")


@pytest.mark.parametrize("missing", OIDC_START_UP_SETTINGS)
def test_start_up_names_the_missing_setting_and_never_a_value(missing):
    message = _refusal(_enterprise(**{missing: None}))

    assert missing in message and "AUTH_EDITION=enterprise" in message
    assert [name for name in OIDC_START_UP_SETTINGS if name in message] == [missing]
    assert "not available in this build" not in message
    _assert_no_value_in(message)


@pytest.mark.parametrize("missing", OIDC_START_UP_SETTINGS)
def test_a_blank_setting_is_unset(missing):
    assert missing in _refusal(_enterprise(**{missing: "   "}))


def test_every_missing_setting_is_named_at_once():
    message = _refusal(_fresh_config(AUTH_EDITION="enterprise"))

    assert all(name in message for name in OIDC_START_UP_SETTINGS)
    assert "not available in this build" not in message


def test_the_public_address_must_be_set_not_defaulted():
    cfg = _enterprise(FRONTEND_URL=None)

    assert cfg.FRONTEND_URL  # the developer's localhost default is still there for other code
    assert "FRONTEND_URL" in _refusal(cfg)


# --------------------------------------------------------------------------- not in this build


def test_a_fully_configured_enterprise_is_not_available_in_this_build():
    message = _refusal(_enterprise())

    assert message == NOT_AVAILABLE and "not available in this build" in message
    _assert_no_value_in(message)


def test_the_not_available_abort_is_the_last_check():
    """With every check before it satisfied, the abort is what stops a boot, and only it."""
    cfg = _enterprise()

    assert _refusal(cfg) == NOT_AVAILABLE  # every OIDC setting set: nothing earlier refused
    for missing in OIDC_START_UP_SETTINGS:  # any earlier refusal wins over it
        assert _refusal(_enterprise(**{missing: None})) != NOT_AVAILABLE


def test_validate_security_runs_the_edition_check_and_lets_its_abort_stop_the_boot(monkeypatch):
    """The boot path: ``validate_security`` calls ``validate_auth_edition`` and does not swallow it.

    The check is replaced with a recorder that raises, and the one check before it (the vector
    plane's) is stubbed, so the result does not depend on any setting CI happens to have.
    """
    cfg = _enterprise()
    calls = []

    def edition_check():
        calls.append("validate_auth_edition")
        raise EditionNotReady(NOT_AVAILABLE)

    monkeypatch.setattr(cfg, "assert_vector_config_integrity", lambda: None)
    monkeypatch.setattr(cfg, "validate_auth_edition", edition_check)

    with pytest.raises(EditionNotReady) as refused:
        cfg.validate_security()

    assert calls == ["validate_auth_edition"] and str(refused.value) == NOT_AVAILABLE


# --------------------------------------------------------------------------- at request time


@pytest.mark.asyncio
async def test_a_request_sees_require_auth_forced_on(monkeypatch):
    """``REQUIRE_AUTH=false`` in an enterprise environment still answers a bare request 401.

    ``config.py`` is reloaded with that environment and the request is made through the real
    dependency. Only the modules that hold the config singleton are pointed at the reloaded
    one, as a fresh process would hold it; ``REQUIRE_AUTH`` itself is never set by the test.
    """
    cfg = _enterprise(REQUIRE_AUTH="false")
    _point_the_request_lane_at(monkeypatch, cfg)

    with pytest.raises(HTTPException) as refused:
        await hybrid.get_request_context_hybrid(_bare_request())

    assert refused.value.status_code == 401 and refused.value.detail == "Authentication required"


@pytest.mark.asyncio
async def test_saas_with_require_auth_off_still_takes_the_bare_request(monkeypatch):
    """The control: the same reload on saas leaves the lane open, so the test above is not vacuous."""
    cfg = _fresh_config(AUTH_EDITION="saas", REQUIRE_AUTH="false")
    _point_the_request_lane_at(monkeypatch, cfg)

    ctx = await hybrid.get_request_context_hybrid(_bare_request())

    assert ctx.auth_type == "anonymous" and str(ctx.workspace_id) == ENTERPRISE_WS


def _point_the_request_lane_at(monkeypatch, cfg):
    for module in (hybrid, bearer_registry):
        monkeypatch.setattr(module, "config", cfg)
    monkeypatch.setattr(hybrid, "SessionLocal", MagicMock(return_value=MagicMock()))
    monkeypatch.setattr(hybrid, "_assert_workspace_usable", lambda db, workspace_id, is_admin=False: None)


def _bare_request():
    """No credential; the workspace is named (which is not a credential) so no setting picks it."""
    return SimpleNamespace(method="GET", headers={"x-workspace-id": ENTERPRISE_WS}, query_params={},
                           state=SimpleNamespace())


def test_the_one_switch_is_all_that_stands_between_a_configured_install_and_a_boot(monkeypatch):
    assert edition_start_up.OIDC_SIGN_IN_IN_THIS_BUILD is False
    monkeypatch.setattr(edition_start_up, "OIDC_SIGN_IN_IN_THIS_BUILD", True)

    _enterprise().validate_auth_edition()  # must not raise
    assert "OIDC_ISSUER" in _refusal(_enterprise(OIDC_ISSUER=None))  # the settings still count


def test_session_mode_is_refused_in_enterprise():
    with pytest.raises(RuntimeError) as refused:
        _enterprise(CLI_RUNTIME_ENABLED="true").validate_auth_edition()

    assert "CLI_RUNTIME_ENABLED" in str(refused.value)


def test_enterprise_needs_no_clerk_and_no_default_workspace(monkeypatch):
    monkeypatch.setattr(edition_start_up, "OIDC_SIGN_IN_IN_THIS_BUILD", True)
    cfg = _enterprise()

    assert not cfg.CLERK_JWKS_URL and not cfg.DEFAULT_WORKSPACE_ID
    cfg.validate_auth_edition()  # must not raise


# --------------------------------------------------------------------------- local and saas


def test_saas_and_local_never_read_the_oidc_settings():
    _fresh_config(AUTH_EDITION="saas", CLERK_JWKS_URL="https://clerk.example.test/.well-known/jwks.json",
                  CLERK_SECRET_KEY=STAND_IN_CLERK_SECRET).validate_auth_edition()
    _fresh_config(AUTH_EDITION="local", DEFAULT_WORKSPACE_ID=LOCAL_WS).validate_auth_edition()


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_check_start_up_passes_every_other_edition_with_nothing_set(edition):
    assert edition_start_up.check_start_up(capabilities_for(edition), {}) is None


def test_check_start_up_reads_only_the_settings_it_is_handed():
    with pytest.raises(EditionNotReady) as refused:
        edition_start_up.check_start_up(capabilities_for("enterprise"), {"OIDC_ISSUER": "x", "FRONTEND_URL": "y"})

    assert "OIDC_CLIENT_ID and OIDC_CLIENT_SECRET are unset" in str(refused.value)
