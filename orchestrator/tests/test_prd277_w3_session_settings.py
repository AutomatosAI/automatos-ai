"""PRD-277 Wave 3: a session clock's value can stop only an edition that issues sessions.

``config.py`` reads ``SESSION_IDLE_MINUTES`` and ``SESSION_ABSOLUTE_HOURS`` so that importing it
never raises: unset is the default, a whole number is that number, and anything else is handed on
as the text it was. Proven by reloading ``config.py`` under each value:

* in ``local`` and ``saas`` a blank and a non-numeric value import and start normally;
* in ``enterprise`` each is refused at start-up, naming the setting and never the value;
* an idle limit below the last-seen interval is refused there too, and the interval itself starts.
"""
from __future__ import annotations

import importlib
import os

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core import edition_start_up  # noqa: E402
from core.edition_start_up import EditionNotReady  # noqa: E402

CLOCKS = ("SESSION_IDLE_MINUTES", "SESSION_ABSOLUTE_HOURS")
ENV_KEYS = (
    "AUTH_EDITION", "REQUIRE_AUTH", "OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "FRONTEND_URL",
    "CLI_RUNTIME_ENABLED", "DEFAULT_WORKSPACE_ID", "CLERK_JWKS_URL", "CLERK_SECRET_KEY",
    "LOCAL_REMOTE_ACCESS", "API_PUBLISH_ADDRESS", "WEB_PUBLISH_ADDRESS", "CLIENT_IP_HEADER", *CLOCKS,
)
OIDC_ENV = {
    "OIDC_ISSUER": "https://idp.example.test/realms/prd277-w3",
    "OIDC_CLIENT_ID": "prd277-w3-client",
    "OIDC_CLIENT_SECRET": "prd277" + "-" + "w3" + "-" + "marker",
    "FRONTEND_URL": "https://automatos.example.test",
    # Wave 4: where sign-in is OIDC, the address it limits and audits by must come from the proxy.
    "CLIENT_IP_HEADER": "X-Real-IP",
    # Wave 4 part 2: the workspace sign-ins are recorded in and the first admins own.
    "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277c2",
}
EDITION_ENV = {
    "local": {"AUTH_EDITION": "local", "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277c1"},
    "saas": {"AUTH_EDITION": "saas", "CLERK_JWKS_URL": "https://clerk.example.test/.well-known/jwks.json",
             "CLERK_SECRET_KEY": "prd277" + "-" + "w3-stand-in"},
    "enterprise": {"AUTH_EDITION": "enterprise", **OIDC_ENV},
}
MALFORMED = ("", "   ", "abc", "1.5", "12h")


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
    os.environ.update(env)
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


def test_unset_clocks_are_the_defaults_and_numbers_are_read_as_numbers():
    cfg = _fresh_config(**EDITION_ENV["saas"])
    assert (cfg.SESSION_IDLE_MINUTES, cfg.SESSION_ABSOLUTE_HOURS) == (120, 12)
    cfg = _fresh_config(**EDITION_ENV["saas"], SESSION_IDLE_MINUTES=" 30 ", SESSION_ABSOLUTE_HOURS="8")
    assert (cfg.SESSION_IDLE_MINUTES, cfg.SESSION_ABSOLUTE_HOURS) == (30, 8)


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize("name", CLOCKS)
@pytest.mark.parametrize("value", MALFORMED)
def test_a_malformed_clock_never_stops_an_edition_that_issues_no_sessions(edition, name, value):
    cfg = _fresh_config(**EDITION_ENV[edition], **{name: value})

    assert getattr(cfg, name) == value
    cfg.validate_auth_edition()  # must not raise


@pytest.mark.parametrize("name", CLOCKS)
@pytest.mark.parametrize("value", MALFORMED)
def test_a_malformed_clock_is_the_named_refusal_in_enterprise(name, value):
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**EDITION_ENV["enterprise"], **{name: value}).validate_auth_edition()

    message = str(refused.value)
    assert message.startswith(f"AUTH_EDITION=enterprise cannot start: {name} ")
    assert "whole number" in message
    assert not value.strip() or value.strip() not in message.split(name, 1)[1]


def test_an_idle_limit_below_the_last_seen_interval_is_refused_in_enterprise():
    below = str(edition_start_up.SESSION_IDLE_MINUTES_MIN - 1)
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**EDITION_ENV["enterprise"], SESSION_IDLE_MINUTES=below).validate_auth_edition()

    assert "SESSION_IDLE_MINUTES" in str(refused.value)
    assert f"at least {edition_start_up.SESSION_IDLE_MINUTES_MIN}" in str(refused.value)

    at_floor = str(edition_start_up.SESSION_IDLE_MINUTES_MIN)
    _fresh_config(**EDITION_ENV["enterprise"], SESSION_IDLE_MINUTES=at_floor).validate_auth_edition()  # starts
