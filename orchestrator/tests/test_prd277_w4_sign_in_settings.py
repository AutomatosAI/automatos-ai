"""PRD-277 Wave 4 (§5.10): the sign-in settings can stop only an edition that signs in with OIDC.

``config.py`` reads ``OIDC_SCOPES``, ``OIDC_CLAIM_EMAIL``, ``OIDC_CLAIM_NAME``,
``OIDC_CA_BUNDLE_FILE``, ``ENTERPRISE_DEFAULT_ROLE``, ``OIDC_ENDPOINT_HOSTS`` and
``OIDC_PENDING_SIGN_INS_MAX`` so that importing it never raises. Proven by reloading ``config.py``
under each value:

* their defaults are the PRD's (``openid profile email``, ``email``, ``name``, none, ``none``, none,
  10000) and every name is in ``reports/config-surface.json``;
* in ``local`` and ``saas`` a blank or malformed value imports and starts normally;
* in ``enterprise`` each is refused at start-up, naming the setting and never the value: an issuer
  or a public address that is not https, scopes without ``openid`` or with a character a scope may
  not hold, an empty claim name, a CA file that cannot be read, a default role that is not
  ``none``, viewer, editor or admin, a role without ``DEFAULT_WORKSPACE_ID``, no
  ``CLIENT_IP_HEADER`` (the address sign-in limits and audits by would be the caller's to write),
  an endpoint host that is not a host name (a scheme, a port, a wildcard or an address), and a
  ceiling on unfinished sign-ins that is not a whole number of at least one;
* a good value of each passes, so only the build switch stops the start.
"""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from core.edition_start_up import NOT_AVAILABLE, OIDC_SIGN_IN_SETTINGS, EditionNotReady  # noqa: E402

ORCH = Path(__file__).resolve().parents[1]
NEW_SETTINGS = ("OIDC_SCOPES", "OIDC_CLAIM_EMAIL", "OIDC_CLAIM_NAME", "OIDC_CA_BUNDLE_FILE", "ENTERPRISE_DEFAULT_ROLE",
                "OIDC_ENDPOINT_HOSTS", "OIDC_PENDING_SIGN_INS_MAX")
ENV_KEYS = (
    "AUTH_EDITION", "REQUIRE_AUTH", "OIDC_ISSUER", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET", "FRONTEND_URL",
    "CLI_RUNTIME_ENABLED", "DEFAULT_WORKSPACE_ID", "CLERK_JWKS_URL", "CLERK_SECRET_KEY",
    "LOCAL_REMOTE_ACCESS", "API_PUBLISH_ADDRESS", "WEB_PUBLISH_ADDRESS", "CLIENT_IP_HEADER", *NEW_SETTINGS,
)
OIDC_ENV = {
    "OIDC_ISSUER": "https://idp.example.test/realms/prd277-w4",
    "OIDC_CLIENT_ID": "prd277-w4-client",
    "OIDC_CLIENT_SECRET": "prd277" + "-" + "w4" + "-" + "marker",
    "FRONTEND_URL": "https://automatos.example.test",
}
EDITION_ENV = {
    "local": {"AUTH_EDITION": "local", "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277d1"},
    "saas": {"AUTH_EDITION": "saas", "CLERK_JWKS_URL": "https://clerk.example.test/.well-known/jwks.json",
             "CLERK_SECRET_KEY": "prd277" + "-" + "w4-stand-in"},
    "enterprise": {"AUTH_EDITION": "enterprise", "CLIENT_IP_HEADER": "X-Real-IP", **OIDC_ENV},
}
# Each a value that must stop enterprise, and must not stop local or saas.
BAD = [
    ("OIDC_SCOPES", "profile email"), ("OIDC_SCOPES", "openid pro\"file"), ("OIDC_SCOPES", "openid \\x"),
    ("OIDC_CLAIM_EMAIL", "   "), ("OIDC_CLAIM_NAME", "   "),
    ("OIDC_CA_BUNDLE_FILE", "/nonexistent/prd277-w4-ca.pem"),
    ("ENTERPRISE_DEFAULT_ROLE", "owner"), ("ENTERPRISE_DEFAULT_ROLE", "super_admin"),
    ("ENTERPRISE_DEFAULT_ROLE", "editorx"),
    ("OIDC_ISSUER", "http://idp.example.test/realms/prd277-w4"), ("OIDC_ISSUER", "idp.example.test"),
    ("FRONTEND_URL", "http://automatos.example.test"),
    ("CLIENT_IP_HEADER", "   "), ("CLIENT_IP_HEADER", "X Real IP"),
    ("OIDC_ENDPOINT_HOSTS", "https://oauth2.example.test"), ("OIDC_ENDPOINT_HOSTS", "*.example.test"),
    ("OIDC_ENDPOINT_HOSTS", "oauth2.example.test:443"), ("OIDC_ENDPOINT_HOSTS", "good.example.test, 10.0.0.1"),
    ("OIDC_PENDING_SIGN_INS_MAX", "0"), ("OIDC_PENDING_SIGN_INS_MAX", "lots"), ("OIDC_PENDING_SIGN_INS_MAX", "1.5"),
]


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


def test_the_defaults_are_the_prds_and_every_name_is_in_the_surface():
    cfg = _fresh_config(**EDITION_ENV["saas"])

    assert (cfg.OIDC_SCOPES, cfg.OIDC_CLAIM_EMAIL, cfg.OIDC_CLAIM_NAME) == ("openid profile email", "email", "name")
    assert (cfg.OIDC_CA_BUNDLE_FILE, cfg.ENTERPRISE_DEFAULT_ROLE) == ("", "none")
    assert (cfg.OIDC_ENDPOINT_HOSTS, cfg.OIDC_PENDING_SIGN_INS_MAX) == ("", 10_000)
    surface = json.loads((ORCH / "reports" / "config-surface.json").read_text(encoding="utf-8"))["settings"]
    assert set(NEW_SETTINGS) <= set(surface) and surface == sorted(surface)
    assert set(NEW_SETTINGS) <= set(OIDC_SIGN_IN_SETTINGS)


def test_values_are_read_trimmed_and_the_role_folded():
    cfg = _fresh_config(**EDITION_ENV["saas"], OIDC_SCOPES="  openid   email ", ENTERPRISE_DEFAULT_ROLE=" Editor ",
                        OIDC_CLAIM_EMAIL=" mail ")
    assert (cfg.OIDC_SCOPES, cfg.ENTERPRISE_DEFAULT_ROLE, cfg.OIDC_CLAIM_EMAIL) == ("openid email", "editor", "mail")


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize(("name", "value"), BAD + [(name, "") for name in NEW_SETTINGS])
def test_no_sign_in_value_ever_stops_local_or_saas(edition, name, value):
    _fresh_config(**EDITION_ENV[edition], **{name: value}).validate_auth_edition()  # must not raise


@pytest.mark.parametrize(("name", "value"), BAD)
def test_a_bad_sign_in_value_is_the_named_refusal_in_enterprise(name, value):
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**{**EDITION_ENV["enterprise"], name: value}).validate_auth_edition()

    message = str(refused.value)
    assert message.startswith(f"AUTH_EDITION=enterprise cannot start: {name} ")
    assert not value.strip() or value.strip() not in message.split(name, 1)[1]
    assert OIDC_ENV["OIDC_CLIENT_SECRET"] not in message


def test_without_the_client_ip_header_enterprise_is_told_why():
    env = {name: value for name, value in EDITION_ENV["enterprise"].items() if name != "CLIENT_IP_HEADER"}
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**env).validate_auth_edition()

    message = str(refused.value)
    assert message.startswith("AUTH_EDITION=enterprise cannot start: CLIENT_IP_HEADER ")
    assert "limits" in message and "audit log" in message


def test_saas_and_local_still_start_without_the_client_ip_header():
    for edition in ("local", "saas"):
        _fresh_config(**EDITION_ENV[edition]).validate_auth_edition()  # must not raise


def test_a_default_role_needs_the_default_workspace():
    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**EDITION_ENV["enterprise"], ENTERPRISE_DEFAULT_ROLE="viewer").validate_auth_edition()
    assert "ENTERPRISE_DEFAULT_ROLE" in str(refused.value) and "DEFAULT_WORKSPACE_ID" in str(refused.value)


@pytest.mark.parametrize("role", ["none", "viewer", "editor", "admin"])
def test_good_values_reach_only_the_build_switch(role, tmp_path):
    bundle = tmp_path / "ca.pem"
    bundle.write_text("placeholder certificate file", encoding="utf-8")
    env = {**EDITION_ENV["enterprise"], "ENTERPRISE_DEFAULT_ROLE": role, "OIDC_CA_BUNDLE_FILE": str(bundle),
           "OIDC_SCOPES": "openid email groups", "DEFAULT_WORKSPACE_ID": "00000000-0000-0000-0000-0000000277d2",
           "OIDC_ENDPOINT_HOSTS": "oauth2.example.test, Keys.Example.Test.", "OIDC_PENDING_SIGN_INS_MAX": "50"}

    with pytest.raises(EditionNotReady) as refused:
        _fresh_config(**env).validate_auth_edition()

    assert str(refused.value) == NOT_AVAILABLE
