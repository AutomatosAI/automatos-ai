"""PRD-278 D-6 (US-301): the enterprise edition starts only on a database connection that checks the server.

``core.edition_start_up.check_database_tls`` reads the edition's ``requires_verified_database_tls``
capability, never its name. Where it is on (``enterprise``), start-up refuses unless
``DATABASE_SSLMODE`` is ``verify-full``, read trimmed and in lower case as ``core.database.tls``
reads it; the refusal names the setting and never its value. ``config.validate_auth_edition`` calls
it last, so every earlier refusal keeps its place. ``local`` and ``saas`` read nothing: with
neither setting they start exactly as before.

``verify-full`` without a CA file is refused in every edition by ``core.database.tls`` itself
(``test_prd278_w3_database_tls.py``), so the two rules together mean the enterprise API checks the
server against the operator's CA. No database is reached.
"""
from __future__ import annotations

import ast
import dataclasses

import pytest

from core import edition_start_up
from core.database import tls
from core.edition_start_up import (
    DATABASE_TLS_START_UP_SETTINGS,
    REQUIRED_DATABASE_SSLMODE,
    EditionNotReady,
    check_database_tls,
)
from core.editions import capabilities_for
from tests.test_prd277_enterprise_start_up import (  # noqa: F401  (the fixture restores config after a reload)
    OIDC_ENV,
    _enterprise,
    _fresh_config,
    _restore_config_module,
)

SETTING = "DATABASE_SSLMODE"
REFUSAL_START = f"AUTH_EDITION=enterprise cannot start: {SETTING} must be {REQUIRED_DATABASE_SSLMODE}"
# Every mode but verify-full, and forms that are not a mode at all.
NOT_VERIFIED = [None, "", "   ", "disable", "allow", "prefer", "require", "verify-ca", "verify", "full"]
LOCAL_WS = "00000000-0000-0000-0000-0000000278d1"
STAND_IN_CLERK_SECRET = "prd278" + "-" + "stand-in"


def _refusal(capabilities, value) -> str:
    with pytest.raises(EditionNotReady) as refused:
        check_database_tls(capabilities, {SETTING: value})
    return str(refused.value)


# --------------------------------------------------------------------------- the rule


def test_the_rule_reads_one_setting_and_requires_the_mode_tls_calls_verifying():
    assert DATABASE_TLS_START_UP_SETTINGS == (SETTING,)
    assert REQUIRED_DATABASE_SSLMODE == tls.SERVER_VERIFYING_MODE
    assert {mode for mode in tls.SSLMODES if tls.DatabaseTLS(sslmode=mode).verifies_server} == {
        REQUIRED_DATABASE_SSLMODE}


@pytest.mark.parametrize("value", NOT_VERIFIED)
def test_enterprise_refuses_any_other_mode_naming_the_setting_never_the_value(value):
    message = _refusal(capabilities_for("enterprise"), value)
    assert message.startswith(REFUSAL_START)
    assert "DATABASE_SSLROOTCERT" in message
    if (value or "").strip():
        assert value not in message.replace(REQUIRED_DATABASE_SSLMODE, "")


@pytest.mark.parametrize("value", ["verify-full", " VERIFY-FULL ", "Verify-Full"])
def test_enterprise_accepts_verify_full_read_as_tls_reads_it(value):
    assert check_database_tls(capabilities_for("enterprise"), {SETTING: value}) is None


@pytest.mark.parametrize("edition", ["local", "saas"])
@pytest.mark.parametrize("value", NOT_VERIFIED)
def test_local_and_saas_read_nothing(edition, value):
    assert check_database_tls(capabilities_for(edition), {SETTING: value}) is None
    assert check_database_tls(capabilities_for(edition), {}) is None


def test_the_rule_follows_the_capability_not_the_edition_name():
    required = dataclasses.replace(capabilities_for("local"), requires_verified_database_tls=True)
    assert _refusal(required, "require").startswith(REFUSAL_START)
    relaxed = dataclasses.replace(capabilities_for("enterprise"), requires_verified_database_tls=False)
    assert check_database_tls(relaxed, {SETTING: None}) is None


def test_the_module_still_imports_neither_config_nor_database_code():
    """config.py hands the value in, so the rule's constant is pinned to tls's above, not imported."""
    with open(edition_start_up.__file__, encoding="utf-8") as module:
        tree = ast.parse(module.read())
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "config" not in imported
    assert not any(name.startswith("core.database") for name in imported)


# --------------------------------------------------------------------------- at boot


@pytest.mark.parametrize("value", [None, "require", "verify-ca"])
def test_an_enterprise_boot_without_verify_full_is_refused(value):
    with pytest.raises(EditionNotReady) as refused:
        _enterprise(DATABASE_SSLMODE=value).validate_auth_edition()
    message = str(refused.value)
    assert message.startswith(REFUSAL_START)
    for other in OIDC_ENV.values():
        assert other == REQUIRED_DATABASE_SSLMODE or other not in message


def test_an_enterprise_boot_with_verify_full_starts():
    _enterprise(DATABASE_SSLMODE="verify-full").validate_auth_edition()  # must not raise


def test_the_database_rule_comes_after_every_sign_in_refusal():
    with pytest.raises(EditionNotReady) as refused:
        _enterprise(DATABASE_SSLMODE=None, OIDC_CLIENT_SECRET=None).validate_auth_edition()
    assert "OIDC_CLIENT_SECRET is unset" in str(refused.value)
    assert SETTING not in str(refused.value)


def test_local_and_saas_boot_with_neither_setting_exactly_as_before():
    _fresh_config(AUTH_EDITION="local", DEFAULT_WORKSPACE_ID=LOCAL_WS).validate_auth_edition()
    _fresh_config(AUTH_EDITION="saas", CLERK_JWKS_URL="https://clerk.example.test/.well-known/jwks.json",
                  CLERK_SECRET_KEY=STAND_IN_CLERK_SECRET).validate_auth_edition()
