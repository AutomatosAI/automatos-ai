"""PRD-277 US-102 and US-103: the capability truth table for every edition.

Every capability is pinned for ``local`` and ``saas`` to what the code does now
(docs/PRDS/PRD-277-EDITION-CHECKS.md, one row per call site), and for ``enterprise``
to what was decided (PRD-277 §5.1, D-7, D-12, D-13, and G-1 to G-8 in that file).
Adding a capability without pinning it here fails ``test_every_capability_is_pinned``;
changing an answer fails it too. ``capabilities_for`` refuses any other edition.
``current()`` follows the flag's own rule instead (PRD-277 §5.1): trimmed, lower-cased,
and ``saas`` for a value with no row, as ``config.py`` does at load.
"""
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from core import editions
from core.editions import (
    CAPABILITIES,
    KNOWN_EDITIONS,
    EditionCapabilities,
    UnknownEdition,
    capabilities_for,
)

TRUTH_TABLE = {
    "local": {
        "sign_in": "none",
        "anonymous_operator": True,
        "multi_person": False,
        "billing": False,
        "profile_editable": True,
        "step_up": "skip",
        "device_pairing": True,
        "device_pairing_by_setting": False,
        "session_mode": True,
        "hosts_automatos_cloud": False,
        "calls_theme_revocations": True,
        "calls_cloud_sync": True,
        "calls_cloud_items": True,
        "hosted_services_by_setting": False,
        "push_needs_consent": True,
        "operator_is_user": True,
        "guards_user_endpoints": False,
        "seeds_first_run_content": True,
        "chat_seeds_user_name": True,
        "install_description": "own_computer",
        "live_agent_named_apart": True,
        "last_workspace_purge_deletes_person": False,
        "issues_sessions": False,
        "agent_http_own_hosts_only": False,
        "agent_http_verifies_tls": False,
        "api_on_public_address": False,
        "requires_verified_database_tls": False,
        "device_tokens_age_out": False,
    },
    "saas": {
        "sign_in": "clerk",
        "anonymous_operator": False,
        "multi_person": True,
        "billing": True,
        "profile_editable": False,
        "step_up": "clerk_freshness",
        "device_pairing": False,
        "device_pairing_by_setting": False,
        "session_mode": False,
        "hosts_automatos_cloud": True,
        "calls_theme_revocations": False,
        "calls_cloud_sync": False,
        "calls_cloud_items": False,
        "hosted_services_by_setting": False,
        "push_needs_consent": False,
        "operator_is_user": False,
        "guards_user_endpoints": True,
        "seeds_first_run_content": False,
        "chat_seeds_user_name": False,
        "install_description": "hosted",
        "live_agent_named_apart": False,
        "last_workspace_purge_deletes_person": True,
        "issues_sessions": False,
        "agent_http_own_hosts_only": False,
        "agent_http_verifies_tls": True,
        "api_on_public_address": False,
        "requires_verified_database_tls": False,
        "device_tokens_age_out": False,
    },
    "enterprise": {
        "sign_in": "oidc",                              # §5.1
        "anonymous_operator": False,                    # §5.1: never
        "multi_person": True,                           # §5.1
        "billing": False,                               # §5.1, FR-24
        "profile_editable": False,                      # §5.1: the IdP owns it
        "step_up": "reauth",                            # D-13
        "device_pairing": True,                         # D-7: on while ENTERPRISE_DEVICE_PAIRING is
        "device_pairing_by_setting": True,              # US-601, §5.10: the setting decides it
        "session_mode": False,                          # §5.1
        "hosts_automatos_cloud": False,                 # §5.1
        "calls_theme_revocations": False,               # D-12, US-704: off until the operator switches it on
        "calls_cloud_sync": False,                      # D-12, US-704
        "calls_cloud_items": False,                     # D-12, US-704
        "hosted_services_by_setting": True,             # D-12, US-704: each one is its own setting
        "push_needs_consent": True,                     # G-1
        "operator_is_user": False,                      # G-2
        "guards_user_endpoints": True,                  # G-3
        "seeds_first_run_content": False,               # G-4
        "chat_seeds_user_name": True,                   # G-6
        "install_description": "organisation",          # G-5
        "live_agent_named_apart": False,                # G-7
        "last_workspace_purge_deletes_person": False,   # G-8: never
        "issues_sessions": True,                        # D-3, §5.3 (Wave 3)
        "agent_http_own_hosts_only": True,              # D-12 (#1605): the platform's API is the install's
        "agent_http_verifies_tls": True,                # #1656: TLS verification stays on (AGENTS.md)
        "api_on_public_address": True,                  # US-602: one public origin, FRONTEND_URL
        "requires_verified_database_tls": True,         # PRD-278 D-6: DATABASE_SSLMODE=verify-full
        "device_tokens_age_out": True,                  # US-601: ENTERPRISE_DEVICE_TOKEN_MAX_AGE_DAYS, 90 by default
    },
}

# Each capability against the comparison its call sites made before they moved. A
# capability that is True on local is ``AUTH_EDITION == "local"``; one True on saas is
# ``AUTH_EDITION == "saas"``. The enum-valued ones are checked by the truth table alone.
LOCAL_TRUE = (
    "anonymous_operator", "profile_editable", "device_pairing", "session_mode",
    "calls_theme_revocations", "calls_cloud_sync", "calls_cloud_items", "push_needs_consent", "operator_is_user",
    "seeds_first_run_content", "chat_seeds_user_name", "live_agent_named_apart",
)
SAAS_TRUE = (
    "multi_person", "billing", "hosts_automatos_cloud", "guards_user_endpoints",
    "last_workspace_purge_deletes_person", "agent_http_verifies_tls",
)
# Capabilities no comparison ever made: new with enterprise, so off in local and saas (PRD-277
# Wave 3 sessions, US-704's settings, D-12's agent host list, US-601's pairing setting, US-602's one
# public address, PRD-278 D-6's verified database connection, US-601's device token maximum age). Each
# is False on both, so neither edition's behaviour moves.
ENTERPRISE_ONLY = ("issues_sessions", "hosted_services_by_setting", "agent_http_own_hosts_only",
                   "device_pairing_by_setting", "api_on_public_address", "requires_verified_database_tls",
                   "device_tokens_age_out")


@pytest.mark.parametrize("edition", ["local", "saas", "enterprise"])
def test_every_capability_is_pinned(edition):
    assert dataclasses.asdict(capabilities_for(edition)) == TRUTH_TABLE[edition]


def test_enterprise_never_has_the_operator_and_never_bills():
    """The two answers §5.1 marks "never": no anonymous operator, no billing (FR-3, FR-24)."""
    caps = capabilities_for("enterprise")
    assert caps.anonymous_operator is False and caps.billing is False
    assert caps.last_workspace_purge_deletes_person is False


def test_the_boolean_lists_cover_every_boolean_capability():
    booleans = {f.name for f in dataclasses.fields(EditionCapabilities) if f.type in (bool, "bool")}
    assert booleans == set(LOCAL_TRUE) | set(SAAS_TRUE) | set(ENTERPRISE_ONLY)
    assert not set(LOCAL_TRUE) & set(SAAS_TRUE)
    assert not (set(LOCAL_TRUE) | set(SAAS_TRUE)) & set(ENTERPRISE_ONLY)


def test_exactly_local_saas_and_enterprise_have_rows():
    assert KNOWN_EDITIONS == ("local", "saas", "enterprise")
    assert set(CAPABILITIES) == set(TRUTH_TABLE)


def test_config_accepts_exactly_the_editions_with_rows():
    """config.py's class body cannot import this module, so it names the editions itself."""
    source = (Path(__file__).resolve().parents[1] / "config.py").read_text(encoding="utf-8")
    accepted = '_AUTH_EDITION_RAW in ("' + '", "'.join(KNOWN_EDITIONS) + '")'
    assert accepted in source


@pytest.mark.parametrize("edition", ["enterprize", "oidc", "", "LOCAL", " saas", " enterprise", None])
def test_an_edition_with_no_row_is_refused(edition):
    with pytest.raises(UnknownEdition):
        capabilities_for(edition)


def test_an_unknown_edition_is_a_value_error():
    assert issubclass(UnknownEdition, ValueError)


@pytest.mark.parametrize("edition", ["local", "saas", "enterprise"])
def test_current_follows_the_configured_edition(monkeypatch, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    assert editions.current() == capabilities_for(edition)


@pytest.mark.parametrize("value, edition", [
    ("LOCAL", "local"), (" local ", "local"), ("Saas", "saas"), (" Enterprise ", "enterprise"),
    ("enterprize", "saas"), ("something-else", "saas"), ("", "saas"), (None, "saas"),
])
def test_current_reads_an_unknown_or_unfolded_value_as_config_does(monkeypatch, value, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", value)
    assert editions.current() == capabilities_for(edition)
    assert editions.current_name() == edition


@pytest.mark.parametrize("edition", ["local", "saas", "enterprise"])
def test_current_name_reports_the_configured_edition(monkeypatch, edition):
    """The R rows (profile, /health, discovery, providers, export, traces) report this name."""
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    assert editions.current_name() == edition


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_capabilities_match_todays_edition_comparisons(monkeypatch, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    caps = editions.current()
    for name in LOCAL_TRUE:
        assert getattr(caps, name) is (config.AUTH_EDITION == "local"), name
    for name in SAAS_TRUE:
        assert getattr(caps, name) is (config.AUTH_EDITION == "saas"), name
    for name in ENTERPRISE_ONLY:
        assert getattr(caps, name) is False, name


def test_only_enterprise_issues_sessions():
    """PRD-277 US-301: the session lane exists in enterprise alone."""
    assert [name for name in KNOWN_EDITIONS if capabilities_for(name).issues_sessions] == ["enterprise"]


def test_the_configured_edition_always_has_a_row():
    from config import config

    assert config.AUTH_EDITION in KNOWN_EDITIONS


def test_capabilities_cannot_be_changed():
    caps = capabilities_for("local")
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.billing = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        CAPABILITIES["enterprise"] = caps  # type: ignore[index]


def test_importing_the_module_reads_no_config():
    """config.py must be able to import the table while it is being built."""
    tree = ast.parse(Path(editions.__file__).read_text(encoding="utf-8"))
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    imported = {alias.name for node in top_level if isinstance(node, ast.Import) for alias in node.names}
    imported |= {node.module for node in top_level if isinstance(node, ast.ImportFrom)}
    assert "config" not in imported
