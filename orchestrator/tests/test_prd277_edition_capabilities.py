"""PRD-277 US-102: the capability truth table for the editions that exist today.

Every capability is pinned for ``local`` and ``saas`` to what the code does now
(docs/PRDS/PRD-277-EDITION-CHECKS.md, one row per call site). Adding a capability
without pinning it here fails ``test_every_capability_is_pinned``; changing an
answer for ``local`` or ``saas`` fails it too. No edition beyond those two has a
row yet: ``enterprise`` arrives with US-103, so ``capabilities_for`` must refuse it.
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
        "session_mode": True,
        "hosts_automatos_cloud": False,
        "calls_automatos_cloud": True,
        "push_needs_consent": True,
        "operator_is_user": True,
        "guards_user_endpoints": False,
        "seeds_first_run_content": True,
        "chat_seeds_user_name": True,
        "install_description": "own_computer",
        "live_agent_named_apart": True,
        "last_workspace_purge_deletes_person": False,
    },
    "saas": {
        "sign_in": "clerk",
        "anonymous_operator": False,
        "multi_person": True,
        "billing": True,
        "profile_editable": False,
        "step_up": "clerk_freshness",
        "device_pairing": False,
        "session_mode": False,
        "hosts_automatos_cloud": True,
        "calls_automatos_cloud": False,
        "push_needs_consent": False,
        "operator_is_user": False,
        "guards_user_endpoints": True,
        "seeds_first_run_content": False,
        "chat_seeds_user_name": False,
        "install_description": "hosted",
        "live_agent_named_apart": False,
        "last_workspace_purge_deletes_person": True,
    },
}

# Each capability against the comparison its call sites made before they moved. A
# capability that is True on local is ``AUTH_EDITION == "local"``; one True on saas is
# ``AUTH_EDITION == "saas"``. The enum-valued ones are checked by the truth table alone.
LOCAL_TRUE = (
    "anonymous_operator", "profile_editable", "device_pairing", "session_mode",
    "calls_automatos_cloud", "push_needs_consent", "operator_is_user",
    "seeds_first_run_content", "chat_seeds_user_name", "live_agent_named_apart",
)
SAAS_TRUE = (
    "multi_person", "billing", "hosts_automatos_cloud", "guards_user_endpoints",
    "last_workspace_purge_deletes_person",
)


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_every_capability_is_pinned(edition):
    assert dataclasses.asdict(capabilities_for(edition)) == TRUTH_TABLE[edition]


def test_the_boolean_lists_cover_every_boolean_capability():
    booleans = {f.name for f in dataclasses.fields(EditionCapabilities) if f.type in (bool, "bool")}
    assert booleans == set(LOCAL_TRUE) | set(SAAS_TRUE)
    assert not set(LOCAL_TRUE) & set(SAAS_TRUE)


def test_only_local_and_saas_have_rows():
    assert KNOWN_EDITIONS == ("local", "saas")
    assert set(CAPABILITIES) == set(TRUTH_TABLE)


@pytest.mark.parametrize("edition", ["enterprise", "oidc", "", "LOCAL", " saas", None])
def test_an_edition_with_no_row_is_refused(edition):
    with pytest.raises(UnknownEdition):
        capabilities_for(edition)


def test_an_unknown_edition_is_a_value_error():
    assert issubclass(UnknownEdition, ValueError)


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_current_follows_the_configured_edition(monkeypatch, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    assert editions.current() == capabilities_for(edition)


@pytest.mark.parametrize("value, edition", [
    ("LOCAL", "local"), (" local ", "local"), ("Saas", "saas"),
    ("enterprise", "saas"), ("something-else", "saas"), ("", "saas"), (None, "saas"),
])
def test_current_reads_an_unknown_or_unfolded_value_as_config_does(monkeypatch, value, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", value)
    assert editions.current() == capabilities_for(edition)
    assert editions.current_name() == edition


@pytest.mark.parametrize("edition", ["local", "saas"])
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
