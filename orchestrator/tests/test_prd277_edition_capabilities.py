"""PRD-277 US-102: the capability truth table for the editions that exist today.

Every capability is pinned for ``local`` and ``saas`` to what the code does now
(docs/PRDS/PRD-277-EDITION-CHECKS.md, one row per call site). Adding a capability
without pinning it here fails ``test_every_capability_is_pinned``; changing an
answer for ``local`` or ``saas`` fails it too. No edition beyond those two has a
row yet: ``enterprise`` arrives with US-103, so here it must be refused.
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
    },
}

# Each capability against the comparison its call sites make today. A capability
# that is True on local is today's ``IS_LOCAL_EDITION``; one True on saas is today's
# ``IS_SAAS_EDITION``. The enum-valued two are checked by the truth table alone.
LOCAL_TRUE = (
    "anonymous_operator", "profile_editable", "device_pairing", "session_mode",
    "calls_automatos_cloud", "push_needs_consent", "operator_is_user",
)
SAAS_TRUE = ("multi_person", "billing", "hosts_automatos_cloud", "guards_user_endpoints")


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


@pytest.mark.parametrize("edition", ["local", "saas"])
def test_capabilities_match_todays_edition_helpers(monkeypatch, edition):
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    caps = editions.current()
    for name in LOCAL_TRUE:
        assert getattr(caps, name) is config.IS_LOCAL_EDITION, name
    for name in SAAS_TRUE:
        assert getattr(caps, name) is config.IS_SAAS_EDITION, name


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
