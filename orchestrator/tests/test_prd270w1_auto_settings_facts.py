"""PRD-270 US-270-05b: Auto knows the local edition's Settings › Automatos Cloud tab.

The tab (where Marketplace themes sync from, the last Sync, Sync now) is the local edition's only,
and the page hides it while the Sync is off (``AUTOMATOS_CLOUD_URL=`` empty). Auto's prompt
(modules/context/sections/product_facts.py) names it in the same cases, after the shared tabs, so
Auto never sends an owner to a tab that isn't there. F232's test holds the label sets to the page.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ORCH = Path(__file__).resolve().parents[1]
SETTINGS_PANEL = ORCH.parent / "frontend" / "components" / "settings" / "SettingsPanel.tsx"
CLOUD = "https://api.automatos.app"


@pytest.fixture
def install(monkeypatch):
    from config import config

    def _set(local: bool, cloud_url: str) -> None:
        monkeypatch.setattr(config, "AUTH_EDITION", "local" if local else "saas")
        monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", cloud_url)
    return _set


def test_a_local_install_with_the_sync_on_names_automatos_cloud_last(install):
    from modules.context.sections.product_facts import edition_facts

    install(True, CLOUD)
    facts = edition_facts()
    assert "Settings has these tabs: Profile, Session mode, Orchestrator," in facts
    assert "Socials, Appearance, Widget SDK, Automatos Cloud. There is no Billing" in facts


@pytest.mark.parametrize("cloud_url", ["", "   "])
def test_a_local_install_with_the_sync_off_does_not_name_it(install, cloud_url):
    from modules.context.sections.product_facts import edition_facts

    install(True, cloud_url)
    facts = edition_facts()
    assert "Automatos Cloud" not in facts
    assert "Appearance, Widget SDK. There is no Billing" in facts
    # A local install has Model keys but not the hosted-only API keys tab.
    assert "Model keys" in facts and "API keys" not in facts


def test_the_hosted_edition_never_names_it(install):
    from modules.context.sections.product_facts import edition_facts

    install(False, CLOUD)
    facts = edition_facts()
    assert "Automatos Cloud" not in facts
    assert "Widget SDK, API keys, Billing." in facts


def test_the_local_tab_list_is_the_local_tabs_and_the_shared_ones():
    from modules.context.sections.product_facts import (
        CLOUD_SYNC_TAB,
        LOCAL_LEADING_TABS,
        LOCAL_SETTINGS_TABS,
        SETTINGS_TABS,
        local_settings_tabs,
    )

    assert LOCAL_SETTINGS_TABS == (*LOCAL_LEADING_TABS, CLOUD_SYNC_TAB)
    assert local_settings_tabs(True) == (*LOCAL_LEADING_TABS, *SETTINGS_TABS, CLOUD_SYNC_TAB)
    assert local_settings_tabs(False) == (*LOCAL_LEADING_TABS, *SETTINGS_TABS)


def test_the_page_shows_the_tab_only_where_the_install_syncs_while_its_gate_says_so():
    """PRD-277 US-704: the tab's one gate is ``useHasAutomatosCloudTab``, which asks only where this install
    syncs (the local edition, or an enterprise install that switched it on), never by the page's edition test."""
    page = SETTINGS_PANEL.read_text(encoding="utf-8")
    panel = (SETTINGS_PANEL.parent / "automatos-cloud-panel.tsx").read_text(encoding="utf-8")
    assert "{ value: 'automatos-cloud', label: 'Automatos Cloud', icon: Cloud }," in page
    assert ".filter((tab) => tab.value !== 'automatos-cloud' || hasCloudTab)" in page
    assert "const hasCloudTab = useHasAutomatosCloudTab()" in page
    assert "const asks = useHostedService('cloudSync') === true && canManageThemes(workspace?.role)" in panel
    assert "return asks && status.data?.enabled === true" in panel
