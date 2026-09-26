"""F200 (a) — a package is installed as a package, never one agent at a time.

Night 6 (04:06Z): the owner said "install Shopify Management as it is". Auto
called platform_install_marketplace_agent with that name; the refusal ("Marketplace
agent not found … Closest: 'Shopify Support Agent', 'Shopify Operations Manager',
'Shopify Business Analyst'") sent it to install those agents one by one (#327-329:
no skills, no tools, no store-URL question, no connect card, no plan check). A
package's name is now answered with the package's own install.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

import modules.tools.discovery.handlers_packages as hp


def _pkg(agents=4):
    members = [{"type": "agent", "ref": f"a{i}"} for i in range(agents)]
    return NS(slug="shopify-management", name="Shopify Management", members=members)


@pytest.fixture
def catalogue(monkeypatch):
    import modules.tools.discovery.not_found_candidates as nf
    import services.marketplace_packages as mp
    import services.package_installer as pi

    async def _no_such_agent(db, workspace_id, ref, user_id=None):
        raise pi.PackageInstallError(f"Marketplace agent not found: {ref}")

    async def _closest(*a, **k):
        return [{"name": "Shopify Support Agent"}, {"name": "Shopify Operations Manager"},
                {"name": "Shopify Business Analyst"}]

    monkeypatch.setattr(pi, "install_marketplace_agent", _no_such_agent)
    monkeypatch.setattr(nf, "find_candidates", _closest)
    monkeypatch.setattr(mp, "get_by_slug", lambda db, slug: _pkg() if slug == "shopify-management" else None)
    monkeypatch.setattr(mp, "list_packages", lambda db: [_pkg()])


@pytest.mark.parametrize("asked", ["Shopify Management", "shopify-management", " shopify management "])
def test_a_package_asked_for_as_an_agent_is_pointed_at_its_install(catalogue, asked):
    out = asyncio.run(hp.install_marketplace_agent_tool(None, "ws-1", {"agent_name": asked}))

    assert out["success"] is False and out.get("use_package") == "shopify-management"   # night: three agents offered
    assert "platform_install_package" in out["error"] and "4 agents" in out["error"]
    assert "one at a time" in out["error"]


def test_an_agent_that_is_simply_missing_still_names_the_closest(catalogue):
    out = asyncio.run(hp.install_marketplace_agent_tool(None, "ws-1", {"agent_name": "Store Poet"}))

    assert out["success"] is False and "use_package" not in out
    assert "Shopify Support Agent" in str(out)


def test_the_single_agent_tool_says_a_package_team_is_installed_whole():
    from modules.tools.discovery.action_registry import get_action_registry

    description = get_action_registry().get("platform_install_marketplace_agent").description
    assert "platform_install_package" in description and "never one agent at a time" in description
