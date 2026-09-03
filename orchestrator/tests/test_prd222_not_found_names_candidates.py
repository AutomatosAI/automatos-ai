"""PRD-222 — a marketplace miss names the closest real entries (live-test 2026-09-02).

Auto invented names ('shopify-store-manager', 'shopify-tools',
'customer-support-agent-skill', 'Restaurant Customer Service Agent'), got a bare
"not found" and guessed again or stalled. The error now says what exists.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock
from uuid import uuid4

from modules.tools.discovery import handlers_marketplace as hm
from modules.tools.discovery import handlers_packages as hp
from modules.tools.discovery.not_found_candidates import candidate_terms, find_candidates, not_found_error


def _run(coro):
    return asyncio.run(coro)


def _db_first_none():
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.query.return_value.filter.return_value.filter.return_value.first.return_value = None
    return db


# ── helper ────────────────────────────────────────────────────────────────

def test_terms_drop_generic_words_and_keep_order():
    assert candidate_terms("shopify-store-manager") == ["shopify", "store"]
    assert candidate_terms("Restaurant Customer Service Agent") == ["restaurant", "customer"]
    assert candidate_terms("customer-support-agent-skill") == ["customer", "support"]
    assert candidate_terms("agent") == ["agent"]  # only generic words → keep the tokens
    assert candidate_terms("") == []


def test_not_found_error_names_candidates_or_says_nothing_resembles():
    out = not_found_error("Plugin", "shopify-tools", [{"slug": "shopify-plugin", "name": "Shopify"}],
                          search_tool="platform_browse_marketplace_plugins")
    assert out["success"] is False
    assert "'shopify-plugin' (Shopify)" in out["error"] and "never guess a name" in out["error"]
    assert out["candidates"][0]["slug"] == "shopify-plugin" and out["requested"] == "shopify-tools"
    out = not_found_error("Plugin", "x", [], search_tool="t")
    assert "nothing in the marketplace resembles it" in out["error"] and out["candidates"] == []


def test_find_candidates_dedupes_and_survives_a_failing_browse():
    calls = []

    async def browse(db, ws, params):
        calls.append(params["search"])
        if params["search"] == "store":
            raise RuntimeError("db down")
        return {"plugins": [{"slug": "shopify-plugin", "name": "Shopify"},
                            {"slug": "shopify-plugin", "name": "Shopify"}]}

    out = _run(find_candidates(browse, None, uuid4(), "shopify-store-manager", list_key="plugins"))
    assert out == [{"slug": "shopify-plugin", "name": "Shopify"}]
    assert calls == ["shopify", "store"]


# ── the four sites ─────────────────────────────────────────────────────────

def test_install_plugin_miss_names_real_plugins(monkeypatch):
    async def browse(db, ws, params):
        return {"plugins": [{"slug": "shopify-plugin", "name": "Shopify"}]}

    monkeypatch.setattr(hm, "browse_marketplace_plugins", browse)
    out = _run(hm.install_plugin(_db_first_none(), uuid4(), {"plugin_slug": "shopify-tools"}))
    assert out["success"] is False and "'shopify-plugin'" in out["error"]
    assert out["requested"] == "shopify-tools"


def test_install_skill_miss_names_real_skills(monkeypatch):
    async def browse(db, ws, params):
        return {"skills": [{"id": 7, "name": "shopify-support"}]}

    monkeypatch.setattr(hm, "browse_marketplace_skills", browse)
    out = _run(hm.install_skill(_db_first_none(), uuid4(), {"skill_name": "customer-support-agent-skill"}))
    assert out["success"] is False and "'shopify-support'" in out["error"]


def test_install_marketplace_agent_miss_names_real_agents(monkeypatch):
    from services.package_installer import PackageInstallError

    async def missing(db, ws, ref, user_id=None):
        raise PackageInstallError(f"Marketplace agent not found: {ref}")

    async def browse(db, ws, params):
        return {"agents": [{"id": 12, "name": "Customer Support Agent"}]}

    monkeypatch.setattr("services.package_installer.install_marketplace_agent", missing)
    monkeypatch.setattr(hm, "browse_marketplace_agents", browse)
    out = _run(hp.install_marketplace_agent_tool(MagicMock(), uuid4(), {"agent_name": "Restaurant Customer Service Agent"}))
    assert out["success"] is False
    assert "'Customer Support Agent'" in out["error"] and "platform_browse_marketplace_agents" in out["error"]


def test_install_marketplace_agent_other_errors_are_unchanged(monkeypatch):
    from services.package_installer import PackageInstallError

    async def quota(db, ws, ref, user_id=None):
        raise PackageInstallError("quota exceeded")

    monkeypatch.setattr("services.package_installer.install_marketplace_agent", quota)
    out = _run(hp.install_marketplace_agent_tool(MagicMock(), uuid4(), {"agent_id": 3}))
    assert out == {"success": False, "error": "quota exceeded"}


def test_install_package_miss_names_the_catalogue(monkeypatch):
    class _WS:
        def __init__(self):
            self.id = uuid4()
            self.onboarding = {}

    class _Pkg:
        def __init__(self, slug, name):
            self.slug, self.name = slug, name

    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: _WS())
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: None)
    monkeypatch.setattr("services.marketplace_packages.list_packages",
                        lambda db: [_Pkg("shopify-management", "Shopify Management"),
                                    _Pkg("shopify-development", "Shopify Development")])
    out = _run(hp.install_package_tool(MagicMock(), uuid4(), {"slug": "shopify-store-manager"}))
    assert out["success"] is False
    assert "'shopify-management' (Shopify Management)" in out["error"]
    assert out["candidates"][0]["slug"] == "shopify-management"
