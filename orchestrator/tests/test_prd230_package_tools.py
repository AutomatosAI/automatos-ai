"""PRD-230 US-006 — package platform tools.

PURE tests: the three tools are registered + walker-clean (asserted here and in
test_prd222_tool_schema_walker), and the tool-layer policies (D6 one-package
restriction, D9 over-quota atomicity, manifest passthrough) are exercised with
stubbed DB reads and a spy installer — no Postgres, no real install.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import modules.tools.discovery.handlers_packages as hp
from modules.tools.discovery.action_registry import get_action_registry
from modules.tools.discovery.platform_executor import PlatformActionExecutor

_TOOLS = ("platform_search_packages", "platform_install_package",
          "platform_install_marketplace_agent")


class FakeWS:
    def __init__(self, stage="questions", plan="basic", funnel=None, segment=None):
        self.id = "ws-1"
        self.plan = plan
        self.onboarding = {"stage": stage, "stages": {}, "segment": segment or {},
                           "funnel": funnel or {}}


class FakeDB:
    def __init__(self):
        self.committed = False

    def commit(self):
        self.committed = True


def _pkg(slug="shopify-management", *, agents=1, showcase=True, name="Shopify Management"):
    members = [{"type": "agent", "ref": f"a{i}"} for i in range(agents)]
    members.append({"type": "playbook", "ref": "weekly-numbers"})
    return SimpleNamespace(
        slug=slug, name=name, description="Run the store.",
        members=members, showcase=showcase,
        setup_manifest={"required_connects": [{"app_name": "SHOPIFY"}]},
        vertical_tags=["shopify"], matching={"platforms": ["shopify"]},
    )


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Registration (AC1) — three tools, registry + dispatch
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tool", _TOOLS)
def test_tool_registered_and_dispatched(tool):
    reg = {a.name for a in get_action_registry().get_all()}
    handlers = PlatformActionExecutor(None, None)._handlers
    assert tool in reg
    assert tool in handlers


def test_install_package_schema_requires_slug():
    action = get_action_registry().get("platform_install_package")
    assert action.parameters["required"] == ["slug"]  # walker (C): hard-fail is required


def test_install_agent_schema_documents_either_of():
    action = get_action_registry().get("platform_install_marketplace_agent")
    # walker (B): "Provide agent_id or agent_name" → both named in the description
    assert "agent_id" in action.description and "agent_name" in action.description


# --------------------------------------------------------------------------- #
# platform_search_packages — ranked matches with a contents summary
# --------------------------------------------------------------------------- #


def test_search_returns_ranked_matches_with_contents(monkeypatch):
    pkgs = [_pkg("shopify-management", agents=4), _pkg("support-desk", agents=2,
            name="Support", showcase=False)]
    pkgs[1].matching = {"platforms": ["zendesk"]}
    pkgs[1].vertical_tags = ["support"]
    monkeypatch.setattr("services.marketplace_packages.list_packages", lambda db: pkgs)

    out = _run(hp.search_packages(FakeDB(), "ws-1", {"platforms": ["shopify"], "text": "store"}))
    assert out["success"] is True
    slugs = [m["slug"] for m in out["matches"]]
    assert slugs and slugs[0] == "shopify-management"     # shopify signal ranks it first
    assert "support-desk" not in slugs                    # no signal → excluded
    top = out["matches"][0]
    assert top["contents"]["agent"] == 4                  # contents summary by type
    assert top["required_connects"] == [{"app_name": "SHOPIFY"}]


# --------------------------------------------------------------------------- #
# D6 — one package during onboarding
# --------------------------------------------------------------------------- #


def test_second_package_during_onboarding_is_rejected(monkeypatch):
    ws = FakeWS(stage="proposal", funnel={"package_installed": {"slug": "x", "at": "t"}})
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    spy = _install_spy(monkeypatch)

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "shopify-management"}))
    assert out["success"] is False
    assert out["onboarding_restricted"] is True
    assert "one package during onboarding" in out["message"].lower()
    assert spy["calls"] == 0                               # nothing installed


def test_first_package_during_onboarding_installs_and_records(monkeypatch):
    ws = FakeWS(stage="proposal")                          # no package yet
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=2))
    spy = _install_spy(monkeypatch)
    recorded = {}
    monkeypatch.setattr("services.onboarding_state.record_package_event",
                        lambda db, w, event, slug, **k: recorded.update({"event": event, "slug": slug}))

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "shopify-management"}))
    assert out["success"] is True
    assert spy["calls"] == 1
    assert recorded == {"event": "package_installed", "slug": "shopify-management"}


def test_post_onboarding_install_is_unrestricted(monkeypatch):
    # terminal stage → onboarding inactive → no restriction, no funnel record
    ws = FakeWS(stage="completed", funnel={"package_installed": {"slug": "old", "at": "t"}})
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=1))
    spy = _install_spy(monkeypatch)

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "shopify-development"}))
    assert out["success"] is True
    assert spy["calls"] == 1                               # installed despite prior package


# --------------------------------------------------------------------------- #
# D9 — over-quota is an honest conversation, NEVER a partial install
# --------------------------------------------------------------------------- #


def test_over_quota_returns_plan_conversation_with_zero_installs(monkeypatch):
    # basic tier max_agents=5; a 6-agent package is over quota.
    ws = FakeWS(stage="proposal", plan="basic")
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=6))
    spy = _install_spy(monkeypatch)

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "big-team"}))
    assert out["success"] is False
    assert out["over_quota"] is True
    assert out["package_agents"] == 6 and out["max_agents"] == 5
    assert out["plan_recommendation"] in ("pro", "business")
    assert spy["calls"] == 0                               # ZERO partial registrations


def test_within_quota_installs(monkeypatch):
    ws = FakeWS(stage="completed", plan="basic")           # 5-cap
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 1)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=3))  # 1+3=4 ≤ 5
    spy = _install_spy(monkeypatch)
    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "fits"}))
    assert out["success"] is True and spy["calls"] == 1


# --------------------------------------------------------------------------- #
# Manifest passthrough (AC3) + missing-slug + install-agent either-of
# --------------------------------------------------------------------------- #


def test_response_carries_the_registration_manifest(monkeypatch):
    ws = FakeWS(stage="completed")
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=1))
    _install_spy(monkeypatch)

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "shopify-management"}))
    assert "registrations" in out and "required_connects" in out and "added_count" in out
    assert out["required_connects"][0]["app_name"] == "SHOPIFY"


def test_install_package_missing_slug():
    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {}))
    assert out["success"] is False and "slug" in out["error"].lower()


def test_install_agent_requires_id_or_name():
    out = _run(hp.install_marketplace_agent_tool(FakeDB(), "ws-1", {}))
    assert out["success"] is False
    assert "agent_id" in out["error"] and "agent_name" in out["error"]


# --------------------------------------------------------------------------- #
# spy helper
# --------------------------------------------------------------------------- #


def _install_spy(monkeypatch):
    from services.package_installer import InstallManifest, Registration

    state = {"calls": 0}

    async def fake_install(db, ws, slug, user_id=None):
        state["calls"] += 1
        m = InstallManifest()
        m.add(Registration("agent", "500", "Agent", "cloned"))
        m.add_required_connect("SHOPIFY")
        return m

    monkeypatch.setattr("services.package_installer.install_package", fake_install)
    return state
