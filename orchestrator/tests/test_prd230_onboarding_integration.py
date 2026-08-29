"""PRD-230 US-009 — onboarding integration: offer ONE package, then the D7 guide.

Two halves, both PURE:
  - SECTION content (modules/context/sections/onboarding.py): the proposal offers a
    matched package BY NAME, carries the defer-pick rule (D5) and the no-match
    custom-design route (D10); the build stage installs the package and narrates
    the manifest with guided connects; the boom stage activates (first Playbook +
    checklist). Budget re-measured (proposal is the new largest variant).
  - FUNNEL emission (handlers_packages.py): a package search DURING onboarding
    stamps ``package_offered``; installing stamps ``package_accepted`` then
    ``package_installed``; post-onboarding search stamps nothing.

Fixtures are reused from the US-002 section tests and the US-006 tool tests so the
render + stub machinery stays in one place.
"""
from __future__ import annotations

from core.context_guard import count_tokens
import modules.tools.discovery.handlers_packages as hp
from tests.test_prd222_onboarding_section import _render_stage
from tests.test_prd230_package_tools import FakeDB, FakeWS, _install_spy, _pkg, _run


# --------------------------------------------------------------------------- #
# Section — proposal offers ONE package by name + defer-pick (D5) + no-match (D10)
# --------------------------------------------------------------------------- #


def test_proposal_searches_and_offers_one_package_by_name():
    out = _render_stage("proposal")
    assert "platform_search_packages" in out           # search the segment first
    assert "Shopify Management" in out                  # offered BY NAME…
    assert "weekly-numbers" in out.lower()              # …with its contents


def test_proposal_carries_defer_pick_rule_owner_to_management():
    low = _render_stage("proposal").lower()
    assert "defer" in low
    assert "owner" in low and "management" in low       # D5: store OWNER → Management


def test_proposal_no_match_custom_designs_marketplace_first():
    low = _render_stage("proposal").lower()
    # D10: never a forced generic package — Auto custom-designs, marketplace-first.
    assert "don't force a package" in low or "custom-design" in low
    assert "marketplace-first" in low


def test_proposal_keeps_the_approval_gate_intact():
    # US-009 must not weaken the PRD-222 approval gate it extends.
    low = _render_stage("proposal").lower()
    assert "approval gate" in low
    assert "nothing is built" in low
    assert "explicit yes" in low


# --------------------------------------------------------------------------- #
# Section — the D7 three-step guide (install → connect → activate)
# --------------------------------------------------------------------------- #


def test_building_installs_package_and_narrates_manifest():
    out = _render_stage("building")
    assert "platform_install_package" in out           # step ① install the package
    low = out.lower()
    assert "manifest" in low and "registered" in low    # narrate what registered


def test_building_guided_connects_are_the_shopify_two_step_never_auto():
    low = _render_stage("building").lower()
    assert "connect card" in low                        # step ② guided connects
    assert "auto-connect" in low                        # "never auto-connect" (FR-4)
    assert "settings" in low and "widget sdk" in low    # the Site two-step truth


def test_boom_activates_with_first_playbook_and_checklist():
    low = _render_stage("boom").lower()
    assert "playbook" in low or "report" in low          # step ③ put agents to work
    assert "checklist" in low                            # checklist carries the rest


def test_proposal_variant_stays_within_the_token_budget():
    # Proposal is the largest variant after US-009 — lock it under the 800 cap.
    out = _render_stage("proposal", comfort="brand new")
    assert count_tokens(out) <= 800


# --------------------------------------------------------------------------- #
# Funnel — package_offered / package_accepted / package_installed
# --------------------------------------------------------------------------- #


def _capture_events(monkeypatch) -> list:
    events: list = []
    monkeypatch.setattr(
        "services.onboarding_state.record_package_event",
        lambda db, w, event, slug, **k: events.append((event, slug)),
    )
    return events


def test_search_during_onboarding_stamps_package_offered(monkeypatch):
    ws = FakeWS(stage="proposal")
    monkeypatch.setattr("services.marketplace_packages.list_packages",
                        lambda db: [_pkg("shopify-management", agents=4)])
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    events = _capture_events(monkeypatch)

    out = _run(hp.search_packages(FakeDB(), "ws-1", {"platforms": ["shopify"]}))
    assert out["success"] is True
    assert ("package_offered", "shopify-management") in events


def test_search_after_onboarding_stamps_no_offer(monkeypatch):
    ws = FakeWS(stage="completed")  # terminal → browsing, not a funnel moment
    monkeypatch.setattr("services.marketplace_packages.list_packages",
                        lambda db: [_pkg("shopify-management", agents=4)])
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    events = _capture_events(monkeypatch)

    _run(hp.search_packages(FakeDB(), "ws-1", {"platforms": ["shopify"]}))
    assert events == []


def test_search_with_no_match_stamps_no_offer(monkeypatch):
    monkeypatch.setattr("services.marketplace_packages.list_packages", lambda db: [])
    events = _capture_events(monkeypatch)

    out = _run(hp.search_packages(FakeDB(), "ws-1", {"platforms": ["shopify"]}))
    assert out["count"] == 0
    assert events == []  # nothing to offer → no funnel stamp


def test_install_during_onboarding_stamps_accepted_then_installed(monkeypatch):
    ws = FakeWS(stage="proposal")
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug",
                        lambda db, slug: _pkg(agents=1))
    _install_spy(monkeypatch)
    events = _capture_events(monkeypatch)

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "shopify-management"}))
    assert out["success"] is True
    # The funnel order matters: acceptance precedes installation.
    assert [e for e, _ in events] == ["package_accepted", "package_installed"]


def test_funnel_side_channel_never_breaks_search(monkeypatch):
    # _load_workspace blows up — the search must still return its matches.
    monkeypatch.setattr("services.marketplace_packages.list_packages",
                        lambda db: [_pkg("shopify-management", agents=4)])
    def _boom(db, wid):
        raise RuntimeError("db down")
    monkeypatch.setattr(hp, "_load_workspace", _boom)

    out = _run(hp.search_packages(FakeDB(), "ws-1", {"platforms": ["shopify"]}))
    assert out["success"] is True
    assert out["matches"][0]["slug"] == "shopify-management"
