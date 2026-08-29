"""PRD-230 US-010 — the invariant guards that keep D1/D2/D3 true.

One dedicated tripwire file locking the registration invariant across the whole
wave, independent of the per-story tests. Five locks, each paired with a
self-check that MUTATES a fixture and proves the lock would fail — a guard that
cannot fail is not a guard.

  Lock 1  D2 full closure through the INSTALLER (not just the resolver): the
          agent-A example (3 tools + 2 skills + 1 LLM ⇒ 7 registrations).
  Lock 2  D3 workspace-owned/editable on every artifact type the installer emits.
  Lock 3  D6 one-package-during-onboarding, enforced at the TOOL layer.
  Lock 4  idempotent re-install (re-install adds nothing).
  Lock 5  no platform-dangling members: every member is registered OR surfaces as
          a required_connect — nothing silently dropped.

PURE: the cascade / leaf-install / DB boundary is stubbed exactly where the
US-005/US-006 tests stub it, so these assert the installer's + tool's OWN
contract. The real-Postgres end-to-end (rows actually written) rides CI.
Machinery is imported from the sibling story tests so it stays in one place.
"""
from __future__ import annotations

from types import SimpleNamespace

import modules.tools.discovery.handlers_packages as hp
import services.package_installer as pi
from modules.tools.discovery import cascade_installer as ci
from services.package_installer import InstallManifest, Registration

# Reuse the exact fixtures the story tests already prove out (no re-invented stubs).
from tests.test_prd230_package_installer import _agent_a_deps, _cascade, _patch_agent_install
from tests.test_prd230_package_tools import FakeDB, FakeWS, _install_spy, _pkg, _run


# --------------------------------------------------------------------------- #
# Lock 1 — D2: the agent-A full closure, end-to-end through install_marketplace_agent
# --------------------------------------------------------------------------- #


def test_lock1_agent_a_closure_of_seven_through_the_installer(monkeypatch):
    # Gerard's canonical example (D2): agent A = 3 tools(plugins) + 2 skills + 1 LLM
    # ⇒ the agent + those 6 = exactly 7 workspace registrations, through the INSTALLER.
    _patch_agent_install(monkeypatch)
    m = _run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))

    assert len(m.registrations) == 7
    assert len(m.by_type("agent")) == 1
    assert len(m.by_type("llm")) == 1
    assert len(m.by_type("skill")) == 2
    assert len(m.by_type("plugin")) == 3


def test_lock1_bites_when_a_closure_member_is_dropped(monkeypatch):
    # Self-check: drop one skill from the closure → 6, not 7. The ==7 lock must be
    # the thing that would fail, proving it is not vacuously satisfied.
    deps = [d for d in _agent_a_deps() if d["name"] != "s2"]
    _patch_agent_install(monkeypatch, cascade=_cascade(deps))
    m = _run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))

    assert len(m.registrations) == 6  # a real drop is visible → Lock 1 would fire


# --------------------------------------------------------------------------- #
# Lock 2 — D3: every registration the installer emits is workspace-owned/editable
# --------------------------------------------------------------------------- #


def _agent_a_deps_with_oauth_tool():
    # The full closure PLUS a connected-app tool (Shopify) — so the run exercises
    # agent, llm, skill, plugin AND tool registration construction in one pass.
    return _agent_a_deps() + [
        {"type": "tool", "name": "SHOPIFY", "status": "assigned", "oauth_required": True}
    ]


def test_lock2_every_installer_registration_is_workspace_owned(monkeypatch):
    _patch_agent_install(monkeypatch, cascade=_cascade(_agent_a_deps_with_oauth_tool()))
    m = _run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))

    # Every type the agent install can emit is present…
    assert {r.type for r in m.registrations} == {"agent", "llm", "skill", "plugin", "tool"}
    # …and D3 holds for each of them.
    assert all(r.workspace_owned for r in m.registrations)


def test_lock2_covers_the_leaf_and_playbook_construction_sites():
    # The remaining Registration construction sites the agent path doesn't hit:
    # leaf skill/plugin/llm results and the playbook member — all default owned.
    leaf = pi._reg_from_install_result("skill", "10", {"success": True})
    assert leaf.workspace_owned is True
    assert Registration("playbook", "5", "Weekly Numbers", "cloned").workspace_owned is True


def test_lock2_bites_on_a_non_owned_registration():
    # Self-check: a registration flagged NOT owned makes the all(...) lock fail.
    m = InstallManifest()
    m.add(Registration("agent", "5", "A", "cloned"))
    m.add(Registration("skill", "s", "s", "installed", workspace_owned=False))
    assert not all(r.workspace_owned for r in m.registrations)  # Lock 2 would fire


# --------------------------------------------------------------------------- #
# Lock 3 — D6: one package during onboarding, enforced at the TOOL layer
# --------------------------------------------------------------------------- #


def test_lock3_second_package_during_onboarding_blocked_at_tool_layer(monkeypatch):
    ws = FakeWS(stage="proposal", funnel={"package_installed": {"slug": "first", "at": "t"}})
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: ws)
    spy = _install_spy(monkeypatch)

    out = _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "second"}))
    assert out["success"] is False
    assert out["onboarding_restricted"] is True
    assert spy["calls"] == 0  # the closure never started — no partial install


def test_lock3_bites_first_package_and_post_onboarding_are_allowed(monkeypatch):
    # Self-check: the guard is CONDITIONAL, not always-on. The same install goes
    # through both on the FIRST package during onboarding and once terminal —
    # proving the block above comes from the D6 state, not a blanket refusal.
    monkeypatch.setattr(hp, "_workspace_agent_count", lambda db, wid: 0)
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: _pkg(agents=1))
    monkeypatch.setattr("services.onboarding_state.record_package_event",
                        lambda db, w, event, slug, **k: None)

    first = FakeWS(stage="proposal")  # no package yet
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: first)
    spy = _install_spy(monkeypatch)
    assert _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "s"}))["success"] is True
    assert spy["calls"] == 1

    done = FakeWS(stage="completed", funnel={"package_installed": {"slug": "old", "at": "t"}})
    monkeypatch.setattr(hp, "_load_workspace", lambda db, wid: done)
    spy2 = _install_spy(monkeypatch)
    assert _run(hp.install_package_tool(FakeDB(), "ws-1", {"slug": "s2"}))["success"] is True
    assert spy2["calls"] == 1  # terminal → unrestricted


# --------------------------------------------------------------------------- #
# Lock 4 — idempotent re-install (re-install = zero dupes, adds nothing)
# --------------------------------------------------------------------------- #


def test_lock4_reinstall_adds_nothing(monkeypatch):
    existing = SimpleNamespace(id=500, name="Agent A")
    already = [
        {"type": "model", "name": "gpt-4o", "status": "already_installed"},
        {"type": "skill", "name": "s1", "status": "already_installed"},
        {"type": "plugin", "name": "p1", "status": "already_installed"},
    ]
    _patch_agent_install(monkeypatch, existing=existing, cascade=_cascade(already))
    monkeypatch.setattr(  # a re-install must NEVER re-clone the agent
        ci, "clone_agent_to_workspace",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-clone")))

    m = _run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))
    assert m.by_type("agent")[0].status == "already_installed"
    assert m.added == []  # nothing new registered on the second install


def test_lock4_bites_first_install_does_add(monkeypatch):
    # Self-check: on a genuine FIRST install everything is new, so `added` is the
    # full closure — proving the `added == []` lock above is a real re-install signal.
    _patch_agent_install(monkeypatch)  # existing=None, all "installed"/"cloned"
    m = _run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))
    assert len(m.added) == 7


# --------------------------------------------------------------------------- #
# Lock 5 — no platform-dangling members: registered OR a required_connect, never lost
# --------------------------------------------------------------------------- #


def _members_for_dangling_check():
    return [
        {"type": "agent", "ref": "1"},
        {"type": "skill", "ref": "10"},
        {"type": "plugin", "ref": "p"},
        {"type": "llm", "ref": "gpt-4o"},
        {"type": "playbook", "ref": "pb"},
    ]


def test_lock5_every_package_member_is_accounted_for(monkeypatch):
    members = _members_for_dangling_check()
    monkeypatch.setattr("services.marketplace_packages.get_by_slug",
                        lambda db, slug: SimpleNamespace(members=members))

    async def fake_member(db, ws, mtype, ref, uid):
        one = InstallManifest()
        one.add(Registration(mtype, ref, ref, "installed"))
        if mtype == "agent":  # an agent carries an app requirement → a connect step
            one.add_required_connect("SHOPIFY")
        return one

    monkeypatch.setattr(pi, "_install_member", fake_member)
    m = _run(pi.install_package(None, "ws", "pkg"))

    registered = {r.ref for r in m.registrations}
    connects = {rc["app_name"].upper() for rc in m.required_connects}
    for mem in members:  # every declared member landed somewhere
        assert mem["ref"] in registered
    assert "SHOPIFY" in connects       # the app surfaced as a guided connect (FR-4)…
    assert "SHOPIFY" not in registered  # …NOT auto-installed as a member
    assert m.warnings == []            # nothing silently skipped


def test_lock5_closure_apps_are_never_dropped(monkeypatch):
    # The closure-level half: an OAuth tool in an agent's cascade is BOTH a
    # registration and a required_connect — absorbed, never dropped on the floor.
    _patch_agent_install(monkeypatch, cascade=_cascade(_agent_a_deps_with_oauth_tool()))
    m = _run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))
    assert m.by_type("tool")[0].name == "SHOPIFY"
    assert [rc["app_name"] for rc in m.required_connects] == ["SHOPIFY"]


def test_lock5_bites_when_a_member_is_dropped(monkeypatch):
    # Self-check: an installer that returns nothing for the plugin leaves "p"
    # dangling — absent from BOTH registrations and required_connects. The
    # accounted-for lock above would fail on exactly this.
    members = _members_for_dangling_check()
    monkeypatch.setattr("services.marketplace_packages.get_by_slug",
                        lambda db, slug: SimpleNamespace(members=members))

    async def dropping_member(db, ws, mtype, ref, uid):
        one = InstallManifest()
        if mtype != "plugin":  # silently drop the plugin
            one.add(Registration(mtype, ref, ref, "installed"))
        return one

    monkeypatch.setattr(pi, "_install_member", dropping_member)
    m = _run(pi.install_package(None, "ws", "pkg"))

    accounted = {r.ref for r in m.registrations} | {rc["app_name"] for rc in m.required_connects}
    assert "p" not in accounted  # the drop is detectable → Lock 5 would fire
