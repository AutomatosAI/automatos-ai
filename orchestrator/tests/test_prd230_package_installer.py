"""PRD-230 US-005 — workspace registration installer (D1/D2/D3).

PURE tests: the installer REUSES the cascade installer + per-type install
functions, so these monkeypatch that boundary and assert the installer's OWN
contract — full-closure manifest (the D2 count), workspace-owned marking (D3),
idempotency, member dispatch, and required_connects (FR-4). The real-DB
end-to-end (rows actually written) is US-010's dedicated guard (+ CI).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import services.package_installer as pi
from modules.tools.discovery import cascade_installer as ci
from services.package_installer import InstallManifest, Registration, PackageInstallError


def _cascade(deps=None, warnings=None, cloned_items=None):
    r = ci.CascadeResult()
    r.installed_dependencies = deps or []
    r.warnings = warnings or []
    r.cloned_items = cloned_items or []
    return r


def _agent_a_deps():
    # D2: agent A = 3 tools (plugins) + 2 skills + 1 LLM.
    return [
        {"type": "model", "name": "gpt-4o", "status": "installed"},
        {"type": "skill", "name": "s1", "status": "installed"},
        {"type": "skill", "name": "s2", "status": "installed"},
        {"type": "plugin", "name": "p1", "status": "installed"},
        {"type": "plugin", "name": "p2", "status": "installed"},
        {"type": "plugin", "name": "p3", "status": "installed"},
    ]


def _patch_agent_install(monkeypatch, *, existing=None, cascade=None, name="Agent A"):
    ma = SimpleNamespace(id=100, name=name, install_count=0)
    cloned = SimpleNamespace(id=500, name=name)
    monkeypatch.setattr(pi, "_find_marketplace_agent", lambda db, ref: ma)
    monkeypatch.setattr(pi, "_existing_workspace_clone", lambda db, ws, m: existing)
    monkeypatch.setattr(ci, "clone_agent_to_workspace", lambda db, ws, m, uid: (cloned, name))

    async def fake_cascade(db, ws, m, c):
        return cascade if cascade is not None else _cascade(_agent_a_deps())

    monkeypatch.setattr(ci, "cascade_agent_dependencies", fake_cascade)
    return ma, cloned


# --------------------------------------------------------------------------- #
# THE closure invariant (D2) + ownership (D3)
# --------------------------------------------------------------------------- #


def test_install_agent_registers_full_closure_of_seven(monkeypatch):
    _patch_agent_install(monkeypatch)
    m = asyncio.run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))

    assert len(m.registrations) == 7  # agent + 1 LLM + 2 skills + 3 plugins
    assert len(m.by_type("agent")) == 1
    assert len(m.by_type("llm")) == 1
    assert len(m.by_type("skill")) == 2
    assert len(m.by_type("plugin")) == 3


def test_every_registration_is_workspace_owned(monkeypatch):
    _patch_agent_install(monkeypatch)
    m = asyncio.run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))
    assert all(r.workspace_owned for r in m.registrations)  # D3
    assert len(m.added) == 7  # all newly registered on first install


def test_unknown_marketplace_agent_raises(monkeypatch):
    monkeypatch.setattr(pi, "_find_marketplace_agent", lambda db, ref: None)
    with pytest.raises(PackageInstallError):
        asyncio.run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="nope"))


# --------------------------------------------------------------------------- #
# Idempotency (re-install = zero dupes)
# --------------------------------------------------------------------------- #


def test_reinstall_reuses_clone_and_adds_nothing(monkeypatch):
    existing = SimpleNamespace(id=500, name="Agent A")
    # cascade on re-install reports everything already present
    already = [
        {"type": "model", "name": "gpt-4o", "status": "already_installed"},
        {"type": "skill", "name": "s1", "status": "already_installed"},
        {"type": "plugin", "name": "p1", "status": "already_installed"},
    ]
    _patch_agent_install(monkeypatch, existing=existing, cascade=_cascade(already))
    # clone must NOT be called on the idempotent path
    monkeypatch.setattr(ci, "clone_agent_to_workspace",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not re-clone")))

    m = asyncio.run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))
    assert m.by_type("agent")[0].status == "already_installed"
    assert m.added == []  # nothing new


# --------------------------------------------------------------------------- #
# required_connects (FR-4) — apps surface as guided connects, never auto
# --------------------------------------------------------------------------- #


def test_oauth_tool_becomes_required_connect(monkeypatch):
    cascade = _cascade([
        {"type": "model", "name": "gpt-4o", "status": "installed"},
        {"type": "tool", "name": "SHOPIFY", "status": "assigned", "oauth_required": True},
    ])
    _patch_agent_install(monkeypatch, cascade=cascade)
    m = asyncio.run(pi.install_marketplace_agent(db=None, workspace_id="ws", agent_ref="100"))

    assert [rc["app_name"] for rc in m.required_connects] == ["SHOPIFY"]
    assert m.required_connects[0]["needs_oauth"] is True
    assert m.by_type("tool")  # the tool is still a registration (assigned)


# --------------------------------------------------------------------------- #
# _absorb_cascade mapping (pure)
# --------------------------------------------------------------------------- #


def test_absorb_maps_model_to_llm_and_dedups():
    m = InstallManifest()
    pi._absorb_cascade(m, _cascade([
        {"type": "model", "name": "gpt-4o", "status": "installed"},
        {"type": "model", "name": "gpt-4o", "status": "installed"},  # dup
    ]))
    assert len(m.by_type("llm")) == 1  # model→llm, deduped


# --------------------------------------------------------------------------- #
# Package dispatch + merge
# --------------------------------------------------------------------------- #


def test_install_package_dispatches_every_member(monkeypatch):
    pkg = SimpleNamespace(members=[
        {"type": "agent", "ref": "1"},
        {"type": "skill", "ref": "10"},
        {"type": "plugin", "ref": "uuid-p"},
        {"type": "llm", "ref": "gpt-4o"},
        {"type": "playbook", "ref": "pb-1"},
    ])
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: pkg)

    calls = []

    async def fake_member(db, ws, mtype, ref, uid):
        calls.append((mtype, ref))
        one = InstallManifest()
        one.add(Registration(mtype, ref, ref, "installed"))
        return one

    monkeypatch.setattr(pi, "_install_member", fake_member)
    m = asyncio.run(pi.install_package(None, "ws", "shopify-management"))

    assert set(calls) == {("agent", "1"), ("skill", "10"), ("plugin", "uuid-p"),
                          ("llm", "gpt-4o"), ("playbook", "pb-1")}
    assert len(m.registrations) == 5


def test_two_agent_package_registers_both_closures_deduped(monkeypatch):
    pkg = SimpleNamespace(members=[{"type": "agent", "ref": "1"}, {"type": "agent", "ref": "2"}])
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: pkg)

    async def fake_member(db, ws, mtype, ref, uid):
        one = InstallManifest()
        one.add(Registration("agent", ref, f"Agent {ref}", "cloned"))
        one.add(Registration("llm", "gpt-4o", "gpt-4o", "installed"))   # SHARED llm
        one.add(Registration("skill", f"s{ref}", f"s{ref}", "installed"))
        return one

    monkeypatch.setattr(pi, "_install_member", fake_member)
    m = asyncio.run(pi.install_package(None, "ws", "two-agents"))

    assert {r.ref for r in m.by_type("agent")} == {"1", "2"}   # both agents
    assert len(m.by_type("llm")) == 1                          # shared llm deduped
    assert {r.ref for r in m.by_type("skill")} == {"s1", "s2"}


def test_unknown_package_raises(monkeypatch):
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: None)
    with pytest.raises(PackageInstallError):
        asyncio.run(pi.install_package(None, "ws", "ghost"))


def test_malformed_member_is_skipped_not_fatal(monkeypatch):
    pkg = SimpleNamespace(members=[{"type": "agent"}, "not-a-dict", {"ref": "x"}])
    monkeypatch.setattr("services.marketplace_packages.get_by_slug", lambda db, slug: pkg)
    m = asyncio.run(pi.install_package(None, "ws", "messy"))
    assert m.registrations == []
    assert len(m.warnings) >= 1  # the malformed members were flagged, not crashed


# --------------------------------------------------------------------------- #
# Leaf install status mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("result,expected", [
    ({"success": True}, "installed"),
    ({"success": True, "already_enabled": True}, "already_installed"),
    ({"success": True, "already_installed": True}, "already_installed"),
    ({"success": True, "reactivated": True}, "reactivated"),
    ({"success": False, "error": "x"}, "failed"),
])
def test_leaf_status_mapping(result, expected):
    reg = pi._reg_from_install_result("skill", "10", result)
    assert reg.status == expected
    assert reg.workspace_owned is True


def test_manifest_to_dict_shape():
    m = InstallManifest()
    m.add(Registration("agent", "5", "A", "cloned"))
    m.add_required_connect("SHOPIFY")
    d = m.to_dict()
    assert d["added_count"] == 1
    assert d["registrations"][0]["workspace_owned"] is True
    assert d["required_connects"][0]["app_name"] == "SHOPIFY"
