"""PRD-221 S4 — page-prior tool exposure (pure: no DB, no network).

The security-critical story. Page-manifest actions are folded into the
narrowed dispatcher enum so page-relevant tools survive semantic narrowing —
but they pass the SAME role gate as ranked actions, so a manifest can never
surface an admin/super-admin tool to an unauthorized principal (PRD-143
fail-closed). Also locks: no page context → identical narrowing; the union
is capped.
"""
from __future__ import annotations

# CI collection-order guard (see PR #434).
import sys as _sys_guard  # noqa: E402
for _name in [n for n, m in list(_sys_guard.modules.items())
              if (n == "modules" or n.startswith("modules.")
                  or n == "consumers" or n.startswith("consumers.")
                  or n == "services" or n.startswith("services."))
              and getattr(m, "__spec__", None) is None]:
    _sys_guard.modules.pop(_name, None)

import modules.tools.tool_router as tr  # noqa: E402


class _FakeAction:
    def __init__(self, admin_only=False, super_admin_only=False):
        self.admin_only = admin_only
        self.super_admin_only = super_admin_only


class _FakeRegistry:
    def __init__(self, mapping):
        self._m = mapping

    def get(self, name):
        return self._m.get(name)


def _patch_registry(monkeypatch, mapping):
    import modules.tools.discovery as disc
    monkeypatch.setattr(disc, "get_action_registry", lambda: _FakeRegistry(mapping))


# --- union + no-op behaviour (gate stubbed permissive) ----------------------

def test_page_actions_unioned(monkeypatch):
    monkeypatch.setattr(tr, "_page_action_passes_gate", lambda n, ia, isa: True)
    narrowing = (["ranked_1", "ranked_2"], "semantic", False)
    out, reason, _pins = tr._apply_page_prior(
        narrowing, ["page_x", "page_y"], is_admin=False, is_super_admin=False
    )
    # ranked survive AND page actions are folded in even though the query
    # ranked them out.
    assert "ranked_1" in out and "ranked_2" in out
    assert "page_x" in out and "page_y" in out
    assert reason  # narrow reason preserved / defaulted


def test_no_page_context_unchanged():
    narrowing = (["a", "b"], "semantic", False)
    # No page actions → identity (same tuple object).
    assert tr._apply_page_prior(narrowing, None, False, False) is narrowing
    assert tr._apply_page_prior(narrowing, [], False, False) is narrowing
    # Full-enum narrowing (allowed is None → every action already exposed):
    # nothing to union, returned unchanged.
    full = (None, "routing_off", False)
    assert tr._apply_page_prior(full, ["x"], False, False) is full


def test_page_actions_dedup_and_order(monkeypatch):
    monkeypatch.setattr(tr, "_page_action_passes_gate", lambda n, ia, isa: True)
    narrowing = (["a", "b"], "s", False)
    out, _, _ = tr._apply_page_prior(narrowing, ["b", "c", "c"], False, False)
    # 'b' already present (not duplicated); 'c' appended once, after ranked.
    assert out == ["a", "b", "c"]


def test_page_actions_cap(monkeypatch):
    monkeypatch.setattr(tr, "_page_action_passes_gate", lambda n, ia, isa: True)
    ranked = [f"r{i}" for i in range(38)]
    page = [f"p{i}" for i in range(10)]
    out, _, _ = tr._apply_page_prior((ranked, "s", False), page, False, False)
    assert len(out) == 40  # TOOL_ROUTING_ENUM_CAP default
    # ranked kept in full; only the first 2 page actions fit under the cap.
    assert out[:38] == ranked
    assert out[38:] == ["p0", "p1"]


# --- the gate itself (real predicate + fake registry) -----------------------

def test_page_actions_respect_super_admin_gate(monkeypatch):
    _patch_registry(monkeypatch, {
        "normal": _FakeAction(),
        "su_only": _FakeAction(super_admin_only=True),
        "admin_x": _FakeAction(admin_only=True),
    })
    narrowing = (["ranked"], "s", False)

    # non-admin, non-super principal: only the ungated action folds in.
    out, _, _ = tr._apply_page_prior(
        narrowing, ["normal", "su_only", "admin_x"],
        is_admin=False, is_super_admin=False,
    )
    assert "normal" in out
    assert "su_only" not in out
    assert "admin_x" not in out

    # super-admin sees the su-only action; admin sees the admin-only action.
    out_su, _, _ = tr._apply_page_prior(
        narrowing, ["su_only"], is_admin=True, is_super_admin=True
    )
    assert "su_only" in out_su
    out_admin, _, _ = tr._apply_page_prior(
        narrowing, ["admin_x"], is_admin=True, is_super_admin=False
    )
    assert "admin_x" in out_admin


def test_gate_drops_unknown_action(monkeypatch):
    _patch_registry(monkeypatch, {"known": _FakeAction()})
    assert tr._page_action_passes_gate("known", True, True) is True
    assert tr._page_action_passes_gate("ghost", True, True) is False


def test_gate_unregistered_yields_no_union(monkeypatch):
    # Every page action unknown → nothing clears the gate → narrowing unchanged.
    _patch_registry(monkeypatch, {})
    narrowing = (["ranked"], "s", False)
    assert tr._apply_page_prior(narrowing, ["ghost1", "ghost2"], True, True) is narrowing
