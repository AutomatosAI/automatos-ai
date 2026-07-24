"""PRD-174 W4 — PolicyGate acceptance (§6.2/§6.5/§6.6/§6.7).

Drives ``PolicyGate.check()`` directly with lightweight fakes for the DB /
registry / autonomy so we exercise the *gate's* decisions with no real DB:

- **F060/F085** — a Composio (and workspace, and registry) tool is evaluated by
  the gate, not routed around it; a destructive/external one is asked, not run.
- **F014** — an ``admin_only`` action is denied when the caller isn't admin;
  the workspace-owner fallback fires only with ``agents_inherit_admin`` on.
- **F043** — a ``super_admin`` passes the super-admin gate; an admin/user does not.
- **Balanced (§5)** — read/internal → allow (auto), destructive/external → ask.

The gate's lazy hooks (registry lookup, ``is_full_autonomy``, workspace-admin
query) are monkeypatched, so this stays DB-free and deterministic.
"""
from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_LEAKED_PARENT_STUBS = {}
for _pkg in ("modules", "modules.tools", "modules.tools.execution"):
    if _pkg not in sys.modules:
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub
        _LEAKED_PARENT_STUBS[_pkg] = _stub


def teardown_module(module):
    for _name, _stub in _LEAKED_PARENT_STUBS.items():
        if sys.modules.get(_name) is _stub:
            del sys.modules[_name]


from modules.policy import gate as gate_mod  # noqa: E402
from modules.policy.gate import PolicyGate, ToolCall  # noqa: E402
from modules.policy.types import Decision  # noqa: E402
from modules.policy import policy_document as pd  # noqa: E402


class _ActionDef:
    """Stand-in for ActionDefinition (only the fields the gate reads)."""

    def __init__(self, permission_level="read", admin_only=False,
                 super_admin_only=False, requires_confirmation=False):
        self.permission_level = permission_level
        self.admin_only = admin_only
        self.super_admin_only = super_admin_only
        self.requires_confirmation = requires_confirmation


@pytest.fixture
def patched_gate(monkeypatch):
    """A PolicyGate whose DB-bound hooks are replaced with in-memory config.

    Returns a helper that builds the gate for a given (action registry map,
    policy document, full_autonomy, has_admin_owner) so each test sets only what
    it cares about.
    """
    def make(*, actions=None, doc=None, full_autonomy=False, has_admin_owner=True,
             budget_allows=True):
        actions = actions or {}
        doc = doc or pd.PolicyDocument(pd.BALANCED, False, {})

        g = PolicyGate(db="fake-db")
        monkeypatch.setattr(g, "_lookup_action", lambda name: actions.get(name))
        monkeypatch.setattr(g, "_full_autonomy", lambda ws: full_autonomy)
        monkeypatch.setattr(g, "_workspace_has_admin_owner", lambda ws: has_admin_owner)
        # policy document + budget are module functions the gate calls.
        monkeypatch.setattr(gate_mod._policy_doc, "load_policy_document",
                            lambda db, ws: doc)
        from modules.policy.budget import BudgetDecision
        monkeypatch.setattr(
            gate_mod._budget, "check_budget",
            lambda db, ws, **kw: BudgetDecision(budget_allows, "test budget"),
        )
        return g

    return make


def _call(name, *, caller=None, ws="ws-1"):
    return ToolCall(tool_name=name, parameters={}, workspace_id=ws, caller_context=caller)


# ---------------------------------------------------------------------------
# F060/F085 — Composio / workspace / registry tools are evaluated (not bypassed)
# ---------------------------------------------------------------------------

def test_composio_send_routes_to_ask_not_bypassed(patched_gate):
    g = patched_gate()  # no action_def (Composio isn't in the platform registry)
    v = g.check(_call("COMPOSIO_GMAIL_SEND_EMAIL"))
    # classified external → Balanced routes to ASK; crucially it was EVALUATED.
    assert v.decision is Decision.ASK
    assert v.error is not None and v.error.code == "approval_required"


def test_workspace_exec_routes_to_ask(patched_gate):
    g = patched_gate()
    v = g.check(_call("workspace_exec_shell"))
    assert v.decision is Decision.ASK  # external-exec → ask under Balanced


def test_registry_read_tool_allowed(patched_gate):
    actions = {"platform_list_agents": _ActionDef(permission_level="read")}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_list_agents"))
    assert v.decision is Decision.ALLOW  # read → auto


def test_internal_write_allowed_under_balanced(patched_gate):
    actions = {"platform_update_task": _ActionDef(permission_level="write")}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_update_task"))
    assert v.decision is Decision.ALLOW  # low-risk internal write → auto


def test_destructive_routes_to_ask(patched_gate):
    actions = {"platform_delete_agent": _ActionDef(permission_level="destructive")}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_delete_agent"))
    assert v.decision is Decision.ASK


# ---------------------------------------------------------------------------
# F043 — super-admin gate
# ---------------------------------------------------------------------------

def test_super_admin_only_denies_non_super_admin(patched_gate):
    actions = {"platform_obs_dump": _ActionDef(super_admin_only=True)}
    g = patched_gate(actions=actions)
    assert g.check(_call("platform_obs_dump", caller={"system_role": "admin"})).decision is Decision.DENY
    assert g.check(_call("platform_obs_dump", caller=None)).decision is Decision.DENY


def test_super_admin_only_allows_super_admin(patched_gate):
    actions = {"platform_obs_dump": _ActionDef(super_admin_only=True, permission_level="read")}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_obs_dump", caller={"system_role": "super_admin"}))
    assert v.decision is Decision.ALLOW  # passes super-admin gate, read → auto


# ---------------------------------------------------------------------------
# F014 — admin_only requires the caller's own role; owner-fallback is opt-in
# ---------------------------------------------------------------------------

def test_admin_only_denied_for_non_admin_caller(patched_gate):
    actions = {"platform_admin_thing": _ActionDef(admin_only=True)}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_admin_thing", caller={"system_role": "user"}))
    assert v.decision is Decision.DENY
    assert v.error.code == "admin_required"


def test_admin_only_allowed_for_admin_caller(patched_gate):
    actions = {"platform_admin_thing": _ActionDef(admin_only=True, permission_level="read")}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_admin_thing", caller={"system_role": "admin"}))
    assert v.decision is Decision.ALLOW


def test_admin_only_super_admin_caller_also_allowed(patched_gate):
    actions = {"platform_admin_thing": _ActionDef(admin_only=True, permission_level="read")}
    g = patched_gate(actions=actions)
    v = g.check(_call("platform_admin_thing", caller={"system_role": "super_admin"}))
    assert v.decision is Decision.ALLOW  # super_admin ⊇ admin


def test_admin_only_no_caller_denied_when_inherit_off(patched_gate):
    # F014: no caller identity + agents_inherit_admin OFF (default) → DENY,
    # even though the workspace HAS an admin owner.
    actions = {"platform_admin_thing": _ActionDef(admin_only=True)}
    doc = pd.PolicyDocument(pd.BALANCED, agents_inherit_admin=False, route_overrides={})
    g = patched_gate(actions=actions, doc=doc, has_admin_owner=True)
    v = g.check(_call("platform_admin_thing", caller=None))
    assert v.decision is Decision.DENY


def test_admin_only_no_caller_allowed_when_inherit_on(patched_gate):
    # With the explicit default-OFF policy turned ON and an admin owner present.
    actions = {"platform_admin_thing": _ActionDef(admin_only=True, permission_level="read")}
    doc = pd.PolicyDocument(pd.BALANCED, agents_inherit_admin=True, route_overrides={})
    g = patched_gate(actions=actions, doc=doc, has_admin_owner=True)
    v = g.check(_call("platform_admin_thing", caller=None))
    assert v.decision is Decision.ALLOW


def test_admin_only_inherit_on_but_no_owner_denied(patched_gate):
    actions = {"platform_admin_thing": _ActionDef(admin_only=True)}
    doc = pd.PolicyDocument(pd.BALANCED, agents_inherit_admin=True, route_overrides={})
    g = patched_gate(actions=actions, doc=doc, has_admin_owner=False)
    assert g.check(_call("platform_admin_thing", caller=None)).decision is Decision.DENY


# ---------------------------------------------------------------------------
# Full autonomy dial — asks are skipped, but the super-admin gate still bites
# ---------------------------------------------------------------------------

def test_full_autonomy_skips_ask_for_destructive(patched_gate):
    actions = {"platform_delete_agent": _ActionDef(permission_level="destructive")}
    g = patched_gate(actions=actions, full_autonomy=True)
    v = g.check(_call("platform_delete_agent"))
    assert v.decision is Decision.ALLOW  # auto dial acts without asking


def test_full_autonomy_still_cannot_pass_super_admin_gate(patched_gate):
    actions = {"platform_obs_dump": _ActionDef(super_admin_only=True)}
    g = patched_gate(actions=actions, full_autonomy=True)
    v = g.check(_call("platform_obs_dump", caller={"system_role": "admin"}))
    assert v.decision is Decision.DENY  # the dial never satisfies super-admin


# ---------------------------------------------------------------------------
# F086 — budget breach denies before the call (gate composes budget)
# ---------------------------------------------------------------------------

def test_budget_breach_denies(patched_gate):
    actions = {"platform_list_agents": _ActionDef(permission_level="read")}
    g = patched_gate(actions=actions, budget_allows=False)
    v = g.check(_call("platform_list_agents"))
    assert v.decision is Decision.DENY
    assert v.error.code == "budget_exceeded"
