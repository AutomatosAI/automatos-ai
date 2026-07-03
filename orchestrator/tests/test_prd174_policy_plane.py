"""PRD-174 W4 — Unified Policy Plane: pure-unit acceptance (§6).

These pin the plane's own semantics with NO database and NO heavy deps — the
verdict algebra (deny > ask > allow), the role/permission helpers (F042/F043),
the Balanced act-vs-ask routing (§5), the model-aware budget gate (F086/F059),
and the errors-as-data envelope (§4.2). The gate/executor *integration* (flag
OFF stays byte-for-byte; a denied call never executes) lives in
``test_prd174_policy_chokepoint.py``.

Stdlib-only import: stub the ``modules`` / ``modules.tools`` package inits so we
never run them (they pull asyncpg/pgvector); ``modules.policy`` is stdlib-only at
import time and loads cleanly via its real path once the parents are stubbed —
the same pattern as ``test_tool_loop_characterization``.
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


from modules.policy.types import (  # noqa: E402
    Decision,
    PolicyError,
    Verdict,
    merge_verdicts,
)
from modules.policy import roles  # noqa: E402
from modules.policy import policy_document as pd  # noqa: E402
from modules.policy import budget as bd  # noqa: E402
from modules.policy.errors import ensure_error_envelope, verdict_to_result  # noqa: E402
from modules.policy.bus import PolicyBus, EventContext  # noqa: E402
from modules.policy.types import Event  # noqa: E402


# ---------------------------------------------------------------------------
# §6.8 — deny > ask > allow
# ---------------------------------------------------------------------------

def test_merge_deny_beats_ask_beats_allow():
    allow = Verdict.allow("ok")
    ask = Verdict.ask(PolicyError("approval_required", "ask a human"))
    deny = Verdict.deny(PolicyError("permission_denied", "no"))

    assert merge_verdicts(allow, ask, deny).decision is Decision.DENY
    assert merge_verdicts(deny, ask, allow).decision is Decision.DENY  # order-independent
    assert merge_verdicts(allow, ask).decision is Decision.ASK
    assert merge_verdicts(allow, allow).decision is Decision.ALLOW


def test_merge_empty_and_defer_are_allow():
    assert merge_verdicts().decision is Decision.ALLOW
    assert merge_verdicts(None, None).decision is Decision.ALLOW
    assert merge_verdicts(Verdict.defer("no opinion")).decision is Decision.ALLOW
    # a defer must never override an explicit allow-with-rewrite
    v = merge_verdicts(Verdict.allow("a", updated_input={"x": 1}), Verdict.defer())
    assert v.decision is Decision.ALLOW
    assert v.updated_input == {"x": 1}


def test_deny_carries_structured_error_the_model_can_read():
    err = PolicyError(
        code="budget_exceeded",
        message_for_model="over budget",
        remediation="raise the ceiling",
        retryable=False,
    )
    v = Verdict.deny(err)
    assert v.blocks_execution is True
    assert v.error is not None
    d = v.error.to_dict()
    assert set(d) == {"code", "message_for_model", "remediation", "retryable"}
    assert d["code"] == "budget_exceeded"


# ---------------------------------------------------------------------------
# §6.6 — F043 (super_admin ⊇ admin ⊇ user) + F042 (empty perms = deny)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("role,expect_admin", [
    ("super_admin", True),
    ("admin", True),
    ("user", False),
    (None, False),
    ("nonsense", False),
])
def test_f043_role_hierarchy(role, expect_admin):
    assert roles.is_admin(role) is expect_admin


def test_f043_super_admin_only_super_admin():
    assert roles.is_super_admin("super_admin") is True
    assert roles.is_super_admin("admin") is False
    assert roles.role_satisfies("super_admin", "admin") is True  # ⊇
    assert roles.role_satisfies("admin", "super_admin") is False  # not ⊇ upward


@pytest.mark.parametrize("perms,required,expect", [
    ([], "widget:chat", False),          # empty = DENY (F042 — the fix)
    (None, "widget:chat", False),        # null = DENY
    (["widget:chat"], "widget:chat", True),
    (["widget:read"], "widget:chat", False),
    (["*"], "anything", True),           # explicit wildcard grant is honoured
])
def test_f042_empty_permission_is_deny(perms, required, expect):
    assert roles.has_permission(perms, required) is expect


# ---------------------------------------------------------------------------
# §6.7 — Balanced act-vs-ask routing (§5)
# ---------------------------------------------------------------------------

def test_balanced_defaults():
    doc = pd.load_policy_document(None, None)  # no DB → Balanced defaults
    assert doc.posture == pd.BALANCED
    assert doc.agents_inherit_admin is False   # F014 default-OFF


@pytest.mark.parametrize("risk,route", [
    (pd.RISK_READ, "auto"),
    (pd.RISK_INTERNAL_WRITE, "auto"),
    (pd.RISK_EXTERNAL, "ask"),
    (pd.RISK_DESTRUCTIVE, "ask"),
    (pd.RISK_PUBLISH, "ask"),
])
def test_balanced_routing_table(risk, route):
    doc = pd.load_policy_document(None, None)
    assert doc.route_for(risk) == route


def test_route_override_wins_over_posture():
    doc = pd.PolicyDocument(
        posture=pd.BALANCED, agents_inherit_admin=False,
        route_overrides={pd.RISK_EXTERNAL: "auto"},
    )
    assert doc.route_for(pd.RISK_EXTERNAL) == "auto"     # override
    assert doc.route_for(pd.RISK_DESTRUCTIVE) == "ask"   # unaffected


@pytest.mark.parametrize("name,perm,composio,expect", [
    ("platform_list_agents", "read", False, pd.RISK_READ),
    ("platform_update_agent", "write", False, pd.RISK_INTERNAL_WRITE),
    ("platform_delete_agent", "destructive", False, pd.RISK_DESTRUCTIVE),
    ("composio_execute", None, True, pd.RISK_EXTERNAL),
    ("COMPOSIO_GMAIL_SEND_EMAIL", None, True, pd.RISK_EXTERNAL),
    ("platform_publish_template", None, False, pd.RISK_PUBLISH),
    # a destructive composio action is destructive, not merely external
    ("composio_shopify_delete", "destructive", True, pd.RISK_DESTRUCTIVE),
])
def test_classify_action(name, perm, composio, expect):
    assert pd.classify_action(name, permission_level=perm, is_composio=composio) == expect


# ---------------------------------------------------------------------------
# §6.3 — F086/F059 model-aware budget admission (pure: no DB, direct numbers)
# ---------------------------------------------------------------------------

class _FakeWorkspace:
    def __init__(self, budget):
        self.plan_limits = {"budget": budget} if budget is not None else {}


class _FakeQuery:
    """Minimal stand-in for a SQLAlchemy query chain used by budget.py."""

    def __init__(self, ws=None, spend=(0.0, 0.0)):
        self._ws = ws
        self._spend = spend
        self._mode = None

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._ws

    def one(self):
        return self._spend


class _FakeDB:
    """Returns a workspace for the Workspace query and a spend tuple for the sum."""

    def __init__(self, ws=None, spend=(0.0, 0.0)):
        self._ws = ws
        self._spend = spend

    def query(self, *entities):
        # budget.load_budget queries Workspace; spend_to_date queries sums.
        first = entities[0] if entities else None
        name = getattr(first, "__name__", "")
        if name == "Workspace":
            return _FakeQuery(ws=self._ws)
        return _FakeQuery(spend=self._spend)


def test_budget_no_ceiling_allows():
    db = _FakeDB(ws=_FakeWorkspace(None))
    d = bd.check_budget(db, "ws", projected_cost_usd=999.0)
    assert d.allowed is True


def test_budget_over_cost_ceiling_denies():
    # ceiling $5, already spent $4.90, this call +$0.20 → over
    db = _FakeDB(ws=_FakeWorkspace({"max_cost_usd": 5.0, "window": "day"}),
                 spend=(4.90, 1000))
    d = bd.check_budget(db, "ws", projected_cost_usd=0.20)
    assert d.allowed is False
    assert d.dimension == "cost_usd"


def test_budget_within_cost_ceiling_allows():
    db = _FakeDB(ws=_FakeWorkspace({"max_cost_usd": 5.0, "window": "day"}),
                 spend=(1.00, 1000))
    d = bd.check_budget(db, "ws", projected_cost_usd=0.20)
    assert d.allowed is True


def test_budget_over_token_ceiling_denies():
    db = _FakeDB(ws=_FakeWorkspace({"max_total_tokens": 10_000, "window": "day"}),
                 spend=(0.0, 9_950))
    d = bd.check_budget(db, "ws", projected_tokens=100)
    assert d.allowed is False
    assert d.dimension == "total_tokens"


# ---------------------------------------------------------------------------
# §4.2 — errors-as-data envelope
# ---------------------------------------------------------------------------

def test_verdict_to_result_is_non_success_with_structured_error():
    v = Verdict.deny(PolicyError("permission_denied", "nope", remediation="escalate"))
    r = verdict_to_result(v, "platform_delete_agent")
    assert r["success"] is False
    assert r["permission_denied"] is True
    assert r["policy_error"]["code"] == "permission_denied"
    assert r["policy_error"]["message_for_model"] == "nope"
    # the model-readable line is also on the legacy key the loop already surfaces
    assert "nope" in r["llm_context"]


def test_verdict_to_result_ask_marks_requires_approval():
    v = Verdict.ask(PolicyError("approval_required", "ask a human", retryable=True))
    r = verdict_to_result(v, "composio_execute")
    assert r["requires_approval"] is True
    assert r["policy_decision"] == "ask"
    assert r["policy_error"]["retryable"] is True


def test_ensure_error_envelope_backfills_failures_only():
    ok = ensure_error_envelope({"success": True, "data": 1})
    assert "policy_error" not in ok  # success untouched

    fail = ensure_error_envelope({"success": False, "error": "boom", "rate_limited": True})
    assert fail["policy_error"]["code"] == "rate_limited"
    assert fail["policy_error"]["retryable"] is True

    # an existing policy_error is preserved, not overwritten
    pre = {"success": False, "policy_error": {"code": "keep", "message_for_model": "m",
                                              "remediation": None, "retryable": False}}
    assert ensure_error_envelope(pre)["policy_error"]["code"] == "keep"


# ---------------------------------------------------------------------------
# Bus fault isolation — a raising handler is no-opinion, never a silent allow
# ---------------------------------------------------------------------------

def test_bus_raising_handler_is_no_opinion():
    bus = PolicyBus()
    bus.register(Event.PRE_TOOL_USE, lambda e, c: Verdict.ask(PolicyError("x", "ask")))
    bus.register(Event.PRE_TOOL_USE, lambda e, c: (_ for _ in ()).throw(RuntimeError("boom")))
    v = bus.fire(Event.PRE_TOOL_USE, EventContext(tool_name="t"))
    assert v.decision is Decision.ASK  # raising handler did not wave it through


def test_bus_no_handlers_allows():
    bus = PolicyBus()
    assert bus.fire(Event.PRE_TOOL_USE, EventContext(tool_name="t")).decision is Decision.ALLOW
