"""PRD-192 S3 (P2-11) — price the budget gate: one pricing source, real
projected cost.

Pins the story's four moves:

1. **Estimates thread** — a ``caller_context`` carrying the driving model +
   turn estimate reaches ``ToolCall`` at the executor chokepoint, and the gate
   prices it (``projected_cost > 0`` — previously structurally $0 because no
   caller passed estimates).
2. **Budget admission binds pre-call** — a projected overage is denied BEFORE
   execution, not caught after the spend.
3. **The board gate gets a real dollar figure** — ``auto_below_budget`` with a
   binding ceiling now asks (C.5 closed).
4. **One pricing source (F059 finished)** — ``COORDINATOR_COST_PER_1K_TOKENS``
   has zero consumers outside ``modules/policy/pricing.py`` + ``config.py``
   (source-grep guard), and the flat rate applies ONLY on a registry miss.

Plus the locked #2a default: autonomy-enabled workspaces with no explicit
``plan_limits.budget`` get the 50 USD/month code-default ceiling.

Pure — registry/session/pricing are mocked at their boundaries; heavy imports
(executor, board router, chat service) are CI-gated like the sibling suites.
"""
from __future__ import annotations

import re
import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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
from modules.policy import policy_document as pd  # noqa: E402
from modules.policy import pricing as pricing_mod  # noqa: E402
from modules.policy.budget import BudgetDecision, check_budget  # noqa: E402
from modules.policy.gate import PolicyGate, ToolCall  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Estimates thread into the gate's budget stage (pure)
# ---------------------------------------------------------------------------


def test_gate_prices_toolcall_estimates(monkeypatch):
    """model+tokens on the ToolCall ⇒ the budget stage sees projected_cost > 0."""
    captured = {}

    def _capture_budget(db, ws, **kw):
        captured.update(kw)
        return BudgetDecision(True, "ok")

    g = PolicyGate(db="fake-db")
    monkeypatch.setattr(g, "_lookup_action", lambda name: None)
    monkeypatch.setattr(g, "_full_autonomy", lambda ws: False)
    monkeypatch.setattr(
        gate_mod._policy_doc, "load_policy_document",
        lambda db, ws: pd.PolicyDocument(pd.BALANCED, False, {}),
    )
    monkeypatch.setattr(
        gate_mod._pricing, "estimate_cost_usd", lambda db, m, i, o: 0.5
    )
    monkeypatch.setattr(gate_mod._budget, "check_budget", _capture_budget)

    g.check(ToolCall(
        tool_name="platform_list_agents", parameters={}, workspace_id="ws-1",
        model_id="openai/gpt-4o", est_input_tokens=1000, est_output_tokens=500,
    ))
    assert captured["projected_cost_usd"] == 0.5
    assert captured["projected_tokens"] == 1500


def test_gate_unpriced_call_stays_spend_to_date_only(monkeypatch):
    """No model/tokens ⇒ projected 0 (spend-to-date still binds) — unchanged."""
    captured = {}

    def _capture_budget(db, ws, **kw):
        captured.update(kw)
        return BudgetDecision(True, "ok")

    g = PolicyGate(db="fake-db")
    monkeypatch.setattr(g, "_lookup_action", lambda name: None)
    monkeypatch.setattr(g, "_full_autonomy", lambda ws: False)
    monkeypatch.setattr(
        gate_mod._policy_doc, "load_policy_document",
        lambda db, ws: pd.PolicyDocument(pd.BALANCED, False, {}),
    )
    monkeypatch.setattr(gate_mod._budget, "check_budget", _capture_budget)

    g.check(ToolCall(tool_name="platform_list_agents", parameters={}, workspace_id="ws-1"))
    assert captured["projected_cost_usd"] == 0.0


# ---------------------------------------------------------------------------
# 2. Budget admission denies a projected overage BEFORE the call
# ---------------------------------------------------------------------------


def test_budget_denies_projected_overage(monkeypatch):
    from modules.policy import budget as budget_mod

    monkeypatch.setattr(
        budget_mod, "load_budget",
        lambda db, ws: {"window": "day", "max_cost_usd": 1.0},
    )
    monkeypatch.setattr(
        budget_mod, "spend_to_date",
        lambda db, ws, window: {"cost_usd": 0.8, "total_tokens": 100.0},
    )

    decision = check_budget(None, "ws-1", projected_cost_usd=0.5, projected_tokens=100)
    assert decision.allowed is False  # 0.8 spent + 0.5 pending > 1.0 ceiling
    assert decision.dimension == "cost_usd"

    ok = check_budget(None, "ws-1", projected_cost_usd=0.1, projected_tokens=100)
    assert ok.allowed is True  # 0.9 <= 1.0 — the same call priced smaller admits


# ---------------------------------------------------------------------------
# 3. One pricing source (F059 finished)
# ---------------------------------------------------------------------------


def test_flat_fallback_only_on_registry_miss(monkeypatch):
    # Registry hit ⇒ blended model rate, NOT the flat rate.
    monkeypatch.setattr(
        pricing_mod, "price_per_1k",
        lambda db, m: pricing_mod.ModelPrice(m, input_per_1k=0.01, output_per_1k=0.03),
    )
    assert pricing_mod.price_total_tokens_usd("db", "gpt-x", 1000) == 0.02

    # Registry miss ⇒ the documented flat last-resort.
    monkeypatch.setattr(pricing_mod, "price_per_1k", lambda db, m: None)
    flat = pricing_mod.flat_rate_per_1k()
    assert pricing_mod.price_total_tokens_usd("db", "gpt-x", 1000) == round(flat, 6)

    # No model at all ⇒ flat, without touching the registry.
    def _boom(db, m):
        raise AssertionError("no model ⇒ registry must not be consulted")

    monkeypatch.setattr(pricing_mod, "price_per_1k", _boom)
    assert pricing_mod.price_total_tokens_usd("db", None, 2000) == round(2 * flat, 6)


def test_single_pricing_source():
    """Source-grep guard (PRD-185 S5 shape): COORDINATOR_COST_PER_1K_TOKENS has
    NO consumer outside modules/policy/pricing.py and config.py. When this
    fails, a new flat-rate consumer crept in — route it through pricing."""
    allowed = {
        Path("modules/policy/pricing.py"),
        Path("config.py"),
    }
    offenders = []
    for path in _ORCH.rglob("*.py"):
        rel = path.relative_to(_ORCH)
        parts = rel.parts
        if parts[0] in ("tests", "alembic") or "node_modules" in parts:
            continue
        if rel in allowed:
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if re.search(r"COORDINATOR_COST_PER_1K_TOKENS", text):
            offenders.append(str(rel))
    assert offenders == [], (
        f"flat-rate consumers outside pricing.py/config.py: {offenders}"
    )


# ---------------------------------------------------------------------------
# 4. The shared turn estimator (chat + agent lanes)
# ---------------------------------------------------------------------------


def test_estimate_turn_budget_prompt_plus_output_cap():
    from core.context_guard import estimate_turn_budget

    manager = SimpleNamespace(
        config=SimpleNamespace(model="openai/gpt-4o", max_tokens=1000)
    )
    messages = [
        {"role": "system", "content": "You are Auto."},
        {"role": "user", "content": "Summarise the Q3 report for me please."},
    ]
    est = estimate_turn_budget(manager, messages)
    assert est["model_id"] == "openai/gpt-4o"
    assert est["est_input_tokens"] > 0
    assert est["est_output_tokens"] == 1000


def test_estimate_turn_budget_no_model_is_empty():
    from core.context_guard import estimate_turn_budget

    assert estimate_turn_budget(SimpleNamespace(config=None), []) == {}
    assert estimate_turn_budget(None, []) == {}


def test_build_tool_caller_context_carries_estimates():
    try:
        from consumers.chatbot.service import build_tool_caller_context
    except Exception as exc:  # pragma: no cover - environment-dependent skip
        pytest.skip(f"chat service unavailable (CI-gated): {type(exc).__name__}")

    ctx = build_tool_caller_context(
        user_query="q", conversation_id="c", turn_id="t",
        driving_clerk="user_1", prior_action=None,
        model_id="openai/gpt-4o", est_input_tokens=1200, est_output_tokens=800,
    )
    assert ctx["model_id"] == "openai/gpt-4o"
    assert ctx["est_input_tokens"] == 1200
    assert ctx["est_output_tokens"] == 800

    # Without a model the estimate keys are omitted (telemetry stays clean).
    ctx = build_tool_caller_context(
        user_query="q", conversation_id="c", turn_id="t",
        driving_clerk="user_1", prior_action=None,
    )
    assert "model_id" not in ctx
    assert "est_input_tokens" not in ctx


# ---------------------------------------------------------------------------
# 5. Executor chokepoint lifts caller_context estimates into ToolCall
# ---------------------------------------------------------------------------

try:
    import modules.tools.execution.unified_executor as unified_executor
    _EXECUTOR_AVAILABLE = True
    _EXECUTOR_SKIP = ""
except Exception as _exc:  # pragma: no cover
    _EXECUTOR_AVAILABLE = False
    _EXECUTOR_SKIP = f"UnifiedToolExecutor unavailable (CI-gated): {type(_exc).__name__}"


@pytest.mark.skipif(not _EXECUTOR_AVAILABLE, reason=_EXECUTOR_SKIP or "executor unavailable")
def test_toolcall_carries_estimates(monkeypatch):
    from modules.policy.types import Verdict

    class _FakeSession:
        def query(self, *a, **k):
            raise AssertionError("gate is monkeypatched")

    monkeypatch.setattr("modules.policy.policy_plane_mode", lambda: "on", raising=False)
    ex = unified_executor.UnifiedToolExecutor(db_session=_FakeSession())
    monkeypatch.setattr(ex, "_policy_action_def", lambda name: None)

    captured = {}

    def _check(self, call):
        captured["call"] = call
        return Verdict.allow("fine")

    monkeypatch.setattr("modules.policy.PolicyGate.check", _check, raising=False)
    fires = []
    monkeypatch.setattr(
        ex, "_fire_policy_bus",
        lambda name, params, verdict, **kw: fires.append(kw),
    )

    ex._policy_gate_check(
        "platform_list_agents", {},
        agent_id=1, workspace_id="ws",
        caller_context={
            "model_id": "openai/gpt-4o",
            "est_input_tokens": 1200,
            "est_output_tokens": 800,
        },
        trace="t",
    )
    call = captured["call"]
    assert call.model_id == "openai/gpt-4o"
    assert call.est_input_tokens == 1200
    assert call.est_output_tokens == 800
    # G.2: the bus fire carries the estimate so the audit row records it.
    assert fires and fires[0]["est_tokens"] == 2000


# ---------------------------------------------------------------------------
# 6. Board gate — real estimate, binding ceiling (C.5)
# ---------------------------------------------------------------------------

try:
    import api.board_tasks as board_tasks_mod
    _BOARD_AVAILABLE = True
    _BOARD_SKIP = ""
except Exception as _exc:  # pragma: no cover
    _BOARD_AVAILABLE = False
    _BOARD_SKIP = f"api.board_tasks unavailable (CI-gated): {type(_exc).__name__}"

_needs_board = pytest.mark.skipif(
    not _BOARD_AVAILABLE, reason=_BOARD_SKIP or "board_tasks unavailable"
)


@_needs_board
def test_board_task_estimate_is_priced(monkeypatch):
    task = SimpleNamespace(raw_prompt="Do the quarterly numbers " * 50,
                           title="t", description="d")
    agent = SimpleNamespace(model_config={"model_id": "openai/gpt-4o", "max_tokens": 500})

    db = MagicMock()
    db.query.return_value.get.side_effect = [task, agent]

    seen = {}

    def _priced(db_, model_id, est_in, est_out):
        seen.update(model=model_id, est_in=est_in, est_out=est_out)
        return 0.42

    monkeypatch.setattr("modules.policy.pricing.estimate_cost_usd", _priced)
    cost = board_tasks_mod._estimate_board_task_cost_usd(db, 7, 1)
    assert cost == 0.42
    assert seen["model"] == "openai/gpt-4o"
    assert seen["est_in"] > 0
    assert seen["est_out"] == 500


@_needs_board
def test_board_gate_receives_real_estimate(monkeypatch):
    """The board consult passes the priced figure — not the 0.0 default."""
    monkeypatch.setattr(
        "core.services.approval_grants.find_active_grant", lambda *a, **k: None
    )
    monkeypatch.setattr(
        board_tasks_mod, "_estimate_board_task_cost_usd", lambda db, t, a: 0.42
    )
    captured = {}

    def _evaluate(db, **kw):
        captured.update(kw)
        return SimpleNamespace(requires_approval=False, reason="auto", grant=None)

    monkeypatch.setattr(
        "services.board_approval.evaluate_board_task_approval", _evaluate
    )

    blocked = board_tasks_mod._board_task_blocked_pending_approval(
        MagicMock(), 7, 1, "ws-1"
    )
    assert blocked is False
    assert captured["estimated_cost_usd"] == 0.42


def test_auto_below_budget_binds_with_real_estimate(monkeypatch):
    """C.5 closed: a priced figure over the ceiling now ASKS instead of
    auto-approving (the 0.0 default made auto_below_budget a rubber stamp)."""
    pytest.importorskip("core.services.approval_grants")
    pytest.importorskip("core.models.approval_grants")
    try:
        from services.board_approval import evaluate_board_task_approval
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"board_approval unavailable (CI-gated): {type(exc).__name__}")

    monkeypatch.setattr(
        "core.services.approval_grants.find_pending_grant", lambda *a, **k: None
    )
    grant = SimpleNamespace(id=11)
    monkeypatch.setattr(
        "core.services.approval_grants.create_grant", lambda *a, **k: grant
    )
    monkeypatch.setattr("services.board_approval._audit_governance", lambda *a, **k: None)

    over = evaluate_board_task_approval(
        MagicMock(), workspace_id="ws-1", task_id=7,
        estimated_cost_usd=0.42,
        _policy_override="auto_below_budget", _ceiling_override=0.10,
    )
    assert over.requires_approval is True  # 0.42 > 0.10 — the ceiling binds

    under = evaluate_board_task_approval(
        MagicMock(), workspace_id="ws-1", task_id=7,
        estimated_cost_usd=0.05,
        _policy_override="auto_below_budget", _ceiling_override=0.10,
    )
    assert under.requires_approval is False


# ---------------------------------------------------------------------------
# 7. Locked #2a — autonomy default ceiling (50 USD / month)
# ---------------------------------------------------------------------------


def _ws_db(plan_limits):
    ws = SimpleNamespace(plan_limits=plan_limits)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = ws
    return db


def test_autonomy_workspace_gets_default_monthly_ceiling(monkeypatch):
    pytest.importorskip("core.models.workspaces")
    pytest.importorskip("core.services.auto_autonomy")
    from modules.policy.budget import load_budget

    monkeypatch.setattr(
        "core.services.auto_autonomy.is_full_autonomy", lambda db, ws: True
    )
    budget = load_budget(_ws_db({}), "ws-1")
    assert budget["max_cost_usd"] == 50.0
    assert budget["window"] == "month"
    assert budget["default_applied"] is True


def test_explicit_budget_wins_over_default(monkeypatch):
    pytest.importorskip("core.models.workspaces")
    pytest.importorskip("core.services.auto_autonomy")
    from modules.policy.budget import load_budget

    def _boom(db, ws):
        raise AssertionError("explicit budget must not consult the autonomy dial")

    monkeypatch.setattr("core.services.auto_autonomy.is_full_autonomy", _boom, raising=False)
    budget = load_budget(
        _ws_db({"budget": {"max_cost_usd": 10, "window": "day"}}), "ws-1"
    )
    assert budget == {"window": "day", "max_cost_usd": 10.0}


def test_supervised_workspace_stays_ceiling_less(monkeypatch):
    pytest.importorskip("core.models.workspaces")
    pytest.importorskip("core.services.auto_autonomy")
    from modules.policy.budget import load_budget

    monkeypatch.setattr(
        "core.services.auto_autonomy.is_full_autonomy", lambda db, ws: False
    )
    assert load_budget(_ws_db({}), "ws-1") == {}
