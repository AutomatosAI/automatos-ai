"""PRD-192 S1 (P2-11) — the staged mode dial, the risk-gated fail-closed branch,
and honest per-action Composio classification.

Pins the three changes that make flipping ``AUTOMATOS_POLICY_PLANE`` safe:

1. **The mode dial** — ``off | shadow | destructive | on`` (legacy booleans
   map). ``off`` is byte-for-byte legacy (no bus fire, no classification);
   ``shadow`` evaluates + audits but NEVER blocks; ``destructive`` enforces
   only the fail-closed risk classes; ``on`` enforces everything blockable.
2. **The fail matrix (locked)** — on a plane fault under enforce modes,
   destructive / external_side_effect / publish ⇒ deny errors-as-data
   (``policy_plane_error``); read / internal_write ⇒ proceed with the greppable
   ``[policy-fail-open]`` marker; unclassifiable ⇒ treated destructive; the
   budget/posture readers stop pre-deciding allow on fault; the board approval
   gate blocks pending approval instead of launching.
3. **Honest classification** — a per-action Composio name routed via the
   executor's ``composio_actions`` dict reaches the gate as
   ``external_side_effect`` (previously: ``internal_write`` ⇒ auto-allowed
   under Balanced even with the plane ON).

Pure at every boundary: ``PolicyGate``/registry/bus are monkeypatched, the DB
session is a fake, config is toggled at its single source of truth. The
executor-level tests are CI-gated on the heavy import, mirroring
``test_prd174_executor_chokepoint.py``.
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
import types as _types
from pathlib import Path
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


import config as _config_mod  # noqa: E402

from modules.policy.types import Decision, PolicyError, Verdict  # noqa: E402


# ---------------------------------------------------------------------------
# The flag module — mode reader fails safe, enforce set is exact
# ---------------------------------------------------------------------------


def test_policy_plane_mode_reads_config(monkeypatch):
    from modules.policy.flag import policy_plane_mode

    for mode in ("off", "shadow", "destructive", "on"):
        monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_MODE", mode, raising=False)
        assert policy_plane_mode() == mode


def test_policy_plane_mode_unknown_value_fails_safe_off(monkeypatch):
    from modules.policy.flag import policy_plane_mode

    monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_MODE", "bogus", raising=False)
    assert policy_plane_mode() == "off"
    monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_MODE", None, raising=False)
    assert policy_plane_mode() == "off"


def test_enforcement_active_only_in_enforce_stages(monkeypatch):
    from modules.policy.flag import enforcement_active

    for mode, expected in (
        ("off", False), ("shadow", False), ("destructive", True), ("on", True),
    ):
        monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_MODE", mode, raising=False)
        assert enforcement_active() is expected, mode


def test_fail_closed_risk_classes_are_exactly_the_locked_three():
    """The destructive stage enforces EXACTLY these classes, and the same set
    fails closed on a plane fault (one frozenset — Gerard's locked #1)."""
    from modules.policy.policy_document import (
        FAIL_CLOSED_RISK_CLASSES,
        RISK_DESTRUCTIVE,
        RISK_EXTERNAL,
        RISK_PUBLISH,
    )

    assert FAIL_CLOSED_RISK_CLASSES == {RISK_DESTRUCTIVE, RISK_EXTERNAL, RISK_PUBLISH}


# ---------------------------------------------------------------------------
# Config parse — stages + legacy boolean mapping (reload pattern from
# test_config_env_centralization.py, blast radius contained)
# ---------------------------------------------------------------------------


@pytest.fixture
def _reload_config_restored():
    """Snapshot + restore the ``config`` module object and os.environ around a
    reload test so downstream suites keep the class identity they bound."""
    env_snapshot = dict(os.environ)
    saved = sys.modules.get("config")
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(env_snapshot)
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)


def _reload_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("AUTOMATOS_POLICY_PLANE", raising=False)
    else:
        monkeypatch.setenv("AUTOMATOS_POLICY_PLANE", value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **kw: False)
    sys.modules.pop("config", None)
    import config  # noqa: WPS433 — intentional re-import after env change

    importlib.reload(config)
    return config


def test_mode_defaults_off(monkeypatch, _reload_config_restored):
    cfg = _reload_config(monkeypatch, None)
    assert cfg.config.POLICY_PLANE_MODE == "off"
    assert cfg.config.POLICY_PLANE_ENABLED is False


@pytest.mark.parametrize(
    "raw,expected_mode,expected_enabled",
    [
        ("true", "on", True),       # legacy boolean maps to on
        ("1", "on", True),
        ("yes", "on", True),
        ("false", "off", False),    # legacy boolean maps to off
        ("0", "off", False),
        ("no", "off", False),
        ("shadow", "shadow", True),
        ("destructive", "destructive", True),
        ("on", "on", True),
        ("off", "off", False),
        ("SHADOW", "shadow", True),  # case-insensitive
        ("bogus", "off", False),     # unknown fails safe to off
    ],
)
def test_legacy_boolean_env_maps(
    monkeypatch, _reload_config_restored, raw, expected_mode, expected_enabled
):
    cfg = _reload_config(monkeypatch, raw)
    assert cfg.config.POLICY_PLANE_MODE == expected_mode
    assert cfg.config.POLICY_PLANE_ENABLED is expected_enabled


# ---------------------------------------------------------------------------
# Budget / posture readers — no silent except swallows deciding policy
# ---------------------------------------------------------------------------


class _RaisingDB:
    def query(self, *a, **k):
        raise RuntimeError("db down")


def test_budget_reader_raises_under_enforce(monkeypatch):
    from modules.policy import budget as budget_mod

    monkeypatch.setattr("modules.policy.flag.enforcement_active", lambda: True)
    with pytest.raises(Exception):
        budget_mod.load_budget(_RaisingDB(), "ws-1")
    with pytest.raises(Exception):
        budget_mod.spend_to_date(_RaisingDB(), "ws-1", "day")


def test_budget_reader_swallows_when_not_enforced(monkeypatch):
    from modules.policy import budget as budget_mod

    monkeypatch.setattr("modules.policy.flag.enforcement_active", lambda: False)
    assert budget_mod.load_budget(_RaisingDB(), "ws-1") == {}
    assert budget_mod.spend_to_date(_RaisingDB(), "ws-1", "day") == {
        "cost_usd": 0.0, "total_tokens": 0.0,
    }


def test_posture_reader_raises_under_enforce(monkeypatch):
    from modules.policy import policy_document as pd

    monkeypatch.setattr("modules.policy.flag.enforcement_active", lambda: True)
    with pytest.raises(Exception):
        pd.load_policy_document(_RaisingDB(), "ws-1")


def test_posture_reader_defaults_when_not_enforced(monkeypatch):
    from modules.policy import policy_document as pd

    monkeypatch.setattr("modules.policy.flag.enforcement_active", lambda: False)
    doc = pd.load_policy_document(_RaisingDB(), "ws-1")
    assert doc.posture == pd.BALANCED


# ---------------------------------------------------------------------------
# Executor mode dial — CI-gated on the heavy import (chokepoint pattern)
# ---------------------------------------------------------------------------

try:
    import modules.tools.execution.unified_executor as unified_executor
    _EXECUTOR_AVAILABLE = True
    _EXECUTOR_SKIP_REASON = ""
except Exception as _exc:  # pragma: no cover - environment-dependent skip
    _EXECUTOR_AVAILABLE = False
    _EXECUTOR_SKIP_REASON = f"UnifiedToolExecutor unavailable (CI-gated): {type(_exc).__name__}"

_needs_executor = pytest.mark.skipif(
    not _EXECUTOR_AVAILABLE, reason=_EXECUTOR_SKIP_REASON or "executor unavailable"
)


class _FakeSession:
    """No-op DB session — the gate is monkeypatched so it's never really used."""

    def query(self, *a, **k):
        raise AssertionError("gate was monkeypatched; DB should not be queried")


def _make_executor(monkeypatch=None):
    ex = unified_executor.UnifiedToolExecutor(db_session=_FakeSession())
    if monkeypatch is not None:
        # Deterministic classification: no registry (permission_level=None), so
        # the pure name/hint classifier decides — the thing under test.
        monkeypatch.setattr(ex, "_policy_action_def", lambda name: None)
    return ex


def _set_mode(monkeypatch, mode: str):
    monkeypatch.setattr("modules.policy.policy_plane_mode", lambda: mode, raising=False)


def _record_bus(monkeypatch, ex):
    fires = []

    def _rec(effective_name, effective_params, verdict, **kw):
        fires.append({"name": effective_name, "verdict": verdict, **kw})

    monkeypatch.setattr(ex, "_fire_policy_bus", _rec)
    return fires


def _gate_returns(monkeypatch, verdict):
    calls = []

    def _check(self, call):
        calls.append(call)
        return verdict

    monkeypatch.setattr("modules.policy.PolicyGate.check", _check, raising=False)
    return calls


def _gate_raises(monkeypatch, exc=None):
    def _check(self, call):
        raise exc or RuntimeError("plane fault")

    monkeypatch.setattr("modules.policy.PolicyGate.check", _check, raising=False)


_DENY = Verdict.deny(PolicyError(
    code="approval_required",
    message_for_model="Blocked by policy; NOT executed.",
    remediation="approve it", retryable=True,
))
_ASK = Verdict.ask(PolicyError(
    code="approval_required",
    message_for_model="Needs approval; NOT executed.",
    remediation="approve it", retryable=True,
))


@_needs_executor
def test_mode_off_is_byte_for_byte(monkeypatch):
    """off ⇒ None, no bus fire, no classification, gate never constructed."""
    _set_mode(monkeypatch, "off")
    ex = _make_executor()

    def _tripwire(*a, **k):
        raise AssertionError("mode off must not classify / fire the bus / check the gate")

    monkeypatch.setattr(ex, "_classify_risk", _tripwire)
    monkeypatch.setattr(ex, "_fire_policy_bus", _tripwire)
    monkeypatch.setattr("modules.policy.PolicyGate.check", _tripwire, raising=False)

    assert ex._policy_gate_check(
        "platform_delete_agent", {"id": 1}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    ) is None


@_needs_executor
def test_shadow_never_blocks_but_audits(monkeypatch):
    """A deny verdict under shadow returns None AND the bus fired with mode=shadow."""
    _set_mode(monkeypatch, "shadow")
    ex = _make_executor(monkeypatch)
    fires = _record_bus(monkeypatch, ex)
    ex.composio_actions["GMAIL_SEND_EMAIL"] = "GMAIL"
    _gate_returns(monkeypatch, _DENY)

    blocked = ex._policy_gate_check(
        "GMAIL_SEND_EMAIL", {"to": "x@y.z"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is None  # shadow NEVER blocks
    assert len(fires) == 1
    assert fires[0]["mode"] == "shadow"
    assert fires[0]["verdict"].decision is Decision.DENY
    assert fires[0]["risk"] == "external_side_effect"


@_needs_executor
def test_destructive_stage_blocks_external_ask(monkeypatch):
    """The destructive stage enforces ask/deny for the closed classes."""
    _set_mode(monkeypatch, "destructive")
    ex = _make_executor(monkeypatch)
    _record_bus(monkeypatch, ex)
    ex.composio_actions["GMAIL_SEND_EMAIL"] = "GMAIL"
    _gate_returns(monkeypatch, _ASK)

    blocked = ex._policy_gate_check(
        "GMAIL_SEND_EMAIL", {"to": "x@y.z"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is not None
    assert blocked["success"] is False
    assert blocked["requires_approval"] is True
    assert blocked["policy_error"]["code"] == "approval_required"


@_needs_executor
def test_destructive_stage_shadow_logs_internal_deny(monkeypatch):
    """The stage boundary is the risk class: an internal_write deny under the
    destructive stage is shadow-logged, not enforced."""
    _set_mode(monkeypatch, "destructive")
    ex = _make_executor(monkeypatch)
    fires = _record_bus(monkeypatch, ex)
    _gate_returns(monkeypatch, _DENY)

    # write_file is builtin-routed ⇒ not composio ⇒ classifies internal_write.
    blocked = ex._policy_gate_check(
        "write_file", {"path": "a", "content": "b"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is None  # open class — below the destructive stage boundary
    assert len(fires) == 1  # still audited
    assert fires[0]["risk"] == "internal_write"


@_needs_executor
def test_enforce_all_blocks_everything_blockable(monkeypatch):
    """Mode on ⇒ every blocking verdict is enforced, open classes included."""
    _set_mode(monkeypatch, "on")
    ex = _make_executor(monkeypatch)
    _record_bus(monkeypatch, ex)
    _gate_returns(monkeypatch, _DENY)

    blocked = ex._policy_gate_check(
        "write_file", {"path": "a", "content": "b"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is not None
    assert blocked["success"] is False
    assert blocked["policy_decision"] == "deny"


@_needs_executor
def test_gate_error_fails_closed_for_destructive(monkeypatch):
    """A raising PolicyGate.check on a closed-class call ⇒ deny errors-as-data,
    never execution — and the synthetic deny is audited."""
    _set_mode(monkeypatch, "on")
    ex = _make_executor(monkeypatch)
    fires = _record_bus(monkeypatch, ex)
    ex.composio_actions["GMAIL_SEND_EMAIL"] = "GMAIL"
    _gate_raises(monkeypatch)

    blocked = ex._policy_gate_check(
        "GMAIL_SEND_EMAIL", {"to": "x@y.z"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is not None
    assert blocked["success"] is False
    assert blocked["policy_error"]["code"] == "policy_plane_error"
    assert blocked["policy_error"]["retryable"] is True
    assert "NOT executed" in blocked["llm_context"]
    assert len(fires) == 1  # the fault denial is recorded (Art.12)
    assert fires[0]["verdict"].decision is Decision.DENY


@_needs_executor
def test_gate_error_fails_open_for_read(monkeypatch, caplog):
    """The same raise on a read-class call ⇒ proceed, with the greppable
    [policy-fail-open] marker (the G.5 rate the shadow report counts)."""
    _set_mode(monkeypatch, "on")
    ex = _make_executor(monkeypatch)
    _record_bus(monkeypatch, ex)
    _gate_raises(monkeypatch)

    with caplog.at_level(logging.WARNING):
        blocked = ex._policy_gate_check(
            "platform_get_agent", {"id": 1}, agent_id=1,
            workspace_id="ws", caller_context=None, trace="t",
        )
    assert blocked is None  # read fails open (marked + counted)
    assert "[policy-fail-open]" in caplog.text


@_needs_executor
def test_gate_error_shadow_never_blocks(monkeypatch):
    """A plane fault in shadow proceeds even for a closed class."""
    _set_mode(monkeypatch, "shadow")
    ex = _make_executor(monkeypatch)
    _record_bus(monkeypatch, ex)
    ex.composio_actions["GMAIL_SEND_EMAIL"] = "GMAIL"
    _gate_raises(monkeypatch)

    assert ex._policy_gate_check(
        "GMAIL_SEND_EMAIL", {"to": "x@y.z"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    ) is None


@_needs_executor
def test_unclassifiable_error_treated_destructive(monkeypatch):
    """Classification itself failing ⇒ the call is treated destructive (closed)."""
    _set_mode(monkeypatch, "on")
    ex = _make_executor(monkeypatch)
    _record_bus(monkeypatch, ex)
    monkeypatch.setattr(ex, "_classify_risk", lambda name, is_composio: None)
    _gate_raises(monkeypatch)

    blocked = ex._policy_gate_check(
        "mystery_tool", {}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is not None
    assert blocked["policy_error"]["code"] == "policy_plane_error"


@_needs_executor
def test_per_action_composio_classifies_external(monkeypatch):
    """A composio_actions-routed name reaches the gate with the is_composio
    hint and classifies external_side_effect (previously internal_write —
    auto-allowed under Balanced even with the plane ON)."""
    _set_mode(monkeypatch, "on")
    ex = _make_executor(monkeypatch)
    _record_bus(monkeypatch, ex)
    ex.composio_actions["GMAIL_SEND_EMAIL"] = "GMAIL"
    calls = _gate_returns(monkeypatch, Verdict.allow("fine"))

    ex._policy_gate_check(
        "GMAIL_SEND_EMAIL", {"to": "x@y.z"}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert len(calls) == 1
    assert calls[0].is_composio is True
    assert ex._classify_risk("GMAIL_SEND_EMAIL", True) == "external_side_effect"


@_needs_executor
def test_composio_execute_resolves_inner_action(monkeypatch):
    """The meta-tool's nested action is what the gate/audit judge — resolved to
    the canonical uppercase per-action name, flagged composio."""
    ex = _make_executor()
    name, params, is_composio = ex._resolve_effective_call(
        "composio_execute",
        {"action": "gmail_send_email", "params": {"to": "x@y.z"}},
    )
    assert name == "GMAIL_SEND_EMAIL"
    assert params == {"to": "x@y.z"}
    assert is_composio is True


@_needs_executor
def test_platform_execute_resolves_inner_action(monkeypatch):
    ex = _make_executor()
    name, params, is_composio = ex._resolve_effective_call(
        "platform_execute", {"action": "platform_delete_agent", "params": {"id": 1}},
    )
    assert name == "platform_delete_agent"
    assert params == {"id": 1}
    assert is_composio is False


# ---------------------------------------------------------------------------
# Board approval gate — fail closed under enforce modes (locked #4)
# ---------------------------------------------------------------------------

try:
    import api.board_tasks as board_tasks_mod
    _BOARD_AVAILABLE = True
    _BOARD_SKIP_REASON = ""
except Exception as _exc:  # pragma: no cover - environment-dependent skip
    _BOARD_AVAILABLE = False
    _BOARD_SKIP_REASON = f"api.board_tasks unavailable (CI-gated): {type(_exc).__name__}"

_needs_board = pytest.mark.skipif(
    not _BOARD_AVAILABLE, reason=_BOARD_SKIP_REASON or "board_tasks unavailable"
)


def _board_db_with_task(status="in_progress"):
    task = MagicMock()
    task.status = status
    db = MagicMock()
    db.query.return_value.get.return_value = task
    return db, task


@_needs_board
def test_board_gate_error_blocks_under_enforce(monkeypatch):
    """An approval-gate ERROR under an enforce stage blocks the task pending
    approval — it never launches (fail closed, locked #4)."""
    monkeypatch.setattr("modules.policy.flag.enforcement_active", lambda: True)
    monkeypatch.setattr(
        "core.services.approval_grants.find_active_grant",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gate fault")),
    )
    db, task = _board_db_with_task()

    blocked = board_tasks_mod._board_task_blocked_pending_approval(db, 7, 1, "ws-1")
    assert blocked is True  # NOT launched
    assert task.status == "blocked"
    assert "fail" in (task.blocked_reason or "").lower()


@_needs_board
def test_board_gate_error_fails_open_when_plane_dark(monkeypatch):
    """off/shadow keep the historical fail-open (per-tool gates still apply)."""
    monkeypatch.setattr("modules.policy.flag.enforcement_active", lambda: False)
    monkeypatch.setattr(
        "core.services.approval_grants.find_active_grant",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gate fault")),
    )
    db, task = _board_db_with_task()

    blocked = board_tasks_mod._board_task_blocked_pending_approval(db, 7, 1, "ws-1")
    assert blocked is False  # historical behaviour preserved
    assert task.status == "in_progress"
