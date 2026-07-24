"""W2-S7 / G3 — fail-closed regression for the platform-write authz gate.

PRD-142 Wave 2 (Test Net). ``PlatformActionExecutor.execute`` gates every
write/destructive platform action on ``can_actor_modify`` (the PRD-140 hierarchy
chokepoint). That helper's DB probes (``_agent_row`` / ``_reports_to_id``) are
not all savepoint-guarded, so a transient DB error mid-check raises *out of* the
helper. Before the gate was hardened the raise propagated past the executor and
the write's fate was decided by whatever caught it upstream — a fail-OPEN risk.

These tests pin the fail-CLOSED contract at the gate:

  * a permission check that **raises** must deny locally — ``permission_denied``
    with reason ``permission_check_failed`` — never fall through to the handler;
  * the ordinary **explicit-deny** path (``decision.allowed is False``) stays
    intact and reports the *decision's own* reason, distinct from the error path;
  * an **allow** decision still flows through the gate to the handler — the
    fail-closed wrap must not turn legitimate writes into denials.

Pure unit tests: ``can_actor_modify`` is patched (raise / deny / allow), so no
DB is touched and the failure injection is deterministic. ``platform_update_agent``
is the gated action under test — it is in ``_HIERARCHY_TARGETS`` and is
``permission_level="write"``, ``requires_confirmation=False``, so it reaches the
gate without confirmation/admin short-circuits.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Lean-venv shim: importing any modules.tools submodule runs modules/tools/__init__.py,
# which eagerly pulls modules.rag's ingestion chain — its multimodal PDF processor does
# ``import camelot`` at module top. camelot is a declared dep installed in CI
# (requirements.txt: camelot-py, opencv-python-headless) but often absent from a lean
# local venv, where it breaks collection — wholly unrelated to the authz gate under test.
#
# Stub the missing *leaf* (camelot), never the whole ``modules.rag`` package. A package
# stub with no ``__path__`` poisons ``sys.modules['modules.rag']`` for every later test
# that imports the real ``modules.rag.service`` (e.g. the W2-S10 RAG suite), failing
# their collection with "'modules.rag' is not a package". Stubbing the leaf lets the real
# package import normally. This is the blessed pattern used across the suite
# (test_w2s9/test_w2s10/test_golden_journeys all setdefault a camelot ModuleType).
#
# Probe defensively. A sibling module (tests/test_l3_distill_input.py) seeds a bare
# ``types.ModuleType("camelot")`` into ``sys.modules`` to dodge the same heavy import.
# That stub's ``__spec__`` is ``None``, and ``find_spec`` *raises*
# ``ValueError: camelot.__spec__ is None`` on a spec-less cached module rather than
# returning it. Treat both "raised" and "found" as "import chain already satisfied"
# (real dep or sibling stub) and skip our shim; only stub when camelot is truly absent.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False  # spec-less camelot stub already present → chain satisfied


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

from core.security.hierarchy_permissions import PermissionDecision
import modules.tools.discovery.platform_executor as pe
from modules.tools.discovery.platform_executor import PlatformActionExecutor

pytestmark = pytest.mark.asyncio

# A write action that IS in _HIERARCHY_TARGETS and reaches the gate.
_GATED_ACTION = "platform_update_agent"
_PARAMS = {"agent_id": 5, "_agent_id": 3}
# Owner role neutralises the US-003 admin gate deterministically, regardless of
# whether the action's admin_only flag defaults differently in future.
_CALLER = {"workspace_role": "owner"}


def _executor() -> PlatformActionExecutor:
    # db is a MagicMock: can_actor_modify is patched so the gate never queries it,
    # and on the deny/raise paths the real handler is never reached.
    return PlatformActionExecutor(MagicMock(), uuid.uuid4())


async def test_permission_check_raise_denies_fail_closed():
    """A permission check that RAISES must deny locally, not propagate or allow."""
    with patch.object(
        pe, "can_actor_modify", side_effect=RuntimeError("db connection lost mid-check")
    ) as cam:
        result = await _executor().execute(_GATED_ACTION, dict(_PARAMS), _CALLER)

    cam.assert_called_once()  # proves we reached the gate, not an earlier return
    assert result["success"] is False
    assert result.get("permission_denied") is True
    assert result.get("reason") == "permission_check_failed"
    assert result.get("escalation_target") == "auto"


async def test_explicit_deny_still_denies_with_its_own_reason():
    """The ordinary deny path stays intact and is distinct from the error path."""
    deny = PermissionDecision(
        allowed=False, reason="out_of_subtree", escalation_target="auto"
    )
    with patch.object(pe, "can_actor_modify", return_value=deny):
        result = await _executor().execute(_GATED_ACTION, dict(_PARAMS), _CALLER)

    assert result["success"] is False
    assert result.get("permission_denied") is True
    # The decision's own reason flows through — NOT the error-path sentinel.
    assert result.get("reason") == "out_of_subtree"


async def test_allow_decision_passes_the_gate():
    """An allow decision must reach the handler — the wrap must not over-deny."""
    allow = PermissionDecision(allowed=True, reason="subtree_authority")
    ex = _executor()
    sentinel = {"success": True, "_sentinel": "handler-ran"}
    ex._handlers[_GATED_ACTION] = AsyncMock(return_value=sentinel)

    with patch.object(pe, "can_actor_modify", return_value=allow), patch(
        "core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)
    ):
        result = await ex.execute(_GATED_ACTION, dict(_PARAMS), _CALLER)

    assert result == sentinel
    assert result.get("permission_denied") is not True
