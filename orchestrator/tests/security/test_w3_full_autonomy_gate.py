"""Full-autonomy dial — executor gate-bypass contract.

The per-workspace ``autonomy`` setting (``standard`` | ``full``) is read by
``PlatformActionExecutor.execute`` via ``self._full_autonomy()``. When ``full``:

  * the US-003 **confirmation gate** is skipped — a ``requires_confirmation``
    action runs instead of returning ``requires_confirmation``;
  * the US-003 **admin gate** treats Auto as admin — an ``admin_only`` action
    runs for a non-admin caller;
  * the destructive deletes (``requires_confirmation=True``) execute without
    asking — confirmation skipped, and the destructive backstop never fires
    because the flag itself is unchanged.

When ``standard`` both gates stand. The dial only relaxes confirmation + admin;
the hierarchy check, rate limits and destructive backstop are untouched (covered
elsewhere). ``_full_autonomy`` is patched directly — the service's own read is
unit-tested in ``tests/test_w3_auto_autonomy_service.py`` — so these tests are
deterministic and never touch the DB.
"""
from __future__ import annotations

import importlib.util as _ilu
import sys as _sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# Lean-venv shim: importing modules.tools.* runs modules/tools/__init__, which pulls
# modules.rag's ingestion chain (camelot at module top). Stub the missing *leaf* only
# when truly absent — never the modules.rag package. Blessed pattern, see W2-S7.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    import types as _types

    _sys.modules.setdefault("camelot", _types.ModuleType("camelot"))

import modules.tools.discovery.platform_executor as pe
from modules.tools.discovery.action_registry import ActionDefinition
from modules.tools.discovery.platform_executor import PlatformActionExecutor

pytestmark = pytest.mark.asyncio

# Stable actions chosen so each reaches exactly the gate under test and nothing
# else: none are in _HIERARCHY_TARGETS, so can_actor_modify is never consulted.
_CONFIRM_WRITE = "platform_update_auto_reporting_prefs"   # write, requires_confirmation
_DESTRUCTIVE_DELETE = "platform_delete_memory"            # destructive, requires_confirmation
# PRD-143 S4 emptied the live admin_only tier (the 7 obs actions became
# super_admin_only, which full autonomy must NOT bypass). The admin-gate
# mechanism is kept for future workspace-tier tools, so its bypass contract is
# pinned with a SYNTHETIC admin_only action via a fake registry (S2 idiom).
_ADMIN_READ = "platform_w3_admin_probe"                  # read, admin_only (synthetic)
_NON_ADMIN_CALLER = {"workspace_role": "member"}         # known non-admin identity

_ADMIN_READ_DEF = ActionDefinition(
    name=_ADMIN_READ,
    description="W3 admin-gate probe",
    category="monitoring",
    parameters={"type": "object", "properties": {}, "required": []},
    permission_level="read",
    admin_only=True,
)


class _FakeRegistry:
    def __init__(self, *defs: ActionDefinition):
        self._defs = {d.name: d for d in defs}

    def get(self, name: str):
        return self._defs.get(name)


def _admin_probe_registry():
    return patch(
        "modules.tools.discovery.get_action_registry",
        return_value=_FakeRegistry(_ADMIN_READ_DEF),
    )


def _executor() -> PlatformActionExecutor:
    return PlatformActionExecutor(MagicMock(), uuid.uuid4())


def _stub_handler(ex: PlatformActionExecutor, action: str):
    sentinel = {"success": True, "_sentinel": "handler-ran"}
    ex._handlers[action] = AsyncMock(return_value=sentinel)
    return sentinel


# ---------------------------------------------------------------------------
# Confirmation gate
# ---------------------------------------------------------------------------

async def test_confirmation_gate_honored_under_standard():
    ex = _executor()
    handler = _stub_handler(ex, _CONFIRM_WRITE)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=False):
        result = await ex.execute(_CONFIRM_WRITE, {"enabled": True})

    assert result.get("requires_confirmation") is True
    assert result["success"] is False
    ex._handlers[_CONFIRM_WRITE].assert_not_awaited()


async def test_confirmation_gate_skipped_under_full():
    ex = _executor()
    sentinel = _stub_handler(ex, _CONFIRM_WRITE)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=True), patch(
        "core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)
    ):
        result = await ex.execute(_CONFIRM_WRITE, {"enabled": True})

    assert result == sentinel
    ex._handlers[_CONFIRM_WRITE].assert_awaited_once()


async def test_destructive_delete_runs_under_full():
    """The headline: a destructive delete executes without asking at full auto.

    Confirmation is skipped AND the destructive backstop stays silent — it only
    fires when requires_confirmation is False, which the delete never changes.
    """
    ex = _executor()
    sentinel = _stub_handler(ex, _DESTRUCTIVE_DELETE)
    with patch.object(PlatformActionExecutor, "_full_autonomy", return_value=True), patch(
        "core.security.rate_limiter.check_rate_limit", new=AsyncMock(return_value=None)
    ):
        result = await ex.execute(_DESTRUCTIVE_DELETE, {"memory_id": "m1"})

    assert result == sentinel
    ex._handlers[_DESTRUCTIVE_DELETE].assert_awaited_once()


# ---------------------------------------------------------------------------
# Admin gate
# ---------------------------------------------------------------------------

async def test_admin_only_denied_for_nonadmin_under_standard():
    ex = _executor()
    _stub_handler(ex, _ADMIN_READ)
    with patch.object(
        PlatformActionExecutor, "_full_autonomy", return_value=False
    ), _admin_probe_registry():
        result = await ex.execute(_ADMIN_READ, {}, _NON_ADMIN_CALLER)

    assert result["success"] is False
    assert result.get("permission_denied") is True
    ex._handlers[_ADMIN_READ].assert_not_awaited()


async def test_admin_only_bypassed_under_full():
    ex = _executor()
    sentinel = _stub_handler(ex, _ADMIN_READ)
    with patch.object(
        PlatformActionExecutor, "_full_autonomy", return_value=True
    ), _admin_probe_registry():
        result = await ex.execute(_ADMIN_READ, {}, _NON_ADMIN_CALLER)

    assert result == sentinel
    ex._handlers[_ADMIN_READ].assert_awaited_once()
