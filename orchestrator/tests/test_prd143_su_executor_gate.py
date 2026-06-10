"""PRD-143 S2 — executor super-admin gate the full-autonomy bypass CANNOT cross.

``PlatformActionExecutor.execute`` must refuse ``super_admin_only`` actions
pre-execution for ANY principal that is not literally
``system_role == 'super_admin'``. The three verified traps (PRD-143 §9):

  1. the ``full_autonomy → is_admin=True`` bypass applies ONLY to
     ``admin_only``, NEVER to ``super_admin_only``;
  2. the ``_workspace_has_admin_owner()`` PRD-122 fallback must NOT satisfy
     the su gate;
  3. ``caller_context=None`` (heartbeat / agent-factory paths) → REFUSE
     (fail-closed), and API-key principals (``system_role='admin'``) refuse.

Synthetic ActionDefinitions are injected via a fake registry so these tests
pin the GATE LOGIC, independent of the live catalogue (S4 reclassifies it).
Idiom mirrors tests/security/test_w3_full_autonomy_gate.py.
"""
from __future__ import annotations

import importlib.util as _ilu
import os
import sys as _sys
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Dummy POSTGRES_* satisfies the config chain (blessed pattern, see
# tests/test_harness_self_management.py) — with one twist: the port points
# at nothing. The modules.tools import chain attempts a fail-soft DB connect
# at import time; a CLOSED port refuses instantly, while a wedged local
# postgres proxy can hang the handshake forever. CI exports real POSTGRES_*
# so these setdefaults no-op there. Nothing in this file touches a DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")


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

from modules.tools.discovery.action_registry import ActionDefinition
from modules.tools.discovery.platform_executor import PlatformActionExecutor

pytestmark = pytest.mark.asyncio

_SU_ACTION = "platform_su_obs_probe"
_ADMIN_ACTION = "platform_admin_probe"


def _action(name: str, *, admin_only: bool = False, super_admin_only: bool = False) -> ActionDefinition:
    return ActionDefinition(
        name=name,
        description="PRD-143 S2 gate probe",
        category="monitoring",
        parameters={"type": "object", "properties": {}, "required": []},
        permission_level="read",
        admin_only=admin_only,
        super_admin_only=super_admin_only,
    )


class _FakeRegistry:
    def __init__(self, *defs: ActionDefinition):
        self._defs = {d.name: d for d in defs}

    def get(self, name: str):
        return self._defs.get(name)


def _executor() -> PlatformActionExecutor:
    return PlatformActionExecutor(MagicMock(), uuid.uuid4())


def _stub_handler(ex: PlatformActionExecutor, action: str):
    sentinel = {"success": True, "_sentinel": "handler-ran"}
    ex._handlers[action] = AsyncMock(return_value=sentinel)
    return sentinel


def _gates(*, full_autonomy: bool, defs: tuple = ()):
    registry = _FakeRegistry(*(defs or (_action(_SU_ACTION, super_admin_only=True),)))
    return (
        patch.object(PlatformActionExecutor, "_full_autonomy", return_value=full_autonomy),
        patch("modules.tools.discovery.get_action_registry", return_value=registry),
    )


def _assert_su_refused(result: dict):
    assert result["success"] is False
    assert result.get("permission_denied") is True
    assert result.get("requires_confirmation") is None
    assert "super admin" in result.get("error", "").lower()


# ---------------------------------------------------------------------------
# Refusals — every non-su principal, every channel
# ---------------------------------------------------------------------------

async def test_su_action_refused_for_member():
    ex = _executor()
    _stub_handler(ex, _SU_ACTION)
    autonomy, registry = _gates(full_autonomy=False)
    with autonomy, registry:
        result = await ex.execute(_SU_ACTION, {}, {"workspace_role": "member"})

    _assert_su_refused(result)
    ex._handlers[_SU_ACTION].assert_not_awaited()


async def test_su_action_refused_for_workspace_admin():
    """Workspace owner/admin roles never satisfy the su gate — it is a SYSTEM role."""
    for ws_role in ("admin", "owner"):
        ex = _executor()
        _stub_handler(ex, _SU_ACTION)
        autonomy, registry = _gates(full_autonomy=False)
        with autonomy, registry:
            result = await ex.execute(_SU_ACTION, {}, {"workspace_role": ws_role})

        _assert_su_refused(result)
        ex._handlers[_SU_ACTION].assert_not_awaited()


async def test_su_action_refused_under_full_autonomy():
    """Trap 1: the full_autonomy → is_admin bypass NEVER crosses the su gate."""
    ex = _executor()
    _stub_handler(ex, _SU_ACTION)
    autonomy, registry = _gates(full_autonomy=True)
    with autonomy, registry:
        result = await ex.execute(_SU_ACTION, {}, {"workspace_role": "member"})

    _assert_su_refused(result)
    ex._handlers[_SU_ACTION].assert_not_awaited()


async def test_su_action_refused_for_api_key_admin():
    """Trap 3: API keys carry system_role='admin' (hybrid auth) — refused."""
    ex = _executor()
    _stub_handler(ex, _SU_ACTION)
    autonomy, registry = _gates(full_autonomy=False)
    with autonomy, registry:
        result = await ex.execute(
            _SU_ACTION, {}, {"system_role": "admin", "workspace_role": "owner"},
        )

    _assert_su_refused(result)
    ex._handlers[_SU_ACTION].assert_not_awaited()


async def test_su_action_refused_when_no_caller_context():
    """Trap 2: no caller_context → refuse; the workspace-owner fallback is
    never consulted for su actions (fail-closed, no identity resolution)."""
    ex = _executor()
    _stub_handler(ex, _SU_ACTION)
    autonomy, registry = _gates(full_autonomy=False)
    with autonomy, registry, patch.object(
        PlatformActionExecutor, "_workspace_has_admin_owner", return_value=True,
    ) as owner_fallback:
        result = await ex.execute(_SU_ACTION, {}, None)

    _assert_su_refused(result)
    ex._handlers[_SU_ACTION].assert_not_awaited()
    owner_fallback.assert_not_called()


# ---------------------------------------------------------------------------
# The one principal that passes
# ---------------------------------------------------------------------------

async def test_su_action_executes_for_super_admin_principal():
    """Gerard in chat: caller_context carries system_role='super_admin' —
    the action runs normally, independent of the autonomy dial."""
    ex = _executor()
    sentinel = _stub_handler(ex, _SU_ACTION)
    autonomy, registry = _gates(full_autonomy=False)
    with autonomy, registry:
        result = await ex.execute(_SU_ACTION, {}, {"system_role": "super_admin"})

    assert result == sentinel
    ex._handlers[_SU_ACTION].assert_awaited_once()


# ---------------------------------------------------------------------------
# The admin gate is unchanged
# ---------------------------------------------------------------------------

async def test_admin_only_gate_unchanged_under_full_autonomy():
    """admin_only (non-su) actions keep the documented full-autonomy bypass."""
    ex = _executor()
    sentinel = _stub_handler(ex, _ADMIN_ACTION)
    autonomy, registry = _gates(
        full_autonomy=True, defs=(_action(_ADMIN_ACTION, admin_only=True),),
    )
    with autonomy, registry:
        result = await ex.execute(_ADMIN_ACTION, {}, {"workspace_role": "member"})

    assert result == sentinel
    ex._handlers[_ADMIN_ACTION].assert_awaited_once()
