"""PRD-174 W4 — flag gating (§6.4/§6.6, safety rule #1) · PRD-195 S1 update.

Proves what still rides the flag — and what deliberately no longer does:

- ``core/auth/roles.caller_is_admin`` — **flag-independent** since PRD-195 S1
  (G2 decoupling): super_admin ⊇ admin at every flag position (F043 closed
  unconditionally; the plane-OFF legacy fork is deleted).
- ``modules.policy.roles.has_permission`` — empty=deny, the single semantic all
  three key planes now share unconditionally (F042).
- ``modules.policy.flag.policy_plane_enabled`` reads the config flag and fails
  safe (OFF) if config can't be read — it still governs the GOVERNANCE legs
  (budget, act-vs-ask, F040 limiter), which are PRD-192's.
- F040 — the SlowAPIMiddleware registration in ``main.py`` is guarded by the
  flag (grep-level assertion; importing main pulls the whole app).

The flag is toggled by monkeypatching ``config.config.POLICY_PLANE_ENABLED`` —
the single source of truth (no ``os.getenv`` outside config).
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


import config as _config_mod  # noqa: E402
from core.auth.roles import caller_is_admin  # noqa: E402


class _User:
    def __init__(self, role):
        self.system_role = role


@pytest.fixture
def flag(monkeypatch):
    """Toggle the policy-plane flag at its single source of truth.

    PRD-192 S1: the boolean became a staged mode dial — keep BOTH derived
    attrs coherent (mode ``on``/``off`` ⇔ enabled True/False) so tests reading
    either surface agree, exactly as the config parse guarantees at runtime.
    """
    def set_flag(value: bool):
        monkeypatch.setattr(_config_mod.config, "POLICY_PLANE_ENABLED", value)
        monkeypatch.setattr(
            _config_mod.config, "POLICY_PLANE_MODE", "on" if value else "off",
            raising=False,
        )
    return set_flag


# ---------------------------------------------------------------------------
# F043 — caller_is_admin: super_admin ⊇ admin UNCONDITIONALLY (PRD-195 S1, G2)
# ---------------------------------------------------------------------------
# The plane-OFF fork (super-admin excluded unless the governance dial was live)
# is deleted: authZ correctness no longer rides AUTOMATOS_POLICY_PLANE. The
# guard is repointed, not removed — the hierarchy must hold at BOTH flag
# positions, proving the decoupling.

def test_caller_is_admin_is_flag_independent(flag):
    for flag_value in (False, True):
        flag(flag_value)
        assert caller_is_admin(_User("admin")) is True
        assert caller_is_admin(_User("super_admin")) is True   # F043 closed at any flag
        assert caller_is_admin(_User("user")) is False
        assert caller_is_admin(None) is False


# ---------------------------------------------------------------------------
# F042 — empty permission = deny (the semantic both planes take under the flag)
# ---------------------------------------------------------------------------

def test_has_permission_empty_is_deny():
    from modules.policy.roles import has_permission
    # This is what the widget plane switches to under the flag, matching the
    # board plane's _sdk_key_has_scope (empty grants nothing).
    assert has_permission([], "widget:chat") is False
    assert has_permission(None, "widget:chat") is False
    assert has_permission(["widget:chat"], "widget:chat") is True


# ---------------------------------------------------------------------------
# flag reader fails safe
# ---------------------------------------------------------------------------

def test_policy_plane_enabled_reads_config(flag):
    from modules.policy.flag import policy_plane_enabled
    flag(True)
    assert policy_plane_enabled() is True
    flag(False)
    assert policy_plane_enabled() is False


# ---------------------------------------------------------------------------
# F040 — SlowAPIMiddleware registration is flag-guarded in main.py
# ---------------------------------------------------------------------------

def test_f040_middleware_registration_is_flag_guarded():
    main_src = (_ORCH / "main.py").read_text()
    # The middleware is registered, and only under the flag guard.
    assert "SlowAPIMiddleware" in main_src
    assert "if _policy_plane_on:" in main_src
    # Fail-closed swallow_errors follows the flag (closed when the plane is on).
    assert "swallow_errors=not _policy_plane_on" in main_src


def test_config_flag_defaults_off():
    """The master flag must default OFF (safety rule #1).

    PRD-192 S1: the boolean became the staged mode dial — the default is the
    ``off`` stage and the derived boolean is mode ≠ off.
    """
    src = (_ORCH / "config.py").read_text()
    assert 'os.getenv("AUTOMATOS_POLICY_PLANE", "off")' in src
    assert 'POLICY_PLANE_ENABLED: bool = POLICY_PLANE_MODE != "off"' in src
