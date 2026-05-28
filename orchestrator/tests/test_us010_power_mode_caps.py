"""
PRD-141 US-010: Configurable power-mode caps via system_settings.
=================================================================

_get_power_mode_caps(power_mode, db) resolves caps as
``system_settings('power_modes', <mode>)`` merged over the hardcoded
``_POWER_MODE_DEFAULTS``. Stored keys win; absent keys fall back; an unknown
mode falls back to 'standard'; any DB error falls back to defaults (never
raises on the mission hot path).
"""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure orchestrator package is importable
_orchestrator_root = Path(__file__).resolve().parent.parent
if str(_orchestrator_root) not in sys.path:
    sys.path.insert(0, str(_orchestrator_root))

from services.coordinator_service import _get_power_mode_caps, _POWER_MODE_DEFAULTS


def _db_returning(setting):
    """Mock db whose query(...).filter(...).first() yields ``setting``."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = setting
    return db


def _setting_with(value_dict):
    setting = MagicMock()
    setting.value = json.dumps(value_dict)
    return setting


def test_power_mode_reads_system_settings():
    """Stored JSON overrides win over the hardcoded defaults."""
    db = _db_returning(_setting_with({"max_tool_iterations": 99, "max_tokens": 12345}))

    caps = _get_power_mode_caps("standard", db)

    assert caps["max_tool_iterations"] == 99  # overridden
    assert caps["max_tokens"] == 12345        # overridden
    assert caps["force_llm_tier"] is None     # untouched 'standard' default


def test_power_mode_partial_override_keeps_other_defaults():
    """A partial override only replaces the keys it names."""
    db = _db_returning(_setting_with({"max_tool_iterations": 25}))

    caps = _get_power_mode_caps("light", db)

    assert caps["max_tool_iterations"] == 25  # overridden
    assert caps["max_tokens"] == _POWER_MODE_DEFAULTS["light"]["max_tokens"]
    assert caps["force_llm_tier"] == "system_llm"  # default kept


def test_power_mode_falls_back_to_defaults():
    """No stored setting → exactly the hardcoded defaults for that mode."""
    db = _db_returning(None)

    caps = _get_power_mode_caps("max", db)

    assert caps == _POWER_MODE_DEFAULTS["max"]


def test_unknown_power_mode_falls_back_to_standard():
    """An unrecognised mode resolves to the 'standard' defaults."""
    db = _db_returning(None)

    caps = _get_power_mode_caps("turbo", db)

    assert caps == _POWER_MODE_DEFAULTS["standard"]


def test_power_mode_db_error_falls_back_to_defaults():
    """A DB failure must not raise on the mission path — defaults are used."""
    db = MagicMock()
    db.query.side_effect = RuntimeError("db down")

    caps = _get_power_mode_caps("light", db)

    assert caps == _POWER_MODE_DEFAULTS["light"]


def test_get_power_mode_caps_does_not_mutate_defaults():
    """Resolution copies the default dict; it never mutates the module const."""
    db = _db_returning(_setting_with({"max_tokens": 999999}))
    before = dict(_POWER_MODE_DEFAULTS["standard"])

    _get_power_mode_caps("standard", db)

    assert _POWER_MODE_DEFAULTS["standard"] == before
