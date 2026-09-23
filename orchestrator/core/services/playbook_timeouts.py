"""F125: a playbook's timeouts, in seconds, read one way everywhere.

``execution_config`` holds seconds: the editor converts on save
(frontend/hooks/use-playbook-form.ts), the API defaults and the seeds write
seconds, and Auto stores what the owner asked for. Until F125 the executor and
the quality score guessed the unit from the size (10,000 or more was taken for
milliseconds), so an owner's 4-hour budget of 14,400 s became 14.4 s and then
the 900 s floor. No unit is guessed now. ``timeout_minutes``, which an owner's
playbook carried and nothing read, is the total when ``total_timeout`` is absent.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

STEP_KEYS = ("timeout_per_step", "per_step_timeout")
TOTAL_KEY = "total_timeout"
TOTAL_MINUTES_KEY = "timeout_minutes"


def _positive(value: Any) -> Optional[float]:
    """A positive number (a numeric string counts), else None: absent, zero,
    negative or not a number all fall back to the default, as before."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def step_timeout_seconds(exec_config: Optional[Mapping[str, Any]], default: float) -> float:
    """The per-step timeout: timeout_per_step, else per_step_timeout, else ``default``."""
    config = exec_config or {}
    for key in STEP_KEYS:
        seconds = _positive(config.get(key))
        if seconds is not None:
            return seconds
    return float(default)


def total_timeout_seconds(exec_config: Optional[Mapping[str, Any]], default: float) -> float:
    """The run's total budget: total_timeout, else timeout_minutes x 60, else ``default``."""
    config = exec_config or {}
    seconds = _positive(config.get(TOTAL_KEY))
    if seconds is not None:
        return seconds
    minutes = _positive(config.get(TOTAL_MINUTES_KEY))
    if minutes is not None:
        return minutes * 60
    return float(default)
