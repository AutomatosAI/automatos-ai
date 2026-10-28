"""PRD-268 §6.2.1, §6.2.4, §12-Q1, K10 — which heartbeats become check-ins.

An agent's heartbeat config (``agents.configuration.heartbeat``) gains ``mode``:

* ``check_in``: the heartbeat runs as a read-only check-in that suggests;
* ``act``: today's tick, with the agent's own tools, for agents an owner allows.

A heartbeat with no stated mode is ``act`` when its ``auto_act`` is set and ``check_in``
otherwise (§12-Q1). A stated value that is neither reads as unset.

The orchestrator heartbeat converts at ``proactive_level`` ``silent`` or ``notify``;
``act_notify`` and ``autonomous`` keep today's tick (§12-Q11). ``silent`` rings no bell,
today or as a check-in.
"""
from __future__ import annotations

from typing import Any, Optional

__all__ = [
    "CHECK_IN_LEVELS",
    "DEFAULT_LEVEL",
    "HEARTBEAT_ACT",
    "HEARTBEAT_CHECK_IN",
    "HEARTBEAT_MODES",
    "MODE_KEY",
    "SILENT_LEVEL",
    "heartbeat_mode",
    "stated_mode",
]

HEARTBEAT_CHECK_IN = "check_in"
HEARTBEAT_ACT = "act"
HEARTBEAT_MODES = (HEARTBEAT_CHECK_IN, HEARTBEAT_ACT)
# The key inside agents.configuration.heartbeat.
MODE_KEY = "mode"

# The orchestrator's proactive levels a check-in replaces, the one that never rings, and the
# level an unset or unreadable setting is (as the orchestrator tick reads it).
SILENT_LEVEL = "silent"
DEFAULT_LEVEL = "notify"
CHECK_IN_LEVELS = frozenset({SILENT_LEVEL, DEFAULT_LEVEL})


def stated_mode(hb_config: Any) -> Optional[str]:
    """The mode the heartbeat states, or None when it states none of ``HEARTBEAT_MODES``."""
    config = hb_config if isinstance(hb_config, dict) else {}
    stated = str(config.get(MODE_KEY) or "").strip().lower()
    return stated if stated in HEARTBEAT_MODES else None


def heartbeat_mode(hb_config: Any) -> str:
    """The heartbeat's mode: the stated one, else ``act`` when ``auto_act`` is set, else
    ``check_in`` (K10)."""
    stated = stated_mode(hb_config)
    if stated is not None:
        return stated
    # Read as the tick reads it: an agent today's tick lets act keeps acting.
    auto_act = isinstance(hb_config, dict) and bool(hb_config.get("auto_act"))
    return HEARTBEAT_ACT if auto_act else HEARTBEAT_CHECK_IN
