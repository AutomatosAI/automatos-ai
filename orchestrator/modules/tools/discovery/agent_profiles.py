"""Ticket #371: what platform_get_agent says about an agent a product calls (PRD-274).

Auto could not read the id a product names an agent by, nor the two profiles its Model tab
sets, so it could not say how Phylax Guide (agent 48) was set up. The lookup now carries:

- ``public_id``: the agent's public UUID, the id a server key's ``POST /api/chat`` takes and
  the one a product's website stores;
- ``runtime_profile``: ``full`` or ``lean``, as the Model tab shows it
  (``savedRuntimeProfile``): ``lean`` only for an agent that is not a system agent and whose
  configuration says so; absent is ``full``. Whether a lean agent's turns are lean also needs
  ``LEAN_TURNS_ENABLED`` on the install (``consumers.chatbot.lean_turn.is_lean``);
- ``memory_profile``: the agent's own memory profile slug, or None for none of its own (its
  workspace's default), as the Model tab shows it (``savedMemoryProfile``). It is the only
  memory setting an agent carries: whether a turn stores memory is decided by the turn (a
  server key's own turn, or a widget turn, stores none), not by the agent.

Read-only: nothing here writes an agent.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def profiles_of(agent: Any) -> Dict[str, Any]:
    """The agent's public id, runtime profile and own memory profile."""
    from consumers.chatbot.lean_turn import PROFILE_FULL, PROFILE_LEAN, RUNTIME_PROFILE_KEY
    from modules.memory.memory_profiles import PROFILE_KEY

    cfg = agent.configuration if isinstance(agent.configuration, dict) else {}
    lean = not getattr(agent, "is_system_agent", False) and cfg.get(RUNTIME_PROFILE_KEY) == PROFILE_LEAN
    return {
        "public_id": str(agent.public_id) if agent.public_id else None,
        "runtime_profile": PROFILE_LEAN if lean else PROFILE_FULL,
        "memory_profile": _slug(cfg.get(PROFILE_KEY)),
    }


def _slug(value: Any) -> Optional[str]:
    return value.strip() if isinstance(value, str) and value.strip() else None
