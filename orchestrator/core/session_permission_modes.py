"""Session permission modes: how much a Claude Code session asks before it acts.

The same four modes Claude Code shows its own users, applied by the CLI host's
gate to every ticket session:

* ``manual`` — a card for every edit and every command off the Bash allowlist.
* ``edits`` ("Edit automatically") — edits inside the session's folders run;
  a command off the allowlist is a card.
* ``plan`` — the session explores read-only and presents a plan; approving the
  plan's card lets it carry on as ``edits``.
* ``auto`` — edits and commands off the allowlist run; only what the gate
  judges risky asks.

Every mode keeps the gate's hard lines: never-allowed commands (push, publish,
escalate) are refused, the platform's secrets stay out of reach, the explicit
ask-list and secret-shaped paths still ask, and the session sandbox bounds what
a command can touch.

Testers of the open-source edition answered a card for every ``mkdir`` and
``npm install``; the local edition now defaults to ``auto``. The hosted edition
defaults to ``edits``. The workspace sets its default on Settings → Session
mode; an agent may override it (``configuration.permission_mode``).
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

PERMISSION_MODE_KEY = "permission_mode"   # in the workspace's session_mode settings and on an agent's configuration
MODE_MANUAL = "manual"
MODE_EDITS = "edits"
MODE_PLAN = "plan"
MODE_AUTO = "auto"
PERMISSION_MODES = (MODE_MANUAL, MODE_EDITS, MODE_PLAN, MODE_AUTO)
LOCAL_EDITION = "local"


def default_permission_mode(edition: str) -> str:
    """Relaxed in the local (open-source) edition, Edit automatically when hosted."""
    return MODE_AUTO if edition == LOCAL_EDITION else MODE_EDITS


def workspace_permission_mode(section: Mapping[str, Any], edition: str) -> str:
    """The mode stored in a workspace's ``session_mode`` settings, else the edition's default."""
    choice = section.get(PERMISSION_MODE_KEY)
    return choice if choice in PERMISSION_MODES else default_permission_mode(edition)


def ticket_permission_mode(agent_configuration: Optional[Mapping[str, Any]], workspace_mode: str) -> str:
    """An agent's own mode wins over the workspace default; anything unknown falls back to it."""
    own = (agent_configuration or {}).get(PERMISSION_MODE_KEY)
    return own if own in PERMISSION_MODES else workspace_mode


def claim_permission_mode(ticket_mode: str, plan_approved: bool) -> str:
    """The mode one claim runs in (PRD-253 Wave P). A Plan ticket plans until the
    operator approves its plan, then works as Edit automatically. Every other mode
    passes through."""
    return MODE_EDITS if ticket_mode == MODE_PLAN and plan_approved else ticket_mode


def validate_permission_mode(value: Any) -> List[str]:
    """Errors for an agent's ``permission_mode``: absent (the workspace default) or one of the modes."""
    if value is None or value in PERMISSION_MODES:
        return []
    return [f"configuration.{PERMISSION_MODE_KEY} must be one of {list(PERMISSION_MODES)}, got {value!r}"]
