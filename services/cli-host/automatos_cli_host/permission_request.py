"""A CLI's own permission prompt, answered (PRD-253 S1.4).

A ``PermissionRequest`` hook fires when the CLI would ask the operator itself —
for Claude Code a prompt that reached the TUI, where nobody watches; it is denied
("sessions are policy-gated, not prompted"). GitHub Copilot can ask AFTER the
gate allowed the call: a path or URL check of its own. Denying that would refuse
a call the gate allowed, so a preset with ``permission_request = "rejudge"``
gets the gate's own verdict on the same call instead: allow when the gate allowed
it at ``PreToolUse`` (by policy or by the operator's answer) or would allow it
now; otherwise deny. Never a card — a held call was already a card at
``PreToolUse``, and a permission prompt is answered at once.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any, Deque, Mapping, Optional, Tuple

PERMISSION_REQUEST_DENY = "deny"
PERMISSION_REQUEST_REJUDGE = "rejudge"
PERMISSION_REQUEST_MODES = (PERMISSION_REQUEST_DENY, PERMISSION_REQUEST_REJUDGE)
TUI_PROMPT_REASON = "a permission prompt reached the TUI — sessions are policy-gated, not prompted"
NOT_ALLOWED_REASON = "{reason} (the CLI asked its own permission; the gate answers it at once)"
MAX_ALLOWED_CALLS = 64


def call_key(tool: str, tool_input: Any) -> str:
    """One call, as the gate saw it: its tool and its exact input."""
    return json.dumps([str(tool or ""), tool_input], sort_keys=True, default=str)


class AllowedCalls:
    """The calls the gate allowed in this turn — the latest ``MAX_ALLOWED_CALLS``."""

    def __init__(self) -> None:
        self._keys: Deque[str] = deque(maxlen=MAX_ALLOWED_CALLS)

    def add(self, tool: str, tool_input: Any) -> None:
        self._keys.append(call_key(tool, tool_input))

    def holds(self, tool: str, tool_input: Any) -> bool:
        return call_key(tool, tool_input) in self._keys


def answer(mode: str, *, allowed_before: bool, behavior: Optional[str], reason: str) -> Tuple[bool, str]:
    """``(allow, reason)`` for one permission request. ``behavior``/``reason`` are
    the gate's verdict on the call now (``rejudge`` only)."""
    if mode != PERMISSION_REQUEST_REJUDGE:
        return False, TUI_PROMPT_REASON
    if allowed_before or behavior == "allow":
        return True, ""
    return False, NOT_ALLOWED_REASON.format(reason=reason or "not allowed by the session's gate")


def request_of(payload: Mapping[str, Any]) -> Tuple[str, Mapping[str, Any]]:
    """The tool and its input, from a permission request in Claude's shape."""
    tool_input = payload.get("tool_input")
    return str(payload.get("tool_name") or ""), tool_input if isinstance(tool_input, Mapping) else {}


__all__ = [
    "AllowedCalls", "MAX_ALLOWED_CALLS", "PERMISSION_REQUEST_DENY", "PERMISSION_REQUEST_MODES",
    "PERMISSION_REQUEST_REJUDGE", "TUI_PROMPT_REASON", "answer", "call_key", "request_of",
]
