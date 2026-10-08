"""F030: the same tool failing the same way is not worth another go.

Night 1 called store_memory seven times in one turn against an identical "backend not
configured": a configuration fault cannot be retried into working. The chat's tool callback
tallies each failure by ``same_failure_key`` and, at ``MAX_IDENTICAL_TOOL_FAILURES``, tells the
model to stop calling the tool.

PRD-256 P256-FIX-RVW-4 (moved here from ``consumers/chatbot/service.py``): an ask for the
owner's click (``requires_confirmation``, inside the chat's envelope) did not fail, it waits on
its card, so it is never counted, and never told "a configuration problem".
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from modules.tools.execution.card_raised import is_waiting

# How many times the same tool may fail the same way in one turn before the
# loop stops handing it back as if the next attempt might differ (F030).
MAX_IDENTICAL_TOOL_FAILURES = 2
FIRST_LINE_CHARS = 120


def same_failure_key(tool_name: str, result: Dict[str, Any]) -> Optional[str]:
    """A key identifying "this tool, failing this way", or None if it succeeded or is waiting
    for the owner's click.

    Keyed on the first line of the error so a retry with different arguments
    that fails for the SAME reason still counts: a missing backend does not
    care what you asked it for.
    """
    if result.get("success") or is_waiting(result):
        return None
    error = str(result.get("error") or (result.get("raw_result") or {}).get("error") or "").strip()
    if not error:
        return None
    return f"{tool_name}:{error.splitlines()[0][:FIRST_LINE_CHARS]}"


__all__ = ["MAX_IDENTICAL_TOOL_FAILURES", "same_failure_key"]
