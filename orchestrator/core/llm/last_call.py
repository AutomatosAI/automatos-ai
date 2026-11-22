"""How the last real model call in this process ended (PRD-282 US-202).

GET /health's model provider entry is never found out by calling the provider. Real
calls already record their outcome, but only in the usage table (one row per call, and
none without a workspace) and in the OpenTelemetry metrics (exported, not readable
back). So the call site that writes the usage row, ``LLMManager._track_usage`` (every
call through the manager, success or failure, async or sync), also notes here whether
the call succeeded and when it ended. One frozen value, replaced whole: safe to read
from any thread.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from core.llm.usage_status import STATUS_SUCCESS


@dataclass(frozen=True)
class LastCall:
    """Whether the call succeeded, and when it ended (UTC)."""

    succeeded: bool
    ended_at: datetime


_last: Optional[LastCall] = None


def note(status: str) -> None:
    """Record a real call's outcome: ``status`` is the usage row's (success or error)."""
    global _last
    _last = LastCall(succeeded=status == STATUS_SUCCESS, ended_at=datetime.now(timezone.utc))


def last() -> Optional[LastCall]:
    """The last real call's outcome in this process, or None before the first."""
    return _last
