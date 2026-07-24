"""Policy plane — the typed event bus (PRD-174 Step 4, §4.2).

Keeps Claude Code's hook *taxonomy* and *verdict semantics*, but execution is
**in-process typed handlers with tenant scope only** — NOT shell commands (the
review is explicit: Bash-string hooks are RCE-by-configuration and don't fit a
SaaS backend). Handlers are plain callables:

    handler(event: Event, ctx: EventContext) -> Optional[Verdict]

registered against an :class:`Event`. On each fire the bus runs every handler
for that event and merges their verdicts under **deny > ask > allow**
(:func:`~modules.policy.types.merge_verdicts`). A handler returning ``None`` (or
a ``defer`` verdict) is "no opinion".

The `PreToolUse` seam is the natural attach point for budget (F086), approval
routing (act-vs-ask), and prerequisite (read-before-write) policy — it sits
beside the dedup check in the tool loop. Post-tool / round / run events exist so
audit + compaction policy can attach later without re-opening the loop.

Stdlib-only: no DB, no config. The bus itself is a thin dispatcher; handlers
bring their own dependencies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from modules.policy.types import Event, Verdict, merge_verdicts

logger = logging.getLogger(__name__)


@dataclass
class EventContext:
    """Everything a handler needs to decide, tenant-scoped.

    Deliberately loose (``dict`` payload) so new events don't force a type
    change, but the common fields are named for the hot path (PreToolUse).
    """

    workspace_id: Any = None
    agent_id: Optional[int] = None
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    caller_context: Optional[Dict[str, Any]] = None
    # Free-form extras (result payloads for post events, round state, etc.).
    data: Dict[str, Any] = field(default_factory=dict)


# handler signature: (event, ctx) -> Optional[Verdict]
Handler = Callable[[Event, EventContext], Optional[Verdict]]


class PolicyBus:
    """In-process typed event bus with deny > ask > allow merge."""

    def __init__(self) -> None:
        self._handlers: Dict[Event, List[Handler]] = {}

    def register(self, event: Event, handler: Handler) -> None:
        """Attach a handler to an event. Order of registration is preserved but
        does not affect the outcome — the merge is rank-based, not first-wins."""
        self._handlers.setdefault(event, []).append(handler)

    def clear(self) -> None:
        """Drop all handlers (test isolation)."""
        self._handlers.clear()

    def fire(self, event: Event, ctx: EventContext) -> Verdict:
        """Run every handler for ``event`` and merge verdicts (deny > ask > allow).

        A handler that raises is treated as **no opinion** (logged), never as a
        silent allow *or* a hard failure — one bad handler must not take the
        loop down, nor must it wave a call through by crashing.
        """
        verdicts: List[Optional[Verdict]] = []
        for handler in self._handlers.get(event, ()):  # empty when unregistered
            try:
                verdicts.append(handler(event, ctx))
            except Exception:
                logger.warning(
                    "[policy.bus] handler raised on %s (treated as no-opinion)",
                    event.value, exc_info=True,
                )
                verdicts.append(None)
        return merge_verdicts(*verdicts)


# ---------------------------------------------------------------------------
# Process-wide singleton. Handlers are registered once at startup (or lazily in
# the tool loop). Kept module-level so the stdlib-only tool_loop can reach it
# without constructing anything.
# ---------------------------------------------------------------------------

_BUS: Optional[PolicyBus] = None


def get_policy_bus() -> PolicyBus:
    global _BUS
    if _BUS is None:
        _BUS = PolicyBus()
    return _BUS


def reset_policy_bus() -> None:
    """Test hook: drop the singleton so a test starts from a clean bus."""
    global _BUS
    _BUS = None
