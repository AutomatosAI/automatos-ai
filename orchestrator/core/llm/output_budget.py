"""F196 (night 6): a model call reserves the output its purpose needs, not the
model's maximum.

From 05:42Z OpenRouter refused calls with 402: "This request requires more
credits, or fewer max_tokens. You requested up to 65535 tokens, but can only
afford 8512". A provider reserves a call's whole max_tokens against the balance,
and calls in flight reserve together. Every service call reserved its settings
category's 8,000, a one-line classifier included. AgentFactory gave Auto, the
system tier and any agent without its own Max Output Tokens the model's ceiling
(65,535 for gemini-2.5-flash), and ignored the 8,000 set in Settings.

Now each purpose has a budget, keyed like llm_usage.request_type:

- A service manager (built from settings) takes the table below for its own
  purpose. Each value is the p99 of nights 3-6's real output, times 1.5, with a
  floor of 1,024. A row in system_settings (category ``llm_output_budget``)
  overrides it.
- Chat and agent runs keep the budget their config was built with: an agent's
  own Max Output Tokens, else 8,000.
- Long deliverables (a report, a generated document, a mission's final write)
  get 16,000.

Budgets are read when a manager is built, never per call, because
read_system_setting is a sync database read (F105). Each is capped by the
model's ceiling when it is known. A call cut at its budget is logged and flagged,
and a text answer carries a note saying so.
"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SETTINGS_CATEGORY = "llm_output_budget"

CHAT = "chat"
AGENT_RUN = "agent_run"
LONG_DELIVERABLE = "long_deliverable"

# Nights 3-6 (llm_usage, 22-26 Sep 2026, successful calls): p99 of output_tokens
# x 1.5, floor 1,024. Chat keeps 8,000: 14 of 3,606 replies ran past the
# formula's 1,931 (the longest 4,164), and a reservation only bites at a
# near-empty balance.
DEFAULT_BUDGETS: Dict[str, int] = {
    CHAT: 8000,
    AGENT_RUN: 8000,
    LONG_DELIVERABLE: 16000,
    "complexity_assessor": 1617,     # p99 1,078
    "entity_extraction": 2457,       # p99 1,638
    "verifier": 2183,                # p99 1,455
    "planner": 2903,                 # p99 1,935 (4 calls)
    "decision": 1024,                # p99 600
    "orchestrator": 1024,            # p99 288
    "digest": 1024,                  # p99 196
    "watch": 1024,                   # p99 198
    "heartbeat": 1024,               # p99 211
    "thread_checkpoint": 1024,       # p99 286
    "consistency_verifier": 1024,    # p99 477
    "graph_community_title": 1024,   # p99 54
}
# The table serves managers built from settings (services). An agent's manager,
# built by AgentFactory, keeps its configured budget whatever lane its run is
# in: its own Max Output Tokens, else AGENT_RUN's. Its lane can share a name
# with a service ("heartbeat", "watch"). graph_extraction has no row: it keeps
# F051's own cap (GRAPH_EXTRACTION_MAX_OUTPUT_TOKENS).
# A generic manager (the default "orchestrator" service) working inside one of
# these lanes does the lane's work, so it keeps its configured budget there.
GENERIC_PURPOSE = "orchestrator"
CONFIGURED_PURPOSES = frozenset({CHAT, AGENT_RUN, "board_task", "recipe", "session", "mission"})
CUT_NOTE = "\n\n[Cut here: this answer reached its {budget:,}-token limit.]"

_purpose: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("llm_output_purpose", default=None)
_call_budget: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar("llm_call_budget", default=None)


_STORED_TTL_S = 60.0
_stored_cache: Dict[str, Tuple[Optional[int], float]] = {}
_refreshing: Set[str] = set()


def _read_stored(purpose: str) -> Optional[int]:
    try:
        from core.llm.manager import read_system_setting

        raw = read_system_setting(SETTINGS_CATEGORY, purpose)
        value = int(raw) if raw not in (None, "") else None
    except Exception:  # noqa: BLE001 — a missing or unreadable row falls back to the table
        value = None
    _stored_cache[purpose] = (value, time.monotonic() + _STORED_TTL_S)
    _refreshing.discard(purpose)
    return value


def _stored(purpose: str) -> Optional[int]:
    """The settings row for ``purpose``, read at most once a minute. The read
    is a sync database round trip, so on an event loop it is never waited for
    (F105): the cached value, or the table's, answers, and the read runs on a
    thread for the next manager."""
    cached = _stored_cache.get(purpose)
    if cached and cached[1] > time.monotonic():
        return cached[0]
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _read_stored(purpose)
    if purpose not in _refreshing:
        _refreshing.add(purpose)
        loop.run_in_executor(None, _read_stored, purpose)
    return cached[0] if cached else None


def budget_for(purpose: Optional[str], ceiling: Optional[int] = None) -> Optional[int]:
    """The budget the table (or its settings row) gives ``purpose``, capped by
    the model's ceiling; None for a purpose the table does not know. Reads
    system_settings: call it when a manager is built, not per call."""
    if not purpose:
        return None
    value = _stored(purpose) or DEFAULT_BUDGETS.get(purpose)
    if value is None:
        return None
    return min(value, ceiling) if ceiling else value


def for_call(lane: Optional[str], configured: int, *, own_purpose: Optional[str],
             service_budget: Optional[int], long_budget: Optional[int]) -> int:
    """What one call asks for. No database read. It is a long deliverable's
    budget when the run writes one. It is the configured budget for an agent's
    manager (no service budget), and for a generic manager inside a chat or
    agent-run lane. Otherwise it is the manager's service budget."""
    if _purpose.get() == LONG_DELIVERABLE and long_budget:
        return max(configured, long_budget)
    if service_budget is None:
        return configured
    if own_purpose == GENERIC_PURPOSE and lane in CONFIGURED_PURPOSES:
        return configured
    return service_budget


def call_purpose(lane: Optional[str], *, own_purpose: Optional[str], from_settings: bool) -> str:
    """What one call is for, resolved as for_call resolves its budget. An
    agent's run is its lane's (a chat turn, a ticket, a mission task). A service
    call is its own purpose, unless a generic manager works inside an agent-run
    lane."""
    if not from_settings:
        return lane or AGENT_RUN
    if own_purpose == GENERIC_PURPOSE and lane in CONFIGURED_PURPOSES:
        return lane
    return own_purpose or lane or GENERIC_PURPOSE


@contextmanager
def output_purpose(purpose: str) -> Iterator[None]:
    """Mark the calls made inside as ``purpose`` (a long deliverable)."""
    token = _purpose.set(purpose)
    try:
        yield
    finally:
        _purpose.reset(token)


@contextmanager
def call_budget(max_tokens: Optional[int]) -> Iterator[None]:
    """The max_tokens the provider sends for the call made inside (None: the config's)."""
    token = _call_budget.set(max_tokens)
    try:
        yield
    finally:
        _call_budget.reset(token)


def current_call_budget() -> Optional[int]:
    return _call_budget.get()


def note_cut(response: Any, *, purpose: str, budget: int, model: str) -> None:
    """A response cut at its budget is logged, and flagged with the budget
    (``response.cut``). Its text is left alone: a caller may continue it (an
    agent run asks for the rest twice), and a JSON answer must stay parseable.
    The caller that writes the finished answer adds the note (cut_note_for)."""
    if getattr(response, "finish_reason", None) != "length":
        return
    logger.warning(f"[F196] {purpose} output cut at its {budget:,}-token budget (model {model})")
    try:
        response.cut = budget
    except Exception:  # noqa: BLE001 — a frozen response is still logged
        pass


def cut_note_for(response: Any) -> Optional[str]:
    """The note a finished text answer carries when it is still cut at its
    budget, else None."""
    budget = getattr(response, "cut", None)
    if (getattr(response, "finish_reason", None) != "length" or not isinstance(budget, int)
            or isinstance(budget, bool) or getattr(response, "tool_calls", None)
            or not getattr(response, "content", None)):
        return None
    return CUT_NOTE.format(budget=budget)
