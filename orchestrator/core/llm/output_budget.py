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

#836: the table measures visible answers. A thinking model's reasoning tokens
count toward max_tokens too, so a call to a model the catalogue says thinks
reserves its purpose's budget plus LLM_THINKING_ALLOWANCE_RATIO times it, capped
by the model's ceiling. A settings row is the operator's whole reservation and
gets no allowance.

Budgets are read when a manager is built, never per call, from the worker's
in-memory snapshot (core.llm.budget_snapshot), which is loaded at boot and kept
fresh on a thread (F105). Each is capped by the model's ceiling when it is
known. A call cut at its budget is logged and flagged, and a text answer carries
a note saying so.
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

from config import config
from core.llm import budget_snapshot

logger = logging.getLogger(__name__)

SETTINGS_CATEGORY = budget_snapshot.SETTINGS_CATEGORY

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


def _stored(purpose: str) -> Optional[int]:
    """The settings row for ``purpose``, from the worker's snapshot (never a
    database read on the event loop)."""
    return budget_snapshot.stored_budget(purpose)


def with_thinking(budget: int, model: Optional[str]) -> int:
    """#836: ``budget`` for the visible answer, plus the thinking allowance when
    the catalogue says ``model`` thinks, capped by the model's own maximum. A
    model that does not think keeps ``budget``."""
    facts = budget_snapshot.model_facts(model)
    if not facts.thinks:
        return budget
    total = budget + int(budget * config.LLM_THINKING_ALLOWANCE_RATIO)
    return min(total, facts.ceiling) if facts.ceiling else total


def budget_for(purpose: Optional[str], ceiling: Optional[int] = None, *, model: Optional[str] = None) -> Optional[int]:
    """The budget the table (or its settings row) gives ``purpose``, capped by
    the model's ceiling; None for a purpose the table does not know. A table
    budget on a thinking ``model`` carries its thinking allowance; a settings
    row is the operator's whole reservation. On the event loop it reads memory
    only (budget_snapshot): call it when a manager is built, not per call."""
    if not purpose:
        return None
    stored = _stored(purpose)
    value = stored or DEFAULT_BUDGETS.get(purpose)
    if value is None:
        return None
    if not stored:
        value = with_thinking(value, model)
    return min(value, ceiling) if ceiling else value


def manager_budgets(purpose: str, config: Any, *, from_settings: bool) -> Tuple[Optional[int], Optional[int]]:
    """F196: a manager's (service budget, long-deliverable budget), read once
    when it is built, never per call. A service manager (built from settings)
    has its purpose's budget; an agent's (given a config) keeps its configured
    one (None here). #836: on a thinking model each carries its allowance."""
    ceiling, model = config.output_ceiling, config.model
    service = budget_for(purpose, ceiling, model=model) if from_settings else None
    return service, budget_for(LONG_DELIVERABLE, ceiling, model=model)


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


def cut_at(response: Any) -> Optional[int]:
    """The budget ``response`` was cut at (note_cut flagged it, and it still
    ends at the limit), else None."""
    budget = getattr(response, "cut", None)
    if getattr(response, "finish_reason", None) != "length" or not isinstance(budget, int) or isinstance(budget, bool):
        return None
    return budget


def cut_note_for(response: Any) -> Optional[str]:
    """The note a finished text answer carries when it is still cut at its
    budget, else None."""
    budget = cut_at(response)
    if budget is None or getattr(response, "tool_calls", None) or not getattr(response, "content", None):
        return None
    return CUT_NOTE.format(budget=budget)
