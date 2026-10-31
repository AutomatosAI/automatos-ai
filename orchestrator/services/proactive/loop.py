"""PRD-268 §6.2.3, §6.9, K8, K9 — a check-in's tool loop, in the read-only lane.

The loop of ``HeartbeatService._orchestrator_tick_llm``, bounded for a check-in, and never
through an agent's own runtime (K8), so no Composio, plugin, memory or sub-agent tool can
reach it:

* the tools are the ``ContextMode.CHECK_IN`` surface: the platform actions that read the
  workspace and ``platform_suggest_next_step`` (``ToolsSection._load_read_only``);
* every call goes through ``PlatformActionExecutor.execute`` with ``caller_context``
  ``{'lane': 'check_in', 'run_id', 'check_in_reads'}`` (and ``check_in_agent_id`` for an
  agent's check-in, ``check_in_user_id`` for a follow-up's person, PRD-268 W3), so the lane
  refuses anything else, and the suggestion tool gets the run and its reads server-side
  (``check_in_lane``);
* ``check_in_reads`` is the run's log of successful reads so far, as the model was shown
  them (``services.proactive.reads``): a suggestion may cite only those;
* at most ``MAX_ITERATIONS`` model calls (4), each asking for at most 800 output tokens
  (the ``check_in`` purpose's budget, ``core.llm.output_budget``), and at most
  ``MAX_CALLS_PER_TURN`` tool calls run per model call; past ``MAX_COST_USD`` ($0.05) the
  run stops and records ``error`` (``cost_ceiling``), that turn's calls unrun;
* a turn whose calls are all suggestions, each recorded, ends the run: nothing is left to
  read. A refused suggestion is answered, and the model may correct it.
* a model response with no call ends the run; its text is the result's ``final_text``
  (a follow-up posts it, ``services.proactive.follow_up_fire``; a check-in ignores it).

Anything that fails after the first model call ends the run as an ``error`` named by its
class (``services.proactive.failures``: a provider's refusal is ``http_<status>``), keeping
what was spent, and no model call follows it. A tool call that fails is answered with the failure, and
the run goes on.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from core.llm.usage_counts import usage_counts
from modules.tools.discovery.check_in_lane import (
    AGENT_KEY,
    CHECK_IN_LANE,
    LANE_KEY,
    READS_KEY,
    RUN_ID_KEY,
    SUGGEST_NEXT_STEP,
    USER_KEY,
)
from services.proactive.failures import failure_class
from services.proactive.reads import read_succeeded, refs_of_read, with_reads

logger = logging.getLogger(__name__)

__all__ = [
    "COST_CEILING",
    "MAX_CALLS_PER_TURN",
    "MAX_COST_USD",
    "MAX_ITERATIONS",
    "LoopResult",
    "LoopSpec",
    "run_loop",
]

MAX_ITERATIONS = 4
MAX_COST_USD = 0.05
MAX_CALLS_PER_TURN = 8
COST_CEILING = "cost_ceiling"
# A tool's answer as the model reads it, cut to this many characters (as the orchestrator tick).
TOOL_RESULT_CHARS = 2000
NO_ARGUMENTS = "{}"
BAD_ARGUMENTS = "The arguments were not valid JSON. Nothing ran."
TOOL_FAILED = "The tool failed ({error}). Nothing more is known."
NOT_RUN = f"Not run: a check-in runs at most {MAX_CALLS_PER_TURN} calls a turn."


@dataclass(frozen=True)
class LoopSpec:
    """What one check-in's loop runs: its run, its prompt, its tools, the reads it starts with,
    and, for a follow-up, its person (``user_id``)."""

    run_id: UUID
    workspace_id: UUID
    agent_id: Optional[int]
    system_prompt: str
    start_message: str
    tools: Tuple[Dict[str, Any], ...]
    reads: Tuple[Dict[str, str], ...] = ()
    user_id: Optional[int] = None


@dataclass(frozen=True)
class LoopResult:
    """How the loop ended: the model, the tokens and cost it spent, what stopped it, if
    anything did (``error_class``), and the last response's text when it made no call
    (``final_text``; None when the loop ended any other way)."""

    model: Optional[str]
    tokens: int
    cost_usd: float
    error_class: Optional[str] = None
    final_text: Optional[str] = None


@dataclass(frozen=True)
class _Call:
    id: str
    name: str
    raw_arguments: Any
    arguments: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class _Turn:
    """One turn's calls answered: the tool messages, the read log after them, and whether
    every call was a suggestion that was recorded."""

    messages: Tuple[Dict[str, Any], ...]
    reads: Tuple[Dict[str, str], ...]
    all_suggested: bool


@dataclass(frozen=True)
class _Spend:
    tokens: int = 0
    cost_usd: float = 0.0

    def plus(self, llm: Any, response: Any) -> "_Spend":
        counts = usage_counts(response)
        cost, _source = llm.call_cost(counts.input_tokens, counts.output_tokens, counts.reported_cost)
        return _Spend(self.tokens + counts.total_tokens, self.cost_usd + float(cost or 0.0))


def _field(item: Any, name: str, default: Any) -> Any:
    return item.get(name, default) if isinstance(item, dict) else getattr(item, name, default)


def _parsed(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw) if isinstance(raw, str) and raw.strip() else {}
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _calls(response: Any) -> Tuple[_Call, ...]:
    """The tool calls a model response made, each with its arguments parsed (None: not JSON)."""
    calls = []
    for item in getattr(response, "tool_calls", None) or []:
        function = _field(item, "function", {}) or {}
        raw = _field(function, "arguments", NO_ARGUMENTS)
        calls.append(_Call(id=str(_field(item, "id", "") or ""), name=str(_field(function, "name", "") or ""),
                           raw_arguments=raw, arguments=_parsed(raw)))
    return tuple(calls)


def _arguments_text(call: _Call) -> str:
    """The arguments as the next call replays them: as sent when they parsed, else ``{}``,
    which every provider parses."""
    if call.arguments is None:
        return NO_ARGUMENTS
    return call.raw_arguments if isinstance(call.raw_arguments, str) else json.dumps(call.arguments, default=str)


def _assistant(response: Any, calls: Tuple[_Call, ...]) -> Dict[str, Any]:
    """The model's turn as the next call reads it: its words and each call it made."""
    made = [{"id": call.id, "type": "function", "function": {"name": call.name, "arguments": _arguments_text(call)}}
            for call in calls]
    return {"role": "assistant", "content": getattr(response, "content", "") or None, "tool_calls": made}


def _shown(result: Any) -> str:
    """A tool's answer as the model reads it."""
    return json.dumps(result, default=str)[:TOOL_RESULT_CHARS]


def _caller_context(spec: LoopSpec, reads: Tuple[Dict[str, str], ...]) -> Dict[str, Any]:
    """The server-built context every call of this check-in carries (``check_in_lane``)."""
    context = {LANE_KEY: CHECK_IN_LANE, RUN_ID_KEY: str(spec.run_id), READS_KEY: [dict(ref) for ref in reads]}
    agent = {AGENT_KEY: spec.agent_id} if spec.agent_id is not None else {}
    person = {USER_KEY: spec.user_id} if spec.user_id is not None else {}
    return {**context, **agent, **person}


async def _execute(db: Session, executor: Any, spec: LoopSpec, call: _Call,
                   reads: Tuple[Dict[str, str], ...]) -> Dict[str, Any]:
    from core.database.session_health import rollback_if_aborted

    if call.arguments is None:
        return {"success": False, "error": BAD_ARGUMENTS}
    try:
        return await executor.execute(call.name, call.arguments, _caller_context(spec, reads))
    except Exception as exc:  # noqa: BLE001 — the model is told the call failed; the run goes on
        logger.exception("[proactive] run %s: %s raised", spec.run_id, call.name)
        rollback_if_aborted(db, f"check-in {spec.run_id} {call.name}")
        return {"success": False, "error": TOOL_FAILED.format(error=type(exc).__name__)}


async def _run_calls(db: Session, executor: Any, spec: LoopSpec, calls: Tuple[_Call, ...],
                     reads: Tuple[Dict[str, str], ...]) -> _Turn:
    """Run one turn's calls in order, the first ``MAX_CALLS_PER_TURN`` of them; each
    successful read's refs join the log before the next call."""
    messages, log, all_suggested = [], reads, True
    for index, call in enumerate(calls):
        ran = index < MAX_CALLS_PER_TURN
        result = await _execute(db, executor, spec, call, log) if ran else {"success": False, "error": NOT_RUN}
        shown = _shown(result)
        if call.name == SUGGEST_NEXT_STEP:
            all_suggested = all_suggested and read_succeeded(result)
        else:
            all_suggested = False
            log = with_reads(log, refs_of_read(call.name, call.arguments, result, shown))
        messages.append({"role": "tool", "tool_call_id": call.id, "content": shown})
    return _Turn(messages=tuple(messages), reads=log, all_suggested=all_suggested)


def _result(llm: Any, spend: _Spend, error_class: Optional[str] = None,
            final_text: Optional[str] = None) -> LoopResult:
    model = getattr(getattr(llm, "config", None), "model", None)
    return LoopResult(model=model, tokens=spend.tokens, cost_usd=spend.cost_usd, error_class=error_class,
                      final_text=final_text)


async def _ask(db: Session, llm: Any, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Any:
    from core.database.database import end_open_transaction

    # Not idle-in-transaction across the model's await (PRD-135 / W1-S9); the commit runs off the loop.
    await asyncio.to_thread(end_open_transaction, db)
    return await llm.generate_response(messages, tools=tools or None)


async def run_loop(db: Session, llm: Any, spec: LoopSpec) -> LoopResult:
    """Run one check-in's turn: at most ``MAX_ITERATIONS`` model calls, each tool call in the
    lane. The suggestions it records are committed by the tool; the caller counts them."""
    from modules.tools.discovery.platform_executor import PlatformActionExecutor

    executor = PlatformActionExecutor(db, spec.workspace_id)
    messages = [{"role": "system", "content": spec.system_prompt}, {"role": "user", "content": spec.start_message}]
    reads, spend = spec.reads, _Spend()
    for _ in range(MAX_ITERATIONS):
        try:
            response = await _ask(db, llm, messages, list(spec.tools))
            spend = spend.plus(llm, response)
            if spend.cost_usd > MAX_COST_USD:
                logger.warning("[proactive] run %s stopped at $%.4f, past its $%.2f ceiling", spec.run_id,
                               spend.cost_usd, MAX_COST_USD)
                return _result(llm, spend, COST_CEILING)
            calls = _calls(response)
            if not calls:
                return _result(llm, spend, final_text=str(getattr(response, "content", "") or ""))
            turn = await _run_calls(db, executor, spec, calls, reads)
        except Exception as exc:  # noqa: BLE001 — the run ends as an error named by its class, its spend kept
            logger.exception("[proactive] run %s failed", spec.run_id)
            return _result(llm, spend, failure_class(exc))
        messages, reads = [*messages, _assistant(response, calls), *turn.messages], turn.reads
        if turn.all_suggested:
            break
    return _result(llm, spend)
