"""PRD-268 §6.2.1, §6.9, FR-2, FR-3, FR-13, K8, K9, K14 — the check-in runner.

Two check-ins, each one run with one ``proactive_runs`` row (``services.proactive.runs``):

* ``run_workspace_check_in`` — **Auto's workspace check-in.** It reads the signals of its
  window (since its last successful run, else the last 24 hours;
  ``services.proactive.signals``). No signal: a ``silent`` run, and no model call at all.
  Signals: one turn of the model in the check-in lane, its prompt listing the signals, the
  owner's open and recently hidden titles, and "most check-ins should record nothing".
* ``run_agent_check_in`` — **an agent's check-in**, its heartbeat converted. Its heartbeat
  prompt and identity are built through ``ContextService`` in ``ContextMode.CHECK_IN``,
  and it runs the same turn. An agent's work is outside what a signal diff can see, so it
  has no pre-check: its interval is its cost lever (§6.9).

Each first lapses the workspace's open suggestions past their 7 days
(``services.proactive.expiry``), so no cap, dedupe read or prompt counts them, then checks,
in order, the switches, ``background_allowed``, the day's spend ceiling and the workspace's
caps (``services.proactive.gates``). A refusal writes its run row and
makes no model call. The turn (``services.proactive.loop``) runs on ``LLMManager(service_name=
'heartbeat')``, the platform's ``orchestrator_llm`` model, never the agent's own (K9), as
one ask in the ``check_in`` lane, which only credits beyond the trial pay for (K14). Its
suggestions are recorded by ``platform_suggest_next_step``, addressed to the workspace
owner (K11); the run row records the signals, the suggestions, the model, tokens and cost.

The runner's own database work before and after the turn (the checks, the signals, the run
row) runs off the event loop; the turn's tool calls run as every platform call does. A
failure anywhere is the run's ``error``, named by its class (``services.proactive.failures``),
with what the turn spent. An agent's check-in also keeps its failing streak
(``services.proactive.failing_streaks``): from the third failure in a row it is held back
at its start until its doubled interval has passed, a failure records the streak's one
card, and a run that works ends the streak. The heartbeat scheduler runs the check-ins
through ``services.proactive.heartbeat_bridge``.

Once the run's row has committed (and an agent's streak has filed its card), the run's new
suggestions are delivered (``services.proactive.delivery``): posted in the addressee's Auto
thread with one bell, in live mode only (K8). A delivery that fails never fails the run.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union
from uuid import UUID, uuid4

from sqlalchemy.orm import Session

from core.models.proactive import RUN_ERROR, RUN_SILENT, RUN_SUGGESTED
from modules.proactive import check_in_window
from services.proactive.delivery import deliver_run
from services.proactive.expiry import expire_lapsed
from services.proactive.failing_streaks import held_back, note_run
from services.proactive.failures import failure_class
from services.proactive.gates import Refusal, start_refusal
from services.proactive.loop import LoopResult, LoopSpec, run_loop
from services.proactive.prompts import MAX_SIGNALS, agent_task, start_message, workspace_task
from services.proactive.runs import (
    KIND_AGENT,
    KIND_WORKSPACE,
    RunRecord,
    known_titles,
    last_success,
    owner_id,
    suggestions_made,
    write_run,
)
from services.proactive.signals import read_signals

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_NOT_FOUND",
    "CHECK_IN_PURPOSE",
    "CheckInResult",
    "heartbeat_llm",
    "run_agent_check_in",
    "run_workspace_check_in",
]

# The model calls' purpose and the ask's lane: llm_usage.request_type, the 800-token output
# budget (core.llm.output_budget), customer work (services.credits.classification), never paid
# from the trial (services.credits.asks BACKGROUND_LANES). The lane's own name (check_in_lane
# CHECK_IN_LANE), written out: tests/test_prd265_call_site_classes.py reads it from this module.
CHECK_IN_PURPOSE = "check_in"
# The service whose settings choose the model: orchestrator_llm (core/llm/manager.py).
HEARTBEAT_SERVICE = "heartbeat"
AGENT_NOT_FOUND = "agent_not_found"
# Auto, as its workspace check-in's identity (as the orchestrator heartbeat names it).
AUTO = SimpleNamespace(id=None, name="Auto", agent_type="orchestrator", use_custom_persona=False, persona=None,
                       description="Auto's workspace check-in: what changed, and what the owner should look at")


@dataclass(frozen=True)
class CheckInResult:
    """How a check-in ended: its run, its outcome, the suggestions it recorded, and what
    stopped or refused it, if anything did."""

    run_id: UUID
    outcome: str
    suggestions: int = 0
    error_class: Optional[str] = None


@dataclass(frozen=True)
class _Plan:
    """A check-in that goes to the model: its run as far as known, its task, its identity,
    the reads it starts with, and its model."""

    record: RunRecord
    task: str
    agent: Any
    reads: Tuple[Dict[str, str], ...]
    llm: Any


@contextmanager
def _session() -> Iterator[Session]:
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def heartbeat_llm(workspace_id: UUID, agent_id: Optional[int]) -> Any:
    """The check-in's model: the heartbeat service's (the platform ``orchestrator_llm``
    setting), never the agent's own (K9), its calls booked as ``check_in``."""
    from core.llm.manager import LLMManager

    return LLMManager(service_name=HEARTBEAT_SERVICE, workspace_id=str(workspace_id), agent_id=agent_id,
                      request_type=CHECK_IN_PURPOSE)


def _refused(db: Session, record: RunRecord, refusal: Refusal) -> CheckInResult:
    write_run(db, replace(record, outcome=refusal.outcome, error_class=refusal.reason))
    return CheckInResult(record.run_id, refusal.outcome, error_class=refusal.reason)


def _errored(db: Session, record: RunRecord, error_class: str) -> CheckInResult:
    """The run as an error: whatever the failure left open is rolled back first."""
    db.rollback()
    write_run(db, replace(record, outcome=RUN_ERROR, error_class=error_class))
    return CheckInResult(record.run_id, RUN_ERROR, error_class=error_class)


def _prepare_workspace(db: Session, record: RunRecord, now: datetime) -> Union[CheckInResult, _Plan]:
    """Auto's check-in up to its model call: refused, silent (no signal), or a plan."""
    workspace_id = record.workspace_id
    expire_lapsed(db, workspace_id, now)
    owner = owner_id(db, workspace_id)
    refusal = start_refusal(db, workspace_id, owner, now, f"Auto's check-in of workspace {workspace_id}")
    if refusal is not None:
        return _refused(db, record, refusal)
    since = check_in_window(last_success(db, workspace_id, KIND_WORKSPACE, None), now)
    signals = read_signals(db, workspace_id, since, now)
    record = replace(record, window_since=since, window_until=now,
                     signals=tuple(signal.as_json() for signal in signals))
    if not signals:
        write_run(db, record)
        return CheckInResult(record.run_id, RUN_SILENT)
    open_titles, hidden_titles = known_titles(db, workspace_id, owner, now)
    # The run read every signal, and may cite those its prompt lists.
    return _Plan(record=record, task=workspace_task(signals, since, now, open_titles, hidden_titles), agent=AUTO,
                 reads=tuple(signal.evidence for signal in signals[:MAX_SIGNALS]),
                 llm=heartbeat_llm(workspace_id, None))


def _prepare_agent(db: Session, record: RunRecord, hb_config: Dict[str, Any],
                   now: datetime) -> Union[CheckInResult, _Plan]:
    """An agent's check-in up to its model call: refused, or a plan."""
    from core.models.core import Agent

    workspace_id, agent_id = record.workspace_id, record.agent_id
    expire_lapsed(db, workspace_id, now)
    owner = owner_id(db, workspace_id)
    refusal = (start_refusal(db, workspace_id, owner, now, f"agent {agent_id}'s check-in", agent_id)
               or held_back(db, workspace_id, agent_id, hb_config, now))
    if refusal is not None:
        return _refused(db, record, refusal)
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    if agent is None:
        return _errored(db, record, AGENT_NOT_FOUND)
    since = check_in_window(last_success(db, workspace_id, KIND_AGENT, agent_id), now)
    open_titles, hidden_titles = known_titles(db, workspace_id, owner, now)
    task = agent_task(hb_config.get("prompt"), since, now, open_titles, hidden_titles)
    return _Plan(record=replace(record, window_since=since, window_until=now), task=task, agent=agent, reads=(),
                 llm=heartbeat_llm(workspace_id, agent_id))


async def _converse(db: Session, plan: _Plan, now: datetime) -> LoopResult:
    """Build the check-in's context and run its turn, as one ask in the ``check_in`` lane."""
    from core.llm.usage_context import usage_scope
    from modules.context import ContextMode, ContextService
    from services.credits.asks import ask_scope

    record = plan.record
    with (
        usage_scope(request_type=CHECK_IN_PURPOSE, agent_id=record.agent_id, workspace_id=str(record.workspace_id),
                    inherit=False),
        ask_scope(CHECK_IN_PURPOSE, workspace_id=record.workspace_id, background=True, new=True),
    ):
        context = await ContextService(db).build_context(
            mode=ContextMode.CHECK_IN, agent=plan.agent, workspace_id=str(record.workspace_id),
            task_description=plan.task, model_id=getattr(getattr(plan.llm, "config", None), "model", None),
        )
        spec = LoopSpec(run_id=record.run_id, workspace_id=record.workspace_id, agent_id=record.agent_id,
                        system_prompt=context.system_prompt, start_message=start_message(now),
                        tools=tuple(context.tools or ()), reads=plan.reads)
        return await run_loop(db, plan.llm, spec)


def _spent(record: RunRecord, result: LoopResult) -> RunRecord:
    """The run with what its turn spent: the model, tokens and cost."""
    return replace(record, model=result.model, tokens=result.tokens, cost_usd=result.cost_usd)


def _finish(db: Session, record: RunRecord, result: LoopResult) -> CheckInResult:
    """Write the run row: its suggestions, and how it ended."""
    made = suggestions_made(db, record.workspace_id, record.run_id)
    outcome = RUN_ERROR if result.error_class else (RUN_SUGGESTED if made else RUN_SILENT)
    write_run(db, replace(record, outcome=outcome, suggestions=made, error_class=result.error_class))
    logger.info("[proactive] %s run %s in %s: %s, %d suggestion(s), $%.4f", record.kind, record.run_id,
                record.workspace_id, outcome, made, result.cost_usd)
    return CheckInResult(record.run_id, outcome, made, result.error_class)


async def _run(db: Session, record: RunRecord, prepare: Callable[[], Union[CheckInResult, _Plan]],
               now: datetime) -> CheckInResult:
    """Prepare, converse and record one check-in; any failure is recorded as its error,
    with its window and what it spent as far as they are known."""
    try:
        planned = await asyncio.to_thread(prepare)
        if isinstance(planned, CheckInResult):
            return planned
        record = planned.record
        result = await _converse(db, planned, now)
        record = _spent(record, result)
        return await asyncio.to_thread(_finish, db, record, result)
    except Exception as exc:  # noqa: BLE001 — recorded as the run's error, by its class
        logger.exception("[proactive] %s run %s in %s failed", record.kind, record.run_id, record.workspace_id)
        return await asyncio.to_thread(_errored, db, record, failure_class(exc))


async def _deliver(db: Session, *run_ids: Optional[UUID]) -> None:
    """Deliver each run's new suggestions once its row has committed (``deliver_run`` never raises)."""
    for run_id in dict.fromkeys(run_id for run_id in run_ids if run_id is not None):
        await deliver_run(db, run_id)


async def run_workspace_check_in(workspace_id: Any) -> CheckInResult:
    """Run Auto's workspace check-in once, record it, and deliver what it suggested."""
    record = RunRecord(run_id=uuid4(), workspace_id=UUID(str(workspace_id)), kind=KIND_WORKSPACE, outcome=RUN_SILENT)
    now = _now()
    with _session() as db:
        result = await _run(db, record, lambda: _prepare_workspace(db, record, now), now)
        await _deliver(db, result.run_id)
        return result


async def run_agent_check_in(agent_id: int, workspace_id: Any, hb_config: Optional[Dict[str, Any]]) -> CheckInResult:
    """Run ``agent_id``'s check-in once, with its heartbeat config, record it, keep its
    failing streak, and deliver what it suggested and the streak's card when it filed one."""
    record = RunRecord(run_id=uuid4(), workspace_id=UUID(str(workspace_id)), kind=KIND_AGENT, outcome=RUN_SILENT,
                       agent_id=int(agent_id))
    now = _now()
    heartbeat = dict(hb_config or {})
    with _session() as db:
        result = await _run(db, record, lambda: _prepare_agent(db, record, heartbeat, now), now)
        card_run = await asyncio.to_thread(note_run, db, result, record.workspace_id, record.agent_id, heartbeat, now)
        await _deliver(db, result.run_id, card_run)
        return result
