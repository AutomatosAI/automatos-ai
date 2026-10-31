"""PRD-268 §6.7 (US-C13b) — the check-in silence eval: Auto's workspace check-in on each gold snapshot.

Each line of ``silence.jsonl`` is one workspace snapshot::

    {"signals": [{"kind": "ticket_failed", "ref": {"type": "ticket", "id": "0846"},
                  "at": "2026-10-30T09:00:00+00:00", "summary": "..."}],
     "open_titles": ["..."], "hidden_titles": ["..."],
     "since": "2026-10-29T09:30:00+00:00", "until": "2026-10-30T09:30:00+00:00",
     "need": ["Slack", "401"]}

``need`` holds the planted need's words, and the snapshot's signals the one that plants it; a
quiet snapshot has none. ``until`` is the check-in's
time (now when left out) and ``since`` its window's start (a day before ``until``, as a first
check-in's, when left out). A naive time is UTC.

The turn is W1's: the prompt is ``prompts.workspace_task`` over the snapshot's signals and its open
and hidden titles, the one message ``prompts.start_message``, and the model the check-in's
(``runner.heartbeat_llm``, the platform ``orchestrator_llm``, booked ``check_in``), run as one
background ask in the ``check_in`` lane. Only ``platform_suggest_next_step``'s schema is offered,
and its calls are captured, never run: nothing is written. A check-in whose calls are all
suggestions ends there (``loop.run_loop``), so one model call is the turn, and of its calls the
first ``MAX_CALLS_PER_TURN`` count, as the loop runs only those. A snapshot with no signal is
silent with no model call, as the runner's own pre-check makes it. A suggestion the platform's
tool would refuse (its evidence, its cap) still counts: the eval measures what the model proposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Mapping, Sequence, Tuple
from uuid import UUID

from core.llm.usage_context import usage_scope
from evals.proactive.gold import GoldSetUnavailable, read_gold
from evals.proactive.runtime import eval_workspace, tool_schema
from evals.proactive.scoring import Score, SilenceRun, Suggested, score_silence
from modules.proactive import check_in_window
from modules.tools.discovery.check_in_lane import SUGGEST_NEXT_STEP
from modules.tools.execution.params_text import json_container
from services.credits.asks import ask_scope
from services.proactive.loop import MAX_CALLS_PER_TURN
from services.proactive.prompts import start_message, workspace_task
from services.proactive.runner import CHECK_IN_PURPOSE, heartbeat_llm
from services.proactive.signals import Signal

__all__ = ["NAME", "Snapshot", "check_in", "evaluate", "read_snapshots", "suggestions_in"]

NAME = "silence"


@dataclass(frozen=True)
class Snapshot:
    """One gold snapshot: its planted need's words (none when quiet), its window, its signals, and
    its open and hidden titles."""

    need_terms: Tuple[str, ...]
    since: datetime
    until: datetime
    signals: Tuple[Signal, ...]
    open_titles: Tuple[str, ...]
    hidden_titles: Tuple[str, ...]


def _moment(value: Any, number: int) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise GoldSetUnavailable(f"{NAME} case {number}: {value!r} is not an ISO time.") from exc
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _given(value: Any) -> bool:
    """A value is given when it says something: 0 does, None and blank text do not."""
    return value is not None and bool(str(value).strip())


def _signal(raw: Any, number: int) -> Signal:
    """One listed signal, as ``signals.read_signals`` would have read it."""
    ref = raw.get("ref") if isinstance(raw, Mapping) else None
    named = isinstance(ref, Mapping) and all(_given(ref.get(key)) for key in ("type", "id"))
    if not named or not _given(raw.get("kind")):
        raise GoldSetUnavailable(f"{NAME} case {number}: each signal has a kind and a ref {{type, id}}.")
    kind, ref_type, ref_id = str(raw["kind"]), str(ref["type"]), str(ref["id"])
    return Signal(kind=kind, subject_key=str(raw.get("subject_key") or f"{ref_type}:{ref_id}"),
                  ref=(ref_type, ref_id), at=_moment(raw.get("at"), number), summary=str(raw.get("summary") or ""))


def _texts(case: Mapping[str, Any], key: str, number: int) -> Tuple[str, ...]:
    values = case.get(key) or []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise GoldSetUnavailable(f"{NAME} case {number}: {key} is a list of text.")
    return tuple(values)


def _need(case: Mapping[str, Any], number: int) -> Tuple[str, ...]:
    """The planted need's words, refused when one says nothing once normalised (``SilenceRun``)."""
    need = _texts(case, "need", number)
    try:
        SilenceRun(need_terms=need, suggested=())
    except ValueError as exc:
        raise GoldSetUnavailable(f"{NAME} case {number}: {exc}.") from exc
    return need


def _snapshot(case: Mapping[str, Any], number: int) -> Snapshot:
    signals = case.get("signals") or []
    if not isinstance(signals, list):
        raise GoldSetUnavailable(f"{NAME} case {number}: signals is a list.")
    need = _need(case, number)
    if need and not signals:
        raise GoldSetUnavailable(f"{NAME} case {number}: a planted need comes with the signal that plants it.")
    until = _moment(case["until"], number) if case.get("until") else datetime.now(timezone.utc)
    since = _moment(case["since"], number) if case.get("since") else check_in_window(None, until)
    return Snapshot(need_terms=need, since=since, until=until,
                    signals=tuple(_signal(raw, number) for raw in signals),
                    open_titles=_texts(case, "open_titles", number),
                    hidden_titles=_texts(case, "hidden_titles", number))


def read_snapshots() -> List[Snapshot]:
    """Every gold snapshot, read and checked before the first model call."""
    return [_snapshot(case, number) for number, case in enumerate(read_gold(NAME), start=1)]


def _field(item: Any, name: str) -> Any:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def suggestions_in(response: Any) -> Tuple[Suggested, ...]:
    """Each ``platform_suggest_next_step`` call the response made, among the calls the loop would run,
    as its title and detail."""
    found = []
    for call in list(_field(response, "tool_calls") or [])[:MAX_CALLS_PER_TURN]:
        function = _field(call, "function") or {}
        if _field(function, "name") != SUGGEST_NEXT_STEP:
            continue
        arguments = json_container(_field(function, "arguments"))
        arguments = arguments if isinstance(arguments, Mapping) else {}
        found.append(Suggested(title=str(arguments.get("title") or ""), detail=str(arguments.get("detail") or "")))
    return tuple(found)


async def check_in(llm: Any, snapshot: Snapshot, tools: Sequence[Mapping[str, Any]],
                   workspace_id: str) -> Tuple[Suggested, ...]:
    """One check-in's turn over ``snapshot``: the suggestions it proposed, never recorded."""
    if not snapshot.signals:
        return ()
    task = workspace_task(snapshot.signals, snapshot.since, snapshot.until, snapshot.open_titles,
                          snapshot.hidden_titles)
    messages = [{"role": "system", "content": task}, {"role": "user", "content": start_message(snapshot.until)}]
    with (
        usage_scope(request_type=CHECK_IN_PURPOSE, workspace_id=workspace_id, inherit=False),
        ask_scope(CHECK_IN_PURPOSE, workspace_id=workspace_id, background=True, new=True),
    ):
        response = await llm.generate_response(messages, tools=[dict(tool) for tool in tools])
    return suggestions_in(response)


def _run(snapshot: Snapshot, suggested: Tuple[Suggested, ...]) -> SilenceRun:
    return SilenceRun(need_terms=snapshot.need_terms, suggested=suggested,
                      taken_titles=(*snapshot.open_titles, *snapshot.hidden_titles))


async def evaluate() -> Score:
    """Run the check-in on every gold snapshot and score silence, the planted need and duplicates."""
    snapshots = read_snapshots()
    workspace_id = eval_workspace()
    llm = heartbeat_llm(UUID(workspace_id), None)
    tools = (tool_schema(SUGGEST_NEXT_STEP),)
    runs = [_run(snapshot, await check_in(llm, snapshot, tools, workspace_id)) for snapshot in snapshots]
    return score_silence(runs)
