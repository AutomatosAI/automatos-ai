"""PRD-263 US-056: each change an owner approved is judged against the weeks before it, and a
rollback comes as its own Approvals card.

DIAGNOSE judges an agent against a reference that trails the judged week (ORG-4). A change
HARNESS applied cannot be judged that way: as the weeks after it enter a trailing
reference, a lasting drop is absorbed (design section 6.8: a 0.8 → 0.5 drop is caught 0.747
of the time within four weeks with a trailing reference, 0.938 with a frozen one). So each
``applied`` change in ``harness_task_ledger`` is judged against its frozen reference, the
HARNESS_REF_DAYS before ``applied_at`` (``frozen_ref_start``/``frozen_ref_end``, stored by
the apply action), week by week over the ISO weeks after the week it applied in (ORG-10):

* **Metrics.** The kind's predicted metric (services.harness_changes) and its guard
  metrics, the agent's task success rate and tool failure rate. A tool removal's predicted
  metric, the app's failure rate, drops to n=0 with the app gone, so its harm shows in the
  guards. Each waits for a week of at least HARNESS_MIN_SAMPLES for up to
  HARNESS_POST_CHANGE_MAX_WEEKS weeks; otherwise the report says it was not judged, with n.
* **Trigger.** Flags in two consecutive weeks, or one below HARNESS_STRONG_P
  (core.stats.harness_verdicts.post_change_verdict, on the flag DIAGNOSE uses).
* **Control.** The agent's drop over those weeks must exceed the median drop of the
  unchanged agents (no HARNESS change since the reference began) judged over the same weeks;
  with none judged, the rollback is suggested labelled "no control".
* **Rollback.** A ``rollback`` suggestion restores the prior value. It rides in DIAGNOSE's
  result under :data:`KEY` to PRESCRIBE (harness_changes.detect_rollbacks), so in a propose
  workspace it is its own Approvals card, applied only on the owner's click; once it
  applies, :func:`close_rolled_back` moves the change it undoes ``applied → rolled_back``
  (FR-15). A rollback is never judged or rolled back (FR-16), and a change gets no new
  rollback while an earlier one for it waits, is applied, or was declined or withdrawn.

Reads only, apart from that one move: no platform action, no tool executor, no model call (FR-12).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.harness import (
    LEDGER_APPLIED,
    LEDGER_APPLYING,
    LEDGER_DENIED,
    LEDGER_PENDING,
    LEDGER_ROLLED_BACK,
    LEDGER_WITHDRAWN,
    HarnessTaskLedger,
)
from core.stats import harness_verdicts as hv
from services import harness_changes as hc
from services import harness_metrics
from services.harness_diagnose import METRIC_BY_NAME, RateMetric
from services.harness_metrics import TOOL_SUCCESS, HarnessMetrics, build_window

logger = logging.getLogger(__name__)

# Where a run's post-change result rides in DIAGNOSE's result (for PRESCRIBE and the report).
KEY = "post_change"
REPORT_HEADING = "## Changes HARNESS made"
# Every apply kind is judged after it applies, except a rollback (FR-16).
JUDGED_KINDS = tuple(name for name in hc.APPLY_KINDS if name != hc.ROLLBACK)
# An earlier rollback of a change in one of these states stops a new one: waiting for its
# click, being applied, applied, or declined or withdrawn by the owner.
SETTLED_ROLLBACK_STATES = frozenset({LEDGER_PENDING, LEDGER_APPLYING, LEDGER_APPLIED, LEDGER_DENIED,
                                     LEDGER_WITHDRAWN})
NO_CONTROL = "no control"
DAYS_PER_WEEK = 7

APPLIED_LINE = "- {change}, applied {day}: {parts}.{tail}"
ROLLBACK_TAIL = " HARNESS suggests undoing it (see the suggested changes): {why}"
SETTLED_TAIL = " An earlier card to undo it is {state}."
FELL = "{label} {moved} from {before:.0%} ({x_ref} of {n_ref}) in the {days} days before the change to " \
       "{after:.0%} ({x} of {n}) in week {week} after it, p={p:.2g}."
# A control compares rates, so its drops are in percentage points.
EXCEEDS = "Over the same weeks it worsened by {treated:.0f} points, against a median of {median:.0f} for the " \
          "{count} unchanged agent(s) judged."
NOT_EXCEEDED = "{label}: worsened by {treated:.0f} points, no more than the {count} unchanged agent(s) judged " \
               "(median {median:.0f}), so nothing to undo"
POINTS = 100
NOTHING_TO_COMPARE = "No unchanged agent had enough data over the same weeks to compare with (no control)."
FELL_PART = "{label}: worse than before the change"
WAITING = "{label}: not judged yet ({n} {noun} since the change, {n_ref} in the {days} days before)"
NOT_JUDGED = "{label}: not judged ({n} {noun} in the {weeks} weeks since the change, {n_ref} in the {days} days " \
             "before; a week needs {min_samples})"
HELD = "{label}: held up"


@dataclass(frozen=True)
class PostChangeSettings:
    """The check's thresholds (config.py): DIAGNOSE's rate thresholds and strong p, how many
    weeks a metric waits, and the frozen reference's length in days (for the report)."""

    rates: hv.ProportionThresholds
    strong_p: float
    max_weeks: int
    ref_days: int


@dataclass(frozen=True)
class Metric:
    """A rate judged after a change: ``count`` gives one agent's (x, n) over some ISO weeks
    of a reading; x counts successes, or with ``higher_is_worse`` failures."""

    name: str
    label: str
    noun: str
    higher_is_worse: bool
    count: Callable[[HarnessMetrics, int, FrozenSet[str]], Tuple[int, int]]


@dataclass(frozen=True)
class Readings:
    """What one change is judged on: its frozen reference, the weeks read since, its own
    weeks after the week it applied in (oldest first), and the unchanged agents."""

    ref: HarnessMetrics
    post: HarnessMetrics
    weeks: Tuple[str, ...]
    unchanged: FrozenSet[int]


@dataclass(frozen=True)
class Judged:
    """One metric of one change: where its check stands, its n before and since, and on a
    trigger the week that triggered and the control."""

    metric: Metric
    outcome: hv.PostChangeOutcome
    n_ref: int
    n_post: int
    trigger: Optional[hv.FlagResult] = None
    control: Optional[hv.DiffInDiffResult] = None

    @property
    def raises_rollback(self) -> bool:
        return self.control is not None and self.control.raises_rollback


@dataclass(frozen=True)
class Check:
    """One applied change and its judged metrics; ``settled`` is the state of an earlier
    rollback of it that stops a new one."""

    rx_id: str
    kind: str
    agent_id: int
    agent_name: str
    prior_value: Mapping[str, Any]
    proposed_value: Mapping[str, Any]
    applied_at: datetime
    judged: Tuple[Judged, ...]
    settled: Optional[str] = None

    @property
    def undo_on(self) -> Optional[Judged]:
        """The first metric whose trigger beat the control, if any."""
        return next((judged for judged in self.judged if judged.raises_rollback), None)


@dataclass(frozen=True)
class PostChange:
    """A run's post-change result: its report lines and the rollbacks it suggests."""

    lines: Tuple[str, ...] = ()
    rollbacks: Tuple[hc.Proposal, ...] = ()


def settings_from_config() -> PostChangeSettings:
    """The HARNESS_* settings the check reads from config.py."""
    from config import config
    from services import harness_diagnose

    diagnose = harness_diagnose.settings_from_config()
    return PostChangeSettings(diagnose.rates, diagnose.strong_p, config.HARNESS_POST_CHANGE_MAX_WEEKS,
                              config.HARNESS_REF_DAYS)


def frozen_reference(applied_at: datetime) -> Tuple[datetime, datetime]:
    """A change's frozen reference, the HARNESS_REF_DAYS before ``applied_at``, which the
    apply action stores with it (``frozen_ref_start``, ``frozen_ref_end``)."""
    from config import config

    return applied_at - timedelta(days=config.HARNESS_REF_DAYS), applied_at


def rollbacks_of(diagnosis: Mapping[str, Any]) -> Tuple[hc.Proposal, ...]:
    """The rollbacks the run's check suggests, from DIAGNOSE's result."""
    result = diagnosis.get(KEY)
    return result.rollbacks if isinstance(result, PostChange) else ()


def report_lines(diagnosis: Mapping[str, Any]) -> List[str]:
    """The audit report's section on the changes HARNESS made; none when none is checked."""
    result = diagnosis.get(KEY)
    if not isinstance(result, PostChange) or not result.lines:
        return []
    return [REPORT_HEADING, *result.lines, ""]


# ------------------------------------------------------------------ metrics


def _rate_counts(metric: RateMetric, reading: HarnessMetrics, agent_id: int,
                 weeks: FrozenSet[str]) -> Tuple[int, int]:
    rows = [row for row in reading.agent_rows if row["agent_id"] == agent_id and row["iso_week"] in weeks]
    return (sum(int(row[metric.x_field]) for row in rows),
            sum(int(row[field]) for row in rows for field in metric.n_fields))


def _app_counts(app: str, reading: HarnessMetrics, agent_id: int, weeks: FrozenSet[str]) -> Tuple[int, int]:
    runs = [run for run in reading.tool_rows if run["agent_id"] == agent_id and run["iso_week"] in weeks
            and str(run["app_name"]).upper() == app]
    return sum(run["calls"] for run in runs if run["status"] != TOOL_SUCCESS), sum(run["calls"] for run in runs)


def _metric(name: str, prior_value: Mapping[str, Any]) -> Metric:
    """A metric by its harness_changes name; the app's failure rate reads the removed app."""
    if name == hc.APP_TOOL_FAILURE:
        app = str(prior_value.get(hc.APP_KEY) or "").upper()
        return Metric(name, f"{app} failures", f"{app} calls", True, partial(_app_counts, app))
    rate = METRIC_BY_NAME[name]
    if not isinstance(rate, RateMetric):
        raise ValueError(f"{name} is not a rate a change can be judged on")
    return Metric(name, rate.label, rate.noun, rate.higher_is_worse, partial(_rate_counts, rate))


def metrics_of(kind: str, prior_value: Mapping[str, Any]) -> Tuple[Metric, ...]:
    """A change kind's predicted metric, then its guard metrics."""
    change = hc.KINDS[kind]
    return tuple(_metric(name, prior_value) for name in (change.predicted_metric, *change.guard_metrics))


# ------------------------------------------------------------------ judging


def _flag(metric: Metric, agent_id: int, readings: Readings, weeks: Iterable[str],
          settings: PostChangeSettings) -> hv.FlagResult:
    """``metric`` for one agent over ``weeks`` after the change, against its frozen reference."""
    x_ref, n_ref = metric.count(readings.ref, agent_id, frozenset(readings.ref.window.weeks))
    x, n = metric.count(readings.post, agent_id, frozenset(weeks))
    return hv.proportion_flag(x, n, x_ref, n_ref, settings.rates, higher_is_worse=metric.higher_is_worse)


def _control(metric: Metric, agent_id: int, readings: Readings, weeks: Tuple[str, ...],
             settings: PostChangeSettings) -> hv.DiffInDiffResult:
    """The agent's worsening over ``weeks`` against each judged unchanged agent's."""
    treated = _flag(metric, agent_id, readings, weeks, settings)
    others = [_flag(metric, other, readings, weeks, settings) for other in sorted(readings.unchanged)]
    return hv.diff_in_diff(treated.worsening or 0.0, [other.worsening for other in others if other.sufficient])


def judge(metric: Metric, agent_id: int, readings: Readings, settings: PostChangeSettings) -> Judged:
    """One metric of one change, week by week against the frozen reference (ORG-10)."""
    weekly = [_flag(metric, agent_id, readings, (week,), settings) for week in readings.weeks]
    outcome = hv.post_change_verdict(weekly, settings.max_weeks, settings.strong_p)
    n_ref = metric.count(readings.ref, agent_id, frozenset(readings.ref.window.weeks))[1]
    n_post = sum(result.n_cur for result in weekly)
    if outcome.status is not hv.PostChangeStatus.ROLLBACK:
        return Judged(metric, outcome, n_ref, n_post)
    week = outcome.trigger_week
    control = _control(metric, agent_id, readings, readings.weeks[:week], settings)
    return Judged(metric, outcome, n_ref, n_post, weekly[week - 1], control)


# ------------------------------------------------------------------ the ledger


def _utc_day(moment: datetime) -> date:
    return (moment.astimezone(timezone.utc) if moment.tzinfo else moment).date()


def post_weeks(applied_at: datetime, cur_week: str, max_weeks: int) -> Optional[Tuple[str, ...]]:
    """The ISO weeks after the week a change applied in, up to the judged week, oldest first;
    None once more than ``max_weeks`` of them have passed (its check is over)."""
    applied = _utc_day(applied_at)
    applied_monday = applied - timedelta(days=applied.weekday())
    count = (build_window(cur_week, 0).start.date() - applied_monday).days // DAYS_PER_WEEK
    if count > max_weeks:
        return None
    return build_window(cur_week, count - 1).weeks if count > 0 else ()


def _judged_row(row: HarnessTaskLedger) -> bool:
    """An applied change of a judged kind, with its target, moment and frozen reference."""
    return (row.state == LEDGER_APPLIED and row.kind in JUDGED_KINDS and row.target_agent_id is not None
            and row.applied_at is not None and row.frozen_ref_start is not None
            and row.frozen_ref_end is not None and row.frozen_ref_start < row.frozen_ref_end)


def _undone_rx(row: HarnessTaskLedger) -> Optional[str]:
    """The rx id of the change a rollback row undoes."""
    value = row.proposed_value if isinstance(row.proposed_value, dict) else {}
    undone = value.get(hc.ROLLBACK_RX_KEY) if row.kind == hc.ROLLBACK else None
    return undone if isinstance(undone, str) else None


def _settled_rollbacks(rows: Iterable[HarnessTaskLedger]) -> Dict[str, str]:
    """rx id of a change -> the state of an earlier rollback of it that stops a new one."""
    return {undone: row.state for row in rows
            if (undone := _undone_rx(row)) is not None and row.state in SETTLED_ROLLBACK_STATES}


def _changed_since(rows: Iterable[HarnessTaskLedger], moment: datetime) -> FrozenSet[int]:
    """Agents HARNESS changed (or rolled back) at or after ``moment``."""
    return frozenset(row.target_agent_id for row in rows
                     if row.state in (LEDGER_APPLIED, LEDGER_ROLLED_BACK) and row.applied_at is not None
                     and row.target_agent_id is not None and row.applied_at >= moment)


def _names(reading: HarnessMetrics) -> Dict[int, str]:
    return {row["agent_id"]: row["agent_name"] for row in reading.agent_rows}


# ------------------------------------------------------------------ the report and the rollback


def _part(judged: Judged, weeks_seen: int, settings: PostChangeSettings) -> str:
    """One metric's standing, in the report's words."""
    metric, status = judged.metric, judged.outcome.status
    figures = {"label": metric.label, "noun": metric.noun, "n": judged.n_post, "n_ref": judged.n_ref,
               "days": settings.ref_days, "weeks": weeks_seen, "min_samples": settings.rates.min_samples}
    if status is hv.PostChangeStatus.ROLLBACK and not judged.raises_rollback:
        control = judged.control
        return NOT_EXCEEDED.format(label=metric.label, treated=control.treated_drop * POINTS, count=control.control_n,
                                   median=(control.control_median_drop or 0.0) * POINTS)
    if status is hv.PostChangeStatus.ROLLBACK:
        return FELL_PART.format(**figures)
    if status is hv.PostChangeStatus.WAITING:
        return WAITING.format(**figures)
    return NOT_JUDGED.format(**figures) if status is hv.PostChangeStatus.NOT_JUDGED else HELD.format(**figures)


def _why_undo(judged: Judged, settings: PostChangeSettings) -> str:
    """A rollback's reason: the week that triggered against the frozen reference, then the control."""
    metric, flag, control = judged.metric, judged.trigger, judged.control
    fell = FELL.format(
        label=metric.label, moved="rose" if metric.higher_is_worse else "fell", before=flag.ref_value,
        x_ref=round(flag.ref_value * flag.n_ref), n_ref=flag.n_ref, days=settings.ref_days, after=flag.cur_value,
        x=round(flag.cur_value * flag.n_cur), n=flag.n_cur, week=judged.outcome.trigger_week, p=flag.p_value,
    )
    if control.status is hv.ControlStatus.NO_CONTROL:
        return f"{fell} {NOTHING_TO_COMPARE}"
    beat = EXCEEDS.format(treated=control.treated_drop * POINTS, median=(control.control_median_drop or 0.0) * POINTS,
                          count=control.control_n)
    return f"{fell} {beat}"


def _change_words(check: Check) -> str:
    from services.harness_cards import change_and_approve

    rx = {"target_name": check.agent_name, "change_type": check.kind, "current_value": check.prior_value,
          "proposed_value": check.proposed_value}
    return change_and_approve(rx)[0]


def _line(check: Check, settings: PostChangeSettings, weeks_seen: int) -> str:
    """The report's line for one change: each metric's standing, and any rollback."""
    undo = check.undo_on
    if check.settled is not None and undo is not None:
        tail = SETTLED_TAIL.format(state=check.settled)
    else:
        tail = ROLLBACK_TAIL.format(why=_why_undo(undo, settings)) if undo is not None else ""
    parts = "; ".join(_part(judged, weeks_seen, settings) for judged in check.judged)
    return APPLIED_LINE.format(change=_change_words(check), day=_utc_day(check.applied_at).isoformat(),
                               parts=parts, tail=tail)


def _rollback(check: Check, settings: PostChangeSettings) -> hc.Proposal:
    """The rollback of a change whose trigger beat the control: restore its prior value."""
    undo = check.undo_on
    control = undo.control
    return hc.Proposal(
        target_id=check.agent_id, target_name=check.agent_name,
        current_value=dict(check.proposed_value),
        proposed_value={hc.ROLLBACK_OF_KEY: check.kind, hc.ROLLBACK_VALUE_KEY: dict(check.prior_value),
                        hc.ROLLBACK_RX_KEY: check.rx_id},
        rationale=_why_undo(undo, settings),
        expected_improvement=f"{check.agent_name} works as it did before the change",
        evidence={"rolls_back": check.rx_id, "metric": undo.metric.name, "trigger_week": undo.outcome.trigger_week,
                  "p_value": undo.trigger.p_value, "control_agents": control.control_n,
                  "control": NO_CONTROL if control.status is hv.ControlStatus.NO_CONTROL else control.status.value},
    )


# ------------------------------------------------------------------ the check


def _check(db: Session, ws: UUID, row: HarnessTaskLedger, post: HarnessMetrics, rows: List[HarnessTaskLedger],
           weeks: Tuple[str, ...], settings: PostChangeSettings) -> Check:
    """Judge one applied change on its frozen reference and the weeks since. The unchanged
    agents are the active roster less every agent HARNESS changed since the reference began."""
    ref = harness_metrics.read_window(db, ws, harness_metrics.span_window(row.frozen_ref_start, row.frozen_ref_end))
    names = _names(post)
    readings = Readings(ref, post, weeks, frozenset(names) - _changed_since(rows, row.frozen_ref_start))
    agent = row.target_agent_id
    prior = row.prior_value if isinstance(row.prior_value, dict) else {}
    judged = tuple(judge(metric, agent, readings, settings) for metric in metrics_of(row.kind, prior))
    proposed = row.proposed_value if isinstance(row.proposed_value, dict) else {}
    return Check(row.rx_id, row.kind, agent, names.get(agent) or f"agent #{agent}", prior, proposed,
                 row.applied_at, judged)


def check_sync(db: Session, workspace_id: Union[UUID, str], cur_week: str,
               settings: PostChangeSettings) -> PostChange:
    """Judge each change HARNESS applied in the workspace whose check is not over, for the
    judged week ``cur_week``; returns the report's lines and the rollbacks to suggest."""
    ws = UUID(str(workspace_id))
    rows = (db.query(HarnessTaskLedger)
            .filter(HarnessTaskLedger.workspace_id == ws, HarnessTaskLedger.kind.in_(hc.APPLY_KINDS))
            .order_by(HarnessTaskLedger.id).all())
    due = [(row, weeks) for row in rows if _judged_row(row)
           and (weeks := post_weeks(row.applied_at, cur_week, settings.max_weeks)) is not None]
    if not due:
        return PostChange()
    post = harness_metrics.read_window(db, ws, build_window(cur_week, settings.max_weeks - 1))
    settled = _settled_rollbacks(rows)
    checks = [(replace(_check(db, ws, row, post, rows, weeks, settings), settled=settled.get(row.rx_id)),
               len(weeks)) for row, weeks in due]
    rollbacks = tuple(_rollback(check, settings) for check, _ in checks
                      if check.undo_on is not None and check.settled is None)
    logger.info("[HARNESS] Post-change check in %s for %s: %d change(s) judged, %d rollback(s) suggested",
                ws, cur_week, len(checks), len(rollbacks))
    return PostChange(tuple(_line(check, settings, seen) for check, seen in checks), rollbacks)


async def check(db: Session, workspace_id: Union[UUID, str], cur_week: str) -> PostChange:
    """:func:`check_sync` with config.py's settings, on a worker thread like COLLECT's read."""
    return await asyncio.to_thread(check_sync, db, workspace_id, cur_week, settings_from_config())


def close_rolled_back(db: Session, rollback: HarnessTaskLedger) -> bool:
    """Once a rollback applies, move the change it undoes ``applied → rolled_back``, the one
    move out of a final state (FR-15). True when it moved. Not committed: the apply action's
    caller owns the transaction."""
    from services import harness_grants

    undone = _undone_rx(rollback)
    original = harness_grants.ledger_row(db, rollback.workspace_id, undone) if undone else None
    if original is None or original.kind not in JUDGED_KINDS:
        logger.warning("[HARNESS] Rollback %s in %s names no change to close", rollback.rx_id, rollback.workspace_id)
        return False
    return harness_grants.move(db, original, LEDGER_ROLLED_BACK, from_state=LEDGER_APPLIED)


__all__ = ["KEY", "Check", "Judged", "Metric", "PostChange", "PostChangeSettings", "Readings", "check", "check_sync",
           "close_rolled_back", "frozen_reference", "judge", "metrics_of", "post_weeks", "report_lines", "rollbacks_of",
           "settings_from_config"]
