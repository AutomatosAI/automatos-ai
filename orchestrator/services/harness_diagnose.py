"""PRD-263 US-014a/US-014b: HARNESS DIAGNOSE judges each agent on its real counts.

DIAGNOSE used to compare four ``platform_list_agents`` fields that are never
returned against last week's baseline file with a flat 10% rule, so every agent
came out STABLE and the report said the organization was stable. It now judges
each agent x metric pair of the ``services.harness_metrics`` rows with the
statistics in ``core.stats.harness_verdicts`` (design section 5.1, ORG-4 and ORG-5):

* **Metrics.** Task success, Mission step success, heartbeat objectives met and
  tool failures are rates. Cost per run and heartbeat cost per run are per-run
  amounts, judged in $ when the agent has a billed run in the weeks compared and
  in tokens when it has none (CLI and subscription agents book $0).
* **Clean reference.** The ISO weeks of ``HARNESS_REF_DAYS`` before the judged
  week, less each week already flagged for that agent and metric (so a bad week
  is never absorbed into the reference) and less the weeks from a change HARNESS
  applied to that agent (``harness_task_ledger``, state ``applied``) onwards. A
  change older than the reference leaves it whole: every week in it is then the
  agent as it now stands.
* **Verdicts.** n below ``HARNESS_MIN_SAMPLES`` in the week or the reference is
  INSUFFICIENT_DATA. Benjamini-Hochberg at ``HARNESS_FDR_Q`` runs across every test
  of the run. A first flag is WATCH; REGRESSION needs a flag in the previous ISO
  week too, or p below ``HARNESS_STRONG_P``. A week already judged keeps its
  verdicts, so a rerun (``platform_harness_trigger``) never counts as a second flag,
  and the figures of its first run, so the report shows the numbers behind them.
* **Coverage** is judged agents over active agents (Auto is not one). Below
  ``HARNESS_MIN_COVERAGE`` the run's convergence reads ``insufficient_data``;
  otherwise ``regression``, ``watching`` or ``steady``.
* **Tool failures** (US-019, FR-8) are judged per agent and app on the judged week's
  production tool logs, never on the routing graph's lifetime ``fails_for_intent``
  counts: a pair flags at n of at least ``HARNESS_MIN_SAMPLES`` with a Wilson lower
  bound of the failure rate at least ``HARNESS_TOOL_FAIL_WILSON_MIN``. A flag that
  sign-in and connection failures carry is "reconnect <app>", never a bad tool.

Agent rows are read only through the names in ``DIAGNOSE_READS``, which
``tests/test_prd263_harness_key_contract.py`` pins to
``harness_metrics.AGENT_ROW_FIELDS``. Pure apart from ``applied_change_weeks``,
which reads the ledger: no write, no platform action, no model call.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.stats import harness_verdicts as hv
from services.harness_metrics import (
    AGENT_ID_PATTERN, RECONNECT_CLASSES, TOOL_SUCCESS, UTC_ZONE, HarnessMetrics, Window, iso_week_label,
)

logger = logging.getLogger(__name__)

DAYS_PER_WEEK = 7

# The agent-row fields DIAGNOSE reads, and the only ones (US-015).
DIAGNOSE_READS: Tuple[str, ...] = (
    "agent_id", "agent_name", "model_id", "heartbeat_enabled", "heartbeat_interval_minutes", "iso_week",
    "tasks_done", "tasks_failed", "mission_tasks_done", "mission_tasks_failed",
    "heartbeat_objective_met", "heartbeat_objective_missed",
    "tool_calls", "tool_failed_calls",
    "llm_run_costs_usd", "llm_run_tokens", "heartbeat_llm_run_costs_usd", "heartbeat_llm_run_tokens",
    "heartbeat_llm_billed_cost_usd",
)

UNIT_RATE = "rate"
UNIT_USD = "usd"
UNIT_TOKENS = "tokens"
_UNITS = frozenset({UNIT_RATE, UNIT_USD, UNIT_TOKENS})

# Run-level convergence (ORG-5).
CONVERGENCE_INSUFFICIENT = "insufficient_data"
CONVERGENCE_REGRESSION = "regression"
CONVERGENCE_WATCHING = "watching"
CONVERGENCE_STEADY = "steady"

# Why a metric's verdict is what it is, where the verdict alone does not say.
REASON_KEPT = "kept"                # the week was already judged; its verdict stands
REASON_SECOND_WEEK = "second_week"  # REGRESSION: flagged in the previous ISO week too
REASON_STRONG_P = "strong_p"        # REGRESSION: one flag below HARNESS_STRONG_P

# The keys of a stored run (harness_runs.verdicts) and of its counts.
AGENTS_KEY = "agents"
CONVERGENCE_KEY = "convergence"
METRICS_KEY = "metrics"
VERDICT_KEY = "verdict"
COUNT_KEYS = {
    hv.Verdict.INSUFFICIENT_DATA: "insufficient_data",
    hv.Verdict.STABLE: "stable",
    hv.Verdict.WATCH: "watch",
    hv.Verdict.REGRESSION: "regression",
}
# What platform_harness_status returns from a completed run, besides coverage and convergence.
STATUS_COUNT_KEYS = ("active_agents", "judged_agents", "insufficient_data", "watch", "regression")

_VERDICT_VALUES = frozenset(v.value for v in hv.Verdict)
_FLAGGED_VALUES = frozenset(v.value for v in hv.FLAGGED_VERDICTS)
_VERDICT_RANK = {
    hv.Verdict.INSUFFICIENT_DATA: 0, hv.Verdict.STABLE: 1, hv.Verdict.WATCH: 2, hv.Verdict.REGRESSION: 3,
}
# Report wording: an agent HARNESS did not judge is never called stable (FR-7).
VERDICT_WORDS = {
    hv.Verdict.INSUFFICIENT_DATA: "Not enough data",
    hv.Verdict.STABLE: "Steady",
    hv.Verdict.WATCH: "Watching",
    hv.Verdict.REGRESSION: "Regression",
}

# ISO week -> agent id -> metric -> that metric's stored entry (its verdict and figures).
History = Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]
# The figures a stored metric entry carries; a kept verdict is reported with its own.
FIGURE_KEYS = ("unit", "n_cur", "n_ref", "x_cur", "x_ref", "cur_value", "ref_value", "p_value", "ref_weeks")

# The root causes of DIAGNOSE's tool issues (US-019).
TOOL_FAILURE = "tool_failure"
RECONNECT_APP = "reconnect_app"
# The platform's own tools, as telemetry names their app: no connection to reconnect.
PLATFORM_APPS = frozenset({"PLATFORM", "WORKSPACE"})


@dataclass(frozen=True)
class RateMetric:
    """A rate: ``x_field`` over the sum of ``n_fields``. ``x`` counts successes, or
    with ``higher_is_worse`` failures, whose rise is the worsening."""

    name: str
    label: str
    noun: str
    x_field: str
    n_fields: Tuple[str, ...]
    higher_is_worse: bool = False


@dataclass(frozen=True)
class AmountMetric:
    """A per-run amount: one sample per run, in $ (``usd_field``) or tokens (``tokens_field``)."""

    name: str
    label: str
    tokens_label: str
    noun: str
    usd_field: str
    tokens_field: str


Metric = Union[RateMetric, AmountMetric]

METRICS: Tuple[Metric, ...] = (
    RateMetric("task_success", "Task success", "Tasks", "tasks_done", ("tasks_done", "tasks_failed")),
    RateMetric("mission_success", "Mission step success", "Mission steps", "mission_tasks_done",
               ("mission_tasks_done", "mission_tasks_failed")),
    RateMetric("heartbeat_objective", "Heartbeat objectives met", "judged heartbeats", "heartbeat_objective_met",
               ("heartbeat_objective_met", "heartbeat_objective_missed")),
    RateMetric("tool_failure", "Tool failures", "tool calls", "tool_failed_calls", ("tool_calls",),
               higher_is_worse=True),
    AmountMetric("run_cost", "Cost per run", "Tokens per run", "runs", "llm_run_costs_usd", "llm_run_tokens"),
    AmountMetric("heartbeat_cost", "Heartbeat cost per run", "Heartbeat tokens per run", "heartbeat runs",
                 "heartbeat_llm_run_costs_usd", "heartbeat_llm_run_tokens"),
)
METRIC_BY_NAME: Dict[str, Metric] = {metric.name: metric for metric in METRICS}


@dataclass(frozen=True)
class DiagnoseSettings:
    """The thresholds DIAGNOSE passes to core.stats.harness_verdicts (PRD-263 section 9)."""

    rates: hv.ProportionThresholds
    amounts: hv.RiseThresholds
    fdr_q: float
    strong_p: float
    min_coverage: float
    tool_fail_wilson_min: float


@dataclass(frozen=True)
class _Test:
    """One agent x metric comparison of the judged week against its clean reference."""

    agent_id: int
    metric: Metric
    unit: str
    ref_weeks: Tuple[str, ...]
    result: hv.FlagResult
    x_cur: Optional[int] = None
    x_ref: Optional[int] = None


def settings_from_config() -> DiagnoseSettings:
    """The HARNESS_* thresholds from config.py. Cost and tokens share the rise
    thresholds; HARNESS_COST_ABS_RISE is in $, so for tokens it is no real floor."""
    from config import config

    return DiagnoseSettings(
        rates=hv.ProportionThresholds(
            min_samples=config.HARNESS_MIN_SAMPLES, min_abs_drop=config.HARNESS_ABS_DROP,
            min_rel_drop=config.HARNESS_REL_DROP, alpha=config.HARNESS_ALPHA,
        ),
        amounts=hv.RiseThresholds(
            min_samples=config.HARNESS_MIN_SAMPLES, min_rel_rise=config.HARNESS_COST_REL_RISE,
            min_abs_rise=config.HARNESS_COST_ABS_RISE, alpha=config.HARNESS_ALPHA,
        ),
        fdr_q=config.HARNESS_FDR_Q,
        strong_p=config.HARNESS_STRONG_P,
        min_coverage=config.HARNESS_MIN_COVERAGE,
        tool_fail_wilson_min=config.HARNESS_TOOL_FAIL_WILSON_MIN,
    )


def reference_weeks() -> int:
    """HARNESS_REF_DAYS in whole ISO weeks: how many weeks before the judged one COLLECT reads."""
    from config import config

    return config.HARNESS_REF_DAYS // DAYS_PER_WEEK


# ------------------------------------------------------------------ inputs


def _stored_metrics(entry: Any) -> Dict[str, Dict[str, Any]]:
    """metric -> its stored entry, from one stored agent entry; a metric without a
    known verdict, and anything else in the entry, is skipped."""
    metrics = entry.get(METRICS_KEY) if isinstance(entry, dict) else None
    if not isinstance(metrics, dict):
        return {}
    return {name: judged for name, judged in metrics.items()
            if isinstance(judged, dict) and judged.get(VERDICT_KEY) in _VERDICT_VALUES}


def verdict_history(runs: Iterable[Tuple[str, Any]]) -> Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]:
    """ISO week -> agent id -> metric -> stored entry, from the (cur_window, verdicts)
    of completed runs, oldest first (``harness_runs.judged_verdicts``). A week judged
    more than once keeps its first run's entries for each agent."""
    history: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    for window, stored in runs:
        agents = stored.get(AGENTS_KEY) if isinstance(stored, dict) else None
        for agent_id, entry in (agents if isinstance(agents, dict) else {}).items():
            history.setdefault(window, {}).setdefault(str(agent_id), _stored_metrics(entry))
    return history


_APPLIED_CHANGES_SQL = text("""
    WITH changes AS (
        SELECT CASE WHEN l.entry ->> 'target_id' ~ :agent_id_pattern
                    THEN CAST(l.entry ->> 'target_id' AS bigint) END AS agent_id,
               l.created_at AT TIME ZONE :tz AS changed_at
        FROM harness_task_ledger l
        WHERE l.workspace_id = CAST(:workspace_id AS uuid)
          AND l.state = :applied
          AND (l.created_at AT TIME ZONE :tz) >= :start
          AND (l.created_at AT TIME ZONE :tz) < :end
    )
    SELECT agent_id, MIN(changed_at) AS changed_at
    FROM changes
    WHERE agent_id IS NOT NULL
    GROUP BY agent_id
""")


def applied_change_weeks(db: Session, workspace_id: Union[UUID, str], window: Window) -> Dict[int, str]:
    """agent id -> the ISO week of the first change HARNESS applied to it inside the
    weeks read (the ledger's ``applied`` rows: the agent from ``entry.target_id``, the
    moment from ``created_at``, until US-051 adds ``applied_at``)."""
    from core.models.harness import LEDGER_APPLIED

    params = {
        "workspace_id": str(UUID(str(workspace_id))), "applied": LEDGER_APPLIED, "tz": UTC_ZONE,
        "agent_id_pattern": AGENT_ID_PATTERN, "start": window.start, "end": window.end,
    }
    return {row.agent_id: iso_week_label(row.changed_at.date()) for row in db.execute(_APPLIED_CHANGES_SQL, params)}


def agent_names(metrics: HarnessMetrics) -> Dict[str, str]:
    """agent id (as a string) -> name, for the roster COLLECT read."""
    return {str(row["agent_id"]): row["agent_name"] for row in metrics.agent_rows}


# ------------------------------------------------------------------ judging


def _rows_by_agent(agent_rows: Iterable[Mapping[str, Any]]) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    """agent id -> ISO week -> that agent-week's row."""
    weeks: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for row in agent_rows:
        weeks.setdefault(row["agent_id"], {})[row["iso_week"]] = row
    return weeks


def _own_history(history: History, agent_id: int, metric: str) -> Dict[str, str]:
    """ISO week -> verdict, for one agent and metric."""
    key = str(agent_id)
    return {week: agents[key][metric][VERDICT_KEY] for week, agents in history.items()
            if metric in agents.get(key, {})}


def _clean_reference(window: Window, own: Mapping[str, str], changed: Optional[str]) -> Tuple[str, ...]:
    """The reference weeks less those already flagged and those from a change onwards.
    ISO-week labels are zero-padded, so they compare in calendar order."""
    return tuple(
        week for week in window.ref_weeks
        if own.get(week) not in _FLAGGED_VALUES and (changed is None or week < changed)
    )


def _rate_counts(row: Mapping[str, Any], metric: RateMetric) -> Tuple[int, int]:
    return int(row[metric.x_field]), sum(int(row[field]) for field in metric.n_fields)


def _judge_rate(weeks: Mapping[str, Mapping[str, Any]], agent_id: int, metric: RateMetric,
                cur_week: str, ref: Tuple[str, ...], settings: DiagnoseSettings) -> _Test:
    x_cur, n_cur = _rate_counts(weeks[cur_week], metric)
    ref_counts = [_rate_counts(weeks[week], metric) for week in ref]
    x_ref, n_ref = sum(x for x, _ in ref_counts), sum(n for _, n in ref_counts)
    result = hv.proportion_flag(x_cur, n_cur, x_ref, n_ref, settings.rates, higher_is_worse=metric.higher_is_worse)
    return _Test(agent_id, metric, UNIT_RATE, ref, result, x_cur=x_cur, x_ref=x_ref)


def _judge_amount(weeks: Mapping[str, Mapping[str, Any]], agent_id: int, metric: AmountMetric,
                  cur_week: str, ref: Tuple[str, ...], settings: DiagnoseSettings) -> _Test:
    rows = [weeks[week] for week in (cur_week, *ref)]
    billed = any(row[metric.usd_field] for row in rows)
    field = metric.usd_field if billed else metric.tokens_field
    cur = [float(value) for value in rows[0][field]]
    ref_values = [float(value) for row in rows[1:] for value in row[field]]
    result = hv.median_rise_flag(cur, ref_values, settings.amounts)
    return _Test(agent_id, metric, UNIT_USD if billed else UNIT_TOKENS, ref, result)


def _judge(weeks: Mapping[str, Mapping[str, Any]], agent_id: int, metric: Metric, window: Window,
           history: History, change_weeks: Mapping[int, str], settings: DiagnoseSettings) -> _Test:
    ref = _clean_reference(window, _own_history(history, agent_id, metric.name), change_weeks.get(agent_id))
    judge = _judge_rate if isinstance(metric, RateMetric) else _judge_amount
    return judge(weeks, agent_id, metric, window.cur_week, ref, settings)


def _reason(verdict: hv.Verdict, cur_week: str, own: Mapping[str, str]) -> Optional[str]:
    if cur_week in own:
        return REASON_KEPT
    if verdict is not hv.Verdict.REGRESSION:
        return None
    return REASON_SECOND_WEEK if own.get(hv.previous_iso_week(cur_week)) in _FLAGGED_VALUES else REASON_STRONG_P


def _metric_entry(test: _Test, result: hv.FlagResult, verdict: hv.Verdict, reason: Optional[str]) -> Dict[str, Any]:
    return {
        VERDICT_KEY: verdict.value, "reason": reason, "unit": test.unit,
        "n_cur": result.n_cur, "n_ref": result.n_ref, "x_cur": test.x_cur, "x_ref": test.x_ref,
        "cur_value": result.cur_value, "ref_value": result.ref_value, "p_value": result.p_value,
        "ref_weeks": list(test.ref_weeks),
    }


def _kept_entry(stored: Mapping[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """A week already judged keeps its first run's figures with its verdict, so the
    report never prints numbers that no longer back it (a late ticket, a backfilled
    llm_usage row). ``fresh`` carries the kept verdict already; its figures stand
    only when the stored entry lacks some."""
    if not all(key in stored for key in FIGURE_KEYS) or stored["unit"] not in _UNITS:
        return fresh
    return {**fresh, **{key: stored[key] for key in FIGURE_KEYS}}


def _agent_entry(row: Mapping[str, Any], metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """One agent's verdict (its worst metric's) with the metrics; ``row`` is its judged week."""
    verdicts = [hv.Verdict(entry[VERDICT_KEY]) for entry in metrics.values()]
    worst = max(verdicts, key=_VERDICT_RANK.__getitem__, default=hv.Verdict.INSUFFICIENT_DATA)
    return {
        "agent_name": row["agent_name"],
        "model_id": row["model_id"],
        VERDICT_KEY: worst.value,
        "heartbeat_enabled": row["heartbeat_enabled"],
        "heartbeat_interval_minutes": row["heartbeat_interval_minutes"],
        "heartbeat_cost_usd": row["heartbeat_llm_billed_cost_usd"],
        METRICS_KEY: metrics,
    }


def _convergence(coverage: Optional[float], tally: Mapping[str, int], min_coverage: float) -> str:
    if coverage is None or coverage < min_coverage:
        return CONVERGENCE_INSUFFICIENT
    if tally[hv.Verdict.REGRESSION.value]:
        return CONVERGENCE_REGRESSION
    if tally[hv.Verdict.WATCH.value]:
        return CONVERGENCE_WATCHING
    return CONVERGENCE_STEADY


def _summary(window: Window, agents: Dict[str, Dict[str, Any]], settings: DiagnoseSettings) -> Dict[str, Any]:
    tally = Counter(entry[VERDICT_KEY] for entry in agents.values())
    active = len(agents)
    judged = active - tally[hv.Verdict.INSUFFICIENT_DATA.value]
    coverage = judged / active if active else None
    counts = {"active_agents": active, "judged_agents": judged,
              **{key: tally[verdict.value] for verdict, key in COUNT_KEYS.items()}}
    return {
        "window": window.cur_week,
        "ref_weeks": list(window.ref_weeks),
        "min_samples": settings.rates.min_samples,
        "coverage": coverage,
        CONVERGENCE_KEY: _convergence(coverage, tally, settings.min_coverage),
        "counts": counts,
        AGENTS_KEY: agents,
    }


def diagnose(metrics: HarnessMetrics, history: History, change_weeks: Mapping[int, str],
             settings: DiagnoseSettings) -> Dict[str, Any]:
    """Judge every agent x metric of one run and summarise it (ORG-4, ORG-5).

    ``history`` is :func:`verdict_history` for the weeks read, ``change_weeks``
    :func:`applied_change_weeks`. Returns the judged week, coverage, convergence,
    the verdict counts and, per agent (keyed by id as a string), its verdict and
    each metric's verdict with its n, values and p-value.
    """
    window = metrics.window
    weeks = _rows_by_agent(metrics.agent_rows)
    tests = [_judge(weeks[agent_id], agent_id, metric, window, history, change_weeks, settings)
             for agent_id in weeks for metric in METRICS]
    results = hv.fdr_filter([test.result for test in tests], settings.fdr_q)
    judged: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for test, result in zip(tests, results):
        own = _own_history(history, test.agent_id, test.metric.name)
        verdict = hv.window_verdict(window.cur_week, result, own, settings.strong_p)
        entry = _metric_entry(test, result, verdict, _reason(verdict, window.cur_week, own))
        if entry["reason"] == REASON_KEPT:
            entry = _kept_entry(history[window.cur_week][str(test.agent_id)][test.metric.name], entry)
        judged.setdefault(test.agent_id, {})[test.metric.name] = entry
    agents = {str(agent_id): _agent_entry(weeks[agent_id][window.cur_week], judged[agent_id]) for agent_id in weeks}
    return _summary(window, agents, settings)


# ------------------------------------------------------------------ tool failures


def _tool_counts(metrics: HarnessMetrics) -> Dict[Tuple[int, str], Tuple[int, int, int]]:
    """(agent id, app) -> (calls, failed calls, failed on the app's sign-in or connection),
    from the judged week's production tool runs."""
    counts: Dict[Tuple[int, str], List[int]] = {}
    for run in metrics.tool_rows:
        if run["iso_week"] != metrics.window.cur_week:
            continue
        tally = counts.setdefault((run["agent_id"], run["app_name"]), [0, 0, 0])
        failed = run["status"] != TOOL_SUCCESS
        tally[0] += run["calls"]
        tally[1] += run["calls"] if failed else 0
        tally[2] += run["calls"] if failed and run["error_class"] in RECONNECT_CLASSES else 0
    return {key: (n, bad, lost) for key, (n, bad, lost) in counts.items()}


def _fails(failed: int, calls: int, settings: DiagnoseSettings) -> bool:
    """FR-8: enough calls, and a failure rate surely at least HARNESS_TOOL_FAIL_WILSON_MIN."""
    return (calls >= settings.rates.min_samples
            and hv.wilson_lower_bound(failed, calls) >= settings.tool_fail_wilson_min)


def _tool_issue(name: str, agent_id: int, app: str, counts: Tuple[int, int, int],
                settings: DiagnoseSettings) -> Dict[str, Any]:
    """One flagged agent-app pair. It is "reconnect <app>" when the sign-in and connection
    failures carry the flag (the other failures alone would not flag) and the app has a
    connection; otherwise a tool failure. Neither proposes removing the tool (US-049, US-057)."""
    calls, failed, reconnect = counts
    reconnects = app not in PLATFORM_APPS and reconnect > 0 and not _fails(failed - reconnect, calls, settings)
    if reconnects:
        detail = f"reconnect {app}: {reconnect} of its {calls} calls this week failed on {app}'s sign-in or connection"
    else:
        bound = hv.wilson_lower_bound(failed, calls)
        detail = f"{app} failed {failed} of {calls} calls this week (a failure rate of at least {bound:.0%})"
    return {
        "agent_id": str(agent_id), "agent_name": name, "root_cause": RECONNECT_APP if reconnects else TOOL_FAILURE,
        "detail": detail, "app_name": app, "calls": calls, "failed_calls": failed, "reconnect_calls": reconnect,
    }


def tool_failure_issues(metrics: HarnessMetrics, settings: DiagnoseSettings) -> List[Dict[str, Any]]:
    """The judged week's tool issues (US-019, FR-8): one per agent and app whose production
    failure rate flags, as a tool failure or as "reconnect <app>". The window is the judged
    ISO week, so an old failure never flags again; asked and refused rows are not in it."""
    names = agent_names(metrics)
    return [_tool_issue(names.get(str(agent_id), "unknown"), agent_id, app, counts, settings)
            for (agent_id, app), counts in sorted(_tool_counts(metrics).items())
            if _fails(counts[1], counts[0], settings)]


def stored_verdicts(diagnosis: Mapping[str, Any]) -> Dict[str, Any]:
    """What a run keeps in ``harness_runs.verdicts``: its convergence and its agents."""
    return {CONVERGENCE_KEY: diagnosis[CONVERGENCE_KEY], AGENTS_KEY: diagnosis[AGENTS_KEY]}


def needs_attention(diagnosis: Mapping[str, Any]) -> bool:
    """Whether any agent is WATCH or REGRESSION."""
    return diagnosis.get(CONVERGENCE_KEY) in (CONVERGENCE_REGRESSION, CONVERGENCE_WATCHING)


def status_fields(coverage: Optional[float], counts: Optional[Mapping[str, Any]],
                  verdicts: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Coverage, convergence and the verdict counts of a completed run, for
    platform_harness_status. A figure the run did not record is None, never 0."""
    counts = counts if isinstance(counts, Mapping) else {}
    verdicts = verdicts if isinstance(verdicts, Mapping) else {}
    return {"coverage": coverage, CONVERGENCE_KEY: verdicts.get(CONVERGENCE_KEY),
            **{key: counts.get(key) for key in STATUS_COUNT_KEYS}}


# ------------------------------------------------------------------ report


def _number(value: float, unit: str) -> str:
    if unit == UNIT_RATE:
        return f"{value:.0%}"
    return f"${value:.4f}" if unit == UNIT_USD else f"{value:,.0f}"


def _label(metric: Metric, unit: str) -> str:
    return metric.tokens_label if unit == UNIT_TOKENS else metric.label


def _why(judged: Mapping[str, Any]) -> str:
    reason = judged.get("reason")
    if reason == REASON_SECOND_WEEK:
        return ", the second week in a row"
    if reason == REASON_STRONG_P:
        return ", strong enough on its own"
    return " (this week's verdict from its first run)" if reason == REASON_KEPT else ""


def _finding(metric: Metric, judged: Mapping[str, Any]) -> str:
    """One flagged metric, e.g. "Task success fell from 80% (64 of 80) to 60% (12 of 20), p=0.031"."""
    unit = judged["unit"]
    if judged["cur_value"] is None:  # a kept verdict stored without figures, its week now short of data
        return f"{_label(metric, unit)}{_why(judged)}"
    worse = "rose" if unit != UNIT_RATE or metric.higher_is_worse else "fell"
    before, after = _number(judged["ref_value"], unit), _number(judged["cur_value"], unit)
    if unit == UNIT_RATE:
        before += f" ({judged['x_ref']} of {judged['n_ref']})"
        after += f" ({judged['x_cur']} of {judged['n_cur']})"
    else:
        after += f" (median of {judged['n_cur']} {metric.noun} against {judged['n_ref']})"
    return f"{_label(metric, unit)} {worse} from {before} to {after}, p={judged['p_value']:.2g}{_why(judged)}"


def _sample_sizes(metrics: Iterable[Tuple[Metric, Mapping[str, Any]]]) -> str:
    """Each metric's n this week and in the reference, for metrics with any data."""
    parts = [f"{judged['n_cur']} {metric.noun} this week, {judged['n_ref']} in the reference"
             for metric, judged in metrics if judged["n_cur"] or judged["n_ref"]]
    return "; ".join(parts) or "nothing recorded"


def _agent_line(entry: Mapping[str, Any], min_samples: int) -> str:
    verdict = hv.Verdict(entry[VERDICT_KEY])
    model = f" ({entry['model_id']})" if entry.get("model_id") else ""
    head = f"- **{entry['agent_name']}**{model}: {VERDICT_WORDS[verdict]}"
    metrics = [(METRIC_BY_NAME[name], judged) for name, judged in entry[METRICS_KEY].items()
               if name in METRIC_BY_NAME]
    if verdict is hv.Verdict.INSUFFICIENT_DATA:
        return f"{head} ({_sample_sizes(metrics)}; each needs {min_samples} in the week and in the reference)."
    flagged = [_finding(metric, judged) for metric, judged in metrics if judged[VERDICT_KEY] in _FLAGGED_VALUES]
    steady = [f"{_label(metric, judged['unit'])} (n={judged['n_cur']}, reference {judged['n_ref']})"
              for metric, judged in metrics if judged[VERDICT_KEY] == hv.Verdict.STABLE.value]
    if steady:
        flagged.append(f"{'Steady on' if flagged else 'Judged on'} {', '.join(steady)}")
    return f"{head}. {'. '.join(flagged)}."


def _coverage_line(diagnosis: Mapping[str, Any]) -> str:
    counts, coverage = diagnosis["counts"], diagnosis["coverage"]
    if coverage is None:
        return "**Coverage:** HARNESS had no active agents to judge (Auto is not judged)."
    return (f"**Coverage:** HARNESS judged {counts['judged_agents']} of {counts['active_agents']} "
            f"agents ({coverage:.0%}).")


def report_lines(diagnosis: Mapping[str, Any]) -> List[str]:
    """The audit report's verdict section: coverage first, then every agent with its n.
    It never calls an agent it did not judge stable, and never says converged."""
    counts = diagnosis["counts"]
    refs = diagnosis["ref_weeks"]
    reference = f"{refs[0]} to {refs[-1]}" if refs else "none"
    agents = sorted(diagnosis[AGENTS_KEY].values(), key=lambda entry: str(entry["agent_name"]).lower())
    return [
        f"**Week judged:** {diagnosis['window']} (reference weeks {reference}, less flagged and "
        "post-change weeks)",
        _coverage_line(diagnosis),
        f"**Convergence:** {diagnosis[CONVERGENCE_KEY]}",
        f"**Verdicts:** {counts['insufficient_data']} not enough data, {counts['watch']} watching, "
        f"{counts['regression']} regression, {counts['stable']} steady",
        "",
        "## Agents",
        *[_agent_line(entry, diagnosis["min_samples"]) for entry in agents],
    ]


__all__ = [
    "DIAGNOSE_READS",
    "DiagnoseSettings",
    "METRICS",
    "RECONNECT_APP",
    "TOOL_FAILURE",
    "agent_names",
    "applied_change_weeks",
    "diagnose",
    "needs_attention",
    "reference_weeks",
    "report_lines",
    "settings_from_config",
    "status_fields",
    "stored_verdicts",
    "tool_failure_issues",
    "verdict_history",
]
