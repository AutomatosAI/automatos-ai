"""PRD-263 US-013: HARNESS's per-agent read model, raw weekly counts from SQL.

DIAGNOSE used to judge four fields ``platform_list_agents`` never returns, so every
agent came out STABLE. This module reads what each agent really did, straight from
the tables that record it, for the ISO week being judged (``cur``) and for each of
the ``ref_weeks`` ISO weeks before it. Every week is its own bucket, so DIAGNOSE can
leave out weeks already flagged for an agent and metric (the clean reference, design
section 5.1 ORG-4) without another read.

Output is raw counts only. A roster agent with no rows in a week still gets that
week's row, with zeros: n=0, never 0%. No rate, mean or percentage is computed here.

The data traps, one per source (design section 4.1 #1):

* ``board_tasks``: done and failed tickets, leaving out heartbeat tickets,
  mission-linked cards and HARNESS's own cards (``tags`` holds ``harness``).
* ``llm_usage``: a per-metric whitelist of ``request_type`` and ``status``. Work is
  ticket, session and Playbook runs; heartbeat is the heartbeat lane. So
  ``reserved`` and ``pending`` (media ledger holds, not model calls) and helpers
  such as ``memory_distill`` and ``complexity_assessor`` never count. Plan-billed
  rows (``tier='subscription'``, booked at $0 for a Claude Code session) are counted
  apart, on tokens. A Claude Code ticket books its session in the ``board_task``
  lane whatever started it, so a row whose ``execution_id`` names a ticket follows
  that ticket: a heartbeat ticket's spend is heartbeat spend, and a mission or
  HARNESS card's spend is left out, like the card itself.
* ``heartbeat_results``: ``source_type='agent'`` only. An orchestrator row keeps the
  workspace UUID in ``source_id`` (heartbeat_service.py, ``emit_primitive_finding``),
  and Postgres does not promise predicate order, so the cast to an agent id sits
  inside a ``CASE``. ``objective_met`` NULL (a silent success) is its own count,
  outside the met/missed denominator.
* ``orchestration_tasks``: no ``workspace_id`` of its own; scoped through
  ``orchestration_runs.workspace_id``.
* ``tool_execution_logs``: ``telemetry_source='production'`` and status success,
  error or timeout (so ``asked``, ``gap``, ``shown`` and ``refused`` rows never
  count), grouped per app. Failures are grouped by ``error_code`` and the start of
  ``error_message``, then classed by :func:`classify_tool_error` (PRD-263 US-019).

Six statements on one session, not one: each source has its own grain (agent-week,
agent-week-run, agent-week-app-class) and its own trap filters, and a fixed set
keeps each filter readable and separately tested. The count never depends on the
roster size (no per-agent loop hits the database).

Read only: no write, no platform action, no model call.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from operator import itemgetter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.orchestration_enums import TaskState

logger = logging.getLogger(__name__)

# The reference period in ISO weeks: design section 5.1 ORG-4, the 28 days before cur.
# PR-1D passes HARNESS_REF_DAYS // 7 from config.
REF_WEEKS = 4
# How the heartbeat service reads an agent's interval when its configuration has none.
DEFAULT_HEARTBEAT_INTERVAL_MINUTES = 60

# One agent-week row. DIAGNOSE reads rows only through these names (US-014, US-015).
AGENT_ROW_FIELDS: Tuple[str, ...] = (
    "agent_id", "agent_name", "model_id", "heartbeat_enabled", "heartbeat_interval_minutes", "iso_week",
    "tasks_done", "tasks_failed",
    "mission_tasks_done", "mission_tasks_failed",
    "heartbeat_runs", "heartbeat_objective_met", "heartbeat_objective_missed",
    "heartbeat_objective_unknown", "heartbeat_tokens",
    "llm_calls", "llm_billed_cost_usd", "llm_billed_tokens", "llm_plan_calls", "llm_plan_tokens",
    "llm_unattributed_calls", "llm_run_costs_usd", "llm_run_tokens",
    "heartbeat_llm_calls", "heartbeat_llm_billed_cost_usd", "heartbeat_llm_billed_tokens",
    "heartbeat_llm_plan_calls", "heartbeat_llm_plan_tokens", "heartbeat_llm_unattributed_calls",
    "heartbeat_llm_run_costs_usd", "heartbeat_llm_run_tokens",
    "tool_calls", "tool_failed_calls",
)
# One agent-week-app-status-class row of production tool runs.
TOOL_ROW_FIELDS: Tuple[str, ...] = ("agent_id", "iso_week", "app_name", "status", "error_class", "calls")

_IDENTITY_FIELDS = frozenset({
    "agent_id", "agent_name", "model_id", "heartbeat_enabled", "heartbeat_interval_minutes", "iso_week",
})
_COST_FIELDS = frozenset({"llm_billed_cost_usd", "heartbeat_llm_billed_cost_usd"})
# Per-run samples for the Mann-Whitney test (ORG-4), one value per execution_id, each
# tuple sorted ascending on its own: the test ranks one metric at a time. Token samples
# hold every run; cost samples only runs with a billed call, because a plan-billed run
# books $0 whatever it used (ORG-4 judges agents with no billed $ on tokens).
_RUN_SAMPLE_FIELDS = frozenset({
    "llm_run_costs_usd", "llm_run_tokens", "heartbeat_llm_run_costs_usd", "heartbeat_llm_run_tokens",
})

ACTIVE_AGENT_STATUS = "active"
TICKET_DONE = "done"
TICKET_FAILED = "failed"
HEARTBEAT_TICKET_SOURCE = "heartbeat"
# A mission's own card, a mission step's card, and a session-lane mission card.
MISSION_TICKET_SOURCES = ("orchestration", "orchestration_task", "mission")
HARNESS_TAG = "harness"
HEARTBEAT_AGENT_SOURCE = "agent"
# Ids are Postgres integers, but the SQL casts the matched digits to bigint: 18 digits
# always fit one (its maximum has 19), so no id-shaped value can overflow the cast, and
# every integer id up to 2,147,483,647 matches.
# An agent id in heartbeat_results.source_id.
AGENT_ID_PATTERN = "^[0-9]{1,18}$"
# An llm_usage execution_id that names a ticket (``board_task:97``).
TICKET_REF_PATTERN = "^board_task:[0-9]{1,18}$"
TICKET_REF_SEPARATOR = ":"
TICKET_REF_PART = 2

# llm_usage vocabulary. Kept literal so this read model never loads the LLM client
# chain (core/llm/__init__ imports the manager); a test pins them to the originals
# in core.llm.usage_context and core.llm.usage_tracker.
USAGE_WORK = "llm"
USAGE_HEARTBEAT = "heartbeat_llm"
# Not 'mission': a mission books its spend per run (``mission:<run_id>``), across its
# steps and agents, so it is not one agent's run. A mission is judged on its steps
# (mission_tasks_done / mission_tasks_failed), as mission-linked tickets are left out.
WORK_REQUEST_TYPES = ("board_task", "session", "recipe")
HEARTBEAT_REQUEST_TYPES = ("heartbeat",)
# Statuses of a model call that happened. Not 'reserved' or 'pending': ledger holds.
WORK_STATUSES = ("success", "error", "timeout", "rate_limited", "cancelled")
HEARTBEAT_STATUSES = WORK_STATUSES
PLAN_TIER = "subscription"
# The _USAGE_SQL columns summed into "<metric>_<column>" row fields.
_USAGE_TOTAL_COLUMNS = ("calls", "plan_calls", "billed_cost_usd", "billed_tokens", "plan_tokens")

PRODUCTION_TELEMETRY = "production"
TOOL_SUCCESS = "success"
TOOL_STATUSES = (TOOL_SUCCESS, "error", "timeout")
# A tool row's error class: ``none`` for a successful run, else classify_tool_error's.
ERROR_CLASS_NONE = "none"
ERROR_CLASS_AUTH = "auth"
ERROR_CLASS_CONNECTION = "connection"
ERROR_CLASS_RATE_LIMIT = "rate_limit"
ERROR_CLASS_OTHER = "other"
# The classes a "reconnect <app>" line answers (US-019): the app's sign-in or its connection.
RECONNECT_CLASSES = frozenset({ERROR_CLASS_AUTH, ERROR_CLASS_CONNECTION})
# How much of a failure's error_message is grouped and classed, lower-cased and trimmed.
ERROR_MESSAGE_PREFIX_CHARS = 200

# error_code -> class. A code is classed here only where the code that writes it is known.
RECONNECT_CODES: Dict[str, str] = {
    # modules/tools/tool_router.py:131 (the constant), written at :202 and :1875: a
    # Composio call refused because the server has no Composio connection.
    "integrations_unavailable": ERROR_CLASS_CONNECTION,
}
# (pattern, class); the first match classes a failure, so the server's own exact words come
# before the generic ones. Each reads the code and the message together, so a row with no
# code (every row written before US-018) still classes.
RECONNECT_PATTERNS: Tuple[Tuple[re.Pattern, str], ...] = (
    # Integrations are off on the server (modules/tools/tool_router.py:197), for rows
    # written before US-018 stored the integrations_unavailable code.
    (re.compile(r"integrations are disabled on this server", re.IGNORECASE), ERROR_CLASS_CONNECTION),
    # The app has no connection: core/composio/tool_executor.py:537 and :769, and Composio's
    # ConnectedAccountNotFound (its words as api/workspace_github.py:72-74 reads them).
    (re.compile(r"is not connected for this workspace|no composio entity id found|connectedaccountnotfound"
                r"|no connected account found|no active connection exists", re.IGNORECASE), ERROR_CLASS_CONNECTION),
    # A platform's rate limit, passed through from Composio's execute
    # (core/composio/tool_executor.py:845; an exception's text at :865). The same words
    # modules/socials/publish_steps.py:65 reads as "the platform did nothing".
    (re.compile(r"\b429\b|too many requests|rate.?limit", re.IGNORECASE), ERROR_CLASS_RATE_LIMIT),
    # A platform refusing the connection's credentials, passed through at
    # core/composio/tool_executor.py:845 and :865; modules/socials/publish_steps.py:68
    # reads the same words as an auth refusal.
    (re.compile(r"\b40[13]\b|unauthori[sz]ed|forbidden|authenticat|credential|invalid.?token|expired.?token",
                re.IGNORECASE), ERROR_CLASS_AUTH),
    # An expired token or connection: core/composio/linkedin_image_workaround.py:228
    # ("access token may be expired"), core/credentials/tester.py:728 ("invalid or
    # expired"), a Composio connection in state EXPIRED (core/composio/pending_connections.py:30).
    (re.compile(r"(?:token|auth\w*|connected.?account|connection)[^.]{0,60}expired\b", re.IGNORECASE),
     ERROR_CLASS_AUTH),
    # The network refused or dropped the call (an exception's text, core/composio/
    # tool_executor.py:865); the words modules/socials/publish_steps.py:58 and :65 read.
    (re.compile(r"connection (?:error|reset|refused|aborted)|connecterror", re.IGNORECASE),
     ERROR_CLASS_CONNECTION),
)

UTC_ZONE = "UTC"
_ISO_WEEK = re.compile(r"^(\d{4})-W(\d{2})$")

AgentWeek = Tuple[int, str]

_ROSTER_SQL = text("""
    SELECT a.id AS agent_id, a.name AS agent_name,
           a.model_config ->> 'model_id' AS model_id,
           a.configuration -> 'heartbeat' AS heartbeat
    FROM agents a
    WHERE a.workspace_id = CAST(:workspace_id AS uuid)
      AND a.status = :active_status
      AND a.is_system_agent IS NOT TRUE
    ORDER BY a.id
""")

_TICKETS_SQL = text("""
    SELECT bt.assigned_agent_id AS agent_id,
           date_trunc('week', COALESCE(bt.completed_at, bt.updated_at) AT TIME ZONE :tz) AS week_start,
           COUNT(*) FILTER (WHERE bt.status = :done) AS tasks_done,
           COUNT(*) FILTER (WHERE bt.status = :failed) AS tasks_failed
    FROM board_tasks bt
    WHERE bt.workspace_id = CAST(:workspace_id AS uuid)
      AND bt.assigned_agent_id = ANY(:agent_ids)
      AND bt.status IN (:done, :failed)
      AND bt.source_type <> :heartbeat_source
      AND bt.source_type <> ALL(:mission_sources)
      AND bt.orchestration_run_id IS NULL
      AND bt.orchestration_task_id IS NULL
      AND NOT (COALESCE(bt.tags, CAST('[]' AS jsonb)) @> CAST(:harness_tags AS jsonb))
      AND (COALESCE(bt.completed_at, bt.updated_at) AT TIME ZONE :tz) >= :start
      AND (COALESCE(bt.completed_at, bt.updated_at) AT TIME ZONE :tz) < :end
    GROUP BY 1, 2
""")

_MISSION_SQL = text("""
    SELECT t.assigned_agent_id AS agent_id,
           date_trunc('week', COALESCE(t.completed_at, t.updated_at) AT TIME ZONE :tz) AS week_start,
           COUNT(*) FILTER (WHERE t.state = :verified) AS mission_tasks_done,
           COUNT(*) FILTER (WHERE t.state = :failed) AS mission_tasks_failed
    FROM orchestration_tasks t
    JOIN orchestration_runs r ON r.id = t.run_id
    WHERE r.workspace_id = CAST(:workspace_id AS uuid)
      AND t.assigned_agent_id = ANY(:agent_ids)
      AND t.state IN (:verified, :failed)
      AND (COALESCE(t.completed_at, t.updated_at) AT TIME ZONE :tz) >= :start
      AND (COALESCE(t.completed_at, t.updated_at) AT TIME ZONE :tz) < :end
    GROUP BY 1, 2
""")

_HEARTBEAT_SQL = text("""
    WITH agent_rows AS (
        SELECT CASE WHEN hr.source_id ~ :agent_id_pattern
                    THEN CAST(hr.source_id AS bigint) END AS agent_id,
               hr.objective_met, hr.tokens_used, hr.created_at
        FROM heartbeat_results hr
        WHERE hr.workspace_id = CAST(:workspace_id AS uuid)
          AND hr.source_type = :agent_source
          AND hr.created_at >= :start AND hr.created_at < :end
    )
    SELECT agent_id, date_trunc('week', created_at) AS week_start,
           COUNT(*) AS heartbeat_runs,
           COUNT(*) FILTER (WHERE objective_met IS TRUE) AS heartbeat_objective_met,
           COUNT(*) FILTER (WHERE objective_met IS FALSE) AS heartbeat_objective_missed,
           COUNT(*) FILTER (WHERE objective_met IS NULL) AS heartbeat_objective_unknown,
           COALESCE(SUM(tokens_used), 0) AS heartbeat_tokens
    FROM agent_rows
    WHERE agent_id = ANY(:agent_ids)
    GROUP BY 1, 2
""")

# The ticket exclusions here must match _TICKETS_SQL (the test pins both).
_USAGE_SQL = text("""
    WITH usage_rows AS (
        SELECT lu.agent_id, lu.created_at, lu.request_type, lu.status, lu.tier,
               lu.execution_id, lu.total_cost, lu.total_tokens,
               CASE WHEN lu.execution_id ~ :ticket_ref_pattern
                    THEN CAST(split_part(lu.execution_id, :ref_separator, :ref_part) AS bigint)
               END AS ticket_id
        FROM llm_usage lu
        WHERE lu.workspace_id = CAST(:workspace_id AS uuid)
          AND lu.agent_id = ANY(:agent_ids)
          AND lu.created_at >= :start AND lu.created_at < :end
    ), tagged AS (
        SELECT u.*, CASE
            WHEN bt.source_type = :heartbeat_source THEN CASE
                 WHEN (u.request_type = ANY(:work_types) OR u.request_type = ANY(:heartbeat_types))
                      AND u.status = ANY(:heartbeat_statuses) THEN :heartbeat_metric END
            WHEN bt.id IS NOT NULL AND (bt.source_type = ANY(:mission_sources)
                 OR bt.orchestration_run_id IS NOT NULL OR bt.orchestration_task_id IS NOT NULL
                 OR COALESCE(bt.tags, CAST('[]' AS jsonb)) @> CAST(:harness_tags AS jsonb)) THEN NULL
            WHEN u.request_type = ANY(:work_types) AND u.status = ANY(:work_statuses) THEN :work_metric
            WHEN u.request_type = ANY(:heartbeat_types)
                 AND u.status = ANY(:heartbeat_statuses) THEN :heartbeat_metric
        END AS metric
        FROM usage_rows u
        LEFT JOIN board_tasks bt
               ON bt.id = u.ticket_id AND bt.workspace_id = CAST(:workspace_id AS uuid)
    )
    SELECT agent_id, date_trunc('week', created_at) AS week_start, metric, execution_id,
           COUNT(*) AS calls,
           COUNT(*) FILTER (WHERE tier = :plan_tier) AS plan_calls,
           COALESCE(SUM(total_cost) FILTER (WHERE tier IS DISTINCT FROM :plan_tier), 0) AS billed_cost_usd,
           COALESCE(SUM(total_tokens) FILTER (WHERE tier IS DISTINCT FROM :plan_tier), 0) AS billed_tokens,
           COALESCE(SUM(total_tokens) FILTER (WHERE tier = :plan_tier), 0) AS plan_tokens
    FROM tagged
    WHERE metric IS NOT NULL
    GROUP BY 1, 2, 3, 4
""")

_TOOLS_SQL = text("""
    SELECT tl.agent_id, date_trunc('week', tl.executed_at) AS week_start, tl.app_name, tl.status,
           CASE WHEN tl.status <> :success THEN tl.error_code END AS error_code,
           CASE WHEN tl.status <> :success
                THEN lower(btrim(left(tl.error_message, :message_chars))) END AS error_said,
           COUNT(*) AS calls
    FROM tool_execution_logs tl
    WHERE tl.workspace_id = CAST(:workspace_id AS uuid)
      AND tl.agent_id = ANY(:agent_ids)
      AND tl.telemetry_source = :production
      AND tl.status = ANY(:tool_statuses)
      AND tl.executed_at >= :start AND tl.executed_at < :end
    GROUP BY 1, 2, 3, 4, 5, 6
""")


@dataclass(frozen=True)
class Window:
    """The judged ISO week and the reference weeks before it, as naive-UTC bounds."""

    cur_week: str
    ref_weeks: Tuple[str, ...]  # oldest first
    start: datetime             # Monday 00:00 UTC of the oldest week
    end: datetime               # Monday 00:00 UTC after the judged week

    @property
    def weeks(self) -> Tuple[str, ...]:
        """Every week read, oldest first, the judged week last."""
        return self.ref_weeks + (self.cur_week,)


@dataclass(frozen=True)
class HarnessMetrics:
    """One workspace's raw counts: a row per roster agent per week, and its tool runs."""

    workspace_id: str
    window: Window
    agent_rows: Tuple[Dict[str, Any], ...]  # keys: AGENT_ROW_FIELDS
    tool_rows: Tuple[Dict[str, Any], ...]   # keys: TOOL_ROW_FIELDS


def iso_week_label(day: date) -> str:
    """The ISO week a day falls in, as ``harness_runs.cur_window`` writes it (``2026-W42``)."""
    year, week, _ = day.isocalendar()
    return f"{year}-W{week:02d}"


def build_window(cur_week: str, ref_weeks: int = REF_WEEKS) -> Window:
    """The window for ``cur_week`` (``YYYY-Www``) and the ``ref_weeks`` weeks before it.

    Raises ValueError on a malformed label or a negative ``ref_weeks``: the caller's
    input is checked here, before any SQL runs.
    """
    match = _ISO_WEEK.match(cur_week or "")
    if match is None:
        raise ValueError(f"cur_week must look like 2026-W42, got {cur_week!r}")
    if ref_weeks < 0:
        raise ValueError(f"ref_weeks must be 0 or more, got {ref_weeks}")
    monday = date.fromisocalendar(int(match.group(1)), int(match.group(2)), 1)
    oldest = monday - timedelta(weeks=ref_weeks)
    refs = tuple(iso_week_label(oldest + timedelta(weeks=i)) for i in range(ref_weeks))
    return Window(
        cur_week=iso_week_label(monday),
        ref_weeks=refs,
        start=datetime.combine(oldest, datetime.min.time()),
        end=datetime.combine(monday + timedelta(weeks=1), datetime.min.time()),
    )


def _week_of(week_start: Any) -> str:
    """The ISO week of a ``date_trunc('week', ...)`` bucket (always a Monday)."""
    day = week_start.date() if isinstance(week_start, datetime) else week_start
    return iso_week_label(day)


def _interval_minutes(heartbeat: Mapping[str, Any], agent_id: int) -> int:
    """The interval the heartbeat service would use: ``interval_minutes``, else 60."""
    raw = heartbeat.get("interval_minutes", DEFAULT_HEARTBEAT_INTERVAL_MINUTES)
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logger.warning("[HARNESS] agent %s has heartbeat interval %r; reading it as %d",
                       agent_id, raw, DEFAULT_HEARTBEAT_INTERVAL_MINUTES)
        return DEFAULT_HEARTBEAT_INTERVAL_MINUTES


def _heartbeat_config(raw: Any, agent_id: int) -> Dict[str, Any]:
    """``configuration -> 'heartbeat'`` as a dict. psycopg2 decodes JSON itself, so
    anything else here (a string such as ``"on"``, a list) is a setting that is not an
    object: it reads as no heartbeat, as the heartbeat service skips it, and is logged
    rather than raised, so one agent's odd setting never aborts the workspace's read."""
    if raw is None or isinstance(raw, dict):
        return raw or {}
    logger.warning("[HARNESS] agent %s has heartbeat setting %r, not an object; reading it as off",
                   agent_id, raw)
    return {}


def _roster(db: Session, workspace_id: str) -> List[Dict[str, Any]]:
    """Active, non-system agents of this workspace (Auto is the workspace's system agent:
    every candidate ``HarnessService._resolve_auto_agent`` can return is one)."""
    rows = db.execute(_ROSTER_SQL, {"workspace_id": workspace_id, "active_status": ACTIVE_AGENT_STATUS})
    roster = []
    for row in rows:
        heartbeat = _heartbeat_config(row.heartbeat, row.agent_id)
        roster.append({
            "agent_id": row.agent_id,
            "agent_name": row.agent_name,
            "model_id": row.model_id,
            "heartbeat_enabled": bool(heartbeat.get("enabled")),
            "heartbeat_interval_minutes": _interval_minutes(heartbeat, row.agent_id),
        })
    return roster


def _base_params(workspace_id: str, agent_ids: List[int], window: Window) -> Dict[str, Any]:
    """The bound parameters every source statement shares (lists bind as Postgres arrays)."""
    return {
        "workspace_id": workspace_id,
        "agent_ids": list(agent_ids),
        "start": window.start,
        "end": window.end,
        "tz": UTC_ZONE,
        "heartbeat_source": HEARTBEAT_TICKET_SOURCE,
        "mission_sources": list(MISSION_TICKET_SOURCES),
        "harness_tags": json.dumps([HARNESS_TAG]),
    }


def _counts(rows: Iterable[Any], fields: Tuple[str, ...]) -> Dict[AgentWeek, Dict[str, Any]]:
    """Rows keyed by agent and ISO week, keeping only the named count columns."""
    return {(r.agent_id, _week_of(r.week_start)): {f: getattr(r, f) for f in fields} for r in rows}


def _ticket_counts(db: Session, params: Mapping[str, Any]) -> Dict[AgentWeek, Dict[str, Any]]:
    rows = db.execute(_TICKETS_SQL, {**params, "done": TICKET_DONE, "failed": TICKET_FAILED})
    return _counts(rows, ("tasks_done", "tasks_failed"))


def _mission_counts(db: Session, params: Mapping[str, Any]) -> Dict[AgentWeek, Dict[str, Any]]:
    states = {"verified": TaskState.VERIFIED.value, "failed": TaskState.FAILED.value}
    rows = db.execute(_MISSION_SQL, {**params, **states})
    return _counts(rows, ("mission_tasks_done", "mission_tasks_failed"))


def _heartbeat_counts(db: Session, params: Mapping[str, Any]) -> Dict[AgentWeek, Dict[str, Any]]:
    extra = {"agent_id_pattern": AGENT_ID_PATTERN, "agent_source": HEARTBEAT_AGENT_SOURCE}
    rows = db.execute(_HEARTBEAT_SQL, {**params, **extra})
    return _counts(rows, ("heartbeat_runs", "heartbeat_objective_met", "heartbeat_objective_missed",
                          "heartbeat_objective_unknown", "heartbeat_tokens"))


def _usage_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        **params,
        "ticket_ref_pattern": TICKET_REF_PATTERN, "ref_separator": TICKET_REF_SEPARATOR,
        "ref_part": TICKET_REF_PART, "work_metric": USAGE_WORK, "heartbeat_metric": USAGE_HEARTBEAT,
        "work_types": list(WORK_REQUEST_TYPES), "work_statuses": list(WORK_STATUSES),
        "heartbeat_types": list(HEARTBEAT_REQUEST_TYPES), "heartbeat_statuses": list(HEARTBEAT_STATUSES),
        "plan_tier": PLAN_TIER,
    }


def _usage_counts(db: Session, params: Mapping[str, Any]) -> Dict[AgentWeek, Dict[str, Any]]:
    """Per metric: calls, billed and plan spend, and each run's cost and tokens (one run =
    one ``execution_id``; rows without one are counted, not sampled). A run with no
    billed call gives a token sample but no cost sample: its $0 is not a price."""
    totals: Dict[AgentWeek, Dict[str, Any]] = defaultdict(lambda: defaultdict(int))
    samples: Dict[AgentWeek, Dict[str, List[Any]]] = defaultdict(lambda: defaultdict(list))
    for row in db.execute(_USAGE_SQL, _usage_params(params)):
        key, prefix = (row.agent_id, _week_of(row.week_start)), row.metric
        for column in _USAGE_TOTAL_COLUMNS:
            totals[key][f"{prefix}_{column}"] += getattr(row, column)
        if row.execution_id is None:
            totals[key][f"{prefix}_unattributed_calls"] += row.calls
            continue
        samples[key][f"{prefix}_run_tokens"].append(row.billed_tokens + row.plan_tokens)
        if row.calls > row.plan_calls:
            samples[key][f"{prefix}_run_costs_usd"].append(row.billed_cost_usd)
    return {
        key: {**dict(totals.get(key, {})), **{f: tuple(sorted(v)) for f, v in samples.get(key, {}).items()}}
        for key in set(totals) | set(samples)
    }


def classify_tool_error(error_code: Optional[str], error_message: Optional[str]) -> str:
    """A failed tool run's class: ``auth``, ``connection``, ``rate_limit`` or ``other``.

    A code in ``RECONNECT_CODES`` decides; otherwise the first of ``RECONNECT_PATTERNS``
    to match the code and the message read together, so a row written before US-018
    (its ``error_code`` always NULL) classes from its message alone.
    """
    code = (error_code or "").strip().lower()
    if code in RECONNECT_CODES:
        return RECONNECT_CODES[code]
    said = f"{code} {error_message or ''}"
    for pattern, error_class in RECONNECT_PATTERNS:
        if pattern.search(said):
            return error_class
    return ERROR_CLASS_OTHER


def _tool_rows(db: Session, params: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Production tool runs per agent, week, app, status and error class (classed here, in
    Python, from each failure group's code and message start)."""
    extra = {
        "success": TOOL_SUCCESS, "production": PRODUCTION_TELEMETRY, "tool_statuses": list(TOOL_STATUSES),
        "message_chars": ERROR_MESSAGE_PREFIX_CHARS,
    }
    calls: Dict[Tuple[Any, ...], int] = defaultdict(int)
    for r in db.execute(_TOOLS_SQL, {**params, **extra}):
        error_class = ERROR_CLASS_NONE if r.status == TOOL_SUCCESS else classify_tool_error(r.error_code, r.error_said)
        calls[(r.agent_id, _week_of(r.week_start), r.app_name, r.status, error_class)] += r.calls
    out = [dict(zip(TOOL_ROW_FIELDS, (*key, count))) for key, count in calls.items()]
    return tuple(sorted(out, key=itemgetter(*TOOL_ROW_FIELDS[:-1])))


def _tool_totals(tool_rows: Iterable[Mapping[str, Any]]) -> Dict[AgentWeek, Dict[str, int]]:
    """Each agent-week's tool runs, all apps together (a guard metric in ORG-10)."""
    totals: Dict[AgentWeek, Dict[str, int]] = defaultdict(lambda: {"tool_calls": 0, "tool_failed_calls": 0})
    for row in tool_rows:
        entry = totals[(row["agent_id"], row["iso_week"])]
        entry["tool_calls"] += row["calls"]
        if row["status"] != TOOL_SUCCESS:
            entry["tool_failed_calls"] += row["calls"]
    return dict(totals)


def _nothing_recorded(field: str) -> Any:
    """A field's value when no row was recorded: no runs, $0.0, or a count of 0."""
    if field in _RUN_SAMPLE_FIELDS:
        return ()
    return 0.0 if field in _COST_FIELDS else 0


def _blank_row(agent: Mapping[str, Any], week: str) -> Dict[str, Any]:
    """An agent-week with nothing recorded: every count 0 (n=0), every run list empty."""
    counts = {f: _nothing_recorded(f) for f in AGENT_ROW_FIELDS if f not in _IDENTITY_FIELDS}
    return {**agent, "iso_week": week, **counts}


def _agent_rows(roster: List[Dict[str, Any]], window: Window,
                sources: Tuple[Mapping[AgentWeek, Mapping[str, Any]], ...]) -> Tuple[Dict[str, Any], ...]:
    """The roster LEFT JOINed to every source, for every week: a missing source row is n=0."""
    rows = []
    for agent in roster:
        for week in window.weeks:
            key = (agent["agent_id"], week)
            found = [source[key] for source in sources if key in source]
            rows.append({k: v for part in (_blank_row(agent, week), *found) for k, v in part.items()})
    return tuple(rows)


def read_sync(db: Session, workspace_id: Union[UUID, str], cur_week: str,
              *, ref_weeks: int = REF_WEEKS) -> HarnessMetrics:
    """Read one workspace's raw per-agent counts for ``cur_week`` and the weeks before it.

    The synchronous core (tests call it on a real session); async callers use
    :func:`read`. Every statement is bound to ``workspace_id``. A database error is
    logged and raised, never turned into zeros.
    """
    workspace = str(UUID(str(workspace_id)))
    window = build_window(cur_week, ref_weeks)
    try:
        roster = _roster(db, workspace)
        if not roster:
            return HarnessMetrics(workspace_id=workspace, window=window, agent_rows=(), tool_rows=())
        params = _base_params(workspace, [a["agent_id"] for a in roster], window)
        tools = _tool_rows(db, params)
        sources = (_ticket_counts(db, params), _mission_counts(db, params), _heartbeat_counts(db, params),
                   _usage_counts(db, params), _tool_totals(tools))
    except SQLAlchemyError:
        logger.exception("[HARNESS] metrics read failed for workspace %s, week %s", workspace, window.cur_week)
        raise
    rows = _agent_rows(roster, window, sources)
    logger.info("[HARNESS] metrics read: workspace %s, week %s, %d agents x %d weeks, %d tool rows",
                workspace, window.cur_week, len(roster), len(window.weeks), len(tools))
    return HarnessMetrics(workspace_id=workspace, window=window, agent_rows=rows, tool_rows=tools)


async def read(db: Session, workspace_id: Union[UUID, str], cur_week: str,
               *, ref_weeks: int = REF_WEEKS) -> HarnessMetrics:
    """:func:`read_sync` on a worker thread, so the six statements never hold the event loop."""
    return await asyncio.to_thread(read_sync, db, workspace_id, cur_week, ref_weeks=ref_weeks)


__all__ = [
    "AGENT_ROW_FIELDS",
    "HarnessMetrics",
    "RECONNECT_CLASSES",
    "RECONNECT_CODES",
    "RECONNECT_PATTERNS",
    "TOOL_ROW_FIELDS",
    "Window",
    "build_window",
    "classify_tool_error",
    "iso_week_label",
    "read",
    "read_sync",
]
