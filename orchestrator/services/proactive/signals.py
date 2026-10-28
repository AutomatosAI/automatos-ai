"""PRD-268 US-C03c (§6.2.1, FR-3) — what changed in a workspace: the signals Auto's check-in starts from.

Auto's workspace check-in begins with a deterministic diff over its window ``[since,
until)`` (``modules.proactive.check_in_window``): what became worth the owner's attention
in that time. No signal, no model call: the runner records the run ``silent`` at zero
cost (US-C03d). Each kind has its own reader, one query, scoped to the workspace:

* ``ticket_failed``: a ticket that failed in the window (``completed_at``, which every
  path that fails a ticket stamps; else ``updated_at``). A mission's step is its
  mission's to handle, as Needs you counts it (``services.needs_you``).
* ``ticket_blocked``: a ticket blocked in the window (``blocked_at``, else ``updated_at``);
  a mission's step is left out the same way.
* ``ticket_stuck``: a ticket nothing will move until the owner does, read as Needs you
  reads it (``needs_you.stuck_tickets``), last changed in the window. Being stuck has no
  timestamp of its own, so an edit to a ticket that was already stuck shows it again.
* ``heartbeat_error`` and ``heartbeat_objective_missed``: a heartbeat result with
  ``status = 'error'``, or with ``objective_met = false``, written in the window. Its
  ``created_at`` has no zone; the query reads it in the session's, as it was written.
* ``watch_missed_run``, ``watch_benched``, ``watch_expired`` and ``watch_budget_exhausted``:
  those watch events (``core.models.watch_enums``), recorded in the window. A watch event
  has no workspace of its own, so it is scoped through its watch.
* ``report_awaiting_approval``: a report filed in the window that asks for approval
  (``requires_approval``) and that nobody has acknowledged.
* ``grant_expiring``: a pending approval that came within 24 hours of lapsing during the
  window (or was asked in it with less than that left) and has not lapsed by ``until``.
  A question stays open past its expiry, so only approvals count, as in Needs you.
* ``scheduled_run_skipped``: a one-off scheduled task whose time passed in the window
  without it running: still active, never run, and its time further back than the
  misfire grace the scheduler allows (``ONE_SHOT_MISFIRE_GRACE_SECONDS``). That covers a
  run the scheduler missed and one skipped because only the trial could pay for it. A
  recurring task's skipped run is logged, not stored, so it cannot be read here; a
  scheduled playbook's are the ``missed_run`` and ``benched`` watch events.
* ``credits_low``: the workspace's AI credits are at the running-low or run-out alert
  level (``credit_balances.alert_level``, ``services.credits.alerts``) and the balance
  moved in the window. An idle balance says nothing new. The level keeps no time of its
  own, so a low balance that keeps being spent says it in each window; its subject is
  always ``credits:low``, for the runner to recognise.

Each reader lists at most ``ROWS_PER_KIND`` rows, newest first. There is one signal per
subject: a ticket both stuck and blocked is one signal, the first of failed, stuck and
blocked, and an agent's latest errored heartbeat stands for its others. Signals come back
newest first.

Everything a signal carries was read from the workspace, so it is untrusted data. Its
summary is one plain line of at most ``SUMMARY_LENGTH`` characters, which a prompt quotes
and never follows. Its ``ref`` is the ``(type, id)`` read a suggestion may cite as
evidence (``modules.proactive.provenance``); a ticket's id is its board number.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.models.approval_grants import KIND_APPROVAL, GrantStatus
from core.models.ticket_numbers import STEP_SOURCE
from core.models.watch_enums import WatchEventType
from modules.proactive.dedupe import board_number, ticket_key
from modules.proactive.provenance import TICKET
from services.credits.alerts import LEVEL_LOW, LEVEL_OUT
from services.needs_you import stuck_tickets
from services.scheduled_task_service import ONE_SHOT_MISFIRE_GRACE_SECONDS
from services.ticket_numbers import ticket_numbers

__all__ = [
    "CREDITS_LOW",
    "GRANT_EXPIRING",
    "HEARTBEAT_ERROR",
    "HEARTBEAT_OBJECTIVE_MISSED",
    "KINDS",
    "REPORT_AWAITING_APPROVAL",
    "SCHEDULED_RUN_SKIPPED",
    "TICKET_BLOCKED",
    "TICKET_FAILED",
    "TICKET_STUCK",
    "WATCH_KINDS",
    "Signal",
    "read_signals",
]

TICKET_FAILED = "ticket_failed"
TICKET_STUCK = "ticket_stuck"
TICKET_BLOCKED = "ticket_blocked"
HEARTBEAT_ERROR = "heartbeat_error"
HEARTBEAT_OBJECTIVE_MISSED = "heartbeat_objective_missed"
REPORT_AWAITING_APPROVAL = "report_awaiting_approval"
GRANT_EXPIRING = "grant_expiring"
SCHEDULED_RUN_SKIPPED = "scheduled_run_skipped"
CREDITS_LOW = "credits_low"
# The watch events a check-in reads (watch_enums.py), each its own kind: watch_<event>.
WATCH_PREFIX = "watch_"
WATCH_EVENTS = tuple(event.value for event in (
    WatchEventType.MISSED_RUN, WatchEventType.BENCHED, WatchEventType.EXPIRED, WatchEventType.BUDGET_EXHAUSTED))
WATCH_KINDS = tuple(f"{WATCH_PREFIX}{event}" for event in WATCH_EVENTS)
KINDS = (TICKET_FAILED, TICKET_STUCK, TICKET_BLOCKED, HEARTBEAT_ERROR, HEARTBEAT_OBJECTIVE_MISSED, *WATCH_KINDS,
         REPORT_AWAITING_APPROVAL, GRANT_EXPIRING, SCHEDULED_RUN_SKIPPED, CREDITS_LOW)

# Rows each reader lists, newest first: a prompt names a handful, never hundreds.
ROWS_PER_KIND = 50
SUMMARY_LENGTH = 240
# A pending approval within this of lapsing is a signal (§6.2.1).
GRANT_HORIZON = timedelta(hours=24)
# How far past its time a one-off may still fire (scheduled_task_service).
SKIP_GRACE = timedelta(seconds=ONE_SHOT_MISFIRE_GRACE_SECONDS)

# heartbeat_results: an agent's row holds its id in source_id; an errored tick's status.
AGENT_SOURCE = "agent"
ERROR_STATUS = "error"
AUTO_NAME = "Auto"
AN_AGENT = "An agent"

# The (type, id) refs a suggestion may cite; a ticket's is provenance's TICKET.
BOARD_TASK_REF = "board_task"
HEARTBEAT_REF = "heartbeat"
WATCH_REF = "watch"
REPORT_REF = "report"
GRANT_REF = "grant"
SCHEDULED_TASK_REF = "scheduled_task"
CREDITS_REF = "credits"
CREDITS_SUBJECT = f"{CREDITS_REF}:low"

# Dropped from a summary: controls, format characters (zero-width, bidi overrides),
# surrogates and private use.
_DROPPED_CATEGORIES = ("Cc", "Cf", "Cs", "Co")
_WATCH_SAYS = "Watch {title}: {event}. {summary}"
_SAYS: Dict[str, str] = {
    TICKET_FAILED: "Ticket {label} failed: {title}. {reason}",
    TICKET_STUCK: "Ticket {label} is stuck ({reason}): {title}",
    TICKET_BLOCKED: "Ticket {label} is blocked: {title}. {reason}",
    HEARTBEAT_ERROR: "{who}'s heartbeat failed",
    HEARTBEAT_OBJECTIVE_MISSED: "{who}'s heartbeat did not meet its objective",
    **{kind: _WATCH_SAYS for kind in WATCH_KINDS},
    REPORT_AWAITING_APPROVAL: "A report from {who} waits for approval: {title}",
    GRANT_EXPIRING: "An approval lapses at {when}: {what}",
    SCHEDULED_RUN_SKIPPED: "A scheduled task for {who} did not run at {when}: {what}",
    CREDITS_LOW: "The workspace's AI credits {state}",
}
_HEARTBEAT_ISSUE = {HEARTBEAT_ERROR: "error", HEARTBEAT_OBJECTIVE_MISSED: "objective_missed"}
_CREDITS_OUT = "have run out"
_CREDITS_RUNNING_LOW = "are running low"


@dataclass(frozen=True)
class Signal:
    """One thing that changed: its kind, what it is about (``subject_key``), the read a
    suggestion cites for it (``ref``, ``(type, id)``), when it happened, and one plain line
    saying what it is, read from untrusted data."""

    kind: str
    subject_key: str
    ref: Tuple[str, str]
    at: datetime
    summary: str

    @property
    def evidence(self) -> Dict[str, str]:
        """``ref`` as a suggestion's evidence cites it: ``{type, id}``."""
        return {"type": self.ref[0], "id": self.ref[1]}

    def as_json(self) -> Dict[str, Any]:
        """The signal as a run row records it (``proactive_runs.signals``)."""
        return {"kind": self.kind, "subject_key": self.subject_key, "ref": self.evidence,
                "at": self.at.isoformat(), "summary": self.summary}


@dataclass(frozen=True)
class _Window:
    """What one read covers: a workspace, from ``since`` (inclusive) to ``until``."""

    workspace_id: Any
    since: datetime
    until: datetime

    def params(self, **extra: Any) -> Dict[str, Any]:
        return {"ws": str(self.workspace_id), "since": self.since, "until": self.until,
                "limit": ROWS_PER_KIND, **extra}

    def holds(self, moment: Optional[datetime]) -> bool:
        return moment is not None and self.since <= _utc(moment) < self.until


def _utc(value: datetime) -> datetime:
    """``value`` in UTC; a naive one is taken as UTC."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _plain_char(char: str) -> str:
    if char.isspace():
        return " "
    return "" if unicodedata.category(char) in _DROPPED_CATEGORIES else char


def _line(value: Any) -> str:
    """``value`` as one plain line of at most ``SUMMARY_LENGTH`` characters."""
    plain = "".join(_plain_char(char) for char in str(value or ""))
    return " ".join(plain.split())[:SUMMARY_LENGTH]


def _when(moment: datetime) -> str:
    return f"{_utc(moment):%Y-%m-%d %H:%M} UTC"


def _signal(kind: str, ref_type: str, ref_id: Any, at: datetime, subject: Optional[str] = None,
            **says: Any) -> Signal:
    """A signal whose summary is its kind's sentence filled with ``says``; its subject is
    its ref (``type:id``) unless given."""
    return Signal(kind=kind, subject_key=subject or f"{ref_type}:{ref_id}", ref=(ref_type, str(ref_id)),
                  at=_utc(at), summary=_line(_SAYS[kind].format(**says)))


# --- Tickets ------------------------------------------------------------------------

_FAILED_TICKETS = text("""
    SELECT bt.id, bt.title, bt.workspace_seq, bt.source_type, bt.parent_task_id,
           bt.error_message AS reason, COALESCE(bt.completed_at, bt.updated_at) AS at
      FROM board_tasks bt
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'failed' AND bt.source_type <> :step
       AND COALESCE(bt.completed_at, bt.updated_at) >= :since AND COALESCE(bt.completed_at, bt.updated_at) < :until
  ORDER BY at DESC LIMIT :limit
""")
_BLOCKED_TICKETS = text("""
    SELECT bt.id, bt.title, bt.workspace_seq, bt.source_type, bt.parent_task_id,
           bt.blocked_reason AS reason, COALESCE(bt.blocked_at, bt.updated_at) AS at
      FROM board_tasks bt
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = 'blocked' AND bt.source_type <> :step
       AND COALESCE(bt.blocked_at, bt.updated_at) >= :since AND COALESCE(bt.blocked_at, bt.updated_at) < :until
  ORDER BY at DESC LIMIT :limit
""")


def _ticket(kind: str, row: Any, number: Optional[str], reason: Any) -> Signal:
    """A ticket's signal: about the ticket by its board number (``ticket:0846``), else by
    its id when it has no number."""
    canonical = board_number(number) if number else None
    says = {"label": number or f"board task {row.id}", "title": row.title, "reason": reason or ""}
    if canonical is None:
        return _signal(kind, BOARD_TASK_REF, row.id, row.at, **says)
    return _signal(kind, TICKET, canonical, row.at, subject=ticket_key(canonical), **says)


def _tickets(db: Session, window: _Window, kind: str, rows: List[Any],
             reason_of: Callable[[Any], Any]) -> List[Signal]:
    numbers = ticket_numbers(db, window.workspace_id, rows) if rows else {}
    return [_ticket(kind, row, numbers.get(row.id), reason_of(row)) for row in rows]


def _reason(row: Any) -> Any:
    return row.reason


def _why(row: Any) -> str:
    return str(row.why or "").replace("_", " ")


def _failed_tickets(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_FAILED_TICKETS, window.params(step=STEP_SOURCE)).all()
    return _tickets(db, window, TICKET_FAILED, rows, _reason)


def _stuck_tickets(db: Session, window: _Window) -> List[Signal]:
    _, rows = stuck_tickets(db, window.workspace_id, limit=ROWS_PER_KIND)
    return _tickets(db, window, TICKET_STUCK, [row for row in rows if window.holds(row.at)], _why)


def _blocked_tickets(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_BLOCKED_TICKETS, window.params(step=STEP_SOURCE)).all()
    return _tickets(db, window, TICKET_BLOCKED, rows, _reason)


# --- Heartbeats and watches ---------------------------------------------------------

# An agent's row names it by id in source_id; Auto's (the orchestrator's) holds the
# workspace id there, so the join compares text and never casts source_id. created_at
# has no zone: it is read in the session's, the zone NOW() wrote it in and the one the
# window's comparison uses.
_HEARTBEATS = text("""
    SELECT hr.id, hr.source_type, hr.source_id, hr.status, CAST(hr.created_at AS timestamptz) AS at,
           a.name AS agent_name
      FROM heartbeat_results hr
      LEFT JOIN agents a ON hr.source_type = :agent_source AND a.workspace_id = hr.workspace_id
                        AND CAST(a.id AS text) = hr.source_id
     WHERE hr.workspace_id = CAST(:ws AS uuid) AND (hr.status = :error OR hr.objective_met IS FALSE)
       AND hr.created_at >= :since AND hr.created_at < :until
  ORDER BY hr.created_at DESC LIMIT :limit
""")
_WATCH_EVENTS = text("""
    SELECT e.event_type, e.summary, e.created_at AS at, w.id AS watch_id, w.title
      FROM watch_events e JOIN watches w ON w.id = e.watch_id
     WHERE w.workspace_id = CAST(:ws AS uuid) AND e.event_type = ANY(:events)
       AND e.created_at >= :since AND e.created_at < :until
  ORDER BY e.created_at DESC LIMIT :limit
""")


def _heartbeat(row: Any) -> Signal:
    kind = HEARTBEAT_ERROR if row.status == ERROR_STATUS else HEARTBEAT_OBJECTIVE_MISSED
    if row.source_type == AGENT_SOURCE:
        about, who = f"{AGENT_SOURCE}:{row.source_id}", row.agent_name or f"Agent {row.source_id}"
    else:
        about, who = AUTO_NAME.casefold(), AUTO_NAME
    subject = f"{HEARTBEAT_REF}:{about}:{_HEARTBEAT_ISSUE[kind]}"
    return _signal(kind, HEARTBEAT_REF, row.id, row.at, subject=subject, who=who)


def _heartbeats(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_HEARTBEATS, window.params(agent_source=AGENT_SOURCE, error=ERROR_STATUS)).all()
    return [_heartbeat(row) for row in rows]


def _watch(row: Any) -> Signal:
    kind = f"{WATCH_PREFIX}{row.event_type}"
    return _signal(kind, WATCH_REF, row.watch_id, row.at, subject=f"{WATCH_REF}:{row.watch_id}:{row.event_type}",
                   title=row.title, event=row.event_type.replace("_", " "), summary=row.summary or "")


def _watch_events(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_WATCH_EVENTS, window.params(events=list(WATCH_EVENTS))).all()
    return [_watch(row) for row in rows]


# --- Reports, approvals, schedules and credits ---------------------------------------

_REPORTS = text("""
    SELECT r.id, r.title, r.created_at AS at, a.name AS agent_name
      FROM agent_reports r LEFT JOIN agents a ON a.id = r.agent_id AND a.workspace_id = r.workspace_id
     WHERE r.workspace_id = CAST(:ws AS uuid) AND r.requires_approval IS TRUE AND r.acknowledged_at IS NULL
       AND r.created_at >= :since AND r.created_at < :until
  ORDER BY r.created_at DESC LIMIT :limit
""")
# When an approval came within the horizon of lapsing: the horizon before its expiry, or
# when it was asked if that is later.
_GRANTS = text("""
    SELECT g.id, g.tool_name, g.reason, g.subject_type, g.subject_id, g.expires_at,
           GREATEST(g.expires_at - :horizon, g.requested_at) AS at
      FROM approval_grants g
     WHERE g.workspace_id = CAST(:ws AS uuid) AND g.status = :pending AND COALESCE(g.kind, :approval) = :approval
       AND g.expires_at > :until
       AND GREATEST(g.expires_at - :horizon, g.requested_at) >= :since
       AND GREATEST(g.expires_at - :horizon, g.requested_at) < :until
  ORDER BY at DESC LIMIT :limit
""")
# A one-off is given up once its time is further back than the grace: that moment falls
# in the window when its time falls in the window moved back by the grace.
_SKIPPED_RUNS = text("""
    SELECT t.id, t.description, t.next_run_at AS at, a.name AS agent_name
      FROM agent_scheduled_tasks t
      LEFT JOIN agents a ON a.id = t.target_agent_id AND a.workspace_id = t.workspace_id
     WHERE t.workspace_id = CAST(:ws AS uuid) AND t.task_type = 'one_shot' AND t.status = 'active'
       AND t.last_run_at IS NULL AND t.next_run_at >= :since_due AND t.next_run_at < :until_due
  ORDER BY t.next_run_at DESC LIMIT :limit
""")
_CREDITS = text("""
    SELECT b.alert_level, b.updated_at AS at
      FROM credit_balances b
     WHERE b.workspace_id = CAST(:ws AS uuid) AND b.alert_level >= :low
       AND b.updated_at >= :since AND b.updated_at < :until
""")


def _reports(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_REPORTS, window.params()).all()
    return [_signal(REPORT_AWAITING_APPROVAL, REPORT_REF, row.id, row.at, who=row.agent_name or AN_AGENT,
                    title=row.title) for row in rows]


def _expiring_grants(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_GRANTS, window.params(horizon=GRANT_HORIZON, pending=GrantStatus.PENDING.value,
                                             approval=KIND_APPROVAL)).all()
    return [_signal(GRANT_EXPIRING, GRANT_REF, row.id, row.at, when=_when(row.expires_at),
                    what=row.tool_name or row.reason or f"{row.subject_type} {row.subject_id}") for row in rows]


def _skipped_runs(db: Session, window: _Window) -> List[Signal]:
    rows = db.execute(_SKIPPED_RUNS, window.params(since_due=window.since - SKIP_GRACE,
                                                   until_due=window.until - SKIP_GRACE)).all()
    return [_signal(SCHEDULED_RUN_SKIPPED, SCHEDULED_TASK_REF, row.id, row.at, who=row.agent_name or AN_AGENT,
                    when=_when(row.at), what=row.description) for row in rows]


def _credits_low(db: Session, window: _Window) -> List[Signal]:
    row = db.execute(_CREDITS, window.params(low=LEVEL_LOW)).first()
    if row is None:
        return []
    state = _CREDITS_OUT if row.alert_level >= LEVEL_OUT else _CREDITS_RUNNING_LOW
    return [_signal(CREDITS_LOW, CREDITS_REF, window.workspace_id, row.at, subject=CREDITS_SUBJECT, state=state)]


# --- The diff -----------------------------------------------------------------------

# In order: the first signal about a subject is the one kept (failed, stuck, blocked).
_READERS: Tuple[Callable[[Session, _Window], List[Signal]], ...] = (
    _failed_tickets, _stuck_tickets, _blocked_tickets, _heartbeats, _watch_events,
    _reports, _expiring_grants, _skipped_runs, _credits_low,
)


def _aware(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} is a timezone-aware datetime")
    return _utc(value)


def _one_per_subject(signals: Iterable[Signal]) -> List[Signal]:
    kept: Dict[str, Signal] = {}
    for signal in signals:
        kept.setdefault(signal.subject_key, signal)
    return list(kept.values())


def _at(signal: Signal) -> datetime:
    return signal.at


def read_signals(db: Session, workspace_id: Any, since: datetime, until: datetime) -> Tuple[Signal, ...]:
    """What changed in ``workspace_id`` from ``since`` (inclusive) to ``until``: one signal
    per subject, newest first. Every query is scoped to the workspace; an empty window
    reads nothing. ``since`` and ``until`` are timezone-aware. A failed read raises, so a
    check-in that could not read records an error rather than a silent run."""
    if workspace_id is None:
        raise ValueError("signals are read for one workspace")
    window = _Window(workspace_id, _aware(since, "since"), _aware(until, "until"))
    if window.until <= window.since:
        return ()
    found = [signal for reader in _READERS for signal in reader(db, window)]
    return tuple(sorted(_one_per_subject(found), key=_at, reverse=True))
