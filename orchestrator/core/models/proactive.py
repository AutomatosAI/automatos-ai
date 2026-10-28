"""PRD-268 §6.2.4, §6.9 — Proactive Auto's records: suggestions, runs and each person's settings.

* ``ProactiveSuggestion`` — one thing a check-in found for one person (the addressee,
  §12-Q2): what Auto says, the reads it cites, what Start does, its dedupe key, its
  streak count and the person's decision. At most one ``open`` row per workspace, person
  and ``subject_key`` (a partial unique index): a hidden row and a new open one may share
  the key.
* ``ProactiveRun`` — the audit: one row per check-in run or refusal, its window, outcome,
  signals, suggestion count, model, tokens and cost.
* ``ProactivePersonSettings`` — one row per workspace and person: the pause and the
  quiet hours. The primary key (workspace_id, user_id) is the unique.

``user_id`` and the agent ids are plain integers, not foreign keys: the run audit and a
decided suggestion outlive the agent or person they name. JSON columns are
``JSON().with_variant(JSONB(), "postgresql")`` and ids the portable ``Uuid``, so the
tables also build under SQLite in the unit tests. The Alembic revision
``prd268_w1_proactive`` declares the same names, and
``tests/test_prd268_proactive_models.py`` holds the two together.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Time,
    Uuid,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = [
    "ProactiveSuggestion",
    "ProactiveRun",
    "ProactivePersonSettings",
    "SUGGESTION_STATUSES",
    "SUGGESTION_OPEN",
    "SUGGESTION_STARTED",
    "SUGGESTION_HIDDEN",
    "SUGGESTION_EXPIRED",
    "RUN_OUTCOMES",
    "RUN_SILENT",
    "RUN_SUGGESTED",
    "RUN_ERROR",
    "RUN_SKIPPED_QUIET",
    "RUN_SKIPPED_CAP",
    "RUN_SKIPPED_CREDITS",
    "RUN_SKIPPED_PAUSED",
    "RUN_SKIPPED_OFF",
]

# A suggestion's life: open until the person starts or hides it, or it expires.
SUGGESTION_OPEN = "open"
SUGGESTION_STARTED = "started"
SUGGESTION_HIDDEN = "hidden"
SUGGESTION_EXPIRED = "expired"
SUGGESTION_STATUSES = (SUGGESTION_OPEN, SUGGESTION_STARTED, SUGGESTION_HIDDEN, SUGGESTION_EXPIRED)

# How a run ended: it found nothing, filed suggestions, failed, or was refused before any model call.
RUN_SILENT = "silent"
RUN_SUGGESTED = "suggested"
RUN_ERROR = "error"
RUN_SKIPPED_QUIET = "skipped_quiet"
RUN_SKIPPED_CAP = "skipped_cap"
RUN_SKIPPED_CREDITS = "skipped_credits"
RUN_SKIPPED_PAUSED = "skipped_paused"
RUN_SKIPPED_OFF = "skipped_off"
RUN_OUTCOMES = (
    RUN_SILENT, RUN_SUGGESTED, RUN_ERROR, RUN_SKIPPED_QUIET,
    RUN_SKIPPED_CAP, RUN_SKIPPED_CREDITS, RUN_SKIPPED_PAUSED, RUN_SKIPPED_OFF,
)

# The severities triage gives a suggestion (§6.7); a check-in that gives none files can_wait.
SEVERITY_DEFAULT = "can_wait"

TITLE_LENGTH = 120
DETAIL_LENGTH = 600
SUBJECT_KEY_LENGTH = 200  # 'title:' + a normalised title, or 'agent:<id>:error:<class>'
SHORT_LENGTH = 16  # source_type, severity, status, decided_via
HIDE_REASON_LENGTH = 32
RUN_KIND_LENGTH = 32
OUTCOME_LENGTH = 32
MODEL_LENGTH = 128
ERROR_CLASS_LENGTH = 64
TIMEZONE_LENGTH = 64  # an IANA name, e.g. 'America/Sao_Paulo'
COST_PRECISION, COST_SCALE = 14, 6  # as credit_ledger.actual_cost_usd

OPEN_ROW = f"status = '{SUGGESTION_OPEN}'"


def _in_list(column: str, values: tuple) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _json_type():
    return JSON().with_variant(JSONB(), "postgresql")


def _workspace_column(primary_key: bool = False) -> Column:
    return Column(
        Uuid(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False, primary_key=primary_key,
    )


def _timestamp(nullable: bool = True) -> Column:
    if nullable:
        return Column(DateTime(timezone=True), nullable=True)
    return Column(DateTime(timezone=True), nullable=False, server_default=func.now())


def _iso(value: Any) -> Any:
    return value.isoformat() if value is not None else None


class ProactiveSuggestion(Base):
    """One thing a check-in found for one person, with what Start does and the person's decision."""

    __tablename__ = "proactive_suggestions"
    __table_args__ = (
        CheckConstraint(_in_list("status", SUGGESTION_STATUSES), name="ck_proactive_suggestions_status"),
        Index(
            "uq_proactive_suggestions_open_subject", "workspace_id", "user_id", "subject_key",
            unique=True, postgresql_where=text(OPEN_ROW), sqlite_where=text(OPEN_ROW),
        ),
        Index("ix_proactive_suggestions_workspace_status_created", "workspace_id", "status", "created_at"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = _workspace_column()
    # The addressee (§12-Q2): users.id, resolved with core.auth.principal.resolve_user_pk.
    user_id = Column(Integer, nullable=False)
    # check_in | heartbeat | follow_up | streak
    source_type = Column(String(SHORT_LENGTH), nullable=False)
    source_agent_id = Column(Integer, nullable=True)
    run_id = Column(Uuid(as_uuid=True), nullable=True)
    title = Column(String(TITLE_LENGTH), nullable=False)
    detail = Column(String(DETAIL_LENGTH), nullable=False, server_default="")
    # Up to 5 {type, id} refs to what the run read: quoted to the person, never followed.
    evidence = Column(_json_type(), nullable=False, server_default="[]")
    # {kind: board_task | mission | playbook_run | chat, payload}: what Start does.
    proposed_action = Column(_json_type(), nullable=False)
    subject_key = Column(String(SUBJECT_KEY_LENGTH), nullable=False)
    # today | can_wait
    severity = Column(String(SHORT_LENGTH), nullable=False, server_default=SEVERITY_DEFAULT)
    status = Column(String(SHORT_LENGTH), nullable=False, server_default=SUGGESTION_OPEN)
    occurrences = Column(Integer, nullable=False, server_default=text("1"))
    last_seen_at = _timestamp(nullable=False)
    # {board_task_id | mission_id | execution_id}: what Start made.
    started_ref = Column(_json_type(), nullable=True)
    # web | go | push | chat | telegram
    decided_via = Column(String(SHORT_LENGTH), nullable=True)
    # not_now | not_useful | wrong_agent
    hide_reason = Column(String(HIDE_REASON_LENGTH), nullable=True)
    decided_at = _timestamp()
    est_credits = Column(Integer, nullable=True)
    created_at = _timestamp(nullable=False)
    delivered_at = _timestamp()
    expires_at = _timestamp()

    def to_dict(self) -> dict:
        """The suggestion as JSON-ready values: ids as strings, times as ISO 8601."""
        return {
            "id": str(self.id) if self.id is not None else None,
            "workspace_id": str(self.workspace_id) if self.workspace_id is not None else None,
            "user_id": self.user_id,
            "source_type": self.source_type,
            "source_agent_id": self.source_agent_id,
            "run_id": str(self.run_id) if self.run_id is not None else None,
            "title": self.title,
            "detail": self.detail,
            "evidence": list(self.evidence or []),
            "proposed_action": dict(self.proposed_action or {}),
            "subject_key": self.subject_key,
            "severity": self.severity,
            "status": self.status,
            "occurrences": self.occurrences,
            "last_seen_at": _iso(self.last_seen_at),
            "started_ref": dict(self.started_ref) if self.started_ref is not None else None,
            "decided_via": self.decided_via,
            "hide_reason": self.hide_reason,
            "decided_at": _iso(self.decided_at),
            "est_credits": self.est_credits,
            "created_at": _iso(self.created_at),
            "delivered_at": _iso(self.delivered_at),
            "expires_at": _iso(self.expires_at),
        }


class ProactiveRun(Base):
    """One check-in run or refusal: its window, outcome, signals, suggestion count, model and cost."""

    __tablename__ = "proactive_runs"
    __table_args__ = (
        CheckConstraint(_in_list("outcome", RUN_OUTCOMES), name="ck_proactive_runs_outcome"),
        Index("ix_proactive_runs_workspace_agent_created", "workspace_id", "agent_id", "created_at"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = _workspace_column()
    agent_id = Column(Integer, nullable=True)
    kind = Column(String(RUN_KIND_LENGTH), nullable=False)
    window_since = _timestamp()
    window_until = _timestamp()
    outcome = Column(String(OUTCOME_LENGTH), nullable=False)
    signals = Column(_json_type(), nullable=False, server_default="[]")
    suggestions = Column(Integer, nullable=False, server_default=text("0"))
    model = Column(String(MODEL_LENGTH), nullable=True)
    tokens = Column(Integer, nullable=False, server_default=text("0"))
    cost_usd = Column(Numeric(COST_PRECISION, COST_SCALE), nullable=False, server_default=text("0"))
    error_class = Column(String(ERROR_CLASS_LENGTH), nullable=True)
    created_at = _timestamp(nullable=False)


class ProactivePersonSettings(Base):
    """One person's proactive settings in one workspace: the pause and the quiet hours."""

    __tablename__ = "proactive_person_settings"
    __table_args__ = ({"extend_existing": True},)

    workspace_id = _workspace_column(primary_key=True)
    # users.id, resolved with core.auth.principal.resolve_user_pk.
    user_id = Column(Integer, primary_key=True, autoincrement=False)
    paused = Column(Boolean, nullable=False, server_default=false())
    quiet_start = Column(Time, nullable=True)
    quiet_end = Column(Time, nullable=True)
    timezone = Column(String(TIMEZONE_LENGTH), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
