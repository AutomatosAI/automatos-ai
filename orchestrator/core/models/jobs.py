"""The durable job queue's table (PRD-276 Wave 1a, §6.2).

A job is "do this, later or now, and don't lose it": queued in the same transaction as the write
that caused it, claimed by a worker with a lease (PRD-161's claim, generalised), retried with its
kind's backoff, and ``dead`` once its attempts are used up. ``core/jobs`` is the only code that
reads or writes it. ``alembic/versions/jobs_20261111.py`` creates the same shape, and
``jobs_schema_20261112.py`` moves it into its own schema, ``jobs`` (§10-7, §12.2): the queue's
write churn sits apart from the business tables in the same database, and every query names it
``jobs.jobs``. ``jobs.job_schedules`` holds each schedule's cursor (Wave 2b, ``job_schedules_20261112.py``).
"""
from __future__ import annotations

from uuid import uuid4

from sqlalchemy import DDL, Column, DateTime, ForeignKey, Index, Integer, String, Text, event, text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
DEAD = "dead"
STATUSES = (QUEUED, RUNNING, DONE, DEAD)
JOBS_SCHEMA = "jobs"


class Job(Base):
    """One unit of background work."""

    __tablename__ = "jobs"
    __table_args__ = (
        # A second queue with the same key is a no-op (§6.2): per kind and tenant, a platform
        # job's NULL workspace counting as one value.
        Index("uq_jobs_kind_workspace_key", "kind", "workspace_id", "idempotency_key", unique=True,
              postgresql_where=text("idempotency_key IS NOT NULL"), postgresql_nulls_not_distinct=True),
        Index("ix_jobs_claim", "status", "run_at"),
        Index("ix_jobs_workspace", "workspace_id"),
        {"schema": JOBS_SCHEMA},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    kind = Column(String(100), nullable=False)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True)
    payload = Column(JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb"))
    status = Column(String(16), nullable=False, default=QUEUED, server_default=text(f"'{QUEUED}'"))
    run_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    attempts = Column(Integer, nullable=False, default=0, server_default=text("0"))
    max_attempts = Column(Integer, nullable=False)
    lease_until = Column(DateTime(timezone=True), nullable=True)
    claimed_by = Column(String(255), nullable=True)
    claim_id = Column(PGUUID(as_uuid=True), nullable=True)
    idempotency_key = Column(String(255), nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
    finished_at = Column(DateTime(timezone=True), nullable=True)


class JobSchedule(Base):
    """A schedule's cursor (PRD-276 §6.4, Wave 2b): the last slot it queued. Each due slot is queued
    in the transaction that moves ``last_slot`` past it, so a slot is queued and recorded, or
    neither; at start-up the slots missed since are queued by the kind's catch-up policy."""

    __tablename__ = "job_schedules"
    __table_args__ = (
        Index("ix_job_schedules_workspace", "workspace_id"),
        {"schema": JOBS_SCHEMA},
    )

    kind = Column(String(100), primary_key=True)
    schedule_id = Column(String(255), primary_key=True)
    workspace_id = Column(PGUUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=True)
    last_slot = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


# create_all (and a test's ``__table__.create``) builds either table only once its schema exists.
for _table in (Job.__table__, JobSchedule.__table__):
    event.listen(_table, "before_create", DDL(f"CREATE SCHEMA IF NOT EXISTS {JOBS_SCHEMA}"))
