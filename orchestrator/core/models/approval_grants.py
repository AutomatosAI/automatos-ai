"""PRD-181 S2 (F060) — durable, scoped, expiring, revocable approval grants.

This is the **deferred W4 slice**: the mission-only approval primitive
(PRD-163's ``workspace.settings.approval_policy``) generalised into a first-class
row so *non-chat* agents — board tasks, playbook runs, scheduled/webhook agents —
that hit an ``ask`` tier get a real approval workflow instead of a hard block or
an auto-allow.

Why a new table (justified per the CLAUDE.md "no new tables" rule): the existing
approval state lives only as transient run/task state and workspace settings.
There is nowhere durable to record "a human granted agent X permission to do Y
on subject Z until time T, revocable" — which is exactly what an auditable,
tool-agnostic grant needs. No existing table fits that shape.

A grant is:
  - **scoped** — to a workspace + a subject (``board_task`` / ``playbook_run`` /
    a specific ``tool`` call) + a tool name + a risk tier.
  - **expiring** — ``expires_at`` bounds how long the authorisation lasts.
  - **revocable** — ``revoked_at`` / ``revoked_by`` retract it before expiry.
  - **auditable** — every state change is written to ``audit_logs`` by the
    services layer (this model is pure data).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


class GrantStatus(str, Enum):
    """Lifecycle of an approval grant.

    ``PENDING``  — created, awaiting a human decision (the subject is blocked).
    ``GRANTED``  — a human approved; authorises the subject until ``expires_at``.
    ``REVOKED``  — retracted before expiry; no longer authorises.
    ``EXPIRED``  — ``expires_at`` passed; no longer authorises (may be lazily set).
    ``DENIED``   — a human explicitly refused; the subject fails, not retries.
    """

    PENDING = "pending"
    GRANTED = "granted"
    REVOKED = "revoked"
    EXPIRED = "expired"
    DENIED = "denied"


# Subject kinds a grant can scope to. Tool-agnostic by design.
SUBJECT_BOARD_TASK = "board_task"
SUBJECT_PLAYBOOK_RUN = "playbook_run"
SUBJECT_TOOL_CALL = "tool_call"

# PRD-225: a grant's ``kind`` — the classic boolean approval, or a free-text ask.
# A question is a grant whose decision is words instead of yes/no; the status
# vocabulary is reused (pending=open, granted=answered, denied=dismissed).
KIND_APPROVAL = "approval"
KIND_QUESTION = "question"


class ApprovalGrant(Base):
    """A durable authorisation record for a scoped, side-effecting action."""

    __tablename__ = "approval_grants"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(
        PGUUID(as_uuid=True),
        ForeignKey("workspaces.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # What this grant authorises.
    subject_type = Column(String(30), nullable=False)  # board_task | playbook_run | tool_call
    subject_id = Column(String(255), nullable=False)   # the task id / run id / call id
    tool_name = Column(String(255), nullable=True)     # the specific tool, when known
    risk_tier = Column(String(40), nullable=True)      # policy_document risk class
    agent_id = Column(Integer, nullable=True)          # the acting agent, when known

    # Lifecycle.
    status = Column(
        String(20), nullable=False,
        default=GrantStatus.PENDING.value, server_default=GrantStatus.PENDING.value,
    )
    reason = Column(Text, nullable=True)               # why approval was needed
    estimated_cost_usd = Column(String(32), nullable=True)  # decimal-as-text, avoids float drift

    # PRD-225: a grant is a question when its decision is words, not a boolean.
    # ``kind`` defaults to 'approval' so every existing row and flow is unchanged.
    kind = Column(
        String(16), nullable=False,
        default=KIND_APPROVAL, server_default=KIND_APPROVAL,
    )
    question_md = Column(Text, nullable=True)          # the ask, markdown
    options = Column(JSONB, nullable=True)             # optional discrete choices
    answer_text = Column(Text, nullable=True)          # the human's free-text answer
    answered_by = Column(String(255), nullable=True)   # actor ref, e.g. 'user:42'
    answered_at = Column(DateTime(timezone=True), nullable=True)
    asked_by_agent_id = Column(Integer, nullable=True)  # who raised the ask
    channel_refs = Column(JSONB, nullable=True)        # outbound delivery correlation

    requested_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    granted_at = Column(DateTime(timezone=True), nullable=True)
    granted_by = Column(String(255), nullable=True)    # actor ref, e.g. 'user:42'
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    revoked_by = Column(String(255), nullable=True)

    details = Column(JSONB, default=dict)

    __table_args__ = (
        # The hot lookup: "is there an active grant for this subject?"
        Index("ix_approval_grants_subject", "workspace_id", "subject_type", "subject_id", "status"),
        {"extend_existing": True},
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "workspace_id": str(self.workspace_id) if self.workspace_id else None,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "tool_name": self.tool_name,
            "risk_tier": self.risk_tier,
            "agent_id": self.agent_id,
            "status": self.status,
            "reason": self.reason,
            "estimated_cost_usd": self.estimated_cost_usd,
            # PRD-225: the ask fields — 'approval' rows leave question_md/answer
            # null, so the classic approval card is unaffected.
            "kind": self.kind or KIND_APPROVAL,
            "question_md": self.question_md,
            "options": self.options,
            "answer_text": self.answer_text,
            "answered_by": self.answered_by,
            "answered_at": self.answered_at.isoformat() if self.answered_at else None,
            "asked_by_agent_id": self.asked_by_agent_id,
            "channel_refs": self.channel_refs or {},
            "requested_at": self.requested_at.isoformat() if self.requested_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "granted_at": self.granted_at.isoformat() if self.granted_at else None,
            "granted_by": self.granted_by,
            "revoked_at": self.revoked_at.isoformat() if self.revoked_at else None,
            "revoked_by": self.revoked_by,
            # PRD-193 S4 (P2-12): the tool_call snapshot (action + params
            # digest source) and the resume outcome (details.executed_result)
            # — the S3 card and the P2-15 approvals queue read these.
            "details": self.details or {},
        }
