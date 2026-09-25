"""F156 — the HARNESS task ledger.

Which done [HARNESS] board tasks self-management has applied (with the value
each change replaced, for US-022 rollback), and which wait for an owner's or
admin's /approve. It is the idempotency key for the weekly tick and the
/approve command. It lives here, written only by services.harness_service,
instead of in a file on the workspace volume that a workspace tool could
write.
"""
from sqlalchemy import CheckConstraint, Column, DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database.base import Base

LEDGER_APPLIED = "applied"
LEDGER_HELD = "held"


class HarnessTaskLedger(Base):
    __tablename__ = "harness_task_ledger"

    id = Column(Integer, primary_key=True)
    workspace_id = Column(UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False)
    board_task_id = Column(Integer, nullable=False)
    state = Column(String(20), nullable=False)
    entry = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint("workspace_id", "board_task_id", name="uq_harness_task_ledger_task"),
        CheckConstraint("state IN ('applied', 'held')", name="ck_harness_task_ledger_state"),
        {"extend_existing": True},
    )
