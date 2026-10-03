"""PRD-252 R4 — every ticket gets a number in its workspace when it is inserted.

The number (``BoardTask.workspace_seq``, shown as #0042) comes from the
workspace's counter row, which an upsert raises by one and returns. The row is
locked until the inserting transaction commits, so two tickets filed at once
in one workspace get two numbers. A rolled-back insert leaves a gap, and a
deleted ticket's number is never given again.

A mission step takes no number: it shows its mission card's number and its
step, #0051.3 (D5, ``services/ticket_numbers.py``). Every insert goes through
the ORM (no raw ``INSERT INTO board_tasks`` outside tests), so the listener
numbers every ticket, wherever it is filed.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import Column, ForeignKey, Integer, event, text
from sqlalchemy.dialects.postgresql import UUID

from core.database.base import Base
from core.models.core import BoardTask

# A mission step's source_type: it is its mission's, and has no number of its own.
STEP_SOURCE = "orchestration_task"

_NEXT_NUMBER = text("""
    INSERT INTO workspace_ticket_counters (workspace_id, last_seq) VALUES (CAST(:ws AS uuid), 1)
    ON CONFLICT (workspace_id) DO UPDATE SET last_seq = workspace_ticket_counters.last_seq + 1
    RETURNING last_seq
""")


class WorkspaceTicketCounter(Base):
    """The last ticket number a workspace gave out."""

    __tablename__ = "workspace_ticket_counters"

    workspace_id = Column(UUID, ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    last_seq = Column(Integer, nullable=False, default=0, server_default="0")


def takes_a_number(task: Any) -> bool:
    """Every ticket but a mission step has a number of its own."""
    return getattr(task, "source_type", None) != STEP_SOURCE


@event.listens_for(BoardTask, "before_insert")
def _number_the_ticket(mapper: Any, connection: Any, task: BoardTask) -> None:
    """Give a new ticket its workspace's next number, in the inserting transaction."""
    if task.workspace_seq is None and task.workspace_id is not None and takes_a_number(task):
        task.workspace_seq = connection.execute(_NEXT_NUMBER, {"ws": str(task.workspace_id)}).scalar()
