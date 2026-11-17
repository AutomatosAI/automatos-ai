"""The new ticket ``cli_ticket_lane.file_cli_ticket`` files: its row, and its consent.

Moved out of ``file_cli_ticket`` unchanged in what they write, so that function fits the
50-line rule (``cli_ticket_lane`` is over 800 lines and does not grow). The consent is now
staged (``commit=False``) for the one commit that files the ticket. Recorded after that
commit, it lost the race: a heartbeat review was claimed 0.66 s after it was filed, its
consent was then skipped because the ticket was no longer ``assigned``, and the gate
parked it.
"""
from __future__ import annotations

from typing import Any, Optional

from core.models.core import BoardTask

__all__ = ["new_lane_ticket", "stage_lane_consent"]


def new_lane_ticket(
    *, workspace_id: Any, agent_id: int, title: str, prompt: str, source_type: str, source_id: str,
    priority: str, review_mode: str, tags: Optional[list], created_by_type: Optional[str],
    created_by_id: Optional[str], resume_session_id: Optional[str], resume_host_id: Optional[str],
    orchestration_run_id: Any, orchestration_task_id: Any,
) -> BoardTask:
    """The lane's new ticket, built and not added; the caller sets its waiting line."""
    task = BoardTask(
        workspace_id=workspace_id,
        title=title[:255],
        description=prompt,
        priority=priority if priority in ("low", "medium", "high", "urgent") else "medium",
        assigned_agent_id=agent_id,
        status="assigned",
        created_by_type=created_by_type or "system",
        created_by_id=created_by_id or source_type,
        source_type=source_type,
        source_id=source_id,
        review_mode=review_mode,
        tags=list(tags or []),
        orchestration_run_id=orchestration_run_id,
        orchestration_task_id=orchestration_task_id,
    )
    if resume_session_id:
        # Read at claim (cli_host_service) and honoured only by the same host —
        # a transcript lives on one machine. The claim rebuilds runtime_ref.
        task.runtime_ref = {
            "resume_session_id": str(resume_session_id),
            "resume_host_id": str(resume_host_id) if resume_host_id else None,
        }
    return task


def stage_lane_consent(db: Any, workspace_id: Any, task: BoardTask, actor: Optional[str], source_type: str) -> None:
    """The new ticket's consent on the local edition, staged for the commit that files it."""
    # PRD-234 (2026-09-07): the operator's standing schedule is their approval on
    # the local edition — the dispatcher's claim finds an active grant instead of
    # parking the heartbeat behind 'always_ask'. No-op on SaaS. PRD-239: a ticket
    # a human typed for in chat carries that human's consent instead (D16).
    from services.board_consent import WHY_ASKED_IN_CHAT, consent_for_created_ticket, consent_for_lane_ticket
    if actor:
        consent_for_created_ticket(db, workspace_id=workspace_id, task=task, actor=actor, why=WHY_ASKED_IN_CHAT,
                                   commit=False)
    else:
        consent_for_lane_ticket(db, workspace_id=workspace_id, task=task, source_type=source_type, commit=False)
