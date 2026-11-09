"""``POST /api/agents/{agent_id}/execute``: start a run of one agent (PRD-274 Wave 1, §8-8).

The run is a board ticket assigned to the agent, so it gets what every board run gets: the
dispatch loop claims and runs it, the spend guard and approval gate apply, a session agent's
host picks it up, and the ticket records its status, result, error and timings. The route
answers at once with the ticket's id; ``GET /api/v1/tasks/{task_id}`` reads the run as it goes.
It replaces a route that answered with a made-up execution id and never ran anything.

A plain ``def``: FastAPI runs it in its threadpool, off the event loop.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db

router = APIRouter(prefix="/api/agents", tags=["agents"])

API_SOURCE_TYPE = "api"
INPUT_MAX = 20_000
TITLE_MAX = 200
TITLE_FROM_INPUT = 100
TASK_ROUTE = "/api/v1/tasks/{task_id}"


class AgentRunRequest(BaseModel):
    """What the agent is asked to do."""

    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1, max_length=INPUT_MAX, description="What the agent should do.")
    title: Optional[str] = Field(default=None, min_length=1, max_length=TITLE_MAX,
                                 description="The ticket's title; the input's first line when left out.")


def _title(body: AgentRunRequest) -> str:
    if body.title and body.title.strip():
        return body.title.strip()
    first_line = body.input.strip().splitlines()[0] if body.input.strip() else ""
    return first_line[:TITLE_FROM_INPUT] or "Run"


def _runnable_agent(db: Session, workspace_id: Any, agent_id: int) -> Any:
    from core.models import Agent

    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if agent.status != "active":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Agent must be active to execute")
    return agent


@router.post("/{agent_id}/execute", status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_workspace_permission("agents:execute"))])
def execute_agent(agent_id: int, body: AgentRunRequest, ctx: RequestContext = Depends(get_request_context_hybrid),
                  db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Files the run as a ticket assigned to the agent and answers with its id; the board runs it."""
    from api.board_task_create import new_ticket_columns
    from api.board_tasks import file_ticket

    agent = _runnable_agent(db, ctx.workspace_id, agent_id)
    columns = new_ticket_columns(db, ctx.workspace_id, {
        "title": _title(body), "raw_prompt": body.input, "assigned_agent_id": agent.id})
    task = file_ticket(db, ctx, {**columns, "source_type": API_SOURCE_TYPE})
    return {"task_id": task.id, "agent_id": agent.id, "agent_name": agent.name, "status": task.status,
            "task_url": TASK_ROUTE.format(task_id=task.id)}
