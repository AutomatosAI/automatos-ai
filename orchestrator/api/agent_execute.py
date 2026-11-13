"""``POST /api/agents/{agent_id}/execute``: start a run of one agent (PRD-274 Wave 1, §8-8).

The run is a board ticket assigned to the agent, so it gets what every board run gets: the
dispatch loop claims and runs it, the spend guard and approval gate apply, a session agent's
host picks it up, and the ticket records its status, result, error and timings. The route
answers at once with the ticket's id; ``GET /api/v1/tasks/{task_id}`` reads the run as it goes.
It replaces a route that answered with a made-up execution id and never ran anything.

A ``callback_url`` (https) is checked before anything is filed, and the run's ending is POSTed
there, signed, when the ticket ends (``services.run_callbacks``).

An ``Idempotency-Key`` header makes a retry safe. The key is kept as the ticket's ``source_id``,
and a partial unique index files one ticket per key in a workspace, even when two requests race.
Repeating the request (same agent, input and callback URL) answers 200 with the run it started and
``Idempotent-Replayed: true``. The same key on a different request is a 409. A key never expires:
it names its ticket for good.

A signed-in member whose role grants ``agents:execute`` may call it, and so may a product's
server key listing ``agents:execute`` (``core.auth.server_key_gate``, PRD-274 §5.2); the run is then
the key's workspace's, and its ticket says a key filed it.

A plain ``def``: FastAPI runs it in its threadpool, off the event loop.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Callable, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.scopes import AGENTS_EXECUTE
from core.auth.server_key_gate import require_workspace_permission_or_server_key
from core.database.database import get_db
from services.run_callbacks import URL_MAX, CallbackUrlRefused, check_callback_url, new_callback

router = APIRouter(prefix="/api/agents", tags=["agents"])

API_SOURCE_TYPE = "api"
INPUT_MAX = 20_000
TITLE_MAX = 200
TITLE_FROM_INPUT = 100
TASK_ROUTE = "/api/v1/tasks/{task_id}"
IDEMPOTENCY_HEADER = "Idempotency-Key"
REPLAYED_HEADER = "Idempotent-Replayed"
IDEMPOTENCY_KEY_MAX = 255  # board_tasks.source_id
IDEMPOTENCY_KEY = re.compile(rf"[\x21-\x7e]{{1,{IDEMPOTENCY_KEY_MAX}}}")


class AgentRunRequest(BaseModel):
    """What the agent is asked to do."""

    model_config = ConfigDict(extra="forbid")

    input: str = Field(min_length=1, max_length=INPUT_MAX, description="What the agent should do.")
    title: Optional[str] = Field(default=None, min_length=1, max_length=TITLE_MAX,
                                 description="The ticket's title; the input's first line when left out.")
    callback_url: Optional[str] = Field(default=None, min_length=1, max_length=URL_MAX, description=(
        "An https URL the run's ending is POSTed to, signed with the workspace's signing secret."))


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


@router.post("/{agent_id}/execute", status_code=status.HTTP_202_ACCEPTED)
def execute_agent(agent_id: int, body: AgentRunRequest, response: Response,
                  idempotency_key: Annotated[Optional[str], Header(alias=IDEMPOTENCY_HEADER)] = None,
                  ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                      "agents:execute", AGENTS_EXECUTE)),
                  db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Files the run as a ticket assigned to the agent and answers with its id; the board runs it.
    A repeat of a request with the same ``Idempotency-Key`` answers with the run it started."""
    from api.board_task_create import file_ticket, new_ticket_columns

    key = _checked_key(idempotency_key)
    earlier = _earlier_run(db, ctx.workspace_id, key)
    if earlier is not None:
        return _replayed(db, earlier, agent_id, body, response)
    agent = _runnable_agent(db, ctx.workspace_id, agent_id)
    attach = _callback(ctx.workspace_id, body.callback_url)
    columns = new_ticket_columns(db, ctx.workspace_id, {
        "title": _title(body), "raw_prompt": body.input, "assigned_agent_id": agent.id})
    try:
        task = file_ticket(db, ctx, {**columns, "source_type": API_SOURCE_TYPE, "source_id": key}, attach=attach)
    except IntegrityError:  # a request with the same key filed it first
        db.rollback()
        earlier = _earlier_run(db, ctx.workspace_id, key)
        if earlier is None:
            raise
        return _replayed(db, earlier, agent_id, body, response)
    return _answer(task, agent.name, body.callback_url)


def _answer(task: Any, agent_name: str, callback_url: Optional[str]) -> Dict[str, Any]:
    return {"task_id": task.id, "agent_id": task.assigned_agent_id, "agent_name": agent_name, "status": task.status,
            "task_url": TASK_ROUTE.format(task_id=task.id), "callback_url": callback_url}


def _checked_key(raw: Optional[str]) -> Optional[str]:
    """The request's ``Idempotency-Key``: 1–255 visible ASCII characters, or none."""
    if raw is None:
        return None
    if not IDEMPOTENCY_KEY.fullmatch(raw):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=(
            f"{IDEMPOTENCY_HEADER} must be 1 to {IDEMPOTENCY_KEY_MAX} visible ASCII characters"))
    return raw


def _earlier_run(db: Session, workspace_id: Any, key: Optional[str]) -> Any:
    from core.models.core import BoardTask

    if key is None:
        return None
    return (db.query(BoardTask)
            .filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == API_SOURCE_TYPE,
                    BoardTask.source_id == key).first())


def _replayed(db: Session, task: Any, agent_id: int, body: AgentRunRequest, response: Response) -> Dict[str, Any]:
    """The run an earlier request with this key started; a different request under the key is a 409."""
    from core.models import Agent
    from core.models.run_callbacks import RunCallback

    callback = db.query(RunCallback.url).filter(RunCallback.task_id == task.id).scalar()
    if (task.assigned_agent_id, task.raw_prompt, callback) != (agent_id, body.input, body.callback_url):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=(
            f"This {IDEMPOTENCY_HEADER} was used for a different request (ticket {task.id})"))
    response.status_code = status.HTTP_200_OK
    response.headers[REPLAYED_HEADER] = "true"
    agent_name = db.query(Agent.name).filter(Agent.id == task.assigned_agent_id).scalar()
    return {**_answer(task, agent_name, callback), "replayed": True}


def _callback(workspace_id: Any, url: Optional[str]) -> Optional[Callable[[Session, Any], None]]:
    """The run's callback, once its URL is checked; a URL the platform won't POST to is a 422."""
    if url is None:
        return None
    try:
        check_callback_url(url)
    except CallbackUrlRefused as refused:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(refused)) from None
    return new_callback(workspace_id, url)
