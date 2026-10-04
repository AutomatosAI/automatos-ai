"""F277 (night 7b): a playbook's address takes the number the list shows, as well as its template id.

To run playbook 102 ("New Cafe Onboarding") from outside the app, the owner had
to read the code: its run address took only the template id (custom-aa07d786),
and ``GET /api/workflow-recipes/102`` answered "Recipe '102' not found". The
routes under ``/api/workflow-recipes/{recipe_id}`` now find the caller's
workspace playbook by either. The template id is tried first, so an address
that is a playbook's template id still means that playbook. A run under it is
found by its execution id (exec-…) or by its number, the same way.

F292 (night 8): the edit (``PUT``) is found by either too. 10 of 10 edits by
number answered 404 "Recipe '113' not found" while reading 113 worked: a timer
switched off from outside the app needed the template id. The edit also takes
the steps as the read gives them (``core.playbook_steps``).
"""
from __future__ import annotations

import functools
import inspect
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.models.core import RecipeExecution, WorkflowTemplate
from core.playbook_steps import steps_as_written

Route = Callable[..., Awaitable[Dict[str, Any]]]

WORKSPACE_PLAYBOOK = "workspace"
# A listed number: ASCII digits that fit the id column (a Postgres integer).
LISTED_NUMBER = re.compile(r"[0-9]{1,10}")
LARGEST_NUMBER = 2_147_483_647


def listed_number(address: Any) -> Optional[int]:
    """The row number ``address`` spells, or None when it spells none."""
    text = str(address or "")
    if not LISTED_NUMBER.fullmatch(text):
        return None
    number = int(text)
    return number if number <= LARGEST_NUMBER else None


def find_playbook(db: Session, workspace_id: Any, address: str) -> Optional[WorkflowTemplate]:
    """The caller's workspace playbook at ``address``: its template id, else its
    number. None when the workspace has neither."""
    mine = db.query(WorkflowTemplate).filter(WorkflowTemplate.owner_type == WORKSPACE_PLAYBOOK,
                                              WorkflowTemplate.workspace_id == workspace_id)
    playbook = mine.filter(WorkflowTemplate.template_id == address).first()
    number = listed_number(address) if playbook is None else None
    return mine.filter(WorkflowTemplate.id == number).first() if number is not None else playbook


def playbook_at(db: Session, workspace_id: Any, address: str) -> WorkflowTemplate:
    """``find_playbook``, or a 404 that names the address."""
    playbook = find_playbook(db, workspace_id, address)
    if playbook is None:
        raise HTTPException(status_code=404, detail=f"Playbook '{address}' not found in this workspace")
    return playbook


def run_of(db: Session, playbook: Any, execution_id: str) -> RecipeExecution:
    """A run of ``playbook`` by its execution id, else by its number; a 404 when
    the playbook has no such run."""
    runs = db.query(RecipeExecution).filter(RecipeExecution.recipe_id == playbook.id,
                                            RecipeExecution.workspace_id == playbook.workspace_id)
    run = runs.filter(RecipeExecution.execution_id == execution_id).first()
    number = listed_number(execution_id) if run is None else None
    run = runs.filter(RecipeExecution.id == number).first() if number is not None else run
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{execution_id}' not found for this playbook")
    return run


def edits_it_by_number_too(update: Route) -> Route:
    """The playbook edit route (``PUT /api/workflow-recipes/{recipe_id}``) at its
    address, by template id or by number, a 404 naming the address when the
    workspace has neither (F292). The steps it is sent are taken as written: each
    with its order, none with the agent summary only a read adds."""
    signature = inspect.signature(update)

    @functools.wraps(update)
    async def edited(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        call = signature.bind(*args, **kwargs)
        given = call.arguments
        playbook = playbook_at(given["db"], given["ctx"].workspace_id, given["recipe_id"])
        data = given["recipe_data"]
        if isinstance(data, dict) and "steps" in data:
            data = {**data, "steps": steps_as_written(data["steps"])}
        given["recipe_id"], given["recipe_data"] = playbook.template_id, data
        return await update(*call.args, **call.kwargs)
    return edited
