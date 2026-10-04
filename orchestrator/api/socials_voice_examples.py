"""
Socials voice examples (PRD-251C C8; US-C406)
=============================================

* ``GET /api/socials/voice-examples``: the workspace's voice examples, newest first: each the
  copy Auto drafted and the copy a person approved instead (``modules/socials/voice_examples.py``).
  The composer reads the same ones.
* ``DELETE /api/socials/voice-examples/{example_id}``: an owner or admin (``workspace:manage``,
  as for the brand kit) removes one; it is never used again. Another workspace's is a 404.

``api/socials_templates.py`` includes this router, so it takes the Socials router's prefix and
gate. Plain ``def`` routes (F105).
"""
from __future__ import annotations

from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.socials import voice_examples

router = APIRouter()
CAN_MANAGE = Depends(require_workspace_permission("workspace:manage"))
NOT_FOUND = "No such voice example in this workspace."


@router.get("/voice-examples")
def list_social_voice_examples(
    db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid),
) -> Dict[str, Any]:
    """The workspace's voice examples, newest first: ``{examples: [{id, post_id, draft, approved, created_at}]}``."""
    return {"examples": [row.to_dict() for row in voice_examples.examples(db, ctx.workspace_id)]}


@router.delete("/voice-examples/{example_id}", status_code=204, dependencies=[CAN_MANAGE])
def delete_social_voice_example(
    example_id: UUID, db: Session = Depends(get_db), ctx: RequestContext = Depends(get_request_context_hybrid),
) -> None:
    """Remove one example: the composer never reads it again."""
    if not voice_examples.remove(db, ctx.workspace_id, example_id):
        raise HTTPException(status_code=404, detail=NOT_FOUND)
