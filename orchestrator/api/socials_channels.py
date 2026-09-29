"""
Socials channels (PRD-251 D8, S3.2)
===================================

``GET /api/socials/channels``: the workspace's connected social channels, each with
the post kinds the channel registry resolves for it (``modules/socials/capabilities.py``:
the seeded adapters, then the generic one), whether each kind is available and why
not, whether it needs public storage (D9), whether the channel is verified, and its
setup note. The composer offers only the available kinds (US-209).

This router has no prefix and no gate of its own: ``api/socials.py`` includes it in
the Socials router, whose ``require_socials_enabled`` answers 404 unless both
switches are on (D1). Never mount it in the app directly.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from modules.socials.capabilities import social_channels

router = APIRouter()


@router.get("/channels")
def list_social_channels(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """The connected channels and what each can post. A plain ``def``: FastAPI runs
    its synchronous database reads in the threadpool (F105)."""
    return [channel.to_dict() for channel in social_channels(db, ctx.workspace_id)]
