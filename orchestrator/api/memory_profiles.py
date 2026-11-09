"""``GET /api/v1/memory/profiles``: the memory profiles a workspace can pick (Wave 4, §5.6).

The platform's seeded profiles and the workspace's own (none until the PRD's §8-7 says owners
may define them), a workspace's own replacing a seeded one with the same slug. Each says what
distillation keeps as a fact with it (``modules.memory.memory_profiles``); an agent picks one in
``configuration.memory_profile``, a workspace in ``settings.memory_profile``. Any member of the
workspace can read the list, as with the rest of the memory explorer's reads. A plain ``def``:
FastAPI runs it in its threadpool, off the event loop.
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db

router = APIRouter(prefix="/api/v1/memory", tags=["Memory Explorer"])


def _as_item(row: Any) -> Dict[str, Any]:
    return {"slug": row.slug, "name": row.name, "description": row.description, "focus": row.focus,
            "keep": list(row.keep or []), "skip": list(row.skip or []), "examples": list(row.examples or []),
            "seeded": row.workspace_id is None}


@router.get("/profiles")
def list_memory_profiles(ctx: RequestContext = Depends(get_request_context_hybrid),
                         db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The profiles this workspace can pick, by slug; the dial says whether they apply yet."""
    from config import config
    from core.models.memory_profiles import MemoryProfile

    rows = (db.query(MemoryProfile)
            .filter((MemoryProfile.workspace_id == ctx.workspace_id) | MemoryProfile.workspace_id.is_(None))
            .order_by(MemoryProfile.slug).all())
    by_slug: Dict[str, Any] = {}
    for row in rows:  # a workspace's own replaces the seeded one with its slug
        if row.slug not in by_slug or row.workspace_id is not None:
            by_slug[row.slug] = row
    profiles: List[Dict[str, Any]] = [_as_item(row) for row in by_slug.values()]
    return {"profiles": profiles, "enabled": bool(getattr(config, "MEMORY_PROFILES_ENABLED", False))}
