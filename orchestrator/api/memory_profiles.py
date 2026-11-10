"""``GET /api/v1/memory/profiles``: the memory profiles a workspace can pick (Wave 4, §5.6).

The platform's seeded profiles and the workspace's own, a workspace's own replacing a seeded one
with the same slug. Each says what distillation keeps as a fact with it
(``modules.memory.memory_profiles``); an agent picks one in ``configuration.memory_profile``, a
workspace in ``settings.memory_profile``. Any member of the workspace can read the list, as with
the rest of the memory explorer's reads.

PRD-274 §8-7: a workspace's managers (``workspace:manage``) create, change and delete the
workspace's own profiles. Seeded profiles stay as the platform ships them; a workspace overrides
one by creating its own with the same slug, and deleting that brings the seeded one back.
``general`` is reserved: it means today's distillation prompt and is never looked up. Every
example's type is one of the platform's fact types: a profile changes what is kept, not how it
is classified. The routes are plain ``def``: FastAPI runs them in its threadpool, off the event
loop.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import BaseModel, Field, StringConstraints, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.memory.memory_profiles import GENERAL, SLUG_MAX
from modules.memory.write_contract import MEMORY_FACT_TYPES

MANAGE = "workspace:manage"
SLUG_PATTERN = rf"^[a-z0-9][a-z0-9-]{{0,{SLUG_MAX - 1}}}$"
NAME_MAX = 255
DESCRIPTION_MAX = 1000
FOCUS_MAX = 2000
LIST_ITEMS_MAX = 20
LIST_ITEM_MAX = 200
EXAMPLES_MAX = 10
EXAMPLE_FACT_MAX = 500

ListItem = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=LIST_ITEM_MAX)]

router = APIRouter(prefix="/api/v1/memory", tags=["Memory Explorer"])


def _as_item(row: Any) -> Dict[str, Any]:
    return {"slug": row.slug, "name": row.name, "description": row.description, "focus": row.focus,
            "keep": list(row.keep or []), "skip": list(row.skip or []), "examples": list(row.examples or []),
            "seeded": row.workspace_id is None}


class ProfileExample(BaseModel):
    """A worked example: a fact as distillation would return it."""

    fact: str = Field(min_length=1, max_length=EXAMPLE_FACT_MAX)
    type: str

    @field_validator("type")
    @classmethod
    def _a_platform_fact_type(cls, value: str) -> str:
        if value not in MEMORY_FACT_TYPES:
            raise ValueError(f"type must be one of {sorted(MEMORY_FACT_TYPES)}")
        return value


class ProfileFields(BaseModel):
    """What a workspace's own profile says; the body of an update."""

    name: str = Field(min_length=1, max_length=NAME_MAX)
    description: str = Field(default="", max_length=DESCRIPTION_MAX)
    focus: str = Field(min_length=1, max_length=FOCUS_MAX)
    keep: List[ListItem] = Field(default_factory=list, max_length=LIST_ITEMS_MAX)
    skip: List[ListItem] = Field(default_factory=list, max_length=LIST_ITEMS_MAX)
    examples: List[ProfileExample] = Field(default_factory=list, max_length=EXAMPLES_MAX)

    def as_columns(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "focus": self.focus,
                "keep": list(self.keep), "skip": list(self.skip),
                "examples": [e.model_dump() for e in self.examples]}


class ProfileCreate(ProfileFields):
    """A new profile of the workspace's own, under ``slug``."""

    slug: str = Field(pattern=SLUG_PATTERN)

    @field_validator("slug")
    @classmethod
    def _not_general(cls, value: str) -> str:
        if value == GENERAL:
            raise ValueError(f"{GENERAL!r} is reserved: it is the distillation prompt without a profile")
        return value


def _own_profile(db: Session, workspace_id: Any, slug: str) -> Optional[Any]:
    from core.models.memory_profiles import MemoryProfile

    return (db.query(MemoryProfile)
            .filter(MemoryProfile.workspace_id == workspace_id, MemoryProfile.slug == slug).first())


def _own_or_404(db: Session, workspace_id: Any, slug: str) -> Any:
    row = _own_profile(db, workspace_id, slug)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"This workspace has no profile of its own called {slug!r}")
    return row


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


@router.post("/profiles", status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_workspace_permission(MANAGE))])
def create_memory_profile(body: ProfileCreate, ctx: RequestContext = Depends(get_request_context_hybrid),
                          db: Session = Depends(get_db)) -> Dict[str, Any]:
    """A profile of the workspace's own; one with a seeded profile's slug overrides it."""
    from core.models.memory_profiles import MemoryProfile

    conflict = HTTPException(status_code=status.HTTP_409_CONFLICT,
                             detail=f"This workspace already has a profile called {body.slug!r}")
    if _own_profile(db, ctx.workspace_id, body.slug) is not None:
        raise conflict
    row = MemoryProfile(workspace_id=ctx.workspace_id, slug=body.slug, **body.as_columns())
    db.add(row)
    try:
        db.commit()
    except IntegrityError:  # created by another request since the check above
        db.rollback()
        raise conflict from None
    db.refresh(row)
    return _as_item(row)


@router.put("/profiles/{slug}", dependencies=[Depends(require_workspace_permission(MANAGE))])
def update_memory_profile(slug: str, body: ProfileFields, ctx: RequestContext = Depends(get_request_context_hybrid),
                          db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Replaces what one of the workspace's own profiles says. Seeded profiles can't be changed."""
    row = _own_or_404(db, ctx.workspace_id, slug)
    for column, value in body.as_columns().items():
        setattr(row, column, value)
    db.commit()
    db.refresh(row)
    return _as_item(row)


@router.delete("/profiles/{slug}", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(require_workspace_permission(MANAGE))])
def delete_memory_profile(slug: str, ctx: RequestContext = Depends(get_request_context_hybrid),
                          db: Session = Depends(get_db)) -> Response:
    """Deletes one of the workspace's own profiles; a seeded one with its slug applies again."""
    db.delete(_own_or_404(db, ctx.workspace_id, slug))
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)
