"""A product's end users and their memories: ``/api/memory/end-users`` (PRD-274 Wave 3, §5.5).

An end user is ``(workspace, end_user_id)`` (§8-17): the id a product's server key asserts, with
their facts in ``mem:{ws}:end_user:{id}``, inside the workspace's tenant boundary. The product is
responsible for consent (§8-6) and records it here before anything is stored for the person.

| Route | Member | Server key |
| --- | --- | --- |
| ``PUT /{end_user_id}``: record consent and retention | ``knowledge:create`` | ``memory:write`` |
| ``POST /{end_user_id}/memories``: store a fact | ``knowledge:create`` | ``memory:write`` |
| ``GET /{end_user_id}/memories``: list | ``knowledge:read`` | ``memory:read`` |
| ``GET /{end_user_id}/memories/search``: their facts and the workspace's | ``knowledge:read`` | ``memory:read`` |
| ``GET /{end_user_id}/resume``: where they left off (§5.7) | ``knowledge:read`` | ``memory:read`` |
| ``DELETE /{end_user_id}``: erase them (GDPR) | ``knowledge:delete`` | ``memory:write`` |
| ``DELETE /{end_user_id}/memories/{id}``: one fact (a key: only one it wrote) | ``knowledge:delete`` | ``memory:write`` |

Reads leave out anything private to a person and, for a key, anything an audience limits (as
``GET /api/memory`` does). A store needs a consent record first. What a key stores names the key
(``written_by``).

The erase deletes everything the end user left: their namespace, their conversations (a server key's
chat turns for them, ``chats.end_user_id``, with their messages) and those conversations' sessions,
then their consent record. Their turns write nothing anywhere else (``core.security.end_user_turn``).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from api.widget_memory import (
    MEMORY_OFF, MEMORY_STORE_FAILED, MEMORY_UNAVAILABLE, MemoryCreate, _item, _service_or_503, _unavailable,
)
from core.auth.dependencies import RequestContext
from core.auth.scopes import MEMORY_READ, MEMORY_WRITE
from core.auth.server_key_gate import is_server_key, require_workspace_permission_or_server_key
from core.database.database import get_db
from core.models.end_users import END_USER_ID_MAX
from core.security.memory_audience import audience_of
from modules.memory.injection_filter import WRITTEN_BY_KEY, visible_to_viewer, without_server_keys

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/memory/end-users", tags=["End-user memory"])

END_USER_ID = Path(..., pattern=rf"^[A-Za-z0-9._:@-]{{1,{END_USER_ID_MAX}}}$",
                   description="The product's own id for the person (letters, digits and . _ : @ -)")
RETENTION_MAX_DAYS = 3650
NO_CONSENT = "Record this end user's consent first: PUT /api/memory/end-users/{end_user_id}."
NO_SUCH_END_USER = "No end user with that id in this workspace."
BEING_ERASED = "This end user is being erased: DELETE them again to finish, then record their consent anew."


class EndUserConsent(BaseModel):
    """The product's consent record for one end user."""

    consent: bool = Field(..., description="True: the product has the person's consent to remember them")
    retention_days: Optional[int] = Field(None, ge=1, le=RETENTION_MAX_DAYS,
                                          description="Days their memories are kept; null: until erased. Left "
                                                      "out: unchanged for an end user who exists; for a new end "
                                                      "user of a product's workspace, the product's default")


def _namespace(ctx: RequestContext, end_user_id: str) -> str:
    from modules.memory.unified_memory_service import MemoryNamespace

    return MemoryNamespace(str(ctx.workspace_id)).end_user(end_user_id)


def _record(db: Session, ctx: RequestContext, end_user_id: str) -> Any:
    from core.models.end_users import EndUser

    return (db.query(EndUser)
            .filter(EndUser.workspace_id == ctx.workspace_id, EndUser.end_user_id == end_user_id).first())


def _known(db: Session, ctx: RequestContext, end_user_id: str) -> Any:
    row = _record(db, ctx, end_user_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NO_SUCH_END_USER)
    return row


def _not_limited_for(ctx: RequestContext, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For a key, nothing an audience limits (as ``_readable``); private facts are left to the viewer rule,
    so an end user's own private facts stay in their resume."""
    return [m for m in rows if not (is_server_key(ctx) and audience_of(m))]


def _readable(ctx: RequestContext, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Nothing private to a person; for a key, nothing an audience limits either."""
    return [m for m in rows if visible_to_viewer(m, None) and not (is_server_key(ctx) and audience_of(m))]


def _retention_for(db: Session, ctx: RequestContext, body: EndUserConsent) -> Optional[int]:
    """A new end user's retention: what the request sent (``null`` included), else their workspace's product's
    default. An end user who already exists never takes it (``record_consent``)."""
    if "retention_days" in body.model_fields_set:
        return body.retention_days
    from services.products import default_retention_of

    return default_retention_of(db, ctx.workspace_id)


@router.put("/{end_user_id}")
def record_consent(body: EndUserConsent, end_user_id: str = END_USER_ID,
                   ctx: RequestContext = Depends(require_workspace_permission_or_server_key("knowledge:create",
                                                                                            MEMORY_WRITE)),
                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Records (or updates) the product's consent and retention for one end user. A new end user of a product's
    workspace recorded with no ``retention_days`` gets the product's default (PRD-274 W5e, decision 28). An
    update changes their retention only when it sends ``retention_days`` (an explicit ``null``: kept until
    erased), so a consent refresh in the same shape keeps what they were told."""
    from core.models.end_users import EndUser

    if not body.consent:
        raise HTTPException(status_code=422, detail="consent must be true; to forget the person, DELETE them.")
    row = _record(db, ctx, end_user_id)
    new = row is None
    row = row or EndUser(workspace_id=ctx.workspace_id, end_user_id=end_user_id)
    if row.erasing_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=BEING_ERASED)
    if new:
        row.retention_days = _retention_for(db, ctx, body)
    elif "retention_days" in body.model_fields_set:
        row.retention_days = body.retention_days
    row.api_key_id = ctx.api_key_id if is_server_key(ctx) else row.api_key_id
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"end_user_id": row.end_user_id, "retention_days": row.retention_days,
            "consent_recorded_at": row.consent_recorded_at}


@router.post("/{end_user_id}/memories", status_code=201)
async def store_end_user_memory(body: MemoryCreate, end_user_id: str = END_USER_ID,
                                ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                                    "knowledge:create", MEMORY_WRITE)),
                                db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Stores one fact about the end user; their consent must be on record, and the write is made as
    theirs (``end_user_turn``): tagged with their subject, and taken back if their erase starts meanwhile."""
    from core.models.end_users import consent_record_id
    from core.security.end_user_turn import ERASED, EndUser, end_user_turn

    record_id = await run_in_threadpool(consent_record_id, db, ctx.workspace_id, end_user_id)
    if record_id is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=NO_CONSENT)
    metadata = {**without_server_keys(body.metadata), "end_user_id": end_user_id}
    if body.tags:
        metadata["tags"] = body.tags
    if is_server_key(ctx):
        metadata[WRITTEN_BY_KEY] = ctx.user.id
    service = _service_or_503()
    try:
        with end_user_turn(EndUser(str(ctx.workspace_id), end_user_id, record_id)):
            stored = await service.store_long_term_messages(
                user_id=_namespace(ctx, end_user_id), messages=[{"role": "user", "content": body.content}],
                metadata=metadata, workspace_id=str(ctx.workspace_id))
    except Exception as exc:
        logger.exception("[end_user_memory] store raised for workspace %s", ctx.workspace_id)
        raise _unavailable(MEMORY_STORE_FAILED) from exc
    stored = stored if isinstance(stored, dict) else {}
    if stored.get("skipped") == ERASED:  # their erase started between the check above and the write
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=BEING_ERASED)
    if not (stored.get("success") and stored.get("id")):
        logger.error("[end_user_memory] store refused for workspace %s: %s", ctx.workspace_id, stored.get("error"))
        raise _unavailable(MEMORY_OFF if stored.get("skipped") else MEMORY_STORE_FAILED)
    return {"id": str(stored["id"]), "end_user_id": end_user_id, "content": body.content}


@router.get("/{end_user_id}/memories")
async def list_end_user_memories(end_user_id: str = END_USER_ID, limit: int = Query(50, ge=1, le=200),
                                 ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                                     "knowledge:read", MEMORY_READ)),
                                 db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The end user's own facts."""
    await run_in_threadpool(_known, db, ctx, end_user_id)
    service = _service_or_503()
    try:
        rows = await service.get_all_memories_scoped(user_id=_namespace(ctx, end_user_id), limit=limit,
                                                     workspace_id=str(ctx.workspace_id))
    except Exception as exc:
        logger.exception("[end_user_memory] list failed for workspace %s", ctx.workspace_id)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc
    items = [_item(m, str(ctx.workspace_id)) for m in _readable(ctx, rows)]
    return {"end_user_id": end_user_id, "memories": items, "total": len(items)}


@router.get("/{end_user_id}/memories/search")
async def search_end_user_memories(q: str = Query(..., min_length=1), end_user_id: str = END_USER_ID,
                                   limit: int = Query(20, ge=1, le=100),
                                   ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                                       "knowledge:read", MEMORY_READ)),
                                   db: Session = Depends(get_db)) -> Dict[str, Any]:
    """The end user's facts and the workspace's, never another end user's (§5.5)."""
    from modules.memory.unified_memory_service import MemoryNamespace

    await run_in_threadpool(_known, db, ctx, end_user_id)
    service = _service_or_503()
    try:
        rows = await service._durable.search(  # the store's own search: the namespace and the workspace tier
            query=q, user_id=_namespace(ctx, end_user_id), limit=limit, workspace_id=str(ctx.workspace_id),
            also_namespaces=(MemoryNamespace(str(ctx.workspace_id)).workspace(),))
    except Exception as exc:
        logger.exception("[end_user_memory] search failed for workspace %s", ctx.workspace_id)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc
    items = [_item(m, str(ctx.workspace_id), with_score=True) for m in _readable(ctx, rows)]
    return {"end_user_id": end_user_id, "query": q, "results": items, "total": len(items)}


@router.get("/{end_user_id}/resume")
async def resume_end_user(end_user_id: str = END_USER_ID,
                          ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                              "knowledge:read", MEMORY_READ)),
                          db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Where the end user left off (§5.7, "where were we"): their recent conversations with each one's
    checkpoint, and the decisions, open loops and next steps in their facts and the workspace's, never
    another end user's. The workspace's are read as the other routes read them (``_readable``); for a key,
    neither theirs nor the workspace's includes anything an audience limits."""
    from modules.memory.resume_context import build_end_user_resume_payload

    row = await run_in_threadpool(_known, db, ctx, end_user_id)
    if row.erasing_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=BEING_ERASED)
    payload = await build_end_user_resume_payload(db, workspace_id=str(ctx.workspace_id), end_user_id=end_user_id,
                                                  readable=lambda rows: _readable(ctx, rows),
                                                  readable_theirs=lambda rows: _not_limited_for(ctx, rows))
    return {"end_user_id": end_user_id, **payload}


@router.delete("/{end_user_id}")
async def erase_end_user(end_user_id: str = END_USER_ID,
                         ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                             "knowledge:delete", MEMORY_WRITE)),
                         db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Erases the end user (GDPR): every memory in their namespace, their conversations' sessions, their
    conversations, then their consent record. Any step that fails is a 503 that keeps the record and its
    erase mark, so the next erase finishes it; nothing is reported erased until all of it is."""
    row = await run_in_threadpool(_known, db, ctx, end_user_id)
    service = _service_or_503()
    await run_in_threadpool(_mark_erasing, db, row)  # first: no new turn, store or write for them from here
    try:
        erased = await service._durable.erase_namespace(_namespace(ctx, end_user_id))
    except Exception as exc:
        logger.exception("[end_user_memory] erase failed for workspace %s", ctx.workspace_id)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc

    chat_ids = await run_in_threadpool(_their_chat_ids, db, ctx, end_user_id)
    if not await _forget_their_sessions(service, str(ctx.workspace_id), chat_ids):
        raise _unavailable(MEMORY_UNAVAILABLE)  # their chats, record and mark stay for the next erase

    def _forget() -> int:
        deleted = _erase_their_chats(db, ctx, end_user_id)
        db.delete(row)
        db.commit()
        return deleted

    chats_deleted = await run_in_threadpool(_forget)
    return {"end_user_id": end_user_id, "erased": True, "memories_deleted": erased, "chats_deleted": chats_deleted}


def _mark_erasing(db: Session, row: Any) -> None:
    """The end user's erase has started (``end_users.erasing_at``): their consent no longer counts, and a
    turn still running writes nothing more for them (``core.security.end_user_turn``). Kept on a failed
    erase, which the next one finishes."""
    from datetime import datetime, timezone

    if row.erasing_at is None:
        row.erasing_at = datetime.now(timezone.utc)
        db.commit()


def _their_chats(db: Session, ctx: RequestContext, end_user_id: str) -> Any:
    from core.models.core import Chat

    return db.query(Chat).filter(Chat.workspace_id == ctx.workspace_id, Chat.end_user_id == end_user_id)


def _their_chat_ids(db: Session, ctx: RequestContext, end_user_id: str) -> List[str]:
    """The ids of the end user's conversations in this workspace, for their sessions."""
    from core.models.core import Chat

    return [str(row.id) for row in _their_chats(db, ctx, end_user_id).with_entities(Chat.id)]


def _erase_their_chats(db: Session, ctx: RequestContext, end_user_id: str) -> int:
    """Deletes the end user's conversations in this workspace; their messages, votes and pinned
    documents go with them (each foreign key cascades). How many were deleted. A chat a turn opened
    after the erase mark is deleted by that turn itself (``api.chat_server_key``)."""
    return _their_chats(db, ctx, end_user_id).delete(synchronize_session=False)


async def _forget_their_sessions(service: Any, workspace_id: str, chat_ids: List[str]) -> bool:
    """Deletes the L1 sessions (Redis) of the end user's conversations, before the conversations go, so a
    failure leaves their ids for the next erase. False, logged, when Redis can't be reached: a client that
    can't be made while Redis is configured is a failure, since their sessions may be there. With no Redis
    configured, no session was ever stored."""
    from config import config

    if not chat_ids:
        return True
    redis_client = service._get_redis()
    if redis_client is None:
        if config.REDIS_URL:
            logger.error("[end_user_memory] Redis is configured but unavailable; sessions not cleared for workspace %s",
                         workspace_id)
            return False
        return True
    keys = [service.namespace(workspace_id).session(chat_id) for chat_id in chat_ids]
    try:
        await asyncio.to_thread(redis_client.get_redis().delete, *keys)
    except Exception:
        logger.exception("[end_user_memory] erased conversations' sessions not cleared for workspace %s", workspace_id)
        return False
    return True


@router.delete("/{end_user_id}/memories/{memory_id}")
async def delete_end_user_memory(memory_id: str, end_user_id: str = END_USER_ID,
                                 ctx: RequestContext = Depends(require_workspace_permission_or_server_key(
                                     "knowledge:delete", MEMORY_WRITE)),
                                 db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Deletes one of the end user's facts. A member may delete any; a key only one it wrote
    (``written_by``), so a product can never delete what the owner or another key stored."""
    await run_in_threadpool(_known, db, ctx, end_user_id)
    service = _service_or_503()
    namespace = _namespace(ctx, end_user_id)
    try:
        if is_server_key(ctx) and not await _written_by_this_key(service, ctx, namespace, memory_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_THIS_KEYS)
        deleted = await service._durable.delete([memory_id], user_id=namespace, workspace_id=str(ctx.workspace_id))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("[end_user_memory] delete failed for workspace %s", ctx.workspace_id)
        raise _unavailable(MEMORY_UNAVAILABLE) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=NOT_THIS_KEYS)
    return {"end_user_id": end_user_id, "id": memory_id, "deleted": True}


NOT_THIS_KEYS = "No such memory of this end user that this caller may delete."


async def _written_by_this_key(service: Any, ctx: RequestContext, namespace: str, memory_id: str) -> bool:
    """Whether the end user's fact ``memory_id`` names this key as its writer. The fact is read by
    its id, however many the end user has; a store that can't be read raises (the route's 503)."""
    row = await service._durable.get_one(memory_id, namespace)
    metadata = (row or {}).get("metadata") or {}
    return row is not None and metadata.get(WRITTEN_BY_KEY) == ctx.user.id
