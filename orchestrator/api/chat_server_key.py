"""A product's headless chat turn on ``POST /api/chat``, through a server key (PRD-274 Wave 1).

A chat turn runs as a person: their ``users.id`` owns the chat, decides which private memories the
turn reads and writes, and whether an admin-only tool may run (the driving user, F145). A server key
has no person, and ``api.chat.get_user_id`` falls back to user 1, the workspace's owner. So a key
never takes that path:

* **It runs as its own service user.** One non-login ``users`` row per key, made on its first turn
  (``sdk_api_keys.service_user_id``): no Clerk id, an address on the reserved ``.invalid`` domain,
  never a workspace member. Its chats are its own; a chat id from anyone else is refused.
* **It chats only with a lean agent** of its own workspace (``consumers.chatbot.lean_turn``), named
  in ``agentId``. Auto, mission, plan and spoken modes are a person's.
* **A model it names** must be on the key's ``model_allow_list`` and on the agent's (§8-5). A key
  with no list names none; the agent answers on its own model.
* **The end user it names** (``endUserId``, Wave 3, §5.5) must have their consent on record
  (``/api/memory/end-users``), or the turn is refused (409). Their turn runs as theirs
  (``core.security.end_user_turn``): what it writes and recalls is their own and the workspace's.
  Their conversation is tagged with them (``chats.end_user_id``), and only turns for them continue
  it; a key's turn for no end user continues none of theirs. A person's turn names no end user.

Three decorators wire this into ``api/chat.py`` without growing it: ``get_user_id``,
``_checked_lean_request`` and ``_open_chat`` are wrapped; everyone else passes straight through.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import replace
from typing import Any, Callable, List, Mapping, Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError

from core.auth.server_key_gate import is_server_key

logger = logging.getLogger(__name__)

SERVICE_USER_NAME = "sdk-{key_id}"
SERVICE_USER_EMAIL = "sdk-{key_id}@keys.automatos.invalid"  # .invalid never resolves: no mail, no sign-in
SERVICE_USER_DISPLAY = "Server key: {name}"
NAME_A_LEAN_AGENT = "A server key chats with a lean agent of its workspace: name one in agentId."
NOT_A_KEYS_MODE = "missionMode, planMode and spokenMode are a person's, not a server key's."
NOT_A_PERSONS = "endUserId is a server key's: a person's chat turn is their own."
NOT_THIS_END_USERS_CHAT = "This conversation is not this end user's: start a new one for them."


def service_user_for(db: Any, key_id: Any) -> int:
    """The ``users.id`` the key's chat turns run as, made on its first turn."""
    from core.models.core import User
    from core.models.sdk_api_keys import SdkApiKey

    key = db.get(SdkApiKey, key_id, with_for_update=True)
    if key.service_user_id is not None:
        return key.service_user_id
    user = User(username=SERVICE_USER_NAME.format(key_id=key.id), email=SERVICE_USER_EMAIL.format(key_id=key.id),
                name=SERVICE_USER_DISPLAY.format(name=key.name)[:255], is_active=True)
    db.add(user)
    try:
        db.flush()
    except IntegrityError:  # another turn of this key made it first
        db.rollback()
        return db.get(SdkApiKey, key_id).service_user_id
    key.service_user_id = user.id
    db.commit()
    logger.info("[chat_server_key] key %s runs its chat turns as users.id %s", key.id, user.id)
    return user.id


def acts_as_its_service_user(get_user_id: Callable[..., int]) -> Callable[..., int]:
    """Wraps ``api.chat.get_user_id``: a server key is its service user, never user 1."""

    @functools.wraps(get_user_id)
    def resolved(db: Any, ctx: Any = None) -> int:
        if is_server_key(ctx):
            return service_user_for(db, ctx.api_key_id)
        return get_user_id(db, ctx)

    return resolved


def what_a_server_key_may_ask(checked: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``api.chat._checked_lean_request``: a key's turn is refused (422) before anything is
    saved unless it names a lean agent of its workspace, in no person's mode, and any model is on
    the key's list as well as the agent's."""

    @functools.wraps(checked)
    def wrapped(db: Any, ctx: Any, request: Any) -> Any:
        if not is_server_key(ctx):
            if _end_user_id(request):
                raise _refuse(NOT_A_PERSONS)
            return checked(db, ctx, request)
        _refuse_unless_allowed(db, ctx, request)
        return _for_its_end_user(db, ctx, request, checked(db, ctx, request))

    return wrapped


def _end_user_id(request: Any) -> Optional[str]:
    """The end user the request names (``ChatRequest.endUserId``); None when it names none."""
    return getattr(request, "endUserId", None) or None


def _for_its_end_user(db: Any, ctx: Any, request: Any, lean: Any) -> Any:
    """``lean`` naming the turn's end user, when the key named one whose consent is on record (409
    otherwise: nothing is saved or recalled for a person the product has no consent for)."""
    end_user_id = _end_user_id(request)
    if not end_user_id:
        return lean
    from api.end_user_memory import NO_CONSENT
    from core.models.end_users import EndUser as EndUserRecord
    from core.security.end_user_turn import EndUser

    known = (db.query(EndUserRecord.id)
             .filter(EndUserRecord.workspace_id == ctx.workspace_id, EndUserRecord.end_user_id == end_user_id)
             .first())
    if known is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=NO_CONSENT)
    return replace(lean, end_user=EndUser(str(ctx.workspace_id), end_user_id))


def keeps_each_end_users_chats_apart(open_chat: Callable[..., str]) -> Callable[..., str]:
    """Wraps ``api.chat._open_chat``: a key's conversation is for one end user or for none, and only
    a turn for the same one continues it (403 otherwise). ``_open_chat`` saves a new conversation with
    its end user in the same commit (``ChatService.create_chat``)."""

    @functools.wraps(open_chat)
    def opened(db: Any, chat_service: Any, request: Any, ctx: Any, user_id: int, parts: Any) -> str:
        if not is_server_key(ctx):
            return open_chat(db, chat_service, request, ctx, user_id, parts)
        wanted = _end_user_id(request)
        chat = chat_service.get_chat(request.id, workspace_id=ctx.workspace_id) if request.id else None
        if chat is not None and chat.user_id == user_id and chat.end_user_id != wanted:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=NOT_THIS_END_USERS_CHAT)
        return open_chat(db, chat_service, request, ctx, user_id, parts)

    return opened


def _refuse(detail: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)


def _refuse_unless_allowed(db: Any, ctx: Any, request: Any) -> None:
    from consumers.chatbot.lean_turn import is_lean
    from core.models.core import Agent

    if request.missionMode or request.planMode or request.spokenMode:
        raise _refuse(NOT_A_KEYS_MODE)
    agent = None
    if request.agentId:
        agent = db.query(Agent).filter(Agent.id == request.agentId, Agent.workspace_id == ctx.workspace_id).first()
    if agent is None or not is_lean(agent):
        raise _refuse(NAME_A_LEAN_AGENT)
    if request.model and request.model not in _key_models(db, ctx.api_key_id):
        raise _refuse(f"{request.model} is not on this key's model allow-list.")


def _key_models(db: Any, key_id: Any) -> List[str]:
    """The model ids on the key's allow-list; none when it has no list."""
    from core.models.sdk_api_keys import SdkApiKey

    row = db.query(SdkApiKey.model_allow_list).filter(SdkApiKey.id == key_id).first()
    entries: Optional[list] = row[0] if row else None
    return [e["model_id"] for e in entries or [] if isinstance(e, Mapping) and isinstance(e.get("model_id"), str)]
