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
* **A turn that names no end user is the key's own, and stores no memory** (``LeanRequest.server_key``,
  ``core.security.end_user_turn.keys_own_turn``), as a widget turn stores none (F154): nothing it says
  is distilled into the workspace's memory, where other turns would recall it.
* **A turn sent with ``Idempotency-Key`` runs once** in a window (``api.chat_turn_keys``), as a run on
  ``/execute`` does: a product's backend that retries after a timeout is answered with the first turn's
  saved answer, or a 409 while it runs, instead of running it again. A person's header is ignored.

Five decorators wire this into ``api/chat.py``, a line each: ``get_user_id``, ``_checked_lean_request``,
``_open_chat``, ``stream_chat`` and ``_after_the_turn`` are wrapped; everyone else passes straight through.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from contextvars import ContextVar
from dataclasses import replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from starlette.concurrency import run_in_threadpool

from api import chat_turn_keys as turn_keys
from api.agent_execute import _checked_key  # one shape check for both routes
from core.auth.server_key_gate import is_server_key

logger = logging.getLogger(__name__)

SERVICE_USER_NAME = "sdk-{key_id}"
SERVICE_USER_EMAIL = "sdk-{key_id}@keys.automatos.invalid"  # .invalid never resolves: no mail, no sign-in
SERVICE_USER_DISPLAY = "Server key: {name}"
NAME_A_LEAN_AGENT = "A server key chats with a lean agent of its workspace: name one in agentId."
NOT_A_KEYS_MODE = "missionMode, planMode and spokenMode are a person's, not a server key's."
NOT_A_PERSONS = "endUserId is a server key's: a person's chat turn is their own."
NOT_THIS_END_USERS_CHAT = "This conversation is not this end user's: start a new one for them."
# The row a keyed turn's request holds, while the route runs (answers_a_repeated_key_once).
_KEYED_TURN: ContextVar[Optional[Any]] = ContextVar("chat_server_key_keyed_turn", default=None)
# The route's own parameter that carries the Idempotency-Key header (declared on api.chat.stream_chat).
IDEMPOTENCY_PARAMETER = "idempotency_key"


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
    the key's list as well as the agent's. What it answers for a key says a key sent it."""

    @functools.wraps(checked)
    def wrapped(db: Any, ctx: Any, request: Any) -> Any:
        if not is_server_key(ctx):
            if _end_user_id(request):
                raise _refuse(NOT_A_PERSONS)
            return checked(db, ctx, request)
        _refuse_unless_allowed(db, ctx, request)
        return _for_its_end_user(db, ctx, request, replace(checked(db, ctx, request), server_key=True))

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
    from core.models.end_users import consent_record_id
    from core.security.end_user_turn import EndUser

    record_id = consent_record_id(db, ctx.workspace_id, end_user_id)
    if record_id is None:  # none on record, or their erase has started
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=NO_CONSENT)
    return replace(lean, end_user=EndUser(str(ctx.workspace_id), end_user_id, record_id))


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
        chat_id = open_chat(db, chat_service, request, ctx, user_id, parts)
        if wanted:
            _unless_their_erase_started(db, chat_service, ctx, wanted, chat_id)
        return chat_id

    return opened


def _unless_their_erase_started(db: Any, chat_service: Any, ctx: Any, end_user_id: str, chat_id: str) -> None:
    """Checked again once the chat is saved: when the end user's erase started after the turn was admitted,
    the turn deletes the chat it opened and stops (409). Otherwise the erase, marked later, deletes it."""
    from api.end_user_memory import NO_CONSENT
    from core.models.end_users import consented

    if not consented(db, ctx.workspace_id, end_user_id):
        chat_service.delete_chat(chat_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=NO_CONSENT)


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


def answers_a_repeated_key_once(stream_chat: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``api.chat.stream_chat``: a server key's turn sent with ``Idempotency-Key`` runs once in the
    window (``api.chat_turn_keys``). A repeat of a turn that answered runs nothing and is answered with that
    answer, framed as the turn framed it; a repeat while it runs, or the key on a different request, is a
    409; a malformed key is a 422, as on ``/execute``. A person's turn, and a key's turn without the header,
    pass straight through: a person's header is ignored, a malformed one included."""
    signature = inspect.signature(stream_chat, eval_str=True)

    @functools.wraps(stream_chat)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        call = signature.bind_partial(*args, **kwargs).arguments
        idempotency_key = call.get(IDEMPOTENCY_PARAMETER)
        if idempotency_key is None or not is_server_key(call.get("ctx")):
            return await stream_chat(*args, **kwargs)
        return await _run_once(stream_chat, call, idempotency_key, args, kwargs)

    return wrapped


async def _run_once(stream_chat: Callable[..., Any], call: Mapping[str, Any], raw_key: str,
                    args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
    """The keyed turn: its saved answer replayed, a refusal, or the route run while this request holds the
    key's row (read by ``records_a_keyed_turn``)."""
    from core.jobs.worker import on_own_session

    key = _checked_key(raw_key)
    held = await run_in_threadpool(on_own_session, turn_keys.claim, call["ctx"], key,
                                   turn_keys.request_hash(call["request"]))
    if isinstance(held, turn_keys.Replay):
        return turn_keys.replayed(held)
    token = _KEYED_TURN.set(held)
    try:
        return await stream_chat(*args, **kwargs)
    except BaseException:  # refused before its turn started (422, 403, 409): nothing ran, so the key is free
        await _free(held)
        raise
    finally:
        _KEYED_TURN.reset(token)


async def _free(row_id: Any) -> None:
    """Deletes the row a refused request held. One that cannot be deleted is logged; it is presumed dead in
    time (``api.chat_turn_keys``), and the key may then run again."""
    from core.jobs.worker import on_own_session

    try:
        await run_in_threadpool(on_own_session, turn_keys.release, row_id)
    except Exception:  # the caller is told its own refusal, never this one
        logger.exception("[chat_server_key] the Idempotency-Key row %s of a refused turn was not deleted", row_id)


def records_a_keyed_turn(after_the_turn: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps ``api.chat._after_the_turn``: a turn run under an ``Idempotency-Key`` records how it ended once
    it has, whatever the client did (``api.chat_turn_keys.record_end``). The route calls it while it runs,
    so the row recorded is the one this request holds; any other turn's hook is the route's own."""

    @functools.wraps(after_the_turn)
    def bound(workspace_id: Any, chat_id: str, user_id: int, message_id: Any) -> Callable[..., Any]:
        on_complete = after_the_turn(workspace_id, chat_id, user_id, message_id)
        row_id = _KEYED_TURN.get()
        return on_complete if row_id is None else _recording(on_complete, row_id, chat_id, message_id)

    return bound


def _recording(on_complete: Callable[..., Any], row_id: Any, chat_id: str, message_id: Any) -> Callable[..., Any]:
    """``on_complete``, then the keyed turn's ending, written off the event loop."""

    async def recorded(*, completed: bool, cancelled: bool, client_gone: bool) -> None:
        try:
            await on_complete(completed=completed, cancelled=cancelled, client_gone=client_gone)
        finally:  # after the route's hook, which saves the note of a turn that did not finish
            await asyncio.to_thread(_record_end, row_id, chat_id, message_id, completed)

    return recorded


def _record_end(row_id: Any, chat_id: str, message_id: Any, completed: bool) -> Optional[str]:
    """How the keyed turn ended, on a session of its own; None (logged) when it could not be written: the
    row is then presumed dead in time, and the key may run the turn again."""
    from core.jobs.worker import on_own_session

    try:
        return on_own_session(turn_keys.record_end, row_id, chat_id, message_id, completed)
    except Exception:  # after the turn: logged with its trace, never raised into the turn's end
        logger.exception("[chat_server_key] the end of keyed turn %s was not recorded", row_id)
        return None
