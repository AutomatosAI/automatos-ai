"""
Retell voice lane (PRD-203 V·S4 seam · PRD-207 Auto Live)
==========================================================

* ``WS /api/voice/retell/llm-websocket/{call_id}`` — Retell's custom-LLM
  transport (WebSocket-ONLY per their integration contract; the HTTP variant
  PRD-203 merged was uncallable by Retell and is deleted). Retell runs
  STT/TTS/turn-taking/barge-in vendor-side and drives Auto's OWN agent loop
  (the same ``StreamingChatService`` text chat uses) through this socket,
  streaming each reply chunk the moment it exists.
* ``POST /api/voice/web-call`` — the S1 session mint (hybrid auth, four
  ordered fail-closed gates); the ``voice_calls`` row born here is BOTH the
  webhook trust boundary and the WS credential.
* ``POST /api/voice/retell/events`` — call-lifecycle webhook (HMAC
  fail-closed), the minute meter's write path.
* ``GET /api/voice/live-status`` — the S7 settings read.

The self-hosted pod path stays as the fallback; retiring the Pipecat/GPU pod
is a cross-repo automatos-voice coordination (§8-Qd/Qe), NOT done here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db, get_db_session
from modules.voice import live_settings, retell_api, voice_meter
from modules.voice.providers.retell import (
    INTERACTION_CALL_DETAILS,
    INTERACTION_PING_PONG,
    INTERACTION_REMINDER,
    INTERACTION_RESPONSE_REQUIRED,
    RetellLLMRequest,
    parse_llm_request,
    retell_response_frames,
    verify_webhook_signature,
    wrap_ws_response,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])


# ---------------------------------------------------------------------------
# PRD-207 S1 — the web-call session mint
# ---------------------------------------------------------------------------

class MintWebCallRequest(BaseModel):
    """The chat screen's ask: bind this call to the thread I'm looking at."""

    chat_id: Optional[str] = Field(default=None, description="The on-screen chat to bind")
    agent_id: Optional[int] = Field(default=None, description="Explicitly selected agent")


def _resolve_caller_int_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """STRICT #513 resolution: the caller's integer ``users.id`` or None.

    Deliberately not ``api.chat.get_user_id`` — that helper falls back to a
    default user for principal-less paths, and a fallback principal must
    never mint a call attributed to somebody. No user → no mint.
    """
    user = getattr(ctx, "user", None)
    if user is None:
        return None
    uid = getattr(user, "id", None)
    if isinstance(uid, int):
        return uid
    clerk_uid = getattr(user, "clerk_user_id", None) or (uid if isinstance(uid, str) else None)
    if not clerk_uid:
        return None
    from core.models.core import User

    row = db.query(User.id).filter(User.clerk_user_id == str(clerk_uid)).first()
    return int(row[0]) if row else None


@router.post("/api/voice/web-call")
async def mint_web_call(
    body: MintWebCallRequest = Body(default=MintWebCallRequest()),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict:
    """Mint a Retell web-call session for the authenticated caller (S1).

    Gates, in order, all fail-closed with honest reasons (§7.5-3):
    platform ``voice.live_enabled`` settings toggle → Retell credentials →
    workspace toggle → cap formula. The ``voice_calls`` row is BORN here
    (status ``minted``) so the webhook's trust boundary has something to
    validate against before any event arrives. The Retell key never reaches
    the browser — only the ~30s-lived ``access_token`` does.
    """
    from core.models.core import Chat
    from core.models.voice_calls import VoiceCall
    from core.models.workspaces import Workspace

    # Gate 1 — the platform master switch (a Settings toggle, never an env var).
    if not live_settings.voice_live_enabled():
        logger.info("voice_live_mint_denied reason=platform_disabled workspace=%s", ctx.workspace_id)
        raise HTTPException(status_code=503, detail="Auto Live is switched off platform-wide")

    # Gate 2 — armed: all three Retell credentials present in system settings.
    creds = live_settings.retell_credentials()
    if not creds.armed:
        logger.warning("voice_live_mint_denied reason=not_armed workspace=%s", ctx.workspace_id)
        raise HTTPException(
            status_code=503,
            detail="Auto Live is not armed — Retell credentials are missing in Settings",
        )

    # The caller must strictly resolve to a real user (no fallback principals).
    user_int_id = _resolve_caller_int_id(db, ctx)
    if user_int_id is None:
        logger.info("voice_live_mint_denied reason=no_user workspace=%s", ctx.workspace_id)
        raise HTTPException(status_code=403, detail="Live voice requires a signed-in user")

    # Gate 3 — the workspace's own toggle.
    workspace = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ws_voice = live_settings.parse_workspace_voice_live(workspace.settings)
    if not ws_voice.enabled:
        logger.info("voice_live_mint_denied reason=workspace_disabled workspace=%s", ctx.workspace_id)
        raise HTTPException(
            status_code=403, detail="Auto Live is not enabled for this workspace"
        )

    # Gate 4 — the cap formula (ended + active×reserve ≥ cap refuses).
    reading = voice_meter.monthly_meter(db, ctx.workspace_id)
    allowed, reason = voice_meter.cap_allows_mint(reading, ws_voice.monthly_cap_minutes)
    if not allowed:
        logger.warning(
            "voice_live_cap_exceeded workspace=%s %s", ctx.workspace_id, reason
        )
        raise HTTPException(status_code=429, detail=reason)

    # Bind-target validation at mint: the chat must be the caller's own thread
    # in THIS workspace — refuse honestly rather than mint a lie the webhook
    # would then reject (§7.5-1/2).
    bound_chat_id: Optional[str] = None
    if body.chat_id:
        try:
            chat_uuid = uuid.UUID(str(body.chat_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="chat_id is not a valid chat")
        chat = (
            db.query(Chat)
            .filter(Chat.id == chat_uuid, Chat.workspace_id == ctx.workspace_id)
            .first()
        )
        if chat is None:
            raise HTTPException(status_code=400, detail="chat_id is not a chat in this workspace")
        if int(chat.user_id) != int(user_int_id):
            raise HTTPException(status_code=403, detail="You can only bind a live call to your own chat")
        bound_chat_id = str(chat_uuid)

    dynamic_vars = {
        "workspace_id": str(ctx.workspace_id),
        "user_id": user_int_id,
        "chat_id": bound_chat_id,
        "agent_id": body.agent_id,
    }
    payload = retell_api.build_web_call_payload(
        agent_id=creds.agent_id,
        dynamic_variables=dynamic_vars,
        voice_id=ws_voice.retell_voice_id,
        max_call_minutes=int(config.VOICE_LIVE_MAX_CALL_MINUTES),
    )

    logger.info(
        "voice_live_mint_requested workspace=%s user=%s chat=%s",
        ctx.workspace_id, user_int_id, bound_chat_id,
    )
    try:
        web_call = await retell_api.create_web_call(creds.api_key, payload)
    except retell_api.RetellApiError as exc:
        logger.error("voice_live_mint_denied reason=vendor_error workspace=%s err=%s", ctx.workspace_id, exc)
        raise HTTPException(status_code=502, detail=f"Could not start the call: {exc}")

    # The row is BORN at mint — the webhook validates against it (S2/S3).
    db.add(
        VoiceCall(
            call_id=web_call.call_id,
            provider="retell",
            workspace_id=ctx.workspace_id,
            user_id=user_int_id,
            chat_id=bound_chat_id,
            status="minted",
        )
    )
    db.commit()

    logger.info(
        "voice_live_minted call=%s workspace=%s user=%s chat=%s",
        web_call.call_id, ctx.workspace_id, user_int_id, bound_chat_id,
    )
    # The token dies in ~30s unused — the client connects immediately.
    return {"call_id": web_call.call_id, "access_token": web_call.access_token}


# ---------------------------------------------------------------------------
# PRD-207 S3 — call lifecycle events + the S7 status read
# ---------------------------------------------------------------------------

def _apply_call_event(db: Session, event: str, call: dict) -> str:
    """Idempotently fold one Retell lifecycle event into voice_calls.

    Returns a disposition string for the log line. Timestamps arrive as
    millisecond epochs (``start_timestamp``/``end_timestamp``); replays
    re-apply the same values — idempotent by construction. An event for a
    call we never minted is stored as a LOUD orphan (phone-lane compat),
    never silently dropped.
    """
    from datetime import datetime, timezone

    from core.models.voice_calls import VoiceCall

    call_id = call.get("call_id")
    if not call_id:
        return "ignored_no_call_id"

    def _ts(ms) -> Optional[Any]:
        try:
            return datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc).replace(tzinfo=None)
        except (TypeError, ValueError, OSError):
            return None

    row = db.query(VoiceCall).filter(VoiceCall.call_id == str(call_id)).first()
    disposition = "updated"
    if row is None:
        logger.warning(
            "voice_live_orphan_event event=%s call=%s — no mint row; storing loud", event, call_id
        )
        row = VoiceCall(call_id=str(call_id), provider="retell", status="minted")
        db.add(row)
        disposition = "orphan_created"

    started = _ts(call.get("start_timestamp"))
    ended = _ts(call.get("end_timestamp"))

    if event == "call_started":
        row.started_at = started or row.started_at
        if row.status in ("minted",):
            row.status = "started"
    elif event == "call_ended":
        row.started_at = started or row.started_at
        row.ended_at = ended or row.ended_at
        if row.started_at is not None and row.ended_at is not None:
            row.duration_seconds = max(
                0, int((row.ended_at - row.started_at).total_seconds())
            )
        reason = call.get("disconnection_reason")
        if reason:
            row.disconnect_reason = str(reason)[:64]
        # ended without ever starting = the call never connected.
        row.status = "ended" if row.started_at is not None else "failed"
    elif event == "call_analyzed":
        # Post-call analysis carries no lifecycle change we bill on — ack only.
        disposition = "acknowledged"
    else:
        disposition = f"ignored_{event or 'unknown'}"

    db.commit()
    return disposition


@router.post("/api/voice/retell/events")
async def retell_events_webhook(request: Request) -> dict:
    """Retell call-lifecycle webhook (S3): the meter's write path.

    HMAC fail-closed like the LLM webhook; idempotent updates keyed by
    ``call_id``. Always answers 2xx fast on verified payloads (Retell
    retries 3× in 10s — a slow handler double-applies, which idempotency
    absorbs, but we stay quick anyway).
    """
    body = await request.body()
    signature = request.headers.get("x-retell-signature")
    secret = live_settings.retell_credentials().webhook_secret
    if not verify_webhook_signature(secret, signature, body):
        raise HTTPException(status_code=401, detail="Invalid Retell webhook signature")

    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    event = str(payload.get("event") or "")
    call = payload.get("call") or {}
    with get_db_session() as db:
        disposition = _apply_call_event(db, event, call)

    logger.info(
        "voice_live_call_event event=%s call=%s disposition=%s",
        event, call.get("call_id"), disposition,
    )
    return {"ok": True}


@router.get("/api/voice/live-status")
async def voice_live_status(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict:
    """The S7 settings card's one read: both gates + the meter, honestly.

    Never returns credential VALUES — only whether they are present.
    """
    from core.models.workspaces import Workspace

    workspace = db.query(Workspace).filter(Workspace.id == ctx.workspace_id).first()
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    ws_voice = live_settings.parse_workspace_voice_live(workspace.settings)
    reading = voice_meter.monthly_meter(db, ctx.workspace_id)
    return {
        "platform_enabled": live_settings.voice_live_enabled(),
        "armed": live_settings.retell_credentials().armed,
        "workspace_enabled": ws_voice.enabled,
        "retell_voice_id": ws_voice.retell_voice_id,
        "used_minutes": reading.ended_minutes,
        "cap_minutes": ws_voice.monthly_cap_minutes,
        "active_calls": reading.active_calls,
        "max_call_minutes": int(config.VOICE_LIVE_MAX_CALL_MINUTES),
    }


async def _agent_retell_stream(req: RetellLLMRequest) -> AsyncIterator[dict[str, Any]]:
    """Drive Auto's agent loop for one Retell turn and yield Retell frames.

    PRD-207 S2 rewired this onto the mint-row trust boundary
    (``resolve_call_binding``): a web call launched from /chat writes into the
    on-screen thread attributed to the REAL user (memory owner/Q7 scoping and
    thread checkpoints now cover spoken conversations); anything unproven
    falls closed to a per-call chat. That also fixes two latent faults in the
    merged-unarmed lane: ``get_chat("retell:{call_id}")`` can never parse as a
    UUID (→ a NEW chat every TURN, multi-turn calls lost their history) and
    ``create_chat(user_id=0)`` violates the NOT-NULL FK to ``users``.

    S6: both turn messages carry the persisted voice source; a
    ``chat_changed`` NOTIFY lets the open thread receive them live.
    S8: one ``voice_turns`` row per turn — the judging metric exists from
    day one (webhook-receipt → stream-close; vendor-side STT/TTS honestly 0).
    """
    from datetime import datetime as _dt
    from time import monotonic

    from consumers.chatbot import ChatService, StreamingChatService
    from modules.voice.call_binding import (
        resolve_call_binding,
        stamp_assistant_voice_source,
        voice_source,
    )

    turn_t0 = monotonic()
    turn_started_at = _dt.utcnow()

    with get_db_session() as db:
        binding = resolve_call_binding(
            db,
            call_id=req.call_id,
            workspace_id=req.workspace_id,
            user_id_var=req.user_id,
            chat_id_var=req.chat_id,
            first_text=req.user_text,
        )
        if binding is None:
            # No safe place to write (logged LOUD upstream) — close the turn.
            yield {"response_id": req.response_id, "content": "", "content_complete": True}
            return

        chat_service = ChatService(db)
        streaming_service = StreamingChatService(db, workspace_id=binding.workspace_id)
        conversation_id = binding.chat_id

        chat_service.save_message(
            chat_id=conversation_id,
            role="user",
            parts=[{"type": "text", "text": req.user_text}],
            workspace_id=binding.workspace_id,
            source=voice_source("user"),
        )

        # Agent selection: an explicit dynamic-variable agent wins; otherwise the
        # workspace default answers (minimal wire — the same default chat uses).
        from api.chat import get_default_agent_id

        agent_id = None
        if req.agent_id:
            try:
                agent_id = int(req.agent_id)
            except (TypeError, ValueError):
                agent_id = None
        if agent_id is None:
            agent_id = get_default_agent_id(db, binding.workspace_id)

        messages = chat_service.get_messages_by_chat_id(conversation_id)
        message_history = [{"role": m.role, "parts": m.parts} for m in messages]

        agent_chunks = streaming_service.stream_response_with_agent(
            chat_id=conversation_id,
            messages=message_history,
            agent_id=agent_id,
            user_id=binding.user_id,
            use_orchestrator_llm=True,
        )

        response_chars = 0
        async for frame in retell_response_frames(req.response_id, agent_chunks):
            response_chars += len(frame.get("content") or "")
            yield frame

        # S6 — stamp this turn's assistant reply + let the open thread hear it.
        try:
            stamp_assistant_voice_source(
                db, chat_id=conversation_id, turn_started_at=turn_started_at
            )
            from services.board_events import notify_chat_event

            notify_chat_event(
                db,
                workspace_id=binding.workspace_id,
                chat_id=conversation_id,
                user_id=binding.user_id,
            )
        except Exception:  # noqa: BLE001 — provenance/notify never fail a turn
            logger.debug("voice turn stamp/notify failed", exc_info=True)

    # S8 — telemetry parity (fire-and-forget, own session, never raises).
    # STT/TTS run vendor-side: honestly 0, not faked. audio_delivered means
    # content frames actually went to the vendor's TTS this turn.
    from modules.voice.telemetry import record_voice_turn

    record_voice_turn(
        workspace_id=binding.workspace_id,
        conversation_id=conversation_id,
        message_id=None,
        call_id=req.call_id,
        stt_latency_ms=0,
        tts_latency_ms=0,
        total_ms=(monotonic() - turn_t0) * 1000.0,
        transcript=req.user_text,
        response_text="x" * response_chars,  # lengths-only discipline: only len() is stored
        truncated=False,
        audio_delivered=response_chars > 0,
    )


@router.websocket("/api/voice/retell/llm-websocket/{call_id}")
async def retell_llm_websocket(websocket: WebSocket, call_id: str) -> None:
    """Retell custom-LLM WebSocket — Auto's brain on Retell's actual transport.

    Retell's dashboard takes a ``wss://…/api/voice/retell/llm-websocket`` URL
    and appends ``/{call_id}``. Protocol: the server speaks FIRST with an
    empty response (the human has the floor); ``call_details`` delivers the
    dynamic variables once; ``ping_pong`` must be echoed; ``update_only``
    owes nothing; ``response_required``/``reminder_required`` start a turn —
    a newer ``response_required`` supersedes a still-streaming one (barge-in),
    so the old stream task is cancelled.

    Auth: a WS handshake carries no HMAC — the MINTED ``call_id`` is the
    credential (born server-side at ``/api/voice/web-call``, unguessable,
    dead in ~30s unused). No mint row → close 4401 before accept. The
    steward/orphan fallback lane stays exclusive to the HMAC-verified events
    webhook; an unauthenticated socket can never conjure attribution.

    The settings toggle gates the socket too — disarming kills in-flight
    speech (§7.5-3), no redeploy.
    """
    if not live_settings.voice_live_enabled():
        await websocket.close(code=4403)
        return

    from core.models.voice_calls import VoiceCall

    with get_db_session() as db:
        minted = db.query(VoiceCall.id).filter(VoiceCall.call_id == str(call_id)).first()
    if minted is None:
        logger.warning("voice_live_ws_rejected reason=unminted_call call=%s", call_id)
        await websocket.close(code=4401)
        return

    await websocket.accept()

    dynamic_vars: dict[str, Any] = {}
    speaking: asyncio.Task | None = None

    async def respond(req: RetellLLMRequest) -> None:
        try:
            async for frame in _agent_retell_stream(req):
                await websocket.send_json(wrap_ws_response(frame))
        except asyncio.CancelledError:
            # Superseded turn (barge-in): Retell has moved on — stop quietly.
            raise
        except Exception:  # noqa: BLE001 — a mid-turn fault closes the turn cleanly
            logger.exception("retell_ws_stream_failed call=%s", call_id)
            try:
                await websocket.send_json(
                    wrap_ws_response(
                        {"response_id": req.response_id, "content": "", "content_complete": True}
                    )
                )
            except Exception:  # noqa: BLE001 — socket already gone
                pass

    try:
        # Server speaks first with an EMPTY response — the human opens.
        await websocket.send_json(
            wrap_ws_response({"response_id": 0, "content": "", "content_complete": True})
        )
        while True:
            message = await websocket.receive_json()
            itype = message.get("interaction_type")
            if itype == INTERACTION_PING_PONG:
                await websocket.send_json(
                    {"response_type": "ping_pong", "timestamp": message.get("timestamp")}
                )
            elif itype == INTERACTION_CALL_DETAILS:
                dynamic_vars = (message.get("call") or {}).get(
                    "retell_llm_dynamic_variables"
                ) or {}
            elif itype in (INTERACTION_RESPONSE_REQUIRED, INTERACTION_REMINDER):
                if speaking is not None and not speaking.done():
                    speaking.cancel()
                req = parse_llm_request(
                    {
                        "response_id": message.get("response_id"),
                        "interaction_type": INTERACTION_RESPONSE_REQUIRED,
                        "transcript": message.get("transcript") or [],
                        "call": {
                            "call_id": call_id,
                            "retell_llm_dynamic_variables": dynamic_vars,
                        },
                    }
                )
                speaking = asyncio.create_task(respond(req))
            # update_only / unknown types: no response owed.
    except WebSocketDisconnect:
        pass
    finally:
        if speaking is not None and not speaking.done():
            speaking.cancel()
