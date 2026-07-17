"""
Retell streaming-voice webhook (PRD-203 V·S4)
=============================================

POST /api/voice/retell/llm — Retell's custom-LLM webhook.

Retell (§8-Qa) runs STT/TTS/turn-taking/barge-in vendor-side and calls THIS
endpoint for the words. We front Auto's own agent loop (the same
``StreamingChatService`` text chat uses) and **stream** the reply back as Retell
custom-LLM frames, so Retell starts speaking (first audio) before the full
generation completes — the streaming posture the blocking self-hosted path
(``chat_voice._collect_streaming_response``) never had.

Auth: the webhook carries no user JWT — it is authenticated by an HMAC signature
(``config.RETELL_WEBHOOK_SECRET``) and fails closed. Per-call workspace/agent
context rides in Retell dynamic variables (set when the call is created).

The self-hosted pod path stays as the fallback; retiring the Pipecat/GPU pod is a
cross-repo automatos-voice coordination (§8-Qa), NOT done here.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db, get_db_session
from modules.voice import live_settings, retell_api, voice_meter
from modules.voice.providers.retell import (
    INTERACTION_RESPONSE_REQUIRED,
    RetellLLMRequest,
    parse_llm_request,
    retell_response_frames,
    verify_webhook_signature,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

_NDJSON = "application/x-ndjson"


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


async def _single_complete(response_id: int) -> AsyncIterator[str]:
    """A one-frame stream that closes the turn without words (non-answer events)."""
    yield json.dumps(
        {"response_id": response_id, "content": "", "content_complete": True}
    ) + "\n"


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


@router.post("/api/voice/retell/llm")
async def retell_llm_webhook(request: Request) -> StreamingResponse:
    """Retell custom-LLM webhook — stream Auto's reply back as Retell frames."""
    # PRD-207 S4: the settings toggle is the master switch for the WHOLE
    # Retell lane — flipping it off in Settings kills in-flight speech too
    # (the "instant platform-wide kill" promise), no redeploy.
    if not live_settings.voice_live_enabled():
        raise HTTPException(status_code=503, detail="Auto Live is switched off platform-wide")

    body = await request.body()
    signature = request.headers.get("x-retell-signature")
    secret = live_settings.retell_credentials().webhook_secret
    if not verify_webhook_signature(secret, signature, body):
        # Fail-closed: unsigned/unconfigured/mismatched → never drive the agent.
        raise HTTPException(status_code=401, detail="Invalid Retell webhook signature")

    try:
        payload = json.loads(body)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    req = parse_llm_request(payload)

    # Non-answer interactions (pings/updates) just close the turn.
    if req.interaction_type != INTERACTION_RESPONSE_REQUIRED:
        return StreamingResponse(_single_complete(req.response_id), media_type=_NDJSON)

    if not req.workspace_id:
        raise HTTPException(
            status_code=422,
            detail="Retell call is missing the workspace_id dynamic variable",
        )

    async def stream() -> AsyncIterator[str]:
        try:
            async for frame in _agent_retell_stream(req):
                yield json.dumps(frame) + "\n"
        except Exception:  # noqa: BLE001 — a mid-stream fault closes the turn cleanly
            logger.exception("retell_llm_stream_failed call=%s", req.call_id)
            yield json.dumps(
                {"response_id": req.response_id, "content": "", "content_complete": True}
            ) + "\n"

    return StreamingResponse(stream(), media_type=_NDJSON)
