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
import contextlib
import json
import logging
import threading
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, WebSocket
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db, get_db_session
from modules.policy import ai_disclosure
from modules.voice import arming, call_events, live_settings, live_socket, mint, retell_api, turn_timing, voice_meter
from modules.voice.caller import named_call_agent
# The mint resolves its caller through this module's name, so a patch here holds.
from modules.voice.caller import resolve_caller_int_id as _resolve_caller_int_id
from modules.voice.providers.retell import (
    RetellLLMRequest,
    retell_response_frames,
    verify_webhook_signature,
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


def _mint_ports() -> mint.MintPorts:
    """What a mint borrows from the route, late-bound through this module's names (the
    ``_socket_ports`` pattern), so a test's patch on ``api.voice_retell`` still holds."""
    return mint.MintPorts(
        voice_live_enabled=lambda: live_settings.voice_live_enabled(),
        retell_credentials=lambda: live_settings.retell_credentials(),
        workspace_voice_live=lambda settings: live_settings.parse_workspace_voice_live(settings),
        monthly_meter=lambda db, workspace_id: voice_meter.monthly_meter(db, workspace_id),
        cap_allows_mint=lambda reading, cap: voice_meter.cap_allows_mint(reading, cap),
        create_web_call=lambda api_key, payload: retell_api.create_web_call(api_key, payload),
    )


def _require_voice_live_plan(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> None:
    """PRD-266 D34: on SaaS the mint needs a plan with the voice family; 403
    ``voice_live_plan_required`` otherwise. The local edition is unrestricted."""
    try:
        mint.require_voice_live_plan(db, ctx.workspace_id)
    except mint.MintRefused as refused:
        raise HTTPException(status_code=refused.status_code, detail=refused.detail) from refused


@router.post("/api/voice/web-call", dependencies=[Depends(_require_voice_live_plan)])
async def mint_web_call(
    body: MintWebCallRequest = Body(default=MintWebCallRequest()),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict:
    """Mint a Retell web-call session for the authenticated caller (S1).

    Gates, in order, all fail-closed with honest reasons (§7.5-3): on SaaS the
    plan (``_require_voice_live_plan``, D34) → platform ``voice.live_enabled``
    settings toggle → Retell credentials → a strictly resolved caller →
    workspace toggle → cap formula. The steps are ``modules/voice/mint.py``'s,
    fed this module's names through ``_mint_ports``. The ``voice_calls`` row is
    BORN here (status ``minted``) so the webhook's trust boundary has something
    to validate against before any event arrives. The Retell key never reaches
    the browser — only the ~30s-lived ``access_token`` does, with Retell's
    transport fields when it returned them (D32).
    """
    ports = _mint_ports()
    try:
        creds = mint.armed_credentials(ports, ctx.workspace_id)
        # The caller must strictly resolve to a real user (no fallback principals).
        user_int_id = _resolve_caller_int_id(db, ctx)
        if user_int_id is None:
            logger.info("voice_live_mint_denied reason=no_user workspace=%s", ctx.workspace_id)
            raise HTTPException(status_code=403, detail="Live voice requires a signed-in user")
        ws_voice = mint.workspace_voice_within_cap(db, ports, ctx.workspace_id)
        target = mint.MintTarget(
            workspace_id=ctx.workspace_id,
            user_id=user_int_id,
            chat_id=mint.bind_chat(db, ctx.workspace_id, user_int_id, body.chat_id),
            agent_id=body.agent_id,
        )
        web_call = await mint.create_call(ports, creds, ws_voice, target)
    except mint.MintRefused as refused:
        raise HTTPException(status_code=refused.status_code, detail=refused.detail) from refused
    mint.record_minted_call(db, web_call, target, is_super_admin=mint.caller_is_super_admin(ctx))
    # The token dies in ~30s unused — the client connects immediately.
    return mint.mint_response(web_call, target.chat_id)


# ---------------------------------------------------------------------------
# PRD-207 S3 — call lifecycle events + the S7 status read
# ---------------------------------------------------------------------------

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
    call = await call_events.with_vendor_latency(event, payload.get("call") or {})
    with get_db_session() as db:
        disposition = call_events.apply_call_event(db, event, call)

    logger.info(
        "voice_live_call_event event=%s call=%s disposition=%s",
        event, call.get("call_id"), disposition,
    )
    return {"ok": True}


def _is_platform_admin(ctx: RequestContext) -> bool:
    """The system-settings admin posture (api-key lane or admin/super_admin)."""
    if getattr(ctx, "auth_type", "") == "api_key":
        return True
    user = getattr(ctx, "user", None)
    return bool(user and getattr(user, "system_role", "user") in ("admin", "super_admin"))


@router.get("/api/voice/live-status")
async def voice_live_status(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict:
    """The S7 settings card's one read: both gates + the meter, honestly.

    Never returns credential VALUES — only whether they are present.
    ``viewer_is_admin`` lets the card show the one-click Arm box to the
    people who can actually use it.
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
        "viewer_is_admin": _is_platform_admin(ctx),
    }


class ArmVoiceRequest(BaseModel):
    """One-click arming from the Auto Live card (S7)."""

    enabled: bool = True
    api_key: Optional[str] = Field(default=None, description="Retell API key (first arm)")
    voice_id: Optional[str] = Field(default=None, description="Voice for the created agent")


@router.post("/api/voice/arm")
async def arm_voice_live(
    body: ArmVoiceRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> dict:
    """Arm (or disarm) Auto Live platform-wide from ONE box on the card.

    Paste the Retell API key, click Arm: the server creates the custom-LLM
    agent in the customer's Retell account (correct wss + webhook URLs —
    nobody hand-copies transport strings), files key/signing-key/agent-id
    into the masked settings slots and flips ``voice.live_enabled`` ON.
    Idempotent: an existing agent id is kept; a re-arm re-tunes it and
    re-points it at the current ``PUBLIC_API_HOST``. ``enabled=false`` flips
    the master toggle only. Admin-gated here; the body is
    ``modules/voice/arming.arm`` (PRD-258 US-003), refusals are honest.
    """
    if not _is_platform_admin(ctx):
        raise HTTPException(status_code=403, detail="Admin access required to arm voice")

    try:
        return await arming.arm(
            db,
            enabled=body.enabled,
            api_key=body.api_key,
            voice_id=body.voice_id,
            workspace_id=ctx.workspace_id,
            actor=getattr(getattr(ctx, "user", None), "id", "?"),
        )
    except arming.ArmRefused as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


# The worker loop gets this long after a turn settles to run its finallys
# (closing the inner generator, its sessions) before the loop is stopped.
_WORKER_STOP_GRACE_SECONDS = 2.0
# A cross-thread frame hand-off that cannot land this long means the caller
# stopped draining (dead socket, cancelled turn) — the pump gives up.
_BRIDGE_PUT_TIMEOUT_SECONDS = 10.0
_STREAM_DONE = object()
_STREAM_ERROR = object()


async def _agent_retell_stream(req: RetellLLMRequest) -> AsyncIterator[dict[str, Any]]:
    """Frames for one turn, generated on a DEDICATED thread + event loop.

    The agent loop inside ``_agent_retell_stream_inner`` performs synchronous
    I/O (ORM reads, memory retrieval, embedding and model-gateway calls). Run
    on the WS event loop those blocks freeze it — and a frozen loop stops
    echoing Retell's 2s ``ping_pong``, which severs the socket after ~5s of
    pong silence (their keepalive contract, up to 2 auto-reconnects). First
    armed morning: every brained turn died code=1006 ~8s after
    ``response_required`` with frames=0 — the reply was killed mid-generation
    by our own starved loop. Isolating the brain on a worker loop keeps THIS
    loop answering pings no matter what the turn is doing.

    Frames hop back through a bounded queue; closing this generator cancels
    the worker task, whose own ``finally`` closes the inner generator
    (sessions and all). The worker thread is daemonic and bounded by the put
    timeout, so an abandoned turn can never pin the process.
    """
    caller_loop = asyncio.get_running_loop()
    frames: asyncio.Queue = asyncio.Queue(maxsize=8)
    worker_loop = asyncio.new_event_loop()

    def _worker_main() -> None:
        try:
            worker_loop.run_forever()
        finally:
            with contextlib.suppress(Exception):
                worker_loop.close()

    worker = threading.Thread(
        target=_worker_main,
        name=f"voice-turn-{(req.call_id or 'call')[-8:]}-{req.response_id}",
        daemon=True,
    )
    worker.start()

    async def pump() -> None:
        inner = _agent_retell_stream_inner(req)
        try:
            async for frame in inner:
                asyncio.run_coroutine_threadsafe(frames.put(frame), caller_loop).result(
                    timeout=_BRIDGE_PUT_TIMEOUT_SECONDS
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — re-raised on the caller loop
            with contextlib.suppress(Exception):
                asyncio.run_coroutine_threadsafe(
                    frames.put((_STREAM_ERROR, exc)), caller_loop
                ).result(timeout=_BRIDGE_PUT_TIMEOUT_SECONDS)
            return
        finally:
            with contextlib.suppress(BaseException):
                await inner.aclose()
        with contextlib.suppress(Exception):
            asyncio.run_coroutine_threadsafe(frames.put(_STREAM_DONE), caller_loop).result(
                timeout=_BRIDGE_PUT_TIMEOUT_SECONDS
            )

    turn = asyncio.run_coroutine_threadsafe(pump(), worker_loop)
    try:
        while True:
            item = await frames.get()
            if item is _STREAM_DONE:
                break
            if isinstance(item, tuple) and len(item) == 2 and item[0] is _STREAM_ERROR:
                raise item[1]
            yield item
    finally:
        # Normal end, aclose() or caller cancellation all land here: cancel
        # the worker task (propagates into the inner generator's finally) and
        # stop the worker loop once those finallys have had their grace.
        turn.cancel()
        worker_loop.call_soon_threadsafe(
            worker_loop.call_later, _WORKER_STOP_GRACE_SECONDS, worker_loop.stop
        )


async def _agent_retell_stream_inner(req: RetellLLMRequest) -> AsyncIterator[dict[str, Any]]:
    """Drive Auto's agent loop for one Retell turn and yield Retell frames.

    PRD-207 S2 rewired this onto the mint-row trust boundary
    (``resolve_call_binding``): a web call launched from /chat writes into the
    on-screen thread attributed to the REAL user (memory owner/Q7 scoping and
    thread checkpoints now cover spoken conversations); anything unproven
    falls closed to a per-call chat.

    S6: both turn messages carry the persisted voice source; a
    ``chat_changed`` NOTIFY lets the open thread receive them live.
    PRD-258 US-001: besides the frames, the stream reports its turn's facts
    in-band (``turn_timing.landed`` once bound, ``turn_timing.finished`` once
    the reply is saved: tool calls before speech, the reply's ``prep_ms``);
    the socket writes the turn's ONE ``voice_turns`` row from them.
    """
    from datetime import datetime as _dt

    from modules.voice.call_binding import assistant_reply_prep_ms, resolve_call_binding

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
        yield turn_timing.landed(binding.workspace_id, binding.chat_id)

        tap = turn_timing.SpeechTap(_start_spoken_turn(db, req, binding))
        async for frame in retell_response_frames(req.response_id, tap):
            yield frame

        _stamp_and_notify(db, req, binding, turn_started_at)
        yield turn_timing.finished(
            tools_before_speech=tap.tally.tools_before_speech,
            prep_ms=assistant_reply_prep_ms(
                db, chat_id=binding.chat_id, workspace_id=binding.workspace_id, turn_started_at=turn_started_at
            ),
        )


def _start_spoken_turn(db: Session, req: RetellLLMRequest, binding: Any) -> AsyncIterator[str]:
    """Save the caller's words and start Auto's agent stream for this turn (AI-SDK lines)."""
    from api.chat import get_default_agent_id
    from consumers.chatbot import ChatService, StreamingChatService
    from modules.voice.call_binding import upsert_voice_user_message

    # Grow-aware: a continued sentence UPDATES the message it grew from
    # (first live use stacked "…single chat" / "…voice chat?" / "…mixed?"
    # as three messages — one voice, one message).
    upsert_voice_user_message(
        db, chat_id=binding.chat_id, workspace_id=binding.workspace_id, text=req.user_text
    )

    # Agent selection: an explicit dynamic-variable agent wins; otherwise the
    # workspace default answers (minimal wire — the same default chat uses).
    agent_id = named_call_agent(db, binding.workspace_id, req.agent_id)
    if agent_id is None:
        agent_id = get_default_agent_id(db, binding.workspace_id)

    # ONE AUTO (Gerard, first live night: "voice auto and text auto are
    # not the same… my auto and voice auto need to be the same"). The
    # spoken turn keeps the SAME BRAIN as the typed turn: full memory
    # retrieval, knowledge search, core tool router, and the caller's real
    # privilege tier. force_text_only is NEVER set — that forces the
    # tool-less ATOM path AND skips memory (the lobotomy Gerard killed).
    #
    # Two voice-specific latency trims, both measured, neither touches the
    # brain: (1) history is the recent conversation, not the archive
    # (rhythm, not memory — memory injection is retrieval, not replay);
    # (2) skip_composio drops the THIRD-PARTY EXECUTION surface (Gmail/
    # Slack/etc.). Root cause of 20-90s first-token: loading all 23
    # Composio apps put ~58 tools / 137 actions / 44 promoted schemas into
    # every call (24-36k input tokens) and drove a ~5-call agentic loop.
    # Without Composio the agent runs ~14 core tools — memory + knowledge
    # + reasoning intact, first token in a conversational range. Config
    # dial VOICE_LIVE_SKIP_COMPOSIO restores full parity if ever wanted.
    # Tool work mid-call still reads as the honest THINKING state.
    messages = ChatService(db).get_messages_by_chat_id(binding.chat_id)
    recent = messages[-int(config.VOICE_LIVE_TURN_HISTORY_MESSAGES):]
    return StreamingChatService(db, workspace_id=binding.workspace_id).stream_response_with_agent(
        chat_id=binding.chat_id,
        messages=[{"role": m.role, "parts": m.parts} for m in recent],
        agent_id=agent_id,
        user_id=binding.user_id,
        use_orchestrator_llm=True,
        # captured at mint (Clerk claims live only there); False for the
        # steward/orphan lanes — fail-closed. The #581 runtime lookup
        # crashed every turn: User has no system_role attribute.
        is_super_admin=binding.is_super_admin,
        # The measured voice-latency lever (memory/tools stay; only
        # third-party Composio EXECUTION is dropped from spoken turns).
        skip_composio=bool(config.VOICE_LIVE_SKIP_COMPOSIO),
        # She is being HEARD, not read: short sentences, no markdown.
        # Same brain — this is a medium instruction, not a lobotomy.
        spoken_mode=True,
    )


def _stamp_and_notify(db: Session, req: RetellLLMRequest, binding: Any, turn_started_at: Any) -> None:
    """S6 — stamp this turn's assistant reply + let the open thread hear it.
    Provenance and the notify never fail a turn: a fault is logged with the call id."""
    from modules.voice.call_binding import stamp_assistant_voice_source
    from services.board_events import notify_chat_event

    try:
        stamp_assistant_voice_source(db, chat_id=binding.chat_id, turn_started_at=turn_started_at)
        notify_chat_event(
            db, workspace_id=binding.workspace_id, chat_id=binding.chat_id, user_id=binding.user_id
        )
    except Exception:
        logger.exception("voice_turn_stamp_notify_failed call=%s", req.call_id)


def _call_status(call_id: str) -> Optional[str]:
    """The call's live ``voice_calls.status`` (None when it has no row)."""
    from core.models.voice_calls import VoiceCall

    with get_db_session() as db:
        return db.query(VoiceCall.status).filter(VoiceCall.call_id == str(call_id)).scalar()


def _socket_ports() -> live_socket.SocketPorts:
    """What a call's socket borrows from the route.

    The turn stream and the status read are late-bound through this module's
    names, so they stay patchable where they live. The AI notice texts
    (PRD-264 EUAI-04) come from the one rules module: ``VOICE_AI_GREETING`` when
    it says it is an AI in the call's language, else the default.
    """
    return live_socket.SocketPorts(
        stream_turn=lambda req: _agent_retell_stream(req),
        call_status=lambda cid: _call_status(cid),
        ai_greeting=ai_disclosure.voice_greeting(config.VOICE_AI_GREETING, config.VOICE_LIVE_LANGUAGE),
        ai_reminder=ai_disclosure.VOICE_REMINDER,
    )


@router.websocket("/api/voice/retell/llm-websocket/{call_id}")
async def retell_llm_websocket(websocket: WebSocket, call_id: str) -> None:
    """Retell custom-LLM WebSocket — Auto's brain on Retell's actual transport.

    Retell's dashboard takes a ``wss://…/api/voice/retell/llm-websocket`` URL
    and appends ``/{call_id}``; the accepted socket is run by
    ``modules/voice/live_socket.LiveCallSocket`` (the protocol, barge-in, the
    slow-turn ack and the per-turn ``voice_turns`` row).

    Auth: a WS handshake carries no HMAC — the MINTED ``call_id`` is the
    credential (born server-side at ``/api/voice/web-call``, unguessable,
    dead in ~30s unused). No mint row → close 4401 before accept; a call
    that is not live (``live_socket.LIVE_CALL_STATUSES``: ended, failed,
    superseded, anything unknown) → close 4409 before accept. The
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
        minted = (
            db.query(VoiceCall.status).filter(VoiceCall.call_id == str(call_id)).first()
        )
    if minted is None:
        logger.warning("voice_live_ws_rejected reason=unminted_call call=%s", call_id)
        await websocket.close(code=4401)
        return
    if not live_socket.is_live(minted[0]):
        # Only a live call opens the socket (PRD-266 US-008b): an ended or failed
        # call's leaked call_id cannot drive Auto's tools after the call, and an
        # auto_reconnect of a superseded call can't loop back in and start a
        # second voice. Refused before accept.
        logger.info("voice_live_ws_rejected reason=not_live status=%s call=%s", minted[0], call_id)
        await websocket.close(code=live_socket.WS_CLOSE_NOT_LIVE)
        return

    await websocket.accept()
    logger.info("voice_live_ws_open call=%s", call_id)
    await live_socket.LiveCallSocket(websocket, call_id, _socket_ports()).run()
