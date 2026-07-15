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
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import config
from core.database.database import get_db_session
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


async def _single_complete(response_id: int) -> AsyncIterator[str]:
    """A one-frame stream that closes the turn without words (non-answer events)."""
    yield json.dumps(
        {"response_id": response_id, "content": "", "content_complete": True}
    ) + "\n"


async def _agent_retell_stream(req: RetellLLMRequest) -> AsyncIterator[dict[str, Any]]:
    """Drive Auto's agent loop for one Retell turn and yield Retell frames.

    Reuses the proven chat bridge (``ChatService`` + ``StreamingChatService``):
    the conversation is keyed by the Retell ``call_id`` so multi-turn calls
    continue with memory/context, and each streamed chunk becomes a Retell frame
    the instant it arrives (``retell_response_frames`` owns that streaming property).
    """
    from consumers.chatbot import ChatService, StreamingChatService

    with get_db_session() as db:
        chat_service = ChatService(db)
        streaming_service = StreamingChatService(db, workspace_id=req.workspace_id)

        ws_uuid = uuid.UUID(req.workspace_id)
        conversation_id = f"retell:{req.call_id or req.workspace_id}"

        chat = chat_service.get_chat(conversation_id, workspace_id=ws_uuid)
        if not chat:
            chat = chat_service.create_chat(
                user_id=0,  # webhook principal — no interactive user
                title=(req.user_text or "Voice call")[:50],
                workspace_id=ws_uuid,
            )
            conversation_id = str(chat.id)

        chat_service.save_message(
            chat_id=conversation_id,
            role="user",
            parts=[{"type": "text", "text": req.user_text}],
            workspace_id=req.workspace_id,
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
            agent_id = get_default_agent_id(db, req.workspace_id)

        messages = chat_service.get_messages_by_chat_id(conversation_id)
        message_history = [{"role": m.role, "parts": m.parts} for m in messages]

        agent_chunks = streaming_service.stream_response_with_agent(
            chat_id=conversation_id,
            messages=message_history,
            agent_id=agent_id,
            user_id=0,
            use_orchestrator_llm=True,
        )

        async for frame in retell_response_frames(req.response_id, agent_chunks):
            yield frame


@router.post("/api/voice/retell/llm")
async def retell_llm_webhook(request: Request) -> StreamingResponse:
    """Retell custom-LLM webhook — stream Auto's reply back as Retell frames."""
    if not config.VOICE_ENABLED:
        raise HTTPException(status_code=503, detail="Voice features are disabled")

    body = await request.body()
    signature = request.headers.get("x-retell-signature")
    if not verify_webhook_signature(config.RETELL_WEBHOOK_SECRET, signature, body):
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
