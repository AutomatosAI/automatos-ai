"""
Voice Chat API (PRD-74)
POST /api/chat/voice   -- Voice-in, voice-out chat endpoint.
GET  /api/chat/voice/audio/{message_id} -- Retrieve voice audio.
GET  /api/voice/health -- Voice service health check.
"""

import json
import logging
import uuid
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from modules.voice.client import VoiceServiceClient
from modules.voice.audio import validate_audio, upload_voice_audio, get_voice_audio_url

logger = logging.getLogger(__name__)
router = APIRouter(tags=["voice"])

_voice_client = VoiceServiceClient()


async def _collect_streaming_response(
    db: Session,
    transcript: str,
    conversation_id: str,
    workspace_id: str,
    user_id: int,
    agent_id: Optional[int],
) -> str:
    """
    Feed a text message through the existing streaming chat pipeline
    and collect the full text response (non-streaming).

    The streaming pipeline emits AI SDK format lines:
      0:"text chunk"   -- text content
      2:[{...}]        -- data events
      d:{...}          -- finish message

    We extract only the text chunks (prefix '0:') to build the response.
    """
    from consumers.chatbot import ChatService, StreamingChatService
    from consumers.chatbot.auto import AutoBrain, Action

    chat_service = ChatService(db)
    streaming_service = StreamingChatService(db, workspace_id=workspace_id)

    # Save the user message (transcript) into chat history
    chat = chat_service.get_chat(conversation_id)
    if not chat:
        chat = chat_service.create_chat(
            user_id=user_id,
            title=transcript[:50],
        )
        conversation_id = str(chat.id)

    chat_service.save_message(
        chat_id=conversation_id,
        role="user",
        parts=[{"type": "text", "text": transcript}],
        workspace_id=workspace_id,
    )

    # Get chat history
    messages = chat_service.get_messages_by_chat_id(conversation_id)
    message_history = [{"role": msg.role, "parts": msg.parts} for msg in messages]

    # Determine agent via AutoBrain (same logic as chat.py)
    use_system_llm = False
    complexity_assessment = None

    if agent_id:
        effective_agent_id = agent_id
    else:
        auto_brain = AutoBrain(db, workspace_id)
        complexity_assessment = await auto_brain.assess(transcript, len(message_history))

        if complexity_assessment.action == Action.RESPOND:
            from api.chat import get_default_agent_id
            effective_agent_id = get_default_agent_id(db, workspace_id)
            use_system_llm = True
        elif complexity_assessment.action in (Action.DELEGATE, Action.MISSION):
            from core.routing.cache import get_routing_cache
            from core.routing.engine import UniversalRouter
            from core.routing.ingestors.chatbot import ChatbotIngestor

            try:
                ingestor = ChatbotIngestor()
                envelope = ingestor.ingest(
                    message=transcript,
                    agent_id=None,
                    session_id=conversation_id,
                    request_context=None,
                )
                universal_router = UniversalRouter(db, cache=get_routing_cache())
                routing_decision = await universal_router.route(envelope)

                if routing_decision and routing_decision.route_type == "agent" and routing_decision.agent_id:
                    effective_agent_id = routing_decision.agent_id
                else:
                    from api.chat import get_default_agent_id
                    effective_agent_id = get_default_agent_id(db, workspace_id)
                    use_system_llm = True
            except Exception:
                logger.exception("[voice] Router failed, falling back to default agent")
                from api.chat import get_default_agent_id
                effective_agent_id = get_default_agent_id(db, workspace_id)
                use_system_llm = True
        else:
            from api.chat import get_default_agent_id
            effective_agent_id = get_default_agent_id(db, workspace_id)
            use_system_llm = True

    # Collect streaming output into a single string
    collected_text = []
    skip_composio = (
        complexity_assessment is not None
        and complexity_assessment.action == Action.RESPOND
    )

    async for chunk in streaming_service.stream_response_with_agent(
        chat_id=conversation_id,
        messages=message_history,
        agent_id=effective_agent_id,
        user_id=user_id,
        use_system_llm=use_system_llm,
        skip_composio=skip_composio,
        complexity_assessment=complexity_assessment,
    ):
        # AI SDK format: text lines are '0:"escaped text"\n'
        if chunk.startswith("0:"):
            try:
                text_part = json.loads(chunk[2:].strip())
                if isinstance(text_part, str):
                    collected_text.append(text_part)
            except (json.JSONDecodeError, ValueError):
                pass

    return "".join(collected_text)


@router.post("/api/chat/voice")
async def voice_chat(
    audio: UploadFile = File(...),
    conversation_id: str = Form(""),
    agent_id: Optional[str] = Form(None),
    response_format: str = Form("both"),
    language: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """
    Voice-in, voice-out chat endpoint.

    Flow:
    1. Validate + STT: audio -> transcript
    2. Chat: transcript -> agent response (via existing pipeline)
    3. TTS: agent response -> audio
    4. Store audio in S3
    5. Return audio URL + transcript + agent text
    """
    # Parse agent_id safely (form values are always strings)
    parsed_agent_id: Optional[int] = None
    if agent_id and agent_id.strip():
        try:
            parsed_agent_id = int(agent_id)
        except (ValueError, TypeError):
            pass

    # Default conversation_id if empty
    if not conversation_id or not conversation_id.strip():
        conversation_id = str(uuid.uuid4())

    if not config.VOICE_ENABLED:
        raise HTTPException(status_code=503, detail="Voice features are disabled")

    # Check voice service health
    is_healthy = await _voice_client.health()
    if not is_healthy:
        raise HTTPException(
            status_code=503,
            detail="Voice service unavailable. Text chat remains functional.",
        )

    workspace_id = str(ctx.workspace_id)
    message_id = str(uuid.uuid4())
    start_time = time.monotonic()

    # 1. Validate audio
    audio_bytes = await validate_audio(audio)

    # 2. STT: audio -> transcript
    try:
        stt_result = await _voice_client.transcribe(
            audio=audio_bytes,
            filename=audio.filename or "audio.webm",
            language=language,
        )
    except Exception as e:
        logger.error("voice_stt_failed", extra={"error": str(e), "workspace_id": workspace_id})
        raise HTTPException(
            status_code=422,
            detail="Failed to transcribe audio. Please try again or use text input.",
        )

    transcript = stt_result.text.strip()
    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="Could not understand audio. Please speak clearly and try again.",
        )

    # 3. Route transcript through existing chat pipeline (collected, not streamed)
    from api.chat import get_user_id
    user_id = get_user_id(db)

    try:
        response_text = await _collect_streaming_response(
            db=db,
            transcript=transcript,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
            user_id=user_id,
            agent_id=parsed_agent_id,
        )
    except Exception as e:
        logger.error(
            "voice_chat_routing_failed",
            extra={"error": str(e), "transcript": transcript[:100]},
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to process message")

    if not response_text:
        response_text = "I received your message but couldn't generate a response. Please try again."

    # 4. TTS: agent response -> audio
    # Look up agent's voice profile for personalized TTS
    tts_voice = voice or config.VOICE_TTS_DEFAULT_VOICE
    tts_model = None
    tts_reference_audio = None

    if parsed_agent_id:
        try:
            from core.models.core import Agent
            agent_row = db.query(Agent).filter(Agent.id == parsed_agent_id).first()
            if agent_row and getattr(agent_row, 'voice_profile_id', None):
                from core.models.voice_profiles import VoiceProfile
                vp = db.query(VoiceProfile).filter(
                    VoiceProfile.id == agent_row.voice_profile_id
                ).first()
                if vp:
                    tts_voice = vp.voice_id
                    tts_model = vp.provider
                    tts_reference_audio = vp.reference_audio
                    logger.info("voice_profile_resolved", extra={
                        "agent_id": parsed_agent_id,
                        "profile_id": str(vp.id),
                        "provider": vp.provider,
                        "voice_id": vp.voice_id,
                    })
        except Exception:
            logger.warning("voice_profile_lookup_failed", exc_info=True)

    tts_latency_ms = 0.0
    audio_s3_key = None

    if response_format in ("audio", "both"):
        try:
            tts_result = await _voice_client.synthesize(
                text=response_text,
                voice=tts_voice,
                model=tts_model,
                reference_audio=tts_reference_audio,
            )
            tts_latency_ms = tts_result.duration_ms

            # Store TTS audio in S3
            audio_s3_key = upload_voice_audio(
                workspace_id=workspace_id,
                message_id=message_id,
                audio_bytes=tts_result.audio,
                audio_format="mp3",
            )
        except Exception as e:
            logger.warning("voice_tts_failed", extra={"error": str(e)})
            # TTS failure is non-fatal -- return text response without audio

    total_ms = (time.monotonic() - start_time) * 1000

    logger.info(
        "voice_chat_complete",
        extra={
            "workspace_id": workspace_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "stt_latency_ms": round(stt_result.duration_ms, 1),
            "tts_latency_ms": round(tts_latency_ms, 1),
            "total_ms": round(total_ms, 1),
            "transcript_length": len(transcript),
            "response_length": len(response_text),
        },
    )

    return JSONResponse({
        "conversation_id": conversation_id,
        "message_id": message_id,
        "transcript": transcript,
        "response_text": response_text,
        "audio_url": f"/api/chat/voice/audio/{message_id}" if audio_s3_key else None,
        "audio_format": "mp3",
        "stt_latency_ms": round(stt_result.duration_ms, 1),
        "tts_latency_ms": round(tts_latency_ms, 1),
        "voice_metadata": {
            "stt_model": config.VOICE_STT_MODEL,
            "tts_model": config.VOICE_TTS_MODEL,
            "tts_voice": voice or config.VOICE_TTS_DEFAULT_VOICE,
            "audio_s3_key": audio_s3_key,
        },
    })


@router.get("/api/chat/voice/audio/{message_id}")
async def get_voice_audio(
    message_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
):
    """Redirect to presigned S3 URL for audio playback."""
    workspace_id = str(ctx.workspace_id)
    s3_key = f"workspaces/{workspace_id}/voice/{message_id}.mp3"

    try:
        url = get_voice_audio_url(s3_key)
        return RedirectResponse(url=url)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=404, detail="Audio not found or expired")


@router.get("/api/voice/health")
async def voice_health():
    """Check voice service health."""
    is_healthy = await _voice_client.health()
    return JSONResponse(
        status_code=200 if is_healthy else 503,
        content={
            "voice_enabled": config.VOICE_ENABLED,
            "voice_service_healthy": is_healthy,
            "voice_service_url": config.VOICE_SERVICE_URL,
        },
    )
