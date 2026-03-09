"""
Voice Service Client (PRD-74)
Async HTTP client for the self-hosted voice service.
OpenAI-compatible API contract.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

from config import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: Optional[str]
    duration_ms: float  # processing time
    audio_duration_ms: Optional[float]  # input audio length


@dataclass(frozen=True)
class SynthesisResult:
    audio: bytes
    format: str  # "mp3", "opus", "wav"
    duration_ms: float  # processing time
    audio_duration_ms: Optional[float]  # output audio length


class VoiceServiceClient:
    """OpenAI-compatible client for the self-hosted voice service."""

    def __init__(self):
        self.base_url = config.VOICE_SERVICE_URL.rstrip("/")
        self.timeout = config.VOICE_SERVICE_TIMEOUT

    async def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.webm",
        language: Optional[str] = None,
    ) -> TranscriptionResult:
        """POST /v1/audio/transcriptions -- returns transcript + metadata."""
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {"file": (filename, audio)}
            data = {"model": config.VOICE_STT_MODEL}
            if language:
                data["language"] = language

            response = await client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                files=files,
                data=data,
            )
            response.raise_for_status()

        elapsed_ms = (time.monotonic() - start) * 1000
        result = response.json()

        logger.info(
            "voice_stt_complete",
            extra={
                "model": config.VOICE_STT_MODEL,
                "processing_ms": round(elapsed_ms, 1),
                "transcript_length": len(result.get("text", "")),
                "language_detected": result.get("language"),
            },
        )

        return TranscriptionResult(
            text=result["text"],
            language=result.get("language"),
            duration_ms=elapsed_ms,
            audio_duration_ms=result.get("duration"),
        )

    async def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        response_format: str = "mp3",
        model: Optional[str] = None,
        reference_audio: Optional[str] = None,
    ) -> SynthesisResult:
        """POST /v1/audio/speech -- returns audio bytes + metadata.

        Args:
            text: Text to synthesize.
            voice: Voice ID (falls back to config default).
            speed: Playback speed multiplier.
            response_format: Output format (mp3, wav, opus).
            model: TTS model/provider override (e.g. "kokoro", "chatterbox").
            reference_audio: S3 key for cloned-voice reference audio.
        """
        effective_model = model or config.VOICE_TTS_MODEL
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            payload = {
                "input": text,
                "model": effective_model,
                "voice": voice or config.VOICE_TTS_DEFAULT_VOICE,
                "speed": speed,
                "response_format": response_format,
            }
            if reference_audio:
                payload["reference_audio"] = reference_audio

            response = await client.post(
                f"{self.base_url}/v1/audio/speech",
                json=payload,
            )
            response.raise_for_status()

        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "voice_tts_complete",
            extra={
                "model": effective_model,
                "voice": voice or config.VOICE_TTS_DEFAULT_VOICE,
                "text_length": len(text),
                "processing_ms": round(elapsed_ms, 1),
                "audio_size_bytes": len(response.content),
                "has_reference": reference_audio is not None,
            },
        )

        return SynthesisResult(
            audio=response.content,
            format=response_format,
            duration_ms=elapsed_ms,
            audio_duration_ms=None,  # Not provided by OpenAI-compatible API
        )

    async def clone_preview(
        self,
        reference_audio_bytes: bytes,
        text: str,
        voice: Optional[str] = None,
    ) -> SynthesisResult:
        """Synthesize speech using inline reference audio for voice cloning.

        Sends the reference audio bytes directly to the voice service
        rather than an S3 key, for real-time preview during upload flow.
        """
        start = time.monotonic()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/v1/audio/speech",
                files={"reference_audio": ("reference.wav", reference_audio_bytes, "audio/wav")},
                data={
                    "input": text,
                    "model": config.AUTO_VOICE_PROVIDER,
                    "voice": voice or "clone_preview",
                    "response_format": "mp3",
                },
            )
            response.raise_for_status()

        elapsed_ms = (time.monotonic() - start) * 1000

        logger.info(
            "voice_clone_preview_complete",
            extra={
                "text_length": len(text),
                "reference_bytes": len(reference_audio_bytes),
                "processing_ms": round(elapsed_ms, 1),
                "audio_size_bytes": len(response.content),
            },
        )

        return SynthesisResult(
            audio=response.content,
            format="mp3",
            duration_ms=elapsed_ms,
            audio_duration_ms=None,
        )

    async def health(self) -> bool:
        """GET /health -- voice service health check."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False
