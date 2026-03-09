"""
Self-Hosted Voice Provider (PRD-74)
Calls the automatos-voice service via OpenAI-compatible API.
"""

from modules.voice.client import VoiceServiceClient, SynthesisResult
from modules.voice.providers.base import TTSProvider, VoiceInfo, ProviderHealth


class SelfHostedProvider(TTSProvider):
    """TTS provider that calls the self-hosted automatos-voice service."""

    def __init__(self):
        self._client = VoiceServiceClient()

    async def synthesize(self, text: str, voice: str, **kwargs) -> bytes:
        speed = kwargs.get("speed", 1.0)
        response_format = kwargs.get("response_format", "mp3")
        result: SynthesisResult = await self._client.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            response_format=response_format,
        )
        return result.audio

    async def list_voices(self) -> list[VoiceInfo]:
        # Voice list is managed by the voice service
        # For now, return known defaults
        return [
            VoiceInfo(id="af_heart", name="Heart (Female)", language="en"),
            VoiceInfo(id="af_star", name="Star (Female)", language="en"),
            VoiceInfo(id="am_adam", name="Adam (Male)", language="en"),
            VoiceInfo(id="am_michael", name="Michael (Male)", language="en"),
        ]

    async def health(self) -> ProviderHealth:
        healthy = await self._client.health()
        return ProviderHealth(
            healthy=healthy,
            provider="self-hosted",
            models_loaded=["kokoro"] if healthy else [],
            error=None if healthy else "Voice service unreachable",
        )
