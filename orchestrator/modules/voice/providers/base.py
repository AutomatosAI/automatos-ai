"""
TTS Provider Abstraction (PRD-74)
Swap TTS engines without changing integration code.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    name: str
    language: Optional[str] = None
    preview_url: Optional[str] = None


@dataclass(frozen=True)
class ProviderHealth:
    healthy: bool
    provider: str
    models_loaded: list[str]
    error: Optional[str] = None


class TTSProvider(ABC):
    """Abstract TTS provider -- swap engines without changing integration code."""

    @abstractmethod
    async def synthesize(self, text: str, voice: str, **kwargs) -> bytes:
        """Convert text to audio bytes."""

    @abstractmethod
    async def list_voices(self) -> list[VoiceInfo]:
        """Available voices for this provider."""

    @abstractmethod
    async def health(self) -> ProviderHealth:
        """Provider health status."""
