"""
Base LLM Provider Interface
===========================

Abstract base class for all LLM provider implementations.
Now also includes embedding provider support.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


def _get_system_dimension() -> int:
    """Get embedding dimension from system settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            # PRD-197 S2: PRD-136 renamed the row to (embeddings, dimensions);
            # the old key-only lookup missed on every read since.
            setting = db.query(SystemSetting).filter(
                SystemSetting.category == "embeddings",
                SystemSetting.key == "dimensions",
            ).first()
            if setting and setting.value:
                return int(setting.value)
        finally:
            db.close()
    except Exception:
        pass
    return 2048  # Fallback if DB unavailable (Qwen3-8B Matryoshka truncated to 2048)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    AWS_BEDROCK = "aws_bedrock"  # Cost-effective gateway to multiple models
    GROK = "grok"  # xAI Grok models
    OPENROUTER = "openrouter"  # OpenRouter aggregator (200+ models)


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE_LOCAL = "huggingface_local"
    HUGGINGFACE_API = "huggingface_api"
    OPENROUTER = "openrouter"  # OpenRouter aggregator (20+ embedding models)
    DISABLED = "disabled"  # Deterministic fallback


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 3000
    api_key: str = None
    base_url: Optional[str] = None  # For custom endpoints
    organization_id: Optional[str] = None  # For OpenAI
    secret_key: Optional[str] = None  # For AWS Bedrock IAM auth
    top_p: Optional[float] = None  # Nucleus sampling (0.0-1.0)
    frequency_penalty: Optional[float] = None  # Reduce repetition (0.0-2.0)
    presence_penalty: Optional[float] = None  # Encourage new topics (0.0-2.0)
    stop: Optional[list] = None  # Stop sequences
    timeout: Optional[int] = None  # Request timeout in seconds


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider - reads dimension from system settings"""
    provider: EmbeddingProvider
    model: str
    dimension: int = field(default_factory=_get_system_dimension)
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    cache_dir: Optional[str] = "./model_cache"  # For local models


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    usage: Dict[str, int] = None
    model: str = None
    provider: str = None
    tool_calls: List[Dict[str, Any]] = None  # PRD-17: Support function calling
    finish_reason: str = None  # PRD-17: Track if stopped for tool use
    additional_blocks: List[Dict[str, Any]] = None  # Additional content blocks (images, documents, etc.)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""

    # Fields that only OpenAI's API actually supports on function defs.
    # Other providers (xAI, Anthropic, Google, etc.) reject or ignore them.
    _OPENAI_ONLY_FUNCTION_FIELDS = {"strict"}

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize_client()

    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client"""
        pass

    @abstractmethod
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response from the LLM (async)"""
        pass

    @abstractmethod
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response from the LLM (synchronous)"""
        pass

    # ------------------------------------------------------------------
    # Shared tool sanitisation
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_tool(tool: Dict) -> Dict:
        """Ensure a tool definition is wrapped in {"type": "function", "function": {...}}."""
        if "type" not in tool:
            return {"type": "function", "function": tool}
        return tool

    @classmethod
    def _sanitize_tools(cls, tools: List[Dict], *, keep_strict: bool = False) -> List[Dict]:
        """
        Normalise tool definitions for safe delivery to any LLM provider.

        - Wraps bare function dicts in the {"type":"function","function":{…}} envelope.
        - Removes ``strict`` when it is ``None`` (always invalid).
        - Removes ``strict`` entirely unless *keep_strict* is True (only
          real OpenAI endpoints honour it; xAI, Anthropic, Google reject it).

        Call this in every client's ``generate_response`` before building the
        API request.  For OpenAI-native calls pass ``keep_strict=True``.
        """
        if not tools:
            return tools
        sanitized = []
        for t in tools:
            t = cls._wrap_tool(t)
            fn = t.get("function")
            if isinstance(fn, dict):
                if "strict" in fn:
                    if fn["strict"] is None or not keep_strict:
                        fn = {k: v for k, v in fn.items() if k != "strict"}
                        t = {**t, "function": fn}
            sanitized.append(t)
        return sanitized


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client = None
        self._initialize_client()
    
    @abstractmethod
    def _initialize_client(self):
        """Initialize the provider-specific client"""
        pass
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector for text (async)"""
        pass
    
    def generate_embedding_sync(self, text: str) -> List[float]:
        """Generate embedding vector for text (synchronous)"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self.generate_embedding(text))
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        pass


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no real embedding provider is configured/reachable (PRD-185 S3).

    The old fallback returned hash-seeded RANDOM vectors, so similarity search ran
    over noise and returned confident-but-meaningless matches with nothing on any
    dashboard. Failing loud lets selection paths return a typed EMPTY result
    (honest "no grounding") instead of silent noise.
    """


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """Degraded no-op provider used when no real embedding provider is available.

    Intentionally does NOT produce vectors — ``generate_embedding`` raises
    ``EmbeddingUnavailableError`` so callers fail loud / return empty rather than
    searching over random noise.
    """

    is_degraded = True
    
    def __init__(self, dimension: Optional[int] = None):
        import numpy as np
        self.dimension = dimension or _get_system_dimension()
        self.config = EmbeddingConfig(
            provider=EmbeddingProvider.DISABLED,
            model="deterministic",
            dimension=dimension
        )
        self.client = None
        logger.warning(
            f"Using DeterministicEmbeddingProvider ({dimension}d). "
            "No semantic meaning - configure real provider in Settings > General > Embedding Provider"
        )
    
    def _initialize_client(self):
        pass  # No client needed
    
    async def generate_embedding(self, text: str) -> List[float]:
        """No real provider configured — fail loud instead of returning noise."""
        raise EmbeddingUnavailableError(
            "No embedding provider configured/reachable; refusing to return random vectors."
        )
    
    def get_dimension(self) -> int:
        return self.dimension
