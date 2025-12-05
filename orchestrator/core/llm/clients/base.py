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
            setting = db.query(SystemSetting).filter(
                SystemSetting.key == "vector_store_dimensions"
            ).first()
            if setting and setting.value:
                return int(setting.value)
        finally:
            db.close()
    except Exception:
        pass
    return 1024  # Fallback if DB unavailable


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    AWS_BEDROCK = "aws_bedrock"  # Cost-effective gateway to multiple models
    GROK = "grok"  # xAI Grok models


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    GOOGLE = "google"
    COHERE = "cohere"
    HUGGINGFACE_LOCAL = "huggingface_local"
    HUGGINGFACE_API = "huggingface_api"
    DISABLED = "disabled"  # Deterministic fallback


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str = None
    base_url: Optional[str] = None  # For custom endpoints
    organization_id: Optional[str] = None  # For OpenAI
    secret_key: Optional[str] = None  # For AWS Bedrock IAM auth


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


class DeterministicEmbeddingProvider(BaseEmbeddingProvider):
    """Fallback provider using deterministic hash-based embeddings"""
    
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
        """Generate deterministic embedding based on text hash"""
        import numpy as np
        text_hash = hash(text)
        rng = np.random.default_rng(abs(text_hash) % (2**32))
        return rng.standard_normal(self.dimension).tolist()
    
    def get_dimension(self) -> int:
        return self.dimension
