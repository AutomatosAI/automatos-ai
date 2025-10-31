"""
Base LLM Provider Interface
===========================

Abstract base class for all LLM provider implementations.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"


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


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    usage: Dict[str, int] = None
    model: str = None
    provider: str = None
    tool_calls: List[Dict[str, Any]] = None  # PRD-17: Support function calling
    finish_reason: str = None  # PRD-17: Track if stopped for tool use


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

