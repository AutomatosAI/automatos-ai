"""
LLM Provider Service
====================

Main entry point for LLM provider functionality.
Supports both LLM completions and embeddings.
Maintains backward compatibility with existing imports.
"""

from .clients.base import (
    LLMProvider, 
    LLMConfig, 
    LLMResponse, 
    BaseLLMProvider,
    EmbeddingProvider,
    EmbeddingConfig,
    BaseEmbeddingProvider,
    DeterministicEmbeddingProvider
)
from .clients import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    AzureProvider,
    HuggingFaceProvider
)
from .manager import LLMManager, create_llm_manager
from .embedding_manager import EmbeddingManager, create_embedding_manager, get_embedding_manager

#Backward compatibility exports
__all__ = [
    # Enums and configs
    'LLMProvider',
    'LLMConfig',
    'LLMResponse',
    'BaseLLMProvider',
    # Embedding classes
    'EmbeddingProvider',
    'EmbeddingConfig',
    'BaseEmbeddingProvider',
    'DeterministicEmbeddingProvider',
    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'GoogleProvider',
    'AzureProvider',
    'HuggingFaceProvider',
    # Managers
    'LLMManager',
    'create_llm_manager',
    'EmbeddingManager',
    'create_embedding_manager',
    'get_embedding_manager',
]
