"""
Shared LLM Infrastructure
=========================

Centralized LLM provider management used by all modules.

Components:
- clients/           - Provider implementations (OpenAI, Anthropic, Google, Azure, etc.)
- manager.py         - LLMManager for completion requests
- embedding_manager.py - EmbeddingManager for vector embeddings

Usage:
    from shared.llm import LLMManager, create_llm_manager
    from shared.llm import EmbeddingManager, create_embedding_manager
    from shared.llm import LLMProvider, LLMConfig, LLMResponse
    
    # Create LLM manager
    manager = create_llm_manager(db_session)
    response = await manager.generate("Hello!")
    
    # Create embedding manager
    emb_manager = create_embedding_manager(db_session)
    vector = await emb_manager.generate_embedding("text to embed")
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
    HuggingFaceProvider,
    BedrockProvider,
    GrokProvider,
)
from .manager import LLMManager, create_llm_manager
from .embedding_manager import EmbeddingManager, create_embedding_manager, get_embedding_manager

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
    'BedrockProvider',
    'GrokProvider',
    # Managers
    'LLMManager',
    'create_llm_manager',
    'EmbeddingManager',
    'create_embedding_manager',
    'get_embedding_manager',
]

