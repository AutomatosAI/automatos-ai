"""
LLM Provider Management - Core Platform Infrastructure
=======================================================

Centralized LLM infrastructure for the Automatos platform.

Components:
- manager.py           - LLM provider management (create_llm_manager)
- embedding_manager.py - Embedding generation
- clients/             - Provider-specific clients (OpenAI, Anthropic, HuggingFace)
- rerank_manager.py    - Reranking

Usage:
    from core.llm import create_llm_manager, get_embedding_manager
    
    llm = create_llm_manager(service_name="my_service")
    response = await llm.complete(prompt)
"""

# Main LLM management (from shared/llm)
from core.llm.manager import (
    LLMManager,
    create_llm_manager,
)
from core.llm.clients.base import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
)
from core.llm.embedding_manager import (
    EmbeddingManager,
    create_embedding_manager,
    get_embedding_manager,
)

# LLM utilities (original core/llm)
from core.llm.rerank_manager import RerankManager, get_rerank_manager

__all__ = [
    # Provider management
    'LLMManager',
    'create_llm_manager',
    'LLMConfig',
    'LLMProvider',
    'LLMResponse',

    # Embeddings
    'EmbeddingManager',
    'create_embedding_manager',
    'get_embedding_manager',

    # Reranking
    'RerankManager',
    'get_rerank_manager',
]
