"""
LLM Provider Management - Core Platform Infrastructure
=======================================================

Centralized LLM infrastructure for the Automatos platform.

Components:
- manager.py           - LLM provider management (create_llm_manager)
- embedding_manager.py - Embedding generation
- clients/             - Provider-specific clients (OpenAI, Anthropic, HuggingFace)
- function_registry.py - LLM function specifications
- function_executor.py - Function execution
- response_parser.py   - Response parsing
- semantic_skill_matcher.py - Skill matching

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
)
from core.llm.embedding_manager import (
    EmbeddingManager,
    create_embedding_manager,
    get_embedding_manager,
)

# LLM utilities (original core/llm)
from core.llm.orchestrator_llm import OrchestratorLLM, LLMResponse, ReasoningMode
from core.llm.function_registry import FunctionRegistry, FunctionSpec, FunctionParameter, FunctionCategory
from core.llm.function_executor import FunctionExecutor, FunctionResult
from core.llm.response_parser import ResponseParser, ParsedResponse
from core.llm.semantic_skill_matcher import SemanticSkillMatcher, get_skill_matcher
from core.llm.rerank_manager import RerankManager, get_rerank_manager

__all__ = [
    # Provider management
    'LLMManager',
    'create_llm_manager',
    'LLMConfig',
    'LLMProvider',
    
    # Embeddings
    'EmbeddingManager',
    'create_embedding_manager',
    'get_embedding_manager',
    
    # Orchestrator LLM
    'OrchestratorLLM',
    'LLMResponse',
    'ReasoningMode',
    
    # Function calling
    'FunctionRegistry',
    'FunctionSpec',
    'FunctionParameter',
    'FunctionCategory',
    'FunctionExecutor',
    'FunctionResult',
    
    # Response parsing
    'ResponseParser',
    'ParsedResponse',
    
    # Skill matching
    'SemanticSkillMatcher',
    'get_skill_matcher',

    # Reranking
    'RerankManager',
    'get_rerank_manager',
]
