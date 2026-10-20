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

The exports load on first use (PEP 562), not when the package is imported (PRD-262
US-031). Importing a light submodule, such as ``core.llm.list_prices`` from the seed
loader, used to import the whole LLM stack (the manager, every provider client and
their SDKs) through this package; now it imports only that submodule.
"""

from importlib import import_module

# Each export and the module it is defined in.
_EXPORTS = {
    # Provider management
    'LLMManager': 'core.llm.manager',
    'create_llm_manager': 'core.llm.manager',
    'LLMConfig': 'core.llm.clients.base',
    'LLMProvider': 'core.llm.clients.base',
    'LLMResponse': 'core.llm.clients.base',

    # Embeddings
    'EmbeddingManager': 'core.llm.embedding_manager',
    'create_embedding_manager': 'core.llm.embedding_manager',
    'get_embedding_manager': 'core.llm.embedding_manager',

    # Reranking
    'RerankManager': 'core.llm.rerank_manager',
    'get_rerank_manager': 'core.llm.rerank_manager',
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load an export from its module on first use, then keep it on the package."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
