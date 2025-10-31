"""
LLM Provider Service
====================

Main entry point for LLM provider functionality.
Maintains backward compatibility with existing imports.
"""

from .clients.base import LLMProvider, LLMConfig, LLMResponse, BaseLLMProvider
from .clients import (
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    AzureProvider,
    HuggingFaceProvider
)
from .manager import LLMManager, create_llm_manager

# Backward compatibility exports
__all__ = [
    # Enums and configs
    'LLMProvider',
    'LLMConfig',
    'LLMResponse',
    'BaseLLMProvider',
    # Providers
    'OpenAIProvider',
    'AnthropicProvider',
    'GoogleProvider',
    'AzureProvider',
    'HuggingFaceProvider',
    # Manager
    'LLMManager',
    'create_llm_manager',
]

