"""
Flexible LLM Provider Abstraction Layer
=======================================

This module provides a unified interface for different LLM providers including:
- OpenAI GPT models
- Anthropic Claude models
- Google Gemini models
- Azure OpenAI models
- HuggingFace Inference API models

Features:
- Environment-based configuration
- System settings integration (per-service configuration)
- Consistent API across providers
- Error handling and fallback mechanisms
- Token usage tracking
- Temperature and max_tokens configuration
- Lazy loading (doesn't fail on startup)

PRD-25: Refactored into modular structure for better maintainability.
Maintains backward compatibility with existing imports.
"""

# Backward compatibility imports - maintain existing API
# Import from the modular subdirectory package
# Python handles both llm_provider.py (this file) and llm_provider/ (directory)
# by making this file take precedence for 'import llm_provider', but we can
# import from the subdirectory package using the full module path
import sys
import os

# Get the current directory
_current_dir = os.path.dirname(os.path.abspath(__file__))
_llm_provider_pkg = os.path.join(_current_dir, 'llm_provider')

# Import from the subdirectory package
if os.path.exists(_llm_provider_pkg):
    # Add parent to path temporarily for clean imports
    if _current_dir not in sys.path:
        sys.path.insert(0, _current_dir)
    
    try:
        # Import from the package subdirectory
        from llm_provider.manager import LLMManager, create_llm_manager
        from llm_provider.clients.base import (
            LLMProvider,
            LLMConfig,
            LLMResponse,
            BaseLLMProvider
        )
        from llm_provider.clients import (
            OpenAIProvider,
            AnthropicProvider,
            GoogleProvider,
            AzureProvider,
            HuggingFaceProvider,
        )
    except ImportError as e:
        # If direct import fails, try relative import
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "llm_provider_pkg",
            os.path.join(_llm_provider_pkg, "__init__.py")
        )
        if spec and spec.loader:
            llm_provider_pkg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_provider_pkg)
            
            # Import from the loaded module
            LLMManager = llm_provider_pkg.LLMManager
            create_llm_manager = llm_provider_pkg.create_llm_manager
            LLMProvider = llm_provider_pkg.LLMProvider
            LLMConfig = llm_provider_pkg.LLMConfig
            LLMResponse = llm_provider_pkg.LLMResponse
            BaseLLMProvider = llm_provider_pkg.BaseLLMProvider
            OpenAIProvider = llm_provider_pkg.OpenAIProvider
            AnthropicProvider = llm_provider_pkg.AnthropicProvider
            GoogleProvider = llm_provider_pkg.GoogleProvider
            AzureProvider = llm_provider_pkg.AzureProvider
            HuggingFaceProvider = llm_provider_pkg.HuggingFaceProvider
        else:
            raise ImportError(f"Failed to load llm_provider package: {e}")
else:
    raise ImportError(f"llm_provider package directory not found at {_llm_provider_pkg}")

# Re-export everything for backward compatibility
__all__ = [
    'LLMProvider',
    'LLMConfig',
    'LLMResponse',
    'BaseLLMProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'GoogleProvider',
    'AzureProvider',
    'HuggingFaceProvider',
    'LLMManager',
    'create_llm_manager',
]

# Convenience function for quick usage (backward compatible)
def create_llm_manager_legacy(provider: str = None, model: str = None) -> LLMManager:
    """
    Create an LLM manager with optional overrides (legacy API).
    
    Note: This function uses 'orchestrator' service by default.
    For per-service configuration, use create_llm_manager(service_name="codegraph").
    
    Args:
        provider: Optional provider override
        model: Optional model override
        
    Returns:
        LLMManager instance configured for 'orchestrator' service
    """
    return create_llm_manager(
        service_name="orchestrator",
        provider=provider,
        model=model
    )

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_llm():
        # New way: Per-service configuration
        llm = create_llm_manager(service_name="orchestrator")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = await llm.generate_response(messages)
        print(f"Response: {response.content}")
        print(f"Provider: {response.provider}")
        print(f"Usage: {response.usage}")
    
    asyncio.run(test_llm())
