
"""
Flexible LLM Provider Abstraction Layer
=======================================

This module provides a unified interface for different LLM providers including:
- OpenAI GPT models
- Anthropic Claude models
- Future extensibility for other providers

Features:
- Environment-based configuration
- Consistent API across providers
- Error handling and fallback mechanisms
- Token usage tracking
- Temperature and max_tokens configuration
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

@dataclass
class LLMConfig:
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str = None

@dataclass
class LLMResponse:
    content: str
    usage: Dict[str, int] = None
    model: str = None
    provider: str = None

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
    async def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response from the LLM (synchronous)"""
        pass

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation"""
    
    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using OpenAI API without blocking the event loop"""
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            def _call():
                return self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            response = await loop.run_in_executor(None, _call)
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                provider="openai"
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using OpenAI API (synchronous)"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                provider="openai"
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def _initialize_client(self):
        if anthropic is None:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        logger.info(f"Initialized Anthropic client with model: {self.config.model}")
    
    def _convert_messages_to_anthropic_format(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert OpenAI-style messages to Anthropic format"""
        system_message = ""
        user_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            elif msg["role"] == "user":
                user_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                user_messages.append({"role": "assistant", "content": msg["content"]})
        
        return system_message, user_messages
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Anthropic API without blocking the event loop"""
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            system_message, user_messages = self._convert_messages_to_anthropic_format(messages)
            def _call():
                return self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_message,
                    messages=user_messages
                )
            response = await loop.run_in_executor(None, _call)
            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=response.model,
                provider="anthropic"
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Anthropic API (synchronous)"""
        try:
            system_message, user_messages = self._convert_messages_to_anthropic_format(messages)
            
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_message,
                messages=user_messages
            )
            
            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=response.model,
                provider="anthropic"
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise

class LLMManager:
    """Main LLM manager that handles provider selection and configuration"""
    
    def __init__(self, config: LLMConfig = None):
        if config is None:
            config = self._load_config_from_env()
        
        self.config = config
        self.provider = self._create_provider()
    
    def _load_config_from_env(self) -> LLMConfig:
        """Load LLM configuration from environment variables"""
        provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
        
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            logger.warning(f"Unknown LLM provider: {provider_str}, defaulting to OpenAI")
            provider = LLMProvider.OPENAI
        
        # Default models for each provider
        default_models = {
            LLMProvider.OPENAI: "gpt-4",
            LLMProvider.ANTHROPIC: "claude-3-sonnet-20240229"
        }
        
        model = os.getenv("LLM_MODEL", default_models[provider])
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        
        # Get API key based on provider
        api_key = None
        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key
        )
    
    def _create_provider(self) -> BaseLLMProvider:
        """Create the appropriate provider instance"""
        if self.config.provider == LLMProvider.OPENAI:
            return OpenAIProvider(self.config)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return AnthropicProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using the configured provider"""
        return await self.provider.generate_response(messages)
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using the configured provider (synchronous)"""
        return self.provider.generate_response_sync(messages)
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider configuration"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

# Convenience function for quick usage
def create_llm_manager(provider: str = None, model: str = None) -> LLMManager:
    """Create an LLM manager with optional overrides"""
    if provider or model:
        config = LLMManager()._load_config_from_env()
        if provider:
            config.provider = LLMProvider(provider.lower())
        if model:
            config.model = model
        return LLMManager(config)
    else:
        return LLMManager()

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_llm():
        llm = create_llm_manager()
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        response = await llm.generate_response(messages)
        print(f"Response: {response.content}")
        print(f"Provider: {response.provider}")
        print(f"Usage: {response.usage}")
    
    asyncio.run(test_llm())
