"""
Azure OpenAI Provider Implementation
====================================

Azure OpenAI service provider (OpenAI-compatible API).
"""

import logging
from typing import Dict, Any, List, Optional

from config import config
from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

logger = logging.getLogger(__name__)


class AzureProvider(BaseLLMProvider):
    """Azure OpenAI provider implementation"""
    
    def _initialize_client(self):
        if AzureOpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Azure requires API key, endpoint, and API version
        api_key = self.config.api_key or config.AZURE_OPENAI_API_KEY
        endpoint = self.config.base_url or config.AZURE_OPENAI_ENDPOINT
        api_version = config.AZURE_OPENAI_API_VERSION
        
        # BOOTSTRAP STRATEGY: Don't require key at initialization
        if not api_key or not endpoint:
            logger.warning(
                "Azure OpenAI credentials not configured. "
                "LLM features will fail until credentials are added. "
                "Configure Azure credential or set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars."
            )
            self.client = None
        else:
            self.client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            logger.info(f"Initialized Azure OpenAI client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using Azure OpenAI API"""
        if self.client is None:
            raise ValueError(
                "Azure OpenAI credentials not configured. Cannot generate response. "
                "Please configure Azure credential or set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars."
            )
        
        import asyncio
        loop = asyncio.get_running_loop()
        
        try:
            def _call():
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                # PRD-17: Add tools if provided
                if tools:
                    kwargs["tools"] = self._sanitize_tools(tools, keep_strict=True)
                    kwargs["tool_choice"] = "auto"
                return self.client.chat.completions.create(**kwargs)
            
            response = await loop.run_in_executor(None, _call)
            
            # Extract tool calls if present
            tool_calls = None
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            if hasattr(response.choices[0].message, 'tool_calls') and response.choices[0].message.tool_calls:
                tool_calls = []
                for tc in response.choices[0].message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    })
            
            return LLMResponse(
                content=content or "",
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                provider="azure",
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Azure OpenAI API (synchronous)"""
        if self.client is None:
            raise ValueError(
                "Azure OpenAI credentials not configured. Cannot generate response. "
                "Please configure Azure credential or set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars."
            )
        
        try:
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens
            }
            
            response = self.client.chat.completions.create(**kwargs)
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                provider="azure"
            )
        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise

