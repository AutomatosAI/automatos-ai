"""
Grok (xAI) Provider Implementation
===================================

xAI Grok models provider using OpenAI-compatible API.
"""

import logging
from typing import Dict, Any, List, Optional

from config import config
from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


class GrokProvider(BaseLLMProvider):
    """xAI Grok provider implementation (OpenAI-compatible API)"""
    
    # xAI API endpoint
    API_BASE_URL = "https://api.x.ai/v1"
    
    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Try multiple sources for API key
        api_key = self.config.api_key or config.XAI_API_KEY
        
        if not api_key:
            logger.warning(
                "xAI API key not configured. "
                "LLM features will fail until key is added. "
                "Configure 'development_xai' credential or set XAI_API_KEY env var."
            )
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.API_BASE_URL,
                timeout=float(self.config.timeout) if self.config.timeout else 180.0,
            )
            logger.info(f"Initialized Grok client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using Grok API"""
        if self.client is None:
            raise ValueError(
                "xAI API key not configured. Cannot generate response. "
                "Please configure 'development_xai' credential or set XAI_API_KEY env var."
            )
        
        import asyncio
        loop = asyncio.get_running_loop()
        
        try:
            def _call():
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                }
                if self.config.top_p is not None:
                    kwargs["top_p"] = self.config.top_p
                if self.config.frequency_penalty is not None:
                    kwargs["frequency_penalty"] = self.config.frequency_penalty
                if self.config.presence_penalty is not None:
                    kwargs["presence_penalty"] = self.config.presence_penalty
                if self.config.stop is not None:
                    kwargs["stop"] = self.config.stop
                if tools:
                    kwargs["tools"] = self._sanitize_tools(tools)
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
            
            # Build usage dict
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=content or "",
                usage=usage,
                model=response.model,
                provider="grok",
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Grok API (synchronous)"""
        if self.client is None:
            raise ValueError(
                "xAI API key not configured. Cannot generate response. "
                "Please configure 'development_xai' credential or set XAI_API_KEY env var."
            )
        
        try:
            sync_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            if self.config.top_p is not None:
                sync_kwargs["top_p"] = self.config.top_p
            if self.config.frequency_penalty is not None:
                sync_kwargs["frequency_penalty"] = self.config.frequency_penalty
            if self.config.presence_penalty is not None:
                sync_kwargs["presence_penalty"] = self.config.presence_penalty
            if self.config.stop is not None:
                sync_kwargs["stop"] = self.config.stop
            response = self.client.chat.completions.create(**sync_kwargs)
            
            usage = {}
            if hasattr(response, 'usage') and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=usage,
                model=response.model,
                provider="grok"
            )
        except Exception as e:
            logger.error(f"Grok API error: {e}")
            raise

