"""
OpenAI Provider Implementation
===============================

OpenAI GPT models provider.
"""

import logging
from typing import Dict, Any, List, Optional

from config import config
from .base import BaseLLMProvider, LLMResponse, accepts_sampling_params, run_blocking
from .openai_chat_request import chat_kwargs, tool_calls_from

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)


def usage_from_openai(usage: Any) -> Dict[str, Any]:
    """OpenAI's accounting: ``prompt_tokens`` already INCLUDES the cached part
    (``prompt_tokens_details.cached_tokens``); the cache figure rides beside it."""
    details = getattr(usage, "prompt_tokens_details", None)
    cached = int(getattr(details, "cached_tokens", 0) or 0) if details is not None else 0
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
        "cache_read_tokens": cached,
        "cache_write_tokens": 0,
    }

class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation"""
    
    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Try multiple sources for API key
        api_key = self.config.api_key or config.OPENAI_API_KEY
        
        # BOOTSTRAP STRATEGY: Don't require key at initialization
        # Only fail when actually making API calls
        if not api_key:
            logger.warning(
                "OpenAI API key not configured. "
                "LLM features will fail until key is added. "
                "Configure 'development_openai' credential or set OPENAI_API_KEY env var."
            )
            self.client = None  # Will fail gracefully on first use
        else:
            client_kwargs = {"api_key": api_key, "timeout": float(self.config.timeout) if self.config.timeout else 180.0}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            if self.config.organization_id:
                client_kwargs["organization"] = self.config.organization_id

            self.client = OpenAI(**client_kwargs)
            logger.info(f"Initialized OpenAI client with model: {self.config.model}")
    
    def _chat_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """#873: a reasoning model (o3, gpt-5) gets no sampling parameters and its
        budget as ``max_completion_tokens``; every other model is sent as before."""
        reasoning = not accepts_sampling_params(self.config.model)
        return chat_kwargs(self.config, messages, sampling=not reasoning, completion_tokens=reasoning)

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using OpenAI API without blocking the event loop"""
        if self.client is None:
            raise ValueError(
                "OpenAI API key not configured. Cannot generate response. "
                "Please configure 'development_openai' credential or set OPENAI_API_KEY env var."
            )
        
        try:
            def _call():
                kwargs = self._chat_kwargs(messages)
                # PRD-17: Add tools if provided
                if tools:
                    formatted_tools = self._sanitize_tools(tools, keep_strict=True)
                    kwargs["tools"] = formatted_tools

                return self.client.chat.completions.create(**kwargs)
            
            try:
                response = await run_blocking(_call)
            except Exception as e:
                msg = str(e)
                if "context_length_exceeded" in msg or "maximum context length" in msg:
                    raise ValueError(
                        f"Your model ({self.config.model}) does not have enough context "
                        f"for this conversation. Please select a model with a larger "
                        f"context window in Settings or start a new chat."
                    ) from e
                raise
            
            # PRD-17: Extract tool calls if present
            tool_calls = tool_calls_from(response.choices[0].message)
            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason

            return LLMResponse(
                content=content or "",  # May be None if tool_calls present
                usage=usage_from_openai(response.usage),
                model=response.model,
                provider="openai",
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using OpenAI API (synchronous)"""
        if self.client is None:
            raise ValueError(
                "OpenAI API key not configured. Cannot generate response. "
                "Please configure 'development_openai' credential or set OPENAI_API_KEY env var."
            )
        
        try:
            response = self.client.chat.completions.create(**self._chat_kwargs(messages))
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=usage_from_openai(response.usage),
                model=response.model,
                provider="openai"
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

