"""
OpenAI Provider Implementation
===============================

OpenAI GPT models provider.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

def _get_model_context_limit(model: str) -> int:
    """
    Best-effort context window sizes for common OpenAI models used in this repo.
    If unknown, default conservatively to 8192 (avoids 400s).
    """
    m = (model or "").lower()
    # GPT-4 Turbo preview is 128k
    if "gpt-4-turbo" in m or "gpt-4.1" in m:
        return 128000
    # Legacy GPT-4 is typically 8k (unless explicitly 32k)
    if "gpt-4-32k" in m:
        return 32768
    if m.startswith("gpt-4"):
        return 8192
    # GPT-3.5 Turbo often 16k, but keep conservative unless explicitly 16k
    if "gpt-3.5-turbo-16k" in m:
        return 16384
    if "gpt-3.5" in m:
        return 8192
    return 8192


def _safe_max_tokens(model: str, requested_max_tokens: int, prompt_tokens: Optional[int]) -> int:
    """
    Compute a safe max_tokens given an estimated/known prompt token count.
    Leaves a small safety margin to avoid context_length_exceeded.
    """
    limit = _get_model_context_limit(model)
    safety = 256
    if not prompt_tokens:
        # No estimate: cap aggressively for 8k-class models.
        return min(int(requested_max_tokens or 0), 1500) if limit <= 8192 else int(requested_max_tokens or 0)
    available = max(0, limit - int(prompt_tokens) - safety)
    if available <= 0:
        return 256
    return min(int(requested_max_tokens or 0), available)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider implementation"""
    
    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")
        
        # Try multiple sources for API key
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        
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
            client_kwargs = {"api_key": api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            if self.config.organization_id:
                client_kwargs["organization"] = self.config.organization_id
            
            self.client = OpenAI(**client_kwargs)
            logger.info(f"Initialized OpenAI client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using OpenAI API without blocking the event loop"""
        if self.client is None:
            raise ValueError(
                "OpenAI API key not configured. Cannot generate response. "
                "Please configure 'development_openai' credential or set OPENAI_API_KEY env var."
            )
        
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            def _call():
                # NOTE: We don't have a tokenizer here, so we do a conservative cap.
                # Also: if OpenAI returns a context_length_exceeded error, we retry once below.
                kwargs = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": _safe_max_tokens(self.config.model, self.config.max_tokens, prompt_tokens=None),
                }
                # PRD-17: Add tools if provided
                if tools:
                    # Auto-Fix: Wrap legacy function definitions in "type": "function" structure
                    formatted_tools = []
                    for t in tools:
                        # If tool is just the function definition (missing 'type' wrapper)
                        if "type" not in t:
                             formatted_tools.append({
                                 "type": "function",
                                 "function": t
                             })
                        else:
                             formatted_tools.append(t)
                    
                    kwargs["tools"] = formatted_tools

                    # Check if tool results already exist in conversation
                    # If so, switch to auto - the LLM has already acted on the request
                    has_tool_results = any(
                        m.get("role") == "tool" for m in (messages or [])
                    )

                    if has_tool_results:
                        # Tool already executed - let LLM decide if more are needed
                        kwargs["tool_choice"] = "auto"
                    else:
                        # First iteration - only force tool use when explicitly required
                        # CHANGED: Removed "Tool candidates for this request" trigger since
                        # that was causing tools to be forced for conversational queries.
                        # Now only "You MUST call" (explicit instruction) triggers forced tools.
                        force_tool_choice = any(
                            (m.get("role") == "system" and "You MUST call" in (m.get("content") or ""))
                            for m in (messages or [])
                        )
                        kwargs["tool_choice"] = "required" if force_tool_choice else "auto"
                    import json
                    tools_json = json.dumps(formatted_tools, default=str)
                    tools_summary = f"{len(formatted_tools)} tools. Sample: {tools_json[:200]}..."
                    logger.info(
                        f"Sending tools to OpenAI (tool_choice={kwargs['tool_choice']}): {tools_summary}"
                    )
                return self.client.chat.completions.create(**kwargs)
            
            try:
                response = await loop.run_in_executor(None, _call)
            except Exception as e:
                # Retry once on context-length errors by lowering max_tokens.
                msg = str(e)
                if "context_length_exceeded" in msg or "maximum context length" in msg:
                    logger.warning(f"OpenAI context length exceeded; retrying with lower max_tokens. Error: {e}")
                    def _retry_call():
                        kwargs = {
                            "model": self.config.model,
                            "messages": messages,
                            "temperature": self.config.temperature,
                            "max_tokens": 512,
                        }
                        if tools:
                            formatted_tools = []
                            for t in tools:
                                if "type" not in t:
                                    formatted_tools.append({"type": "function", "function": t})
                                else:
                                    formatted_tools.append(t)
                            kwargs["tools"] = formatted_tools
                            # Check if tool results already exist
                            has_tool_results = any(
                                m.get("role") == "tool" for m in (messages or [])
                            )
                            if has_tool_results:
                                kwargs["tool_choice"] = "auto"
                            else:
                                # Only force tool use when explicitly required
                                force_tool_choice = any(
                                    (m.get("role") == "system" and "You MUST call" in (m.get("content") or ""))
                                    for m in (messages or [])
                                )
                                kwargs["tool_choice"] = "required" if force_tool_choice else "auto"
                        return self.client.chat.completions.create(**kwargs)
                    response = await loop.run_in_executor(None, _retry_call)
                else:
                    raise
            
            # PRD-17: Extract tool calls if present
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
                content=content or "",  # May be None if tool_calls present
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
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

