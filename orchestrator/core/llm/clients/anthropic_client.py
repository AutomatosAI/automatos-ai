"""
Anthropic Provider Implementation
=================================

Anthropic Claude models provider.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    import anthropic
except ImportError:
    anthropic = None

logger = logging.getLogger(__name__)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider implementation"""
    
    def _initialize_client(self):
        if anthropic is None:
            raise ImportError("Anthropic package not installed. Run: pip install anthropic")
        
        # Try multiple sources for API key
        api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
        
        # BOOTSTRAP STRATEGY: Don't require key at initialization
        # Only fail when actually making API calls
        if not api_key:
            logger.warning(
                "Anthropic API key not configured. "
                "LLM features will fail until key is added. "
                "Configure 'development_anthropic' credential or set ANTHROPIC_API_KEY env var."
            )
            self.client = None  # Will fail gracefully on first use
        else:
            client_kwargs = {"api_key": api_key}
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            
            self.client = anthropic.Anthropic(**client_kwargs)
            logger.info(f"Initialized Anthropic client with model: {self.config.model}")
    
    def _convert_messages_to_anthropic_format(self, messages: List[Dict[str, Any]]) -> tuple:
        """
        Convert OpenAI-style messages to Anthropic format.
        
        Handles:
        - System messages (becomes system parameter)
        - User messages (can contain text or tool results)
        - Assistant messages (can contain text or tool use)
        
        Anthropic tool result format:
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "...", "content": "..."}
            ]
        }
        """
        system_message = ""
        user_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
            elif role == "user":
                # Handle tool results (Anthropic expects array of content blocks)
                if isinstance(content, str):
                    try:
                        # Try parsing as JSON (might be tool results array as JSON string)
                        parsed = json.loads(content)
                        if isinstance(parsed, list):
                            # It's an array - likely tool results
                            user_messages.append({"role": "user", "content": parsed})
                        else:
                            # Regular text content
                            user_messages.append({"role": "user", "content": content})
                    except (json.JSONDecodeError, TypeError):
                        # Not JSON, regular text
                        user_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Already formatted as content blocks
                    user_messages.append({"role": "user", "content": content})
                else:
                    # Fallback: convert to string
                    user_messages.append({"role": "user", "content": str(content)})
            elif role == "assistant":
                # Assistant messages can have text or tool use
                if isinstance(content, str):
                    user_messages.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    # Content blocks (might include tool use)
                    user_messages.append({"role": "assistant", "content": content})
                else:
                    user_messages.append({"role": "assistant", "content": str(content)})
                # Note: Anthropic handles tool_use blocks in assistant messages automatically
        
        return system_message, user_messages
    
    def _convert_tools_to_anthropic_format(self, tools: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style tools to Anthropic format.
        
        OpenAI format:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "...",
                "parameters": {...}
            }
        }
        
        Anthropic format:
        {
            "name": "tool_name",
            "description": "...",
            "input_schema": {...}  # Note: Anthropic uses input_schema, not parameters
        }
        """
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                # Handle OpenAI format: {"type": "function", "function": {...}}
                if tool.get("type") == "function" and "function" in tool:
                    func_def = tool["function"]
                    anthropic_tools.append({
                        "name": func_def.get("name"),
                        "description": func_def.get("description", ""),
                        "input_schema": func_def.get("parameters", {})
                    })
                # Handle direct function format: {"name": ..., "description": ..., "parameters": ...}
                elif "name" in tool and ("parameters" in tool or "input_schema" in tool):
                    anthropic_tools.append({
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters") or tool.get("input_schema", {})
                    })
        return anthropic_tools
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using Anthropic API without blocking the event loop"""
        if self.client is None:
            raise ValueError(
                "Anthropic API key not configured. Cannot generate response. "
                "Please configure 'development_anthropic' credential or set ANTHROPIC_API_KEY env var."
            )
        
        import asyncio
        loop = asyncio.get_running_loop()
        try:
            system_message, user_messages = self._convert_messages_to_anthropic_format(messages)
            def _call():
                kwargs = {
                    "model": self.config.model,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "system": system_message,
                    "messages": user_messages
                }
                # PRD-17: Add tools if provided (convert to Anthropic format)
                if tools:
                    anthropic_tools = self._convert_tools_to_anthropic_format(tools)
                    if anthropic_tools:
                        kwargs["tools"] = anthropic_tools
                return self.client.messages.create(**kwargs)
            
            response = await loop.run_in_executor(None, _call)
            
            # PRD-17: Extract tool calls if present
            tool_calls = None
            content = ""
            finish_reason = response.stop_reason
            
            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input)
                        }
                    })
            
            return LLMResponse(
                content=content or "",
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=response.model,
                provider="anthropic",
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Anthropic API (synchronous)"""
        if self.client is None:
            raise ValueError(
                "Anthropic API key not configured. Cannot generate response. "
                "Please configure 'development_anthropic' credential or set ANTHROPIC_API_KEY env var."
            )
        
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

