"""
OpenRouter Provider Implementation
====================================

OpenRouter aggregator — access 200+ models via OpenAI-compatible API.
Uses the same chat completions format as OpenAI with extra headers.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter aggregator provider — OpenAI-compatible API with 200+ models"""

    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")

        if not api_key:
            logger.warning(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY env var or add credential."
            )
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url or OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": "https://automatos.app",
                    "X-Title": "Automatos AI",
                },
            )
            logger.info(f"Initialized OpenRouter client with model: {self.config.model}")

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response via OpenRouter (OpenAI-compatible)"""
        if self.client is None:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY or add an openrouter credential."
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
                if tools:
                    formatted_tools = []
                    for t in tools:
                        if "type" not in t:
                            formatted_tools.append({"type": "function", "function": t})
                        else:
                            formatted_tools.append(t)
                    kwargs["tools"] = formatted_tools

                    has_tool_results = any(
                        m.get("role") == "tool" for m in (messages or [])
                    )
                    if has_tool_results:
                        kwargs["tool_choice"] = "auto"
                    else:
                        force_tool_choice = any(
                            (m.get("role") == "system" and "You MUST call" in (m.get("content") or ""))
                            for m in (messages or [])
                        )
                        kwargs["tool_choice"] = "required" if force_tool_choice else "auto"

                return self.client.chat.completions.create(**kwargs)

            response = await loop.run_in_executor(None, _call)

            tool_calls = None
            finish_reason = response.choices[0].finish_reason

            # Handle multipart content (e.g., Gemini image models return text + images)
            content = ""
            additional_blocks = []
            msg = response.choices[0].message

            # Debug: log raw message structure to understand multipart responses
            try:
                raw_msg = msg.model_dump()
                raw_content = raw_msg.get("content")
                logger.info(
                    f"[multipart-debug] content type={type(raw_content).__name__}, "
                    f"content_is_none={raw_content is None}, "
                    f"msg_keys={list(raw_msg.keys())}"
                )
                # Log truncated content preview for debugging
                if raw_content and isinstance(raw_content, str) and len(raw_content) > 0:
                    logger.info(f"[multipart-debug] content preview: {raw_content[:200]}")
                elif isinstance(raw_content, list):
                    logger.info(f"[multipart-debug] content is list with {len(raw_content)} parts")
                    for i, part in enumerate(raw_content[:3]):
                        logger.info(f"[multipart-debug] part[{i}]: {str(part)[:200]}")
            except Exception as e:
                logger.warning(f"[multipart-debug] model_dump failed: {e}")
                raw_content = msg.content

            if isinstance(raw_content, list):
                # Multipart response — extract text and inline images
                text_parts = []
                for part in raw_content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "text":
                        text_parts.append(part.get("text", ""))
                    elif ptype == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url:
                            text_parts.append(f"\n\n![Generated Image]({url})\n\n")
                            additional_blocks.append({"type": "image", "url": url})
                content = "\n".join(text_parts)
                if additional_blocks:
                    logger.info(f"Extracted {len(additional_blocks)} image(s) from multipart response")
            else:
                content = msg.content or ""
                # Check if content is empty but raw response might have data elsewhere
                if not content:
                    try:
                        # Some models return image data in non-standard fields
                        raw_choices = response.model_dump().get("choices", [{}])
                        if raw_choices:
                            raw_message = raw_choices[0].get("message", {})
                            logger.info(f"[multipart-debug] empty content, full message keys: {list(raw_message.keys())}")
                            # Log any non-standard fields
                            for k, v in raw_message.items():
                                if k not in ("role", "content", "tool_calls", "refusal"):
                                    val_preview = str(v)[:200] if v else "None"
                                    logger.info(f"[multipart-debug] extra field '{k}': {val_preview}")
                    except Exception as e:
                        logger.warning(f"[multipart-debug] response dump failed: {e}")

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
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) or 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) or 0,
                    "total_tokens": getattr(response.usage, 'total_tokens', 0) or 0,
                },
                model=response.model,
                provider="openrouter",
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                additional_blocks=additional_blocks or None,
            )
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response via OpenRouter (synchronous)"""
        if self.client is None:
            raise ValueError(
                "OpenRouter API key not configured. "
                "Set OPENROUTER_API_KEY or add an openrouter credential."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0) or 0,
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 0) or 0,
                    "total_tokens": getattr(response.usage, 'total_tokens', 0) or 0,
                },
                model=response.model,
                provider="openrouter",
            )
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise
