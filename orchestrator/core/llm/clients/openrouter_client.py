"""
OpenRouter Provider Implementation
====================================

OpenRouter aggregator — access 200+ models via OpenAI-compatible API.
Uses the same chat completions format as OpenAI with extra headers.
"""

import json
import logging
from typing import Dict, Any, List, Optional

from config import config
from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = config.OPENROUTER_BASE_URL


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter aggregator provider — OpenAI-compatible API with 200+ models"""

    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = self.config.api_key or config.OPENROUTER_API_KEY

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
                    "HTTP-Referer": config.OPENROUTER_SITE_URL,
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
                    kwargs["tools"] = self._sanitize_tools(tools)

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

                try:
                    return self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    err_str = str(exc)
                    if tools and ("not support tool use" in err_str or "No endpoints found that support tool" in err_str):
                        logger.warning(
                            "Model %s does not support tool use — retrying without tools",
                            self.config.model,
                        )
                        kwargs.pop("tools", None)
                        kwargs.pop("tool_choice", None)
                        return self.client.chat.completions.create(**kwargs)
                    if tools and "Tool choice must be auto" in err_str and kwargs.get("tool_choice") != "auto":
                        logger.warning(
                            "Model %s provider requires tool_choice=auto — retrying",
                            self.config.model,
                        )
                        kwargs["tool_choice"] = "auto"
                        return self.client.chat.completions.create(**kwargs)
                    raise

            response = await loop.run_in_executor(None, _call)

            if not response.choices:
                raise ValueError(
                    f"OpenRouter returned empty choices (model={self.config.model}). "
                    "This usually means the provider rejected the request."
                )

            tool_calls = None
            finish_reason = response.choices[0].finish_reason

            # Handle multipart/image content from models like Gemini via OpenRouter.
            # OpenRouter returns images in a separate `images` field on the message
            # (not inside `content`), with structure:
            #   images: [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
            content = ""
            additional_blocks = []
            msg = response.choices[0].message

            try:
                raw_msg = msg.model_dump()
            except Exception:
                raw_msg = {}

            # 1. Extract text content (may be string, list of parts, or empty)
            raw_content = raw_msg.get("content")
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)
            else:
                content = msg.content or ""

            # 2. Extract images from the `images` field (OpenRouter/Gemini image models)
            raw_images = raw_msg.get("images") or []
            for img in raw_images:
                if not isinstance(img, dict):
                    continue
                url = ""
                if img.get("type") == "image_url":
                    url = (img.get("image_url") or {}).get("url", "")
                elif "url" in img:
                    url = img["url"]
                if url:
                    content += f"\n\n![Generated Image]({url})\n\n"
                    additional_blocks.append({"type": "image", "url": url})

            # 3. Also check content list for inline image_url parts
            if isinstance(raw_content, list):
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url:
                            content += f"\n\n![Generated Image]({url})\n\n"
                            additional_blocks.append({"type": "image", "url": url})

            if additional_blocks:
                logger.info(f"Extracted {len(additional_blocks)} image(s) from OpenRouter response")

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
