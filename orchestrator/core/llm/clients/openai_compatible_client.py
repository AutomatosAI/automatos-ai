"""
OpenAI-compatible provider adapter (PRD-236 S0.2)
=================================================

One client for every provider that speaks the OpenAI chat-completions
protocol behind its own base URL: OpenRouter, NVIDIA (build.nvidia.com),
DeepSeek. Which one is decided by ``LLMConfig.provider`` through the
registry (``core/llm/providers.py``): base URL, env key, attribution headers
and the rate-limit note all come from the spec, never from a per-provider
subclass. This file is the former ``openrouter_client.py`` generalised; the
OpenRouter-specific behaviours (Referer/Title headers, the ``images`` field
on image models, the two tool-choice retries) are kept because they are
harmless on the other providers.

Rate limits are honest: a 429 raises ``ProviderRateLimitError`` carrying the
spec's note. There is no retry and no reroute here — the manager has no
silent fallbacks (PRD-236 Q2: free must never silently become paid).
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseLLMProvider, LLMConfig, LLMResponse
from core.llm import providers as registry
from core.llm.reasoning import coalesce_reasoning, reasoning_from_fields, split_think_tags

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover — the SDK is a hard dependency in production
    OpenAI = None

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 180.0


class ProviderRateLimitError(ValueError):
    """The serving provider refused the call with HTTP 429."""


def _is_rate_limited(exc: Exception) -> bool:
    if getattr(exc, "status_code", None) == 429:
        return True
    text = str(exc)
    return "429" in text or "rate limit" in text.lower() or "rate_limit" in text.lower()


class _StreamAssembler:
    """Folds chat-completion stream chunks into one ``LLMResponse``.

    Text and reasoning deltas are yielded as they arrive; tool-call argument
    fragments (keyed by ``index``) are concatenated; the last chunk's usage and
    finish_reason win. Inline ``<think>`` tags are lifted out of the final text
    for models that think in-band.
    """

    def __init__(self) -> None:
        self.text: List[str] = []
        self.reasoning: List[str] = []
        self.tool_calls: Dict[int, Dict[str, Any]] = {}
        self.finish_reason: Optional[str] = None
        self.usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model: Optional[str] = None

    def feed(self, chunk: Any) -> List[Tuple[str, str]]:
        """Absorb one chunk; return the (kind, text) deltas to deliver live."""
        out: List[Tuple[str, str]] = []
        self.model = getattr(chunk, "model", None) or self.model
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self.usage = {
                "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(usage, "total_tokens", 0) or 0,
            }
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return out
        choice = choices[0]
        self.finish_reason = getattr(choice, "finish_reason", None) or self.finish_reason
        delta = getattr(choice, "delta", None)
        if delta is None:
            return out
        try:
            raw = delta.model_dump()
        except Exception:
            raw = {k: getattr(delta, k, None) for k in ("content", "reasoning_content", "reasoning", "tool_calls")}
        reasoning = reasoning_from_fields(raw)
        if reasoning:
            self.reasoning.append(reasoning)
            out.append(("reasoning", reasoning))
        content = raw.get("content")
        if isinstance(content, str) and content:
            self.text.append(content)
            out.append(("text", content))
        for tc in raw.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            index = int(tc.get("index") or 0)
            slot = self.tool_calls.setdefault(index, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("type"):
                slot["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
        return out

    def response(self, provider: str, *, streamed: bool) -> LLMResponse:
        content, inline_reasoning = split_think_tags("".join(self.text))
        reasoning = coalesce_reasoning("".join(self.reasoning) or None, inline_reasoning)
        calls = [self.tool_calls[i] for i in sorted(self.tool_calls)] or None
        return LLMResponse(
            content=content,
            usage=self.usage,
            model=self.model,
            provider=provider,
            tool_calls=calls,
            finish_reason=self.finish_reason,
            reasoning=reasoning,
            streamed=streamed,
        )


class OpenAICompatibleProvider(BaseLLMProvider):
    """Chat completions against any OpenAI-compatible base URL, spec-driven."""

    def __init__(self, config: LLMConfig):
        self.spec = registry.get_spec(config.provider.value if config.provider else None)
        if self.spec is None or self.spec.adapter != registry.ADAPTER_OPENAI_COMPATIBLE:
            raise ValueError(
                f"Provider '{getattr(config.provider, 'value', config.provider)}' is not an "
                "OpenAI-compatible provider in the registry (core/llm/providers.py)"
            )
        super().__init__(config)

    # ------------------------------------------------------------------ #
    # Client
    # ------------------------------------------------------------------ #

    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = self.config.api_key or registry.env_api_key(self.spec.slug)
        base_url = self.config.base_url or registry.base_url_for(self.spec.slug)

        if not api_key:
            logger.warning(
                "%s API key not configured. Set %s or add a key in Settings → API Keys.",
                self.spec.label, self.spec.env_key or "the provider key",
            )
            self.client = None
            return

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            "timeout": float(self.config.timeout) if self.config.timeout else DEFAULT_TIMEOUT_SECONDS,
        }
        headers = registry.headers_for(self.spec.slug)
        if headers:
            client_kwargs["default_headers"] = headers
        self.client = OpenAI(**client_kwargs)
        logger.info("Initialized %s client with model: %s", self.spec.label, self.config.model)

    def _require_client(self):
        if self.client is None:
            raise ValueError(
                f"{self.spec.label} API key not configured. "
                f"Set {self.spec.env_key or 'the provider key'} or add a {self.spec.slug} key "
                "in Settings → API Keys."
            )

    def _rate_limit_error(self, exc: Exception) -> ProviderRateLimitError:
        note = self.spec.rate_limit_note or "The provider's rate limit was reached."
        return ProviderRateLimitError(
            f"{self.spec.label} rate limit reached for model '{self.config.model}'. {note}"
        )

    # ------------------------------------------------------------------ #
    # Requests
    # ------------------------------------------------------------------ #

    def _base_kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
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
        return kwargs

    def _request_kwargs(self, messages: List[Dict[str, str]], tools: Optional[List[Dict]]) -> Dict[str, Any]:
        """The chat-completions request for ``messages`` (+ tools and tool_choice)."""
        kwargs = self._base_kwargs(messages)
        if tools:
            kwargs["tools"] = self._sanitize_tools(tools)
            has_tool_results = any(m.get("role") == "tool" for m in (messages or []))
            if has_tool_results:
                kwargs["tool_choice"] = "auto"
            else:
                force_tool_choice = any(
                    (m.get("role") == "system" and "You MUST call" in (m.get("content") or ""))
                    for m in (messages or [])
                )
                kwargs["tool_choice"] = "required" if force_tool_choice else "auto"
        return kwargs

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate a response (OpenAI-compatible chat completions)."""
        self._require_client()

        import asyncio
        loop = asyncio.get_running_loop()

        try:
            def _call():
                kwargs = self._request_kwargs(messages, tools)

                try:
                    return self.client.chat.completions.create(**kwargs)
                except Exception as exc:
                    if _is_rate_limited(exc):
                        raise self._rate_limit_error(exc) from exc
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
                    f"{self.spec.label} returned empty choices (model={self.config.model}). "
                    "This usually means the provider rejected the request."
                )

            tool_calls = None
            finish_reason = response.choices[0].finish_reason

            # Multipart/image content (OpenRouter image models return an
            # `images` field on the message; content may be a list of parts).
            content = ""
            additional_blocks = []
            msg = response.choices[0].message

            try:
                raw_msg = msg.model_dump()
            except Exception:
                raw_msg = {}

            raw_content = raw_msg.get("content")
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = "\n".join(text_parts)
            else:
                content = msg.content or ""

            # PRD-238 S1: the reasoning channel — a dedicated field on the
            # message (DeepSeek/Kimi/NVIDIA NIM) or inline <think> tags — is
            # lifted out so it is never the answer.
            content, inline_reasoning = split_think_tags(content)
            reasoning = coalesce_reasoning(reasoning_from_fields(raw_msg), inline_reasoning)

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

            if isinstance(raw_content, list):
                for part in raw_content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url:
                            content += f"\n\n![Generated Image]({url})\n\n"
                            additional_blocks.append({"type": "image", "url": url})

            if additional_blocks:
                logger.info("Extracted %d image(s) from %s response", len(additional_blocks), self.spec.label)

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
                usage=self._usage(response),
                model=response.model,
                provider=self.spec.slug,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                additional_blocks=additional_blocks or None,
                reasoning=reasoning,
            )
        except Exception as e:
            logger.error("%s API error: %s", self.spec.label, e)
            raise

    # ------------------------------------------------------------------ #
    # Streaming (PRD-238 S2)
    # ------------------------------------------------------------------ #

    async def stream_response(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict] = None,
        on_delta=None,
    ) -> LLMResponse:
        """Generate with ``stream=True``, handing every text / reasoning delta to
        ``on_delta(kind, text)`` (``kind`` is ``"text"`` or ``"reasoning"``) as it
        arrives, and return the same ``LLMResponse`` the non-streaming path
        would — tool calls re-assembled from their argument fragments, the
        reasoning channel separated, ``streamed=True`` when anything was
        delivered live. The sync SDK iterator runs in a worker thread; deltas
        cross to the event loop through a queue so nothing blocks it.
        """
        self._require_client()

        import asyncio
        import queue as _queue

        loop = asyncio.get_running_loop()
        handoff: "asyncio.Queue[Any]" = asyncio.Queue()
        _END = object()

        def _run_stream():
            kwargs = self._request_kwargs(messages, tools)
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
            try:
                iterator = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                if _is_rate_limited(exc):
                    raise self._rate_limit_error(exc) from exc
                if "stream_options" in str(exc):
                    kwargs.pop("stream_options", None)
                    iterator = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            for chunk in iterator:
                loop.call_soon_threadsafe(handoff.put_nowait, chunk)

        async def _produce():
            try:
                await loop.run_in_executor(None, _run_stream)
            except Exception as exc:  # surfaced to the consumer below
                loop.call_soon_threadsafe(handoff.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(handoff.put_nowait, _END)

        producer = asyncio.ensure_future(_produce())
        assembler = _StreamAssembler()
        streamed = False
        try:
            while True:
                item = await handoff.get()
                if item is _END:
                    break
                if isinstance(item, Exception):
                    raise item
                for kind, text in assembler.feed(item):
                    streamed = True
                    if on_delta is not None:
                        await on_delta(kind, text)
        finally:
            await producer
        return assembler.response(self.spec.slug, streamed=streamed)

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate a response (synchronous)."""
        self._require_client()
        try:
            try:
                response = self.client.chat.completions.create(**self._base_kwargs(messages))
            except Exception as exc:
                if _is_rate_limited(exc):
                    raise self._rate_limit_error(exc) from exc
                raise
            return LLMResponse(
                content=response.choices[0].message.content,
                usage=self._usage(response),
                model=response.model,
                provider=self.spec.slug,
            )
        except Exception as e:
            logger.error("%s API error: %s", self.spec.label, e)
            raise

    @staticmethod
    def _usage(response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }
