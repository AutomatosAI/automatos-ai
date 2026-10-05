"""
Azure OpenAI (Microsoft Foundry) Provider
=========================================

#873: this client used the legacy ``AzureOpenAI`` client pinned to
``api-version=2024-02-15-preview`` and sent ``max_tokens`` and ``temperature`` on
every call, which reasoning-model deployments refuse.

It now calls Foundry's v1 route (``<resource>/openai/v1/``) with the plain
OpenAI client and no api-version, which also reaches Foundry's non-OpenAI
models through Chat Completions. ``model`` is the deployment name. The output
budget goes out as ``max_completion_tokens``; temperature and the other sampling
parameters go out unless the deployment is known to refuse them, and a 400 that
names one as unsupported is retried once without it (``azure_v1``).
"""

import logging
from typing import Any, Dict, List, Optional

from config import config
from .azure_v1 import QUIRKS, DeploymentKey, quirks_after_refusal, quirks_after_response, refused_params, v1_base_url
from .base import BaseLLMProvider, LLMResponse, run_blocking
from .openai_chat_request import chat_kwargs, tool_calls_from
from .openai_client import usage_from_openai

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

PROVIDER = "azure"
DEFAULT_TIMEOUT_SECONDS = 180.0
NOT_CONFIGURED = (
    "Azure OpenAI credentials not configured. Cannot generate response. "
    "Please configure Azure credential or set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars."
)


class AzureProvider(BaseLLMProvider):
    """Azure OpenAI (Microsoft Foundry) over the v1 route."""

    def _initialize_client(self):
        if OpenAI is None:
            raise ImportError("OpenAI package not installed. Run: pip install openai")

        api_key = self.config.api_key or config.AZURE_OPENAI_API_KEY
        endpoint = self.config.base_url or config.AZURE_OPENAI_ENDPOINT
        self.base_url: Optional[str] = None

        # BOOTSTRAP STRATEGY: Don't require key at initialization
        if not api_key or not endpoint:
            logger.warning(
                "Azure OpenAI credentials not configured. "
                "LLM features will fail until credentials are added. "
                "Configure Azure credential or set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT env vars."
            )
            self.client = None
            return
        self.base_url = v1_base_url(endpoint)
        timeout = float(self.config.timeout) if self.config.timeout else DEFAULT_TIMEOUT_SECONDS
        self.client = OpenAI(api_key=api_key, base_url=self.base_url, timeout=timeout, **self._pinned(timeout))
        logger.info(f"Initialized Azure OpenAI v1 client at {self.base_url} for deployment: {self.config.model}")

    def _pinned(self, timeout: float) -> Dict[str, Any]:
        """A workspace key's own endpoint is user input: on saas every call to it is pinned (#873).

        An operator's endpoint (env, credential store) is not, and keeps the SDK's client.
        """
        from core.llm.byok_endpoint import endpoint_http_client

        http_client = endpoint_http_client(timeout) if self.config.endpoint_from_key else None
        return {"http_client": http_client} if http_client else {}

    def _deployment(self) -> DeploymentKey:
        return (self.base_url or "", self.config.model)

    def _request(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]], quirks) -> Dict[str, Any]:
        kwargs = chat_kwargs(
            self.config, messages, sampling=quirks.sampling, completion_tokens=quirks.completion_tokens,
        )
        if tools:
            kwargs["tools"] = self._sanitize_tools(tools, keep_strict=True)
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _complete(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> Any:
        """One Chat Completions call (blocking), retried once when the deployment
        refuses a parameter, with what it refuses remembered."""
        key = self._deployment()
        quirks = QUIRKS.recall(key)
        kwargs = self._request(messages, tools, quirks)
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            refused = refused_params(exc, kwargs.keys())
            if not refused:
                raise
            quirks = quirks_after_refusal(quirks, refused)
            QUIRKS.remember(key, quirks)
            logger.warning(f"Azure deployment {key[1]} refuses {sorted(refused)}; retrying once without them")
            response = self.client.chat.completions.create(**self._request(messages, tools, quirks))
        learned = quirks_after_response(quirks, getattr(response, "model", None))
        if learned != quirks:
            QUIRKS.remember(key, learned)
        return response

    def _require_client(self) -> None:
        if self.client is None:
            raise ValueError(NOT_CONFIGURED)

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate a response through the Azure v1 route."""
        self._require_client()
        try:
            response = await run_blocking(self._complete, messages, tools)
        except Exception:
            logger.exception("Azure OpenAI API error")
            raise
        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            usage=usage_from_openai(response.usage),
            model=response.model,
            provider=PROVIDER,
            tool_calls=tool_calls_from(choice.message),
            finish_reason=choice.finish_reason,
        )

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate a response through the Azure v1 route (synchronous)."""
        self._require_client()
        try:
            response = self._complete(messages)
        except Exception:
            logger.exception("Azure OpenAI API error")
            raise
        return LLMResponse(
            content=response.choices[0].message.content,
            usage=usage_from_openai(response.usage),
            model=response.model,
            provider=PROVIDER,
        )
