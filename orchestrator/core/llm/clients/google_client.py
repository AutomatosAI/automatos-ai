"""
Google Gemini Provider Implementation
=====================================

Google Gemini models, through the Google Gen AI SDK (``google-genai``) (#1169).

The SDK this replaced, ``google-generativeai``, is retired: Google ended its support on
30 Nov 2025. It also held one API key for the whole process, so two workspaces' own
keys could each be used for the other's calls. Here every provider has a client of its
own (:func:`gemini_client`), and system messages go to Gemini as its system instruction.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from config import config
from .base import BaseLLMProvider, LLMResponse, request_max_tokens, run_blocking

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = types = None

logger = logging.getLogger(__name__)

# Gemini's turn roles; a tool turn has no Gemini equivalent here and is left out, as before.
_ROLES = {"user": "user", "assistant": "model"}
# Gemini's finish reasons, named as the other providers name them (the tool loop reads "length";
# an answer withheld for its content is OpenAI's "content_filter", not an empty "stop").
_CONTENT_FILTER = "content_filter"
_FINISH_REASONS = {
    "STOP": "stop", "MAX_TOKENS": "length",
    **dict.fromkeys(("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"), _CONTENT_FILTER),
}
_NOT_CONFIGURED = (
    "Google API key not configured. Cannot generate response. "
    "Please configure 'development_google' credential or set GOOGLE_API_KEY env var."
)


def gemini_client(api_key: str, timeout_seconds: Optional[float] = None) -> Any:
    """A Gemini API client of its own for ``api_key``: never the process-wide key the old
    SDK kept, and never Vertex AI, whatever GOOGLE_GENAI_USE_VERTEXAI says."""
    options = types.HttpOptions(timeout=int(timeout_seconds * 1000)) if timeout_seconds else None
    return genai.Client(api_key=api_key, vertexai=False, http_options=options)


def _text(content: Any) -> str:
    """A message's text: a string, or the text blocks of an OpenAI-style content list."""
    if isinstance(content, list):
        return "\n".join(str(block.get("text", "")) for block in content
                         if isinstance(block, dict) and block.get("type") == "text")
    return "" if content is None else str(content)


def gemini_request(messages: List[Dict[str, Any]]) -> Tuple[List[Any], Optional[str]]:
    """OpenAI-style messages as Gemini ``contents`` and a system instruction.

    System messages become the instruction. User and assistant turns become ``user`` and
    ``model`` contents; tool turns and empty turns are left out. Gemini needs a turn to
    answer, so a request of system messages alone sends them as the user's turn."""
    instruction = "\n\n".join(text for text in (_text(m.get("content")) for m in messages
                                                 if m.get("role") == "system") if text) or None
    contents = [
        types.Content(role=_ROLES[m.get("role", "user")], parts=[types.Part(text=_text(m.get("content")))])
        for m in messages if m.get("role", "user") in _ROLES and _text(m.get("content"))
    ]
    if not contents and instruction:
        return [types.Content(role="user", parts=[types.Part(text=instruction)])], None
    return contents, instruction


def _finish_reason(reason: Any) -> Optional[str]:
    """Gemini's finish reason ("STOP", "MAX_TOKENS", …) as the other providers name it."""
    name = getattr(reason, "value", reason)
    return _FINISH_REASONS.get(name, str(name).lower()) if name else None


def _why_it_stopped(response: Any) -> Optional[str]:
    """Why Gemini stopped: its candidate's finish reason, or "content_filter" when it
    refused the prompt itself (no candidate, a ``block_reason`` instead)."""
    if response.prompt_feedback and response.prompt_feedback.block_reason:
        return _CONTENT_FILTER
    candidates = response.candidates or []
    return _finish_reason(candidates[0].finish_reason) if candidates else None


def _llm_response(response: Any, model: str) -> LLMResponse:
    """The platform's response from Gemini's: text, token usage and why it stopped."""
    usage = response.usage_metadata
    return LLMResponse(
        content=response.text or "",
        usage={
            "prompt_tokens": usage.prompt_token_count or 0,
            "completion_tokens": usage.candidates_token_count or 0,
            "total_tokens": usage.total_token_count or 0,
        } if usage else None,
        model=model,
        provider="google",
        finish_reason=_why_it_stopped(response),
    )


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""

    def _initialize_client(self):
        if genai is None:
            raise ImportError("Google Gen AI package not installed. Run: pip install google-genai")

        api_key = self.config.api_key or config.GOOGLE_API_KEY

        # BOOTSTRAP STRATEGY: Don't require key at initialization
        if not api_key:
            logger.warning(
                "Google API key not configured. "
                "LLM features will fail until key is added. "
                "Configure 'development_google' credential or set GOOGLE_API_KEY env var."
            )
            self.client = None
        else:
            self.client = gemini_client(api_key, self.config.timeout)
            logger.info(f"Initialized Google Gemini client with model: {self.config.model}")

    def _generate(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        """One Gemini call. Runs in the caller's thread, or the executor's (run_blocking)."""
        contents, instruction = gemini_request(messages)
        settings = types.GenerateContentConfig(
            system_instruction=instruction,
            temperature=self.config.temperature,
            max_output_tokens=request_max_tokens(self.config),
        )
        response = self.client.models.generate_content(model=self.config.model, contents=contents, config=settings)
        return _llm_response(response, self.config.model)

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using Google Gemini API. ``tools`` are not sent: this provider
        has never offered function calling."""
        if self.client is None:
            raise ValueError(_NOT_CONFIGURED)
        try:
            return await run_blocking(self._generate, messages)
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Google Gemini API (synchronous)"""
        if self.client is None:
            raise ValueError(_NOT_CONFIGURED)
        try:
            return self._generate(messages)
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise
