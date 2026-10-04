"""A model call that failed only because the provider's web search did (F264, night 7b).

PRD-240 puts the provider's own web search (OpenRouter's ``openrouter:web_search``
server tool) on every call while web access is on, and the model decides when to
search. On night 7b most of Auto's replies ended "Auto could not finish this reply:
the AI provider's web search failed on its side", replies that had finished. The
search had failed on a later call in the turn (F205's re-prompt that takes the
platform's tool names out of a reply, F187's re-prompt about an id), and the
provider's 502 failed the whole call, so the turn was reported as failed and the
re-prompt's fix never landed.

A call refused that way is sent once more without the provider's search: the model
answers without searching, rather than not at all. Any other failure, and a call
that carried no such tool, is the caller's as before. The retry is logged.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

import openai

logger = logging.getLogger(__name__)

# The provider's own words: 'Server tool "openrouter:web_search" failed: upstream returned an invalid response'.
SERVER_TOOL_FAILED = re.compile(r'Server tool \\?"[\w.-]+:[\w.-]+\\?" failed')
SERVER_TOOL_PREFIX = "openrouter:"


def _server_tool(tool: Any) -> bool:
    """A provider-executed tool (PRD-240), not one of the platform's functions."""
    return isinstance(tool, dict) and str(tool.get("type", "")).startswith(SERVER_TOOL_PREFIX)


def without_server_tools(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """The call's arguments without the provider's own tools (a new dict)."""
    kept: List[Any] = [t for t in kwargs.get("tools") or [] if not _server_tool(t)]
    if kept:
        return {**kwargs, "tools": kept}
    return {k: v for k, v in kwargs.items() if k not in ("tools", "tool_choice")}


class _Completions:
    """``chat.completions`` with the one retry; anything else is the SDK's own."""

    def __init__(self, completions: Any) -> None:
        self._completions = completions

    def create(self, **kwargs: Any) -> Any:
        try:
            return self._completions.create(**kwargs)
        except openai.APIError as exc:
            if not SERVER_TOOL_FAILED.search(str(exc)) or not any(_server_tool(t) for t in kwargs.get("tools") or []):
                raise
            logger.warning("[F264] the provider's web search failed (%s); sending the call again without it", exc)
            return self._completions.create(**without_server_tools(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._completions, name)


class _Chat:
    def __init__(self, chat: Any) -> None:
        self._chat = chat
        self.completions = _Completions(chat.completions)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat, name)


class SearchRecoveringClient:
    """An OpenAI SDK client whose chat completions are sent once more without the
    provider's web search when only that search failed. Everything else passes
    through to the client."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self.chat = _Chat(client.chat)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = ["SERVER_TOOL_FAILED", "SearchRecoveringClient", "without_server_tools"]
