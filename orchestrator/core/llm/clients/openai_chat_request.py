"""
OpenAI-shaped Chat Completions requests (#873)
==============================================

#873: the Azure client sent ``max_tokens`` and ``temperature`` on every call.
Reasoning-model deployments (the o-series, GPT-5) take ``max_completion_tokens``
and refuse a custom temperature, so they answered with a 400. The direct OpenAI
client sent the same two parameters to o3 and gpt-5 and had the same fault.

Both clients now build the request here. ``sampling`` says whether temperature,
top_p and the penalties go out; ``completion_tokens`` says which name the output
budget travels under. Which models refuse sampling parameters is one rule,
``accepts_sampling_params`` in ``base.py``.
"""
from typing import Any, Dict, List, Optional

from .base import request_max_tokens

SAMPLING_PARAMS = ("temperature", "top_p", "frequency_penalty", "presence_penalty")
COMPLETION_TOKENS = "max_completion_tokens"
LEGACY_MAX_TOKENS = "max_tokens"


def _sampling_kwargs(config: Any) -> Dict[str, Any]:
    """The config's sampling settings; temperature always, the rest when set."""
    values = {
        "temperature": config.temperature,
        "top_p": getattr(config, "top_p", None),
        "frequency_penalty": getattr(config, "frequency_penalty", None),
        "presence_penalty": getattr(config, "presence_penalty", None),
    }
    return {k: v for k, v in values.items() if k == "temperature" or v is not None}


def chat_kwargs(
    config: Any,
    messages: List[Dict[str, Any]],
    *,
    sampling: bool,
    completion_tokens: bool,
) -> Dict[str, Any]:
    """Keyword arguments for ``client.chat.completions.create`` (tools excluded).

    The output budget is this call's (``request_max_tokens``), sent as
    ``max_completion_tokens`` when ``completion_tokens`` is true, else as the
    older ``max_tokens``. Sampling parameters go out only when ``sampling`` is
    true. ``stop`` goes out whenever the config sets it.
    """
    token_key = COMPLETION_TOKENS if completion_tokens else LEGACY_MAX_TOKENS
    kwargs: Dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        token_key: request_max_tokens(config),
    }
    if sampling:
        kwargs.update(_sampling_kwargs(config))
    stop = getattr(config, "stop", None)
    if stop is not None:
        kwargs["stop"] = stop
    return kwargs


def tool_calls_from(message: Any) -> Optional[List[Dict[str, Any]]]:
    """The message's tool calls as plain dicts, or None when it made none."""
    calls = getattr(message, "tool_calls", None)
    if not calls:
        return None
    return [
        {
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in calls
    ]
