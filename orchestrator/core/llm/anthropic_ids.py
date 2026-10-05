"""Anthropic model ids and their OpenRouter twins (#829).

Anthropic's Models API names a model ``claude-sonnet-4-5-20250929`` or
``claude-opus-5``; OpenRouter names the same model ``anthropic/claude-sonnet-4.5``
or ``anthropic/claude-opus-5``. The Anthropic catalogue sync borrows a new
row's price from the OpenRouter twin, and the usage tracker prices a call on an
unpriced direct Anthropic route from it, so both read the mapping from here
(``core.llm`` sits below ``core.services``).
"""
from __future__ import annotations

import re
from typing import List

OPENROUTER_PREFIX = "anthropic/"
DIRECT_ID_PREFIX = "claude-"
DATE_SUFFIX = re.compile(r"-\d{8}$")
_VERSION_HYPHEN = re.compile(r"(?<=\d)-(?=\d)")


def openrouter_twin_id(model_id: str) -> str:
    """Anthropic's id → OpenRouter's id for the same model.

    ``claude-sonnet-4-5-20250929`` → ``anthropic/claude-sonnet-4.5``;
    ``claude-3-5-sonnet-20241022`` → ``anthropic/claude-3.5-sonnet``;
    ``claude-opus-5`` → ``anthropic/claude-opus-5``.
    """
    undated = DATE_SUFFIX.sub("", model_id)
    return f"{OPENROUTER_PREFIX}{_VERSION_HYPHEN.sub('.', undated)}"


def openrouter_twin_ids(model_id: str) -> List[str]:
    """The OpenRouter ids to try for a direct Anthropic id, best first; [] for any other id."""
    if "/" in model_id or not model_id.startswith(DIRECT_ID_PREFIX):
        return []
    twins = [openrouter_twin_id(model_id), f"{OPENROUTER_PREFIX}{model_id}"]
    return list(dict.fromkeys(twins))


__all__ = ["DATE_SUFFIX", "openrouter_twin_id", "openrouter_twin_ids"]
