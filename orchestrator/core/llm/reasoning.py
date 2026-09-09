"""PRD-238 S1 — the reasoning channel, separated from the answer.

Reasoning models deliver their deliberation two ways: a dedicated field on
the message/delta (``reasoning_content`` on DeepSeek, Kimi and NVIDIA NIM;
``reasoning`` on some OpenRouter routes) or inline ``<think>…</think>`` tags
in the content. Either way it must never be shown or stored as the answer.
These helpers are pure so every client can share them.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Tuple

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_UNCLOSED = re.compile(r"<think>(.*)\Z", re.DOTALL | re.IGNORECASE)

#: Message/delta fields that carry reasoning on OpenAI-compatible providers.
REASONING_FIELDS: Tuple[str, ...] = ("reasoning_content", "reasoning")


def split_think_tags(content: Optional[str]) -> Tuple[str, Optional[str]]:
    """Return ``(answer, reasoning)`` with every ``<think>`` block lifted out.

    An unclosed ``<think>`` (the model ran out of tokens mid-thought) is
    treated as reasoning too, so a truncated thought never leaks as an answer.
    """
    if not content or "<think>" not in content.lower():
        return content or "", None
    thoughts = [m.group(1).strip() for m in _THINK_BLOCK.finditer(content)]
    answer = _THINK_BLOCK.sub("", content)
    tail = _THINK_OPEN_UNCLOSED.search(answer)
    if tail:
        thoughts.append(tail.group(1).strip())
        answer = answer[: tail.start()]
    reasoning = "\n\n".join(t for t in thoughts if t) or None
    return answer.strip(), reasoning


def reasoning_from_fields(raw: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The reasoning text a provider put on a message/delta dict, if any."""
    if not raw:
        return None
    for key in REASONING_FIELDS:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def coalesce_reasoning(*chunks: Optional[str]) -> Optional[str]:
    """Join non-empty reasoning fragments; None when there are none."""
    kept = [c.strip() for c in chunks if isinstance(c, str) and c.strip()]
    return "\n\n".join(kept) if kept else None
