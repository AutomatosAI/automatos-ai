"""Prompt-cache helpers — the Anthropic ``cache_control`` seam (PRD-201 S4).

Pure, provider-agnostic string→block transforms. The assembler (PRD-201 S4)
orders sections static-first / volatile-last and hands back a ``cacheable_prefix``
(the leading, cache-stable bytes). This module turns a system prompt into the
Anthropic ``system`` block list with a single ``cache_control`` breakpoint on the
last stable block, so the static identity/skills/catalog prefix is billed once
and re-read at ~0.1× within the 5-minute TTL instead of re-billed every turn.

Kept out of ``anthropic_client.py`` so it is unit-testable without importing the
``anthropic`` SDK, and so the shape is asserted directly (tests assert request
*shape*, never call a provider). ``cache_control: {"type": "ephemeral"}`` is GA
Anthropic API; the optional 1-hour TTL is ``{"type": "ephemeral", "ttl": "1h"}``.
Max 4 breakpoints per request — this emits exactly one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

# GA Anthropic parameter shapes (confirmed via the claude-api skill). Do not
# invent variants beyond these.
EPHEMERAL: Dict[str, str] = {"type": "ephemeral"}
EPHEMERAL_1H: Dict[str, str] = {"type": "ephemeral", "ttl": "1h"}

SystemParam = Union[str, List[Dict[str, Any]]]


def build_cached_system(
    system_text: str,
    cacheable_prefix: Optional[str] = None,
    *,
    cache_control: Optional[Dict[str, str]] = None,
) -> SystemParam:
    """Return the Anthropic ``system`` param with ONE ``cache_control`` breakpoint.

    - When ``cacheable_prefix`` is a non-empty leading slice of ``system_text``
      with a non-empty remainder, return a **two-block** list: the stable prefix
      (carrying the breakpoint) then the volatile remainder. A change in the
      volatile tail then leaves the cached prefix bytes untouched.
    - Otherwise return a **single** cached block over the whole system (still the
      first-ever ``cache_control`` on the Anthropic route; caches tools+system
      together because render order is tools → system → messages).
    - Empty ``system_text`` → ``""`` (nothing to cache).

    Anthropic silently declines to cache a prefix below the model minimum
    (~4096 tokens on Opus-class), so emitting the marker on a short prefix is a
    harmless no-op, never an error.
    """
    if not system_text:
        return ""

    cc = cache_control or EPHEMERAL

    if (
        cacheable_prefix
        and system_text.startswith(cacheable_prefix)
        and len(cacheable_prefix) < len(system_text)
    ):
        remainder = system_text[len(cacheable_prefix):]
        return [
            {"type": "text", "text": cacheable_prefix, "cache_control": cc},
            {"type": "text", "text": remainder},
        ]

    return [{"type": "text", "text": system_text, "cache_control": cc}]


def count_cache_breakpoints(system: SystemParam) -> int:
    """How many ``cache_control`` markers a system param carries (test helper)."""
    if not isinstance(system, list):
        return 0
    return sum(1 for block in system if isinstance(block, dict) and "cache_control" in block)


def read_cache_usage(usage: Any) -> Dict[str, int]:
    """Extract the Anthropic prompt-cache usage fields from a response ``usage``.

    Reads ``cache_read_input_tokens`` / ``cache_creation_input_tokens`` (the
    measured-prize fields, PRD-201 S4 §7) plus ``input_tokens``. Tolerates a
    dict or an SDK object, and providers that don't report them (→ 0). The
    realized cache-hit fraction is ``cache_read / (cache_read + cache_creation +
    input)`` — surfaced through S1's trace, not asserted here.
    """
    def _get(name: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            val = usage.get(name)
        else:
            val = getattr(usage, name, None)
        try:
            return int(val or 0)
        except (TypeError, ValueError):
            return 0

    return {
        "cache_read_input_tokens": _get("cache_read_input_tokens"),
        "cache_creation_input_tokens": _get("cache_creation_input_tokens"),
        "input_tokens": _get("input_tokens"),
    }
