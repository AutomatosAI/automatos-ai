"""The ``--brand-*`` tokens that set a social still the way the kit's documents are set (F376, night 11).

F376 (7 Oct): the stills looked like "a cousin of my invoices, not the same company".
``core/media_render_bundle.brand_tokens`` gave the templates the kit's colours, its
dark stage and its paper; what the documents also do with the kit never reached a
post. :func:`kit_render_tokens` adds it, read from the tokens ``brand_tokens`` has
already checked (so nothing unsafe comes back) and from the kit:

* ``mono-font``: the kit has no mono font, and the documents set the small print
  (an attribution, a handle, a counter, a footer) in the body font. Every template
  reads ``var(--brand-mono-font, monospace)`` for those, which printed in a
  Courier-like stand-in; the token is the body font's stack.

A template never reads the kit itself: what differs between kits reaches it only as a
token, so the brand rule (``core/social_brand_rule.py``) holds. Pure.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

# The checked body font (``media_render_bundle.BODY_FONT_TOKEN``), and the tokens added here.
BODY_FONT_TOKEN = "body-font"
MONO_FONT_TOKEN = "mono-font"
SOCIAL_KIT_TOKENS = (MONO_FONT_TOKEN,)


def kit_render_tokens(kit: Mapping[str, Any], tokens: Mapping[str, str]) -> Dict[str, str]:
    """The tokens that set a still as the kit's documents are set, from ``tokens`` (already checked) and ``kit``."""
    out: Dict[str, str] = {}
    body = tokens.get(BODY_FONT_TOKEN)
    if body:
        out[MONO_FONT_TOKEN] = body
    return out


__all__ = ["MONO_FONT_TOKEN", "SOCIAL_KIT_TOKENS", "kit_render_tokens"]
