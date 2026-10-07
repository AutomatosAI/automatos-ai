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
* ``body-font`` and ``heading-font`` end in a generic family: a stack without one
  (a kit saved as bare ``Newsreader`` before its save kept fallbacks,
  ``modules/documents/font_fallbacks.py``) gets the generic its first family belongs
  to (``core/font_stacks.py``), so a missing face never falls to the browser's Times.

A template never reads the kit itself: what differs between kits reaches it only as a
token, so the brand rule (``core/social_brand_rule.py``) holds. Pure.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from core.font_stacks import with_generic

# The checked font tokens (``media_render_bundle.BODY_FONT_TOKEN`` and ``HEADING_FONT_TOKEN``),
# and the tokens added here.
BODY_FONT_TOKEN, HEADING_FONT_TOKEN = "body-font", "heading-font"
MONO_FONT_TOKEN = "mono-font"
SOCIAL_KIT_TOKENS = (MONO_FONT_TOKEN,)
# media-render's cap on one token's value (services/media-render/media_render/bundle.py).
MAX_TOKEN_CHARS = 200


def font_tokens(tokens: Mapping[str, str]) -> Dict[str, str]:
    """The body and heading stacks, each ending in a generic family, and ``mono-font``: the body's."""
    out: Dict[str, str] = {}
    for name in (BODY_FONT_TOKEN, HEADING_FONT_TOKEN):
        stack = tokens.get(name)
        if stack:
            ended = with_generic(stack)
            out[name] = ended if len(ended) <= MAX_TOKEN_CHARS else stack
    if BODY_FONT_TOKEN in out:
        out[MONO_FONT_TOKEN] = out[BODY_FONT_TOKEN]
    return out


def kit_render_tokens(kit: Mapping[str, Any], tokens: Mapping[str, str]) -> Dict[str, str]:
    """The tokens that set a still as the kit's documents are set, from ``tokens`` (already checked) and ``kit``."""
    return font_tokens(tokens)


__all__ = ["MONO_FONT_TOKEN", "SOCIAL_KIT_TOKENS", "font_tokens", "kit_render_tokens"]
