"""A font stack a brand-kit save sets keeps its fallbacks (F376, night 11).

F376 (7 Oct): the Brand Designer's approved "make my socials match my documents"
(proposal #1812) saved ``font_family: "Geist"`` and ``heading_font: "Newsreader"``,
replacing ``Geist, Inter, 'Segoe UI', system-ui, sans-serif`` and ``Newsreader,
Georgia, serif``. A bare name is one valid CSS value, so nothing refused it, and
every render without that face fell to the browser's default serif: the approval
made the posts match the documents less.

:func:`with_font_fallbacks` is applied to the patch before the kit validates it
(``brand_kit.validate_brand_kit``): a body or heading stack the patch sets that ends
without a generic family keeps the stack it replaces after it (each family once,
case aside); when the result still has no generic family (the stack it replaced had
none, or there was none), the generic its first family belongs to is appended
(``core/font_stacks.py``: Newsreader is a serif, Geist and Inter sans-serif, any
other sans-serif). A stack with a generic family, or an empty one, is kept as sent.
A result too long to be one brand token is cut back to the stack and its generic.
Pure.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from core.font_stacks import has_generic, merged_stack, with_generic
from core.media_render_bundle import MAX_TOKEN_CHARS

FONT_STACK_FIELDS = ("font_family", "heading_font")


def kept_fallbacks(stack: str, replaced: str) -> str:
    """``stack`` with the fallbacks of the stack it ``replaced``, ending in a generic family."""
    if not stack.strip() or has_generic(stack):
        return stack
    merged = with_generic(merged_stack(stack, replaced))
    return merged if len(merged) <= MAX_TOKEN_CHARS else with_generic(stack)


def with_font_fallbacks(patch: Mapping[str, Any], base: Mapping[str, Any]) -> Dict[str, Any]:
    """A copy of ``patch`` whose body and heading font stacks keep their fallbacks over ``base`` (the stored kit)."""
    kept = {
        field: kept_fallbacks(patch[field], str(base.get(field) or ""))
        for field in FONT_STACK_FIELDS
        if isinstance(patch.get(field), str)
    }
    return {**patch, **kept}


__all__ = ["FONT_STACK_FIELDS", "kept_fallbacks", "with_font_fallbacks"]
