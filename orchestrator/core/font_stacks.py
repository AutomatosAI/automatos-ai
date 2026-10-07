"""CSS font stacks: whether one ends in a generic family, and the generic a family falls back to (F376).

F376 (night 11, 7 Oct): an approved Brand Designer proposal saved ``font_family:
"Geist"`` and ``heading_font: "Newsreader"``, bare names with no fallback. A bare
name is a valid ``--brand-heading-font``, so the template's own ``var()`` fallback
never applied, and Chrome fell to its default (Times) wherever the face was missing.
A stack that ends in a generic family (``serif``, ``sans-serif``) always lands on a
face of the right kind. The kit's save (``modules/documents/font_fallbacks.py``) and
the render bundle (``core/social_kit_tokens.py``) both keep one at the end. Pure.
"""
from __future__ import annotations

from typing import List

# CSS Fonts 4's generic families: a stack ending in one always resolves.
GENERIC_FAMILIES = frozenset({
    "serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui", "math", "emoji", "fangsong",
    "ui-serif", "ui-sans-serif", "ui-monospace", "ui-rounded",
})
SERIF, SANS_SERIF = "serif", "sans-serif"
# The families the code ships (``modules/documents/bundled_fonts.BUNDLED_FAMILIES``), by their generic.
FAMILY_GENERICS = {"newsreader": SERIF, "geist": SANS_SERIF, "inter": SANS_SERIF}
# A family the code does not know: the generic the templates' own fallbacks use.
DEFAULT_GENERIC = SANS_SERIF
STACK_SEPARATOR = ", "


def stack_parts(stack: str) -> List[str]:
    """The comma-separated entries of a stack as written (quotes kept), blanks dropped."""
    return [part.strip() for part in str(stack or "").split(",") if part.strip()]


def family_key(part: str) -> str:
    """A stack entry's family name, unquoted and casefolded, for comparing two entries."""
    return part.strip().strip("'\"").strip().casefold()


def has_generic(stack: str) -> bool:
    """Whether the stack names a generic family anywhere (an unquoted one: a quoted name is a font)."""
    return any(part.casefold() in GENERIC_FAMILIES for part in stack_parts(stack))


def generic_for(stack: str) -> str:
    """The generic the stack's first family belongs to: a shipped family's own, else :data:`DEFAULT_GENERIC`."""
    parts = stack_parts(stack)
    return FAMILY_GENERICS.get(family_key(parts[0]), DEFAULT_GENERIC) if parts else DEFAULT_GENERIC


def merged_stack(stack: str, fallback: str) -> str:
    """``stack`` followed by every entry of ``fallback`` it does not already name (by family, case aside)."""
    parts = stack_parts(stack)
    named = {family_key(part) for part in parts}
    extra = [part for part in stack_parts(fallback) if family_key(part) not in named]
    return STACK_SEPARATOR.join(dict.fromkeys(parts + extra))


def with_generic(stack: str) -> str:
    """``stack`` as it is when it is empty or names a generic family; else with its generic at the end."""
    if not stack_parts(stack) or has_generic(stack):
        return stack
    return STACK_SEPARATOR.join([*stack_parts(stack), generic_for(stack)])


__all__ = [
    "DEFAULT_GENERIC", "FAMILY_GENERICS", "GENERIC_FAMILIES",
    "family_key", "generic_for", "has_generic", "merged_stack", "stack_parts", "with_generic",
]
