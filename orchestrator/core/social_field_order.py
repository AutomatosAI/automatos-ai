"""The order a social template's fields are shown in: the order they appear on the image.

``document_templates.blocks`` is JSONB, which stores an object's keys sorted by length and
then by name. A template written "eyebrow, headline, card 1 title, card 1 text, …" came back
as "cta, handle, eyebrow, headline, card_1_text, card_2_text, card_3_text, card_1_title, …":
the button first, and every card's text before any card's title (the owner's screenshot,
16 Nov 2026). The column cannot keep the author's order, so the fields are put in the order
the composition itself uses them: each at its first ``{{ name }}`` in the HTML (top to bottom
on the image; a video's in the order its scenes play), then any the HTML never shows (a
spoken line, a chart's data), in the order they were stored.

The editor's form and the composer's prompt both read the schema through here, so a person
and Auto see a card's title next to its text.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from core.social_templates import PLACEHOLDER


def ordered_schema(blocks: Any) -> Dict[str, Any]:
    """``blocks.variables_schema`` with its fields in the order the composition shows them;
    ``{}`` when the blocks declare none."""
    declared = blocks.get("variables_schema") if isinstance(blocks, Mapping) else None
    schema = declared if isinstance(declared, Mapping) else {}
    html = blocks.get("html") if isinstance(blocks, Mapping) else None
    shown = [name for name in dict.fromkeys(PLACEHOLDER.findall(html if isinstance(html, str) else "")) if name in schema]
    unseen = [name for name in schema if name not in shown]
    return {name: schema[name] for name in [*shown, *unseen]}
