"""F376 (night 11, 7 Oct), mono: a post's small print is set in the kit's body font, as the documents' is.

Every paper and photo template reads ``--mono: var(--brand-mono-font, monospace)``
for its attribution, handle, counters, pills and footer. Nothing emitted
``mono-font``, so they printed in a Courier-like stand-in where the invoice sets the
same lines in the body font. ``brand_tokens`` now emits ``mono-font``: the kit's
checked body font stack (the kit has no mono field).
"""
from __future__ import annotations

from core import media_render_bundle, social_kit_tokens
from core.media_render_bundle import brand_tokens
from core.social_kit_tokens import MONO_FONT_TOKEN
from modules.documents.social_starters import social_starters

NIGHT_BODY = "Geist, Inter, 'Segoe UI', system-ui, sans-serif"
NIGHT_KIT = {"primary_color": "#c44a1a", "secondary_color": "#dcd2bd", "text_color": "#1a1814",
             "font_family": NIGHT_BODY, "heading_font": "Newsreader, Georgia, serif"}
MONO_ALIAS = "--mono: var(--brand-mono-font, monospace);"


def test_the_mono_token_is_the_kits_body_font():
    tokens = brand_tokens(NIGHT_KIT)
    assert tokens[MONO_FONT_TOKEN] == tokens["body-font"] == NIGHT_BODY
    assert social_kit_tokens.BODY_FONT_TOKEN == media_render_bundle.BODY_FONT_TOKEN


def test_a_body_font_that_is_not_one_css_value_gives_no_mono_token_either():
    tokens = brand_tokens({**NIGHT_KIT, "font_family": "Inter; } body { color: red"})
    assert "body-font" not in tokens and MONO_FONT_TOKEN not in tokens  # the template's fallback applies


def test_every_template_that_sets_small_print_in_mono_reads_the_token():
    readers = [s["slug"] for s in social_starters() if MONO_ALIAS in s["blocks"]["html"]]
    assert len(readers) >= 18  # every seeded template but the app promo, which has no small print in mono
    for starter in social_starters():
        html = starter["blocks"]["html"]
        if "var(--mono)" in html:
            assert MONO_ALIAS in html, starter["slug"]
