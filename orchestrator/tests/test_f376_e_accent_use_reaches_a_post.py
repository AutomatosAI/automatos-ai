"""F376 (night 11, 7 Oct), B3: a post follows the kit's ``accent_use``, as the invoice does.

Sparing and bold renders of the Announcement and the Carousel were pixel-identical
while the invoice changed (bold fills its table header with the accent). Socials
now get it as tokens (``core/social_kit_tokens.accent_tokens``), never by reading the
kit: under bold a card's solid accent surfaces (the numbered circles, the status
pill, the call-to-action pill) are filled with the accent and their words are the
paper; under sparing they are clear, ringed in the accent, and their words are the
accent. Pins: the two kits give different tokens, each pair reads (WCAG AA), sparing
never makes the accent a background, and the templates with accent surfaces read the
tokens.
"""
from __future__ import annotations

import re

import pytest

from core.brand_palette import PAPER, PRIMARY_ON_PAPER, contrast, parse_hex
from core.media_render_bundle import brand_tokens
from core.social_kit_tokens import ACCENT_BOLD, ACCENT_FILL_TOKEN, ON_ACCENT_FILL_TOKEN
from modules.documents.brand_system import ACCENT_BOLD as DOCUMENTS_BOLD
from modules.documents.brand_system import ACCENT_SPARING
from modules.documents.social_starters import social_starters

NIGHT_KIT = {"primary_color": "#c44a1a", "secondary_color": "#dcd2bd", "text_color": "#1a1814",
             "font_family": "Geist, Inter, sans-serif"}
KITS = [
    pytest.param(NIGHT_KIT, id="night-11"),
    pytest.param({"primary_color": "#e96235", "secondary_color": "#1a1714", "text_color": "#f0e8db"}, id="studio-dark"),
    pytest.param({"primary_color": "#ffcc00", "secondary_color": "#f3ede2", "text_color": "#222222"}, id="a-light-primary"),
    pytest.param({"primary_color": "#1E3A5F", "secondary_color": "#C26A2E", "text_color": "#1a1a2e",
                  "palette": {"paper": "#faf7f2", "accent": "#1e3a5f"}}, id="harbourline"),
]
ACCENT_SURFACES = {"definition-card": [".num"], "announcement-card": [".num", ".status"], "carousel": [".cta-pill"]}
FILL_ALIASES = ("--accent-fill: var(--brand-accent-fill, transparent);",
                "--on-accent-fill: var(--brand-on-accent-fill, var(--accent));")
AA_TEXT = 4.5


def _starter(slug):
    return next(s for s in social_starters() if s["slug"] == slug)


def _rule(html: str, selector: str) -> str:
    """The body of ``selector``'s first rule in the template's style."""
    match = re.search(rf"\n\s*{re.escape(selector)} \{{([^{{}}]*)\}}", html)
    assert match, selector
    return match.group(1)


def test_sparing_and_bold_give_different_tokens():
    assert ACCENT_BOLD == DOCUMENTS_BOLD
    sparing = brand_tokens({**NIGHT_KIT, "accent_use": ACCENT_SPARING})
    bold = brand_tokens({**NIGHT_KIT, "accent_use": ACCENT_BOLD})
    assert (sparing[ACCENT_FILL_TOKEN], sparing[ON_ACCENT_FILL_TOKEN]) == ("transparent", sparing[PRIMARY_ON_PAPER])
    assert (bold[ACCENT_FILL_TOKEN], bold[ON_ACCENT_FILL_TOKEN]) == (bold[PRIMARY_ON_PAPER], bold[PAPER])
    assert brand_tokens(NIGHT_KIT)[ACCENT_FILL_TOKEN] == "transparent"  # sparing is every kit's default


@pytest.mark.parametrize("kit", KITS)
def test_each_pair_reads(kit):
    sparing = brand_tokens(kit)
    assert contrast(parse_hex(sparing[ON_ACCENT_FILL_TOKEN]), parse_hex(sparing[PAPER])) >= AA_TEXT
    bold = brand_tokens({**kit, "accent_use": ACCENT_BOLD})
    assert contrast(parse_hex(bold[ON_ACCENT_FILL_TOKEN]), parse_hex(bold[ACCENT_FILL_TOKEN])) >= AA_TEXT


def test_without_a_paper_there_is_no_fill_and_the_templates_fall_back_to_sparing():
    assert ACCENT_FILL_TOKEN not in brand_tokens({"accent_use": ACCENT_BOLD})
    for slug in ACCENT_SURFACES:
        html = _starter(slug)["blocks"]["html"]
        assert all(alias in html for alias in FILL_ALIASES), slug


@pytest.mark.parametrize("slug", list(ACCENT_SURFACES))
def test_the_templates_with_accent_surfaces_read_the_tokens(slug):
    html = _starter(slug)["blocks"]["html"]
    for selector in ACCENT_SURFACES[slug]:
        rule = _rule(html, selector)
        assert "background: var(--accent-fill);" in rule and "color: var(--on-accent-fill);" in rule, selector
    assert "background: var(--accent);" not in html  # no surface is filled with the accent outright
