"""PRD-255 Wave 1, US-003 — every existing kit becomes a v2 kit, derived.

Pins (``core/brand_palette.derive_palette`` / ``effective_palette``, pure):

* **Every kit derives every role.** The Automatos kit, the Harbourline night
  kit, a dark-only kit and a light-only kit each get the role palette, every
  role a 6-digit hex.
* **Readable (FR-5).** Every text role (ink, heading, muted, the accents as text)
  reads at 4.5:1 or more on ``paper`` AND on ``surface_2``; body ink at 10:1.
* **The Automatos kit stops being orange.** Near-black headings, a white or
  off-white page, and the orange only in ``accent``.
* **The derivation rules.** Surfaces are the paper tinted 4% / 8% toward the
  secondary; the accent is the primary darkened only as far as AA needs;
  ``accent_2`` only for a second hue; the page is the kit's light colour when it
  has one.
* **A stored role always wins** (FR-2), and ``effective_palette`` says which
  roles are set and which derived; a stored role that isn't a hex colour is
  ignored. No kit is mutated (derivation runs at read time).
"""
from __future__ import annotations

import colorsys
import copy
import re

import pytest

from core.brand_palette import (
    ACCENT_2_MIN_HUE_DEGREES,
    ACCENT_2_MIN_SATURATION,
    ACCENT_TEXT_MIN_CONTRAST,
    HEADING_MAX_LUMINANCE,
    INK_ON_PAPER_MIN_CONTRAST,
    PALETTE_ROLES,
    PAPER_MIN_LUMINANCE,
    ROLE_ACCENT,
    ROLE_ACCENT_2,
    ROLE_DERIVED,
    ROLE_SET,
    SURFACE_2_TINT,
    SURFACE_TINT,
    TEXT_ROLES,
    WHITE,
    contrast,
    derive_palette,
    effective_palette,
    luminance,
    mix,
    parse_hex,
)

ORANGE, NAVY, AUTOMATOS_TEXT = "#c44a1a", "#1d3658", "#1a1a2e"
V1_DEFAULT_ACCENT = "#0f3460"
AUTOMATOS = {"primary_color": ORANGE, "secondary_color": NAVY, "accent_color": V1_DEFAULT_ACCENT, "text_color": AUTOMATOS_TEXT}
HARBOURLINE = {"primary_color": "#1E3A5F", "secondary_color": "#C26A2E", "accent_color": V1_DEFAULT_ACCENT, "text_color": AUTOMATOS_TEXT}
DARK_ONLY = {"primary_color": "#111111", "secondary_color": "#1b1b1b", "accent_color": "#222222", "text_color": "#000000"}
LIGHT_ONLY = {"primary_color": "#ffe9a8", "secondary_color": "#f5f0e6", "accent_color": "#e8f4ff", "text_color": "#fafafa"}
# The v1 kit's own defaults (modules/documents/brand_kit.py): every colour dark navy.
V1_DEFAULTS = {"primary_color": "#1a1a2e", "secondary_color": "#16213e", "accent_color": V1_DEFAULT_ACCENT, "text_color": "#1a1a2e"}
KITS = {"automatos": AUTOMATOS, "harbourline": HARBOURLINE, "dark_only": DARK_ONLY, "light_only": LIGHT_ONLY}

AA_TEXT = 4.5
HEX6 = re.compile(r"^#[0-9a-f]{6}$")
REQUIRED_ROLES = tuple(role for role in PALETTE_ROLES if role != ROLE_ACCENT_2)
# A search step rounds each channel, so "only as far as AA needs" allows this much over the target.
ACCENT_SEARCH_SLACK = 0.15


def _rgb(palette, role):
    return parse_hex(palette[role])


def _hue_saturation(rgb):
    hue, _lightness, saturation = colorsys.rgb_to_hls(*(channel / 255 for channel in rgb))
    return hue * 360, saturation


def _hue_gap(a, b) -> float:
    gap = abs(_hue_saturation(a)[0] - _hue_saturation(b)[0]) % 360
    return min(gap, 360 - gap)


@pytest.mark.parametrize("name", sorted(KITS))
def test_every_kit_derives_every_role_as_a_six_digit_hex(name):
    palette = derive_palette(KITS[name])
    assert set(REQUIRED_ROLES) <= set(palette)
    assert set(palette) <= set(PALETTE_ROLES)
    assert all(HEX6.match(value) for value in palette.values()), palette


@pytest.mark.parametrize("name", sorted(KITS))
def test_every_text_role_reads_on_the_paper_and_on_surface_2(name):
    palette = derive_palette(KITS[name])
    paper, surface_2 = _rgb(palette, "paper"), _rgb(palette, "surface_2")
    for role in (role for role in TEXT_ROLES if role in palette):
        colour = _rgb(palette, role)
        assert contrast(colour, paper) >= AA_TEXT, (role, palette)
        assert contrast(colour, surface_2) >= AA_TEXT, (role, palette)
    assert contrast(_rgb(palette, "ink"), paper) >= INK_ON_PAPER_MIN_CONTRAST


def test_the_automatos_kit_gets_near_black_headings_a_light_page_and_orange_only_as_the_accent():
    palette = derive_palette(AUTOMATOS)
    orange = parse_hex(ORANGE)
    heading = _rgb(palette, "heading")
    assert luminance(heading) <= HEADING_MAX_LUMINANCE
    assert palette["heading"] != ORANGE
    assert luminance(_rgb(palette, "paper")) >= PAPER_MIN_LUMINANCE
    assert _hue_gap(_rgb(palette, ROLE_ACCENT), orange) < ACCENT_2_MIN_HUE_DEGREES
    for role in (role for role in palette if role != ROLE_ACCENT):
        colour = _rgb(palette, role)
        orange_like = _hue_saturation(colour)[1] >= ACCENT_2_MIN_SATURATION and _hue_gap(colour, orange) < ACCENT_2_MIN_HUE_DEGREES
        assert not orange_like, (role, palette[role])


def test_the_surfaces_are_the_paper_tinted_toward_the_secondary():
    palette = derive_palette(AUTOMATOS)
    paper, navy = _rgb(palette, "paper"), parse_hex(NAVY)
    assert _rgb(palette, "surface") == mix(paper, navy, SURFACE_TINT)
    assert _rgb(palette, "surface_2") == mix(paper, navy, SURFACE_2_TINT)


def test_the_accent_is_the_primary_darkened_only_as_far_as_aa_needs():
    palette = derive_palette(AUTOMATOS)
    accent = _rgb(palette, ROLE_ACCENT)
    surface_2 = _rgb(palette, "surface_2")
    assert luminance(accent) < luminance(parse_hex(ORANGE))
    assert contrast(accent, surface_2) <= ACCENT_TEXT_MIN_CONTRAST + ACCENT_SEARCH_SLACK
    # A primary that already reads is the accent exactly as it is.
    assert derive_palette(HARBOURLINE)[ROLE_ACCENT] == "#1e3a5f"


def test_accent_2_is_the_secondary_only_when_it_is_a_second_hue():
    assert derive_palette(AUTOMATOS)[ROLE_ACCENT_2] == NAVY
    harbourline = derive_palette(HARBOURLINE)
    assert _hue_gap(_rgb(harbourline, ROLE_ACCENT_2), parse_hex("#C26A2E")) < ACCENT_2_MIN_HUE_DEGREES
    assert ROLE_ACCENT_2 not in derive_palette(V1_DEFAULTS)
    assert ROLE_ACCENT_2 not in derive_palette(DARK_ONLY)
    assert ROLE_ACCENT_2 not in derive_palette({**AUTOMATOS, "secondary_color": "#777777"})


def test_the_page_is_the_kits_light_colour_or_white():
    assert derive_palette(LIGHT_ONLY)["paper"] == "#fafafa"
    assert parse_hex(derive_palette(DARK_ONLY)["paper"]) == WHITE
    assert parse_hex(derive_palette(AUTOMATOS)["paper"]) == WHITE


def test_a_kit_with_no_colours_still_derives_a_readable_palette():
    palette = derive_palette({})
    assert set(REQUIRED_ROLES) <= set(palette)
    paper = _rgb(palette, "paper")
    assert all(contrast(_rgb(palette, role), paper) >= AA_TEXT for role in TEXT_ROLES if role in palette)


def test_a_stored_role_always_wins_and_is_reported_as_set():
    kit = {**AUTOMATOS, "palette": {"paper": "#FAF7F2", "accent": "#C44A1A"}}
    roles, sources = effective_palette(kit)
    assert roles["paper"] == "#faf7f2"
    assert roles[ROLE_ACCENT] == ORANGE
    assert sources["paper"] == ROLE_SET and sources[ROLE_ACCENT] == ROLE_SET
    assert {role for role, source in sources.items() if source == ROLE_DERIVED} == set(roles) - {"paper", ROLE_ACCENT}
    # The derived roles are measured against the stored paper.
    assert contrast(parse_hex(roles["ink"]), parse_hex("#faf7f2")) >= INK_ON_PAPER_MIN_CONTRAST


@pytest.mark.parametrize("stored", [{"ink": "zz", "paper": 7}, "not a palette", None, ["#ffffff"]])
def test_a_stored_role_that_is_not_a_hex_colour_is_derived(stored):
    roles, sources = effective_palette({**AUTOMATOS, "palette": stored})
    assert roles == derive_palette(AUTOMATOS)
    assert set(sources.values()) == {ROLE_DERIVED}


def test_a_v1_kit_reads_as_v2_with_every_role_derived_and_is_never_mutated():
    kit = copy.deepcopy(AUTOMATOS)
    roles, sources = effective_palette(kit)
    assert kit == AUTOMATOS
    assert set(sources) == set(roles)
    assert set(sources.values()) == {ROLE_DERIVED}
