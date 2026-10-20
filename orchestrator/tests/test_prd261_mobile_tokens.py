"""PRD-261 M-01 — the phone's colours (``services/themes/mobile_tokens.py``) and the mobile block's lists.

* ``parse_colour`` reads a mobile token value as ``#RRGGBB`` exactly where look-kit's ``hsl.ts`` and
  ``contrast.ts`` read it, and every ``contrast`` case of the shared corpus parses to the hex it names
  (M-03's vitest holds the same hexes to ``tripletToHex``);
* ``project_look`` paints a look through a prepaint table built here: knob tables, then pair tables,
  in table order; a value the table lacks takes the defaults'; only mobile tokens whose value parses
  are kept, as written;
* ``look_tone`` takes the table's tone rule, else theme.json's fixed tone, else light;
* a checked manifest's mobile fonts are files it names (``manifest.referenced_files``), so a package
  without one is ``MISSING_FILE``.

The block's manifest verdicts are the corpus's (``test_prd261_theme_package.py`` runs every
``manifests`` case; ``validator-parity.test.ts`` runs them in TypeScript).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes import contract  # noqa: E402
from services.themes.manifest import referenced_files, validate_manifest  # noqa: E402
from services.themes.mobile_tokens import look_tone, parse_colour, project_look  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

CASES = json.loads((Path(build.__file__).with_name("cases.json")).read_text(encoding="utf-8"))
MOBILE_ENGINES = {"automatos": "^1.1.0", "sdk": "^1.0.0"}

PAPER = {
    "--background": "40 33.33% 94.12%", "--foreground": "#1a1814", "--card": "0 0% 100%",
    "--paper": "hsl(0 0% 100%)", "--zinc-ground": "#f4efe6", "--border": "30 10% 80%",
}
INK = {"--background": "210 20% 7.84%", "--foreground": "#f0f4f8", "--card": "hsl(210 20% 12%)", "--border": "#2a3038"}
BLUE = {"--primary": "#2f6fed", "--ring": "#2f6fed", "--border": "221 50% 70%"}
GREEN = {"--primary": "#2e9e5b", "--ring": "#2e9e5b"}
PAPER_BLUE = {"--primary": "221.21 83.19% 53.33%", "--primary-foreground": "0 0% 100%"}
INK_GREEN = {"--primary-foreground": "#04201d", "--muted": "#fff"}
TABLE = {
    "knobs": {"ground": {"paper": PAPER, "ink": INK}, "accent": {"blue": BLUE, "green": GREEN}},
    "pairs": [{"a": "ground", "b": "accent", "table": {"paper": {"blue": PAPER_BLUE}, "ink": {"green": INK_GREEN}}}],
    "numeric": [{"var": "--ring", "knob": "glow", "min": 0, "max": 30, "integer": True, "scale": 100,
                 "template": "rgba-alpha", "color": {"knob": "accent", "rgb": {"blue": "47,111,237", "green": "46,158,91"}}}],
    "defaults": {"ground": "paper", "accent": "blue", "glow": 10},
    "tone": {"knob": "ground", "values": {"paper": "light", "ink": "dark"}},
}


# ---------------------------------------------------------------------------
# The lists (contract.py; validator-parity.test.ts holds the TypeScript twins equal)
# ---------------------------------------------------------------------------


def test_the_mobile_lists_hold_together():
    names = contract.MOBILE_TOKEN_NAMES
    assert len(names) == len(set(names)) == 24
    assert set(contract.MOBILE_REQUIRED_TOKENS) == {"--background", "--foreground"} <= set(names)
    kinds = [kind for _, _, kind in contract.MOBILE_CONTRAST_PAIRS]
    assert (kinds.count("text"), kinds.count("ui")) == (8, 2)
    assert {token for pair in contract.MOBILE_CONTRAST_PAIRS for token in pair[:2]} <= set(names)
    assert (contract.MOBILE_AA_TEXT, contract.MOBILE_AA_UI, contract.MOBILE_MIN_HOST) == (4.5, 3.0, "1.1.0")


# ---------------------------------------------------------------------------
# parse_colour: the hex look-kit gives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, hex_value", [
    ("0 0% 100%", "#FFFFFF"),
    ("0 0% 0%", "#000000"),
    # PRD-260's Paper coral: lib/looks/__tests__/studio-tokens.test.ts names this triplet #A54B49.
    ("1.3 38.66% 46.67%", "#A54B49"),
    ("221.21 83.19% 53.33%", "#2563EB"),
    ("359.99 100% 100%", "#FFFFFF"),
    ("007 0% 0%", "#000000"),
    ("#2f6fed", "#2F6FED"),
    ("#ABCDEF", "#ABCDEF"),
])
def test_a_triplet_or_a_hex_parses_to_its_hex(value, hex_value):
    assert parse_colour(value) == hex_value


@pytest.mark.parametrize("value", [
    "360 0% 0%", "361 0% 0%", "0 101% 0%", "0 0% 100.01%", "0 0% 99.999%", "-1 0% 0%", ".5 0% 0%", "0 0 0",
    "0  0% 0%", " 0 0% 0%", "0 0% 0%\n", "hsl(0 0% 100%)", "#fff", "#2f6fed80", "#gggggg", "", "red",
    "\u0661 0% 0%",  # an Arabic-Indic one: Python's \d would take it, JavaScript's never does
    0, None, True, ["0 0% 0%"], {"h": 0},
])
def test_anything_else_does_not_parse(value):
    assert parse_colour(value) is None


@pytest.mark.parametrize("case", CASES["contrast"]["cases"], ids=[case["name"] for case in CASES["contrast"]["cases"]])
def test_every_contrast_case_value_parses_to_the_hex_the_case_names(case):
    assert parse_colour(case["foreground"]) == case["foregroundHex"]
    assert parse_colour(case["background"]) == case["backgroundHex"]


def test_the_contrast_section_measures_text_and_ui_minimums_both_ways():
    ratios = [case["ratio"] for case in CASES["contrast"]["cases"]]
    assert any(ratio >= contract.MOBILE_AA_TEXT for ratio in ratios)
    assert any(contract.MOBILE_AA_UI <= ratio < contract.MOBILE_AA_TEXT for ratio in ratios)
    assert any(ratio < contract.MOBILE_AA_UI for ratio in ratios)


# ---------------------------------------------------------------------------
# project_look: the web boot's walk, mobile tokens only
# ---------------------------------------------------------------------------


def test_a_look_projects_to_mobile_tokens_only_knob_tables_then_pair_tables():
    tokens = project_look(TABLE, {"ground": "paper", "accent": "blue", "glow": 10})
    assert tokens == {
        "--background": "40 33.33% 94.12%", "--foreground": "#1a1814", "--card": "0 0% 100%",
        # the accent table comes after the ground table, and the pair table after both
        "--border": "221 50% 70%", "--primary": "221.21 83.19% 53.33%", "--ring": "#2f6fed",
        "--primary-foreground": "0 0% 100%",
    }
    assert set(tokens) <= set(contract.MOBILE_TOKEN_NAMES)


def test_the_knob_tables_merge_in_the_table_s_own_order():
    accent_first = {**TABLE, "knobs": {"accent": TABLE["knobs"]["accent"], "ground": TABLE["knobs"]["ground"]}}
    assert project_look(accent_first, {"ground": "paper", "accent": "blue"})["--border"] == "30 10% 80%"


def test_a_value_the_table_lacks_takes_the_defaults_value():
    tokens = project_look(TABLE, {"ground": "dusk", "accent": "green"})
    assert tokens["--background"] == PAPER["--background"]
    assert tokens["--ring"] == GREEN["--ring"]
    # the pair table has no paper/green cell: its second value falls back to the defaults' blue
    assert tokens["--primary"] == PAPER_BLUE["--primary"]
    assert project_look(TABLE, {}) == project_look(TABLE, TABLE["defaults"])


def test_a_value_that_does_not_parse_is_dropped():
    tokens = project_look(TABLE, {"ground": "ink", "accent": "green"})
    assert "--card" not in tokens and "--muted" not in tokens
    assert tokens["--primary-foreground"] == "#04201d"
    assert tokens["--ring"] == GREEN["--ring"]  # the numeric rule on --ring paints a glow, never a token


@pytest.mark.parametrize("table, look", [
    ([], {"ground": "paper"}),
    ({"knobs": ["ground"], "pairs": {"a": "ground"}}, {"ground": "paper"}),
    ({"knobs": {"ground": {"paper": "#ffffff"}}, "pairs": [None, {"a": 1, "b": "x", "table": []}]}, {"ground": "paper"}),
    (TABLE, ["paper"]),
])
def test_a_table_or_a_look_of_the_wrong_shape_projects_without_raising(table, look):
    tokens = project_look(table, look)
    assert set(tokens) <= set(contract.MOBILE_TOKEN_NAMES)


# ---------------------------------------------------------------------------
# look_tone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("look, manifest_tone, tone", [
    ({"ground": "ink"}, "light", "dark"),
    ({"ground": "paper"}, "dark", "light"),
    ({"ground": "dusk"}, "dark", "light"),  # the defaults' paper
    ({}, "user", "light"),
])
def test_the_tone_comes_from_the_table_s_rule(look, manifest_tone, tone):
    assert look_tone(TABLE, look, manifest_tone) == tone


@pytest.mark.parametrize("manifest_tone, tone", [("dark", "dark"), ("light", "light"), ("user", "light"), ("from-look", "light"), (None, "light")])
def test_without_a_rule_the_tone_is_the_theme_s_fixed_one_else_light(manifest_tone, tone):
    no_rule = {key: value for key, value in TABLE.items() if key != contract.TONE_RULE_KEY}
    assert look_tone(no_rule, {"ground": "ink"}, manifest_tone) == tone


def test_a_rule_that_gives_the_look_no_tone_falls_back_to_the_theme_s_tone():
    partial = {**TABLE, "tone": {"knob": "ground", "values": {"paper": "light"}}}
    assert look_tone(partial, {"ground": "ink"}, "dark") == "dark"
    assert look_tone({**TABLE, "tone": "dark"}, {"ground": "paper"}, "user") == "light"


# ---------------------------------------------------------------------------
# The manifest names the mobile fonts as package files
# ---------------------------------------------------------------------------


def test_a_checked_manifest_names_its_mobile_fonts_after_its_other_files():
    fonts = {"sans": "assets/fonts/mobile/inter.ttf", "mono": "assets/fonts/mobile/mono.otf"}
    document = {
        **build.hello_manifest(), "engines": MOBILE_ENGINES, "fonts": ["assets/fonts/brand.woff2"],
        "mobile": {"fromLooks": True, "fonts": fonts},
    }
    result = validate_manifest(document)
    assert result.errors == []
    assert referenced_files(result.manifest) == ["assets/fonts/brand.woff2", *fonts.values()]


def test_a_manifest_without_mobile_names_what_it_named_before():
    assert referenced_files(validate_manifest(build.hello_manifest()).manifest) == []
