"""PRD-261 M-01 — the phone's colours (``services/themes/mobile_tokens.py``) and the mobile block's lists.

* ``parse_colour`` reads a mobile token value as ``#RRGGBB`` exactly where look-kit's ``hsl.ts`` and
  ``contrast.ts`` read it, and every ``contrast`` case of the shared corpus parses to the hex it names
  (M-03's vitest holds the same hexes to ``tripletToHex``);
* ``paint_look`` paints a look through a prepaint table built here exactly as the web's first paint
  does (``frontend/lib/theme-host/boot.ts`` THEME_BOOT_SCRIPT over ``boot-sanitize.ts``
  ``sanitizePrepaint``): knob tables, then pair tables, in table order; a look that doesn't fit the
  table (boot.ts :158-:159 ``fits``) paints the defaults WHOLE (boot.ts :164), the cases
  ``frontend/lib/looks/__tests__/prepaint.test.ts`` holds the script to; a numeric rule's variable is
  painted last with rgba() (boot.ts :167); only mobile tokens whose final value parses are kept;
* ``missing_tokens`` names the required tokens a painting lacks;
* ``look_tone``: theme.json's fixed tone first (boot.ts :57-:60, :69), then under ``from-look`` the
  table's tone rule, and otherwise the painted ground (look-kit ``isDarkGround``);
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
from services.themes.mobile_tokens import (  # noqa: E402
    boot_table, ground_tone, look_fits, look_tone, missing_tokens, paint_look, parse_colour,
)
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
GLOW = {"var": "--hello-glow", "knob": "glow", "min": 0, "max": 30, "integer": True, "scale": 100, "capTable": None,
        "template": "rgba-alpha", "color": {"knob": "accent", "rgb": {"blue": "47,111,237", "green": "46,158,91"}}}
# The pair table has a cell for paper/blue and ink/green only, so paper/green and ink/blue don't fit it.
TABLE = {
    "knobs": {"ground": {"paper": PAPER, "ink": INK}, "accent": {"blue": BLUE, "green": GREEN}},
    "pairs": [{"a": "ground", "b": "accent", "table": {"paper": {"blue": PAPER_BLUE}, "ink": {"green": INK_GREEN}}}],
    "numeric": [GLOW],
    "defaults": {"ground": "paper", "accent": "blue", "glow": 10},
    "tone": {"knob": "ground", "values": {"paper": "light", "ink": "dark"}},
}
PAPER_LOOK = {"ground": "paper", "accent": "blue", "glow": 10}
INK_LOOK = {"ground": "ink", "accent": "green", "glow": 3}
NO_RULE = {key: value for key, value in TABLE.items() if key != contract.TONE_RULE_KEY}


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
# paint_look: the web's first paint (boot.ts THEME_BOOT_SCRIPT over boot-sanitize.ts sanitizePrepaint)
# ---------------------------------------------------------------------------


def test_a_look_paints_mobile_tokens_only_knob_tables_then_pair_tables():
    tokens = paint_look(TABLE, PAPER_LOOK)
    assert tokens == {
        "--background": "40 33.33% 94.12%", "--foreground": "#1a1814", "--card": "0 0% 100%",
        # the accent table comes after the ground table, and the pair table after both (boot.ts :165-:166)
        "--border": "221 50% 70%", "--primary": "221.21 83.19% 53.33%", "--ring": "#2f6fed",
        "--primary-foreground": "0 0% 100%",
    }
    assert set(tokens) <= set(contract.MOBILE_TOKEN_NAMES)


def test_the_knob_tables_merge_in_the_table_s_own_order():
    accent_first = {**TABLE, "knobs": {"accent": TABLE["knobs"]["accent"], "ground": TABLE["knobs"]["ground"]}}
    assert paint_look(accent_first, PAPER_LOOK)["--border"] == "30 10% 80%"


@pytest.mark.parametrize("look", [
    # a value the schema may allow but the table lacks: the web never paints ink with the default accent
    {"ground": "ink", "accent": "violet", "glow": 3},
    {"ground": "ink", "accent": "blue", "glow": 3},  # the pair table has no ink/blue cell
    # prepaint.test.ts's tampered looks: a glow out of range, not whole, text, missing; names that aren't own keys
    {**INK_LOOK, "glow": 31}, {**INK_LOOK, "glow": -1}, {**INK_LOOK, "glow": 2.5}, {**INK_LOOK, "glow": "3"},
    {**INK_LOOK, "glow": True}, {"ground": "ink", "accent": "green"},
    {**INK_LOOK, "ground": "__proto__"}, {**INK_LOOK, "accent": "toString"}, {**INK_LOOK, "ground": ["ink"]},
    None, [], "ink", {},
])
def test_a_look_that_does_not_fit_the_table_paints_the_defaults_whole(look):
    # boot.ts :164: `if(!fits(l)||!guarded(l)){l=P.defaults;...}` — the whole default look, never value by value.
    assert paint_look(TABLE, look) == paint_look(TABLE, TABLE["defaults"]) == paint_look(TABLE, PAPER_LOOK)
    assert paint_look(TABLE, look)["--background"] == PAPER["--background"]


def test_a_look_fits_when_it_names_an_option_and_a_cell_of_every_table_and_a_glow_in_range():
    table = boot_table(TABLE)
    assert look_fits(table, INK_LOOK) and look_fits(table, {**PAPER_LOOK, "glow": 30.0, "radius": 4})
    assert not look_fits(table, {**INK_LOOK, "glow": 30.5}) and not look_fits(table, {**INK_LOOK, "accent": "blue"})


def test_a_value_that_does_not_parse_is_dropped():
    tokens = paint_look(TABLE, INK_LOOK)
    assert tokens["--background"] == INK["--background"]
    assert "--card" not in tokens and "--muted" not in tokens  # hsl(...) and #fff: CSS, but no phone colour
    assert tokens["--primary-foreground"] == "#04201d" and tokens["--ring"] == GREEN["--ring"]


def test_a_numeric_rule_paints_its_variable_last_with_rgba_so_the_phone_gets_no_value_for_it():
    on_ring = {**TABLE, "numeric": [{**GLOW, "var": "--ring"}]}
    # boot.ts :167 sets --ring to rgba(...) after the knob tables, which is no phone colour
    assert "--ring" not in paint_look(on_ring, PAPER_LOOK)
    assert paint_look(on_ring, PAPER_LOOK)["--primary"] == PAPER_BLUE["--primary"]


@pytest.mark.parametrize("rule", [
    {key: value for key, value in GLOW.items() if key != "capTable"},  # capTable must be null or two knobs
    {**GLOW, "template": "rgba"}, {**GLOW, "integer": 1}, {**GLOW, "min": 40}, {**GLOW, "scale": 0},
    {**GLOW, "color": {"knob": "accent-2", "rgb": {}}}, {**GLOW, "knob": "glow level"},
])
def test_a_numeric_rule_the_boot_s_gate_drops_neither_paints_nor_holds_a_look_to_its_range(rule):
    # boot-sanitize.ts sanitizeNumericRule drops the rule, so the script's fits never sees it
    table = {**TABLE, "numeric": [{**rule, "var": "--ring"}]}
    tokens = paint_look(table, {"ground": "ink", "accent": "green"})
    assert tokens["--background"] == INK["--background"] and tokens["--ring"] == GREEN["--ring"]


def test_a_knob_whose_name_is_no_look_field_is_dropped_and_paints_nothing():
    table = {**TABLE, "knobs": {**TABLE["knobs"], "ground-2": {"x": {"--card": "#000000"}}}}
    assert paint_look(table, PAPER_LOOK) == paint_look(TABLE, PAPER_LOOK)


def test_a_later_value_the_boot_s_gate_refuses_never_overwrites_and_one_it_passes_does():
    knobs = {"ground": {"paper": {**PAPER, "--primary": "#2f6fed"}}, "accent": {"blue": {}}}
    refused = {"knobs": {**knobs, "accent": {"blue": {"--primary": "color-mix(in srgb, red, blue)"}}}, "defaults": PAPER_LOOK}
    passed = {"knobs": {**knobs, "accent": {"blue": {"--primary": "hsl(221 83% 53%)"}}}, "defaults": PAPER_LOOK}
    assert paint_look(refused, PAPER_LOOK)["--primary"] == "#2f6fed"  # the web drops color-mix(), keeps paper's
    assert "--primary" not in paint_look(passed, PAPER_LOOK)  # the web paints hsl(), which no phone reads


@pytest.mark.parametrize("defaults", [
    {"ground": "dusk", "accent": "blue", "glow": 10}, {"ground": "paper", "accent": "blue", "glow": 40},
    {"ground": "paper", "accent": "green", "glow": 10}, {"ground": "paper", "accent": "blue"},
])
def test_a_table_whose_defaults_do_not_fit_it_paints_nothing_for_any_look(defaults):
    # boot-sanitize.ts :231: the web drops the whole table, so not even a fitting look paints from it
    table = {**TABLE, "defaults": defaults}
    assert boot_table(table) is None
    assert paint_look(table, INK_LOOK) == paint_look(table, None) == {}


@pytest.mark.parametrize("table, look", [
    ([], {"ground": "paper"}),
    ({"knobs": ["ground"], "pairs": {"a": "ground"}}, {"ground": "paper"}),
    ({"knobs": {"ground": {"paper": "#ffffff"}}, "pairs": [None, {"a": 1, "b": "x", "table": []}]}, {"ground": "paper"}),
    ({"knobs": {"ground": {"paper": PAPER}}, "numeric": [None, {"capTable": ["a"]}], "defaults": "paper"}, None),
    (TABLE, ["paper"]),
])
def test_a_table_or_a_look_of_the_wrong_shape_paints_without_raising(table, look):
    tokens = paint_look(table, look)
    assert set(tokens) <= set(contract.MOBILE_TOKEN_NAMES)


def test_missing_tokens_names_the_required_tokens_a_painting_gives_no_colour():
    assert missing_tokens(paint_look(TABLE, INK_LOOK)) == []
    assert missing_tokens({}) == list(contract.MOBILE_REQUIRED_TOKENS) == missing_tokens(None)
    assert missing_tokens({"--background": "hsl(0 0% 100%)", "--foreground": "#000000"}) == ["--background"]


# ---------------------------------------------------------------------------
# look_tone: theme.json's fixed tone, else the tone rule under from-look, else the painted ground
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("policy, look, tone", [
    # boot.ts :57-:60, :69: a fixed tone is the boot's tone value, whatever the look
    ("light", INK_LOOK, "light"), ("dark", PAPER_LOOK, "dark"),
    ("from-look", INK_LOOK, "dark"), ("from-look", PAPER_LOOK, "light"),
    ("from-look", {"ground": "ink", "accent": "violet"}, "light"),  # doesn't fit: the defaults' paper
    ("user", INK_LOOK, "dark"), ("user", PAPER_LOOK, "light"), (None, INK_LOOK, "dark"),
])
def test_the_tone_follows_the_theme_s_policy(policy, look, tone):
    assert look_tone(TABLE, look, policy) == tone


def test_from_look_takes_the_rule_where_user_takes_the_painted_ground():
    inverted = {**TABLE, "tone": {"knob": "ground", "values": {"paper": "dark", "ink": "light"}}}
    assert look_tone(inverted, INK_LOOK, "from-look") == "light"
    assert look_tone(inverted, INK_LOOK, "user") == "dark"  # the phone has no user tone cookie


@pytest.mark.parametrize("table", [
    NO_RULE,
    {**TABLE, "tone": {"knob": "ground", "values": {"paper": "light"}}},  # a choice without a tone: dropped
    {**TABLE, "tone": {"knob": "glow", "values": {}}}, {**TABLE, "tone": "dark"},
])
def test_from_look_without_a_rule_the_boot_keeps_takes_the_painted_ground(table):
    assert look_tone(table, INK_LOOK, "from-look") == "dark"
    assert look_tone(table, PAPER_LOOK, "from-look") == "light"


def test_the_ground_is_dark_when_its_background_is_darker_than_its_text_and_light_without_one():
    assert ground_tone({"--background": "#101418", "--foreground": "#f0f4f8"}) == "dark"
    assert ground_tone({"--background": "#f0f4f8", "--foreground": "#101418"}) == "light"
    assert ground_tone({"--background": "#101418"}) == ground_tone({}) == "light"
    assert look_tone({**TABLE, "defaults": {"ground": "dusk"}}, INK_LOOK, "user") == "light"  # paints nothing


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
