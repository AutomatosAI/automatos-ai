"""PRD-261 M-03 — WCAG AA contrast for theme.json's mobile block (``services/themes/mobile_contrast.py``).

* ``contrast_ratio`` gives every ``contrast`` case of the shared corpus its ratio to two decimals, over
  the hexes ``mobile_tokens.parse_colour`` reads (``frontend/lib/look-kit/__tests__/contrast-parity.test.ts``
  holds look-kit's ``contrastRatio`` and ``tripletToHex`` to the same cases);
* ``measure_mobile`` measures the block's ten pairs (eight text pairs at 4.5:1, two UI pairs at 3:1) per
  scheme of ``mobile.tokens``, or under ``fromLooks`` per preset painted through ``boot/prepaint.json``: a
  pair under its minimum fails, named by its scheme or preset, both tokens and the ratio; a pair whose two
  tokens are not both set is not measured; a manifest without the block is left to the review (None).

The level and the verdict the review gives these outcomes are ``test_prd261_review_stage.py``'s.
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
from services.themes.mobile_contrast import (  # noqa: E402
    contrast_ratio, measure_mobile, measure_pair, outcome_of, relative_luminance,
)
from services.themes.mobile_tokens import parse_colour  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

CASES = json.loads(Path(build.__file__).with_name("cases.json").read_text(encoding="utf-8"))["contrast"]["cases"]
WHITE, INK_TEXT = "0 0% 100%", "#14181c"
PASSING_LIGHT = {"--background": WHITE, "--foreground": INK_TEXT}
# The corpus's colours at about 3:1 (3.03), 3.2:1 (3.19) and under 3:1 (2.85) on white.
ABOUT_THREE, ABOUT_THREE_TWO, UNDER_THREE = "0 0% 58%", "0 0% 56.5%", "0 0% 60%"
DARK_GROUND, DARK_TEXT, DIM_TEXT = "220 13% 9%", "210 20% 96%", "0 0% 30%"
BLUE, GREEN = "221.21 83.19% 53.33%", "#2e9e5b"
PREPAINT = "boot/prepaint.json"
TABLE = {
    "knobs": {
        "ground": {
            "paper": {"--background": WHITE, "--foreground": INK_TEXT, "--hello-bg": "#f4efe6"},
            "ink": {"--background": DARK_GROUND, "--foreground": DARK_TEXT},
        },
        # White on this green is 3.41:1: a text pair under AA.
        "accent": {
            "blue": {"--primary": BLUE, "--primary-foreground": WHITE, "--ring": BLUE},
            "green": {"--primary": GREEN, "--primary-foreground": WHITE, "--ring": GREEN},
        },
    },
    "defaults": {"accent": "blue", "ground": "paper", "radius": 6},
    "tone": {"knob": "ground", "values": {"paper": "light", "ink": "dark"}},
}


def _tokens(**schemes):
    return {"mobile": {"tokens": schemes}}


def _pairs(subject):
    return {(pair["foreground"], pair["background"]): pair for pair in subject["pairs"]}


def _look_files(table=TABLE, **presets):
    files = {f"looks/presets/{preset}.look.json": json.dumps({"id": preset, **look}).encode("utf-8")
             for preset, look in presets.items()}
    return {**files, PREPAINT: json.dumps(table).encode("utf-8")} if table is not None else files


# ---------------------------------------------------------------------------
# The ratio: look-kit's, case for case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_every_corpus_contrast_case_has_its_ratio_to_two_decimals(case):
    foreground, background = parse_colour(case["foreground"]), parse_colour(case["background"])
    assert round(contrast_ratio(foreground, background), 2) == case["ratio"]
    assert round(contrast_ratio(background, foreground), 2) == case["ratio"]  # either way round


def test_luminance_runs_from_black_to_white_and_refuses_what_is_not_a_hex():
    assert (relative_luminance("#000000"), relative_luminance("#FFFFFF"), relative_luminance("#ffffff")) == (0, 1, 1)
    assert contrast_ratio("#000000", "#FFFFFF") == 21 and contrast_ratio("#2F6FED", "#2f6fed") == 1
    for value in ("0 0% 100%", "#fff", "#2f6fed80", "", None):
        with pytest.raises(ValueError):
            relative_luminance(value)


def test_the_ten_pairs_are_measured_against_their_own_minimum():
    pairs = [measure_pair({}, pair) for pair in contract.MOBILE_CONTRAST_PAIRS]
    assert [pair["minimum"] for pair in pairs] == [contract.MOBILE_AA_TEXT] * 8 + [contract.MOBILE_AA_UI] * 2
    assert [pair["kind"] for pair in pairs] == ["text"] * 8 + ["ui"] * 2
    assert {pair["status"] for pair in pairs} == {"not_measured"} and {pair["ratio"] for pair in pairs} == {None}


def test_the_outcome_is_fail_on_any_failure_pass_on_any_pass_else_not_measured():
    assert outcome_of(["pass", "fail", "not_measured"]) == "fail"
    assert outcome_of(["pass", "not_measured"]) == "pass"
    assert outcome_of(["not_measured"]) == outcome_of([]) == "not_measured"


# ---------------------------------------------------------------------------
# mobile.tokens: per scheme
# ---------------------------------------------------------------------------


def test_text_at_about_3_to_1_on_the_light_ground_fails_naming_light_both_tokens_and_the_ratio():
    measured = measure_mobile(_tokens(light={"--background": WHITE, "--foreground": ABOUT_THREE}), {})
    light = measured["schemes"][0]
    assert (measured["source"], measured["status"], light["id"], light["status"]) == ("tokens", "fail", "light", "fail")
    assert _pairs(light)[("--foreground", "--background")]["ratio"] == 3.03
    assert measured["problems"] == [
        "mobile.tokens.light: --foreground on --background is 3.03:1, under the AA minimum of 4.5:1 for text.",
    ]
    assert measured["reason"].startswith("1 measured colour pair(s) of the mobile block are under WCAG AA")


def test_a_failing_dark_scheme_is_named_and_the_passing_light_one_is_not():
    measured = measure_mobile(_tokens(light=PASSING_LIGHT, dark={"--background": DARK_GROUND, "--foreground": DIM_TEXT}), {})
    assert [(scheme["id"], scheme["status"]) for scheme in measured["schemes"]] == [("light", "pass"), ("dark", "fail")]
    assert measured["status"] == "fail" and len(measured["problems"]) == 1
    assert measured["problems"][0].startswith("mobile.tokens.dark: --foreground on --background is 2.14:1")
    assert "light" not in measured["problems"][0]


def test_a_ring_at_about_3_2_to_1_passes_as_a_ui_mark_where_text_at_the_same_ratio_fails():
    ring = measure_mobile(_tokens(light={**PASSING_LIGHT, "--ring": ABOUT_THREE_TWO}), {})
    assert (ring["status"], ring["problems"]) == ("pass", [])
    assert _pairs(ring["schemes"][0])[("--ring", "--background")] == {
        "foreground": "--ring", "background": "--background", "kind": "ui", "minimum": 3.0, "status": "pass", "ratio": 3.19,
    }
    text = measure_mobile(_tokens(light={"--background": WHITE, "--foreground": ABOUT_THREE_TWO}), {})
    assert text["status"] == "fail" and "3.19:1" in text["problems"][0]


def test_a_ui_mark_under_3_to_1_fails_naming_its_minimum():
    measured = measure_mobile(_tokens(light={**PASSING_LIGHT, "--primary": UNDER_THREE}), {})
    assert measured["problems"] == [
        "mobile.tokens.light: --primary on --background is 2.85:1, under the AA minimum of 3:1 for UI marks.",
    ]


def test_a_pair_with_one_token_unset_is_not_measured_and_never_fails_the_block():
    measured = measure_mobile(_tokens(light={**PASSING_LIGHT, "--card": WHITE}), {})
    pairs = _pairs(measured["schemes"][0])
    assert (measured["status"], measured["problems"]) == ("pass", [])
    assert pairs[("--card-foreground", "--card")] == {
        "foreground": "--card-foreground", "background": "--card", "kind": "text", "minimum": 4.5,
        "status": "not_measured", "ratio": None,
    }
    assert sum(1 for pair in pairs.values() if pair["status"] == "not_measured") == 9
    assert measured["reason"].endswith("9 pair(s) are not measured: their two tokens are not both set.")


# ---------------------------------------------------------------------------
# fromLooks: per preset, painted through the prepaint table
# ---------------------------------------------------------------------------


def test_from_looks_measures_every_preset_and_names_the_failing_one():
    files = _look_files(default={"accent": "blue", "ground": "paper"}, night={"accent": "green", "ground": "ink"})
    measured = measure_mobile({"mobile": {"fromLooks": True}}, files)
    presets = {preset["id"]: preset for preset in measured["presets"]}

    assert (measured["source"], measured["status"], list(presets)) == ("fromLooks", "fail", ["default", "night"])
    assert (presets["default"]["status"], presets["night"]["status"]) == ("pass", "fail")
    assert _pairs(presets["night"])[("--foreground", "--background")]["ratio"] == 16.57  # ink's own text
    assert measured["problems"] == [
        'Preset "night": --primary-foreground on --primary is 3.41:1, under the AA minimum of 4.5:1 for text.',
    ]
    assert "schemes" not in measured


def test_from_looks_a_knob_value_the_table_lacks_is_painted_from_the_defaults():
    files = _look_files(odd={"accent": "violet", "ground": "paper"})
    odd = measure_mobile({"mobile": {"fromLooks": True}}, files)["presets"][0]
    assert odd["status"] == "pass" and _pairs(odd)[("--primary", "--background")]["ratio"] == 5.17  # blue, the default


def test_from_looks_with_no_table_measures_nothing_and_never_raises():
    measured = measure_mobile({"mobile": {"fromLooks": True}}, _look_files(table=None, default={"ground": "paper"}))
    assert measured["status"] == "not_measured" and measured["presets"][0]["status"] == "not_measured"
    assert measured["reason"].startswith("Not measured") and measured["problems"] == []
    broken = {**_look_files(default={"ground": "paper"}), PREPAINT: b"{not json"}
    assert measure_mobile({"mobile": {"fromLooks": True}}, broken)["status"] == "not_measured"


def test_a_manifest_without_the_block_is_left_to_the_review():
    assert measure_mobile(build.hello_manifest(), {}) is None
    assert measure_mobile({"mobile": None}, {}) is None
