"""PRD-271 C01 — the phone layout contracts (``services/themes/data/phone-layouts.json``).

The table's own shape; every contract's ``example`` passing its contract (each value of its slot's kind
and in range, every pair at AA); every pair naming a colour slot or a fixed colour; the loader's answers;
and the browser's copy, ``frontend/lib/theme-host/phone-layouts.json``, byte for byte the host's. §4.3's
four contracts at major 1 are pinned by their slot lists and pair counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes.contract import MOBILE_AA_TEXT, MOBILE_AA_UI, MOBILE_FONT_ROLES  # noqa: E402
from services.themes.layout_contracts import (  # noqa: E402
    LAYOUT_PATTERN, LAYOUTS_FILE, MASCOT_KINDS, PAIR_KINDS, SLOT_KINDS, TONE_GROUND, TONE_RULES, contract_of,
    home_layouts, layout_names, layouts_table, required_slots,
)
from services.themes.layout_slots import slot_value  # noqa: E402
from services.themes.mobile_contrast import contrast_ratio  # noqa: E402
from services.themes.mobile_tokens import parse_colour  # noqa: E402

# The browser's copy of the table. This path, named here literally, is what makes the backend planner
# (scripts/ci/backend_test_plan.py) select this file on a frontend change.
FRONTEND_COPY = ORCH_ROOT.parent / "frontend/lib/theme-host/phone-layouts.json"
SCHEMA = "automatos-phone-layouts/v1"
LAYOUT_KEYS = {"name", "home", "versions"}
VERSION_KEYS = {"revision", "tone", "slots", "fixed", "pairs", "fonts", "mascot", "example"}
SLOT_KEYS = {"colour": {"kind", "group", "default"}, "tint": {"kind", "group", "default"},
             "number": {"kind", "group", "min", "max", "default"}, "choice": {"kind", "group", "choices", "default"}}
MINIMUMS = {"text": MOBILE_AA_TEXT, "ui": MOBILE_AA_UI}
CONTRACTS = layout_names()
# §4.3, read from Go's tokens and the kit's themes: the required slots and the number of pairs of each contract.
CANDY_STOPS = tuple(f"g{card}{stop}" for card in (1, 2, 3) for stop in "abc")
REQUIRED = {
    "play@1": ("bg", "surface", "text", "text2", "line", "chip", "panel", "accent", "accentInk", "autoLight"),
    "hud@1": ("bg", "accent", "accentInk"),
    "arcade@1": (
        "page", "win", "tx", "tx2", "bd", "sh", "bevl", "bevd", "bar", "barTx", "ac", "aci", "hl", "hlt", "hp", "xp",
        "mp", "qs",
    ),
    "candy@1": (
        "page", "bandA", "bandB", "band1", "band2", "band3", "win", "chip", "text", "quiet", "line", "border",
        "bandQuiet", "accent", "accentInk", "accentText", "userA", "userB", "userInk", "stageInk", "cardInk",
        *CANDY_STOPS, "tint",
    ),
}
PAIR_COUNTS = {"play@1": 5, "hud@1": 4, "arcade@1": 15, "candy@1": 34}
HOMES = {"automatos/play": "play", "automatos/hud": "hud", "automatos/arcade": "arcade", "automatos/candy": "candy"}


def _raw() -> dict:
    return json.loads(LAYOUTS_FILE.read_text(encoding="utf-8"))


def _versions():
    return [(f"{layout_id}@{major}", version) for layout_id, entry in _raw()["layouts"].items()
            for major, version in entry["versions"].items()]


# ---------------------------------------------------------------------------
# The table's shape
# ---------------------------------------------------------------------------


def test_the_table_names_its_schema_and_each_layout_its_name_home_and_majors_in_order():
    table = _raw()
    assert table["schema"] == SCHEMA and table["_comment"].strip()
    assert set(table) == {"_comment", "schema", "layouts"}
    for layout_id, entry in table["layouts"].items():
        assert set(entry) == LAYOUT_KEYS, layout_id
        assert entry["name"].strip() and entry["home"].count("/") == 1, layout_id
        majors = list(entry["versions"])
        assert majors == sorted(majors, key=int) and all(LAYOUT_PATTERN.fullmatch(f"{layout_id}@{m}") for m in majors)
    homes = [entry["home"] for entry in table["layouts"].values()]
    assert len(homes) == len(set(homes))


@pytest.mark.parametrize("layout, version", _versions(), ids=[name for name, _ in _versions()])
def test_each_contract_has_its_fields_and_well_formed_slots(layout, version):
    assert set(version) == VERSION_KEYS and isinstance(version["revision"], int) and version["revision"] >= 1
    assert version["mascot"] in MASCOT_KINDS and set(version["fonts"]) <= set(MOBILE_FONT_ROLES)
    for name, spec in version["slots"].items():
        assert spec["kind"] in SLOT_KINDS and set(spec) <= SLOT_KEYS[spec["kind"]], name
        assert spec["group"].strip(), name
        if spec["kind"] == "number":
            assert isinstance(spec["min"], int) and isinstance(spec["max"], int) and spec["min"] <= spec["max"], name
        if spec["kind"] == "choice":
            assert spec["choices"] and len(set(spec["choices"])) == len(spec["choices"]), name
            assert all(isinstance(choice, str) and choice for choice in spec["choices"]), name
        if "default" in spec:
            assert slot_value(spec, spec["default"]) is not None, f"{layout}: {name}'s default doesn't fit its kind"
    assert not set(version["fixed"]) & set(version["slots"]), "a fixed colour can't also be a slot"


@pytest.mark.parametrize("layout, version", _versions(), ids=[name for name, _ in _versions()])
def test_each_tone_rule_reads_a_ground_and_a_text_colour_slot(layout, version):
    tone = version["tone"]
    assert tone["rule"] in TONE_RULES
    if tone["rule"] == TONE_GROUND:
        assert set(tone) == {"rule", "ground", "text"}
        assert {version["slots"][tone[key]]["kind"] for key in ("ground", "text")} == {"colour"}
    else:
        assert set(tone) == {"rule"}


@pytest.mark.parametrize("layout, version", _versions(), ids=[name for name, _ in _versions()])
def test_every_pair_names_a_colour_slot_or_a_fixed_colour(layout, version):
    colours = {name for name, spec in version["slots"].items() if spec["kind"] == "colour"}
    fixed = {name for name, value in version["fixed"].items() if parse_colour(value) is not None}
    for foreground, background, kind in version["pairs"]:
        assert kind in PAIR_KINDS
        assert {foreground, background} <= colours | fixed, (layout, foreground, background)
    assert len({tuple(pair) for pair in version["pairs"]}) == len(version["pairs"])


# ---------------------------------------------------------------------------
# Every example passes its own contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout, version", _versions(), ids=[name for name, _ in _versions()])
def test_every_example_fills_every_slot_with_a_value_of_its_kind(layout, version):
    example = version["example"]
    assert list(example) == list(version["slots"]), "the example lists every slot, in the contract's order"
    for name, spec in version["slots"].items():
        assert slot_value(spec, example[name]) is not None, f"{layout}: example {name}={example[name]!r}"


@pytest.mark.parametrize("layout, version", _versions(), ids=[name for name, _ in _versions()])
def test_every_example_meets_aa_at_every_pair(layout, version):
    colours = {**version["fixed"], **version["example"]}
    for foreground, background, kind in version["pairs"]:
        ratio = contrast_ratio(parse_colour(colours[foreground]), parse_colour(colours[background]))
        assert ratio >= MINIMUMS[kind], f"{layout}: {foreground} on {background} is {ratio:.2f}:1"


@pytest.mark.parametrize("layout, version", _versions(), ids=[name for name, _ in _versions()])
def test_every_default_is_the_home_themes_default_look(layout, version):
    # §4.3: defaults are the home theme's default preset's values, and the example is that look as slots.
    for name, spec in version["slots"].items():
        if "default" in spec:
            assert spec["default"] == version["example"][name], f"{layout}: {name}"


# ---------------------------------------------------------------------------
# §4.3: the four contracts at major 1
# ---------------------------------------------------------------------------


def test_the_host_knows_the_four_contracts_of_section_4_3():
    assert CONTRACTS == ("play@1", "hud@1", "arcade@1", "candy@1")
    for layout in CONTRACTS:
        contract = contract_of(layout)
        assert (contract.layout, contract.revision) == (layout, 1)
        assert required_slots(contract) == REQUIRED[layout]
        assert len(contract.pairs) == PAIR_COUNTS[layout]


def test_hud_fixes_its_text_and_the_others_take_their_tone_from_their_ground():
    assert contract_of("hud@1").tone == {"rule": "dark"}
    assert set(contract_of("hud@1").fixed) == {"text", "text2", "line", "track", "error"}
    grounds = {layout: dict(contract_of(layout).tone) for layout in ("play@1", "arcade@1", "candy@1")}
    assert grounds == {
        "play@1": {"rule": "ground", "ground": "bg", "text": "text"},
        "arcade@1": {"rule": "ground", "ground": "page", "text": "tx"},
        "candy@1": {"rule": "ground", "ground": "page", "text": "text"},
    }


def test_candy_roundness_is_a_number_from_16_to_40_and_its_blobs_may_be_transparent():
    slots = contract_of("candy@1").slots
    assert dict(slots["roundness"]) == {"kind": "number", "min": 16, "max": 40, "default": 28, "group": "Shape"}
    assert {slots[name]["kind"] for name in ("blob1", "blob2", "blob3")} == {"tint"}
    assert slot_value(slots["blob1"], "transparent") == "transparent"


def test_the_drawn_auto_layouts_and_the_bundled_ones():
    assert {layout: contract_of(layout).mascot for layout in CONTRACTS} == {
        "play@1": "bundled", "hud@1": "bundled", "arcade@1": "drawn", "candy@1": "drawn",
    }


# ---------------------------------------------------------------------------
# The loader
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [None, 1, "", "hud", "hud@", "hud@0", "hud@01", "hud@2", "HUD@1", "disco@1", " hud@1",
                                    "hud@1 ", "constructor@1", ["hud@1"]])
def test_contract_of_answers_none_for_anything_but_a_known_contract(layout):
    assert contract_of(layout) is None


def test_home_layouts_maps_each_first_party_theme_to_its_layout():
    assert home_layouts() == HOMES


def test_the_loaded_table_is_read_only():
    contract = contract_of("hud@1")
    with pytest.raises(TypeError):
        contract.slots["bg"] = {"kind": "colour"}
    with pytest.raises(TypeError):
        layouts_table()["layouts"]["hud"] = {}


def test_the_frontend_copy_is_byte_for_byte_the_hosts():
    assert FRONTEND_COPY.read_bytes() == LAYOUTS_FILE.read_bytes()
