"""PRD-271 C02 — checking a ``mobile`` block's ``layout`` and ``slots`` (``mobile_layout.py``, ``layout_slots.py``).

The manifest's half is held to the shared corpus (``tests/fixtures/themes/cases.json``, the cases named
``mobile layout``; ``test_prd261_theme_package.py`` runs them here, ``validator-parity.test.ts`` in the
browser); this file pins what the corpus can't say alone: each slot kind's values, a number slot read from
a choice key (Candy's ``roundness`` since kit 1.0.1), the look painted as the web paints it, and the
reader's half (``MOBILE_CANNOT_PAINT`` naming the slot) on real packages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes.contract import LAYOUT_MIN_HOST, THEME_HOST_VERSION  # noqa: E402
from services.themes.layout_contracts import contract_of  # noqa: E402
from services.themes.layout_paint import layout_tone, resolve_slots  # noqa: E402
from services.themes.layout_slots import is_tint, knob_number, knob_slot_value, slot_value  # noqa: E402
from services.themes.manifest import validate_manifest  # noqa: E402
from services.themes.mobile_manifest import MOBILE_FIELDS  # noqa: E402
from services.themes.mobile_tokens import paint_vars, painted_look  # noqa: E402
from services.themes.package import ThemePackageError, read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

CASES = json.loads((Path(build.__file__).with_name("cases.json")).read_text(encoding="utf-8"))
ENGINES = {"automatos": f"^{LAYOUT_MIN_HOST}", "sdk": "^1.0.0"}
SKIN = {"tokens": {"light": {"--background": "0 0% 100%", "--foreground": "#14181c"}}}
HUD = {"bg": "#030507", "accent": "#3FE0FF", "accentInk": "#021116"}
LOOKS = {"bg": {"look": "--hello-bg"}, "accent": {"look": "--hello-accent"}, "accentInk": {"look": "--hello-on-accent"},
         "glow": {"look": "radius"}}
# hello's three look knobs painted (the reader needs each, first_paint.py): accent and ground by choice tables,
# radius by build.HELLO_RADIUS_RULE, whose --hello-glow is painted from a number.
ACCENTS = {"blue": {"--hello-accent": "#2F6FED", "--hello-on-accent": "#FFFFFF"},
           "green": {"--hello-accent": "#1E7F3C", "--hello-on-accent": "#0B1A07"}}
GROUNDS = {"paper": {"--hello-bg": "#14181C"}, "ink": {"--hello-bg": "#030507", "--hello-sky": "#0B2340"}}
TABLE = {"knobs": {"accent": ACCENTS, "ground": GROUNDS}, "pairs": [], "numeric": [build.HELLO_RADIUS_RULE],
         "defaults": dict(build.HELLO_DEFAULTS)}
NIGHT = {"accent": "green", "ground": "ink", "radius": 0}


def _block(slots: Any, layout: str = "hud@1") -> Dict[str, Any]:
    return {**SKIN, "layout": layout, "slots": slots}


def _package_errors(slots: Any, table: Any = TABLE) -> List[Dict[str, str]]:
    fields = {"engines": ENGINES, "mobile": _block(slots)}
    files = {build.MANIFEST: build.changed_manifest(fields)}
    files = {**files, "boot/prepaint.json": json.dumps(table).encode("utf-8")} if table is not None else files
    try:
        read_theme_package(build.build_package(replace=files))
    except ThemePackageError as exc:
        return exc.errors
    return []


# ---------------------------------------------------------------------------
# The block's fields and the host it needs
# ---------------------------------------------------------------------------


def test_the_block_gains_layout_and_slots_on_host_1_2_0():
    assert MOBILE_FIELDS[-2:] == ("layout", "slots")
    assert THEME_HOST_VERSION == LAYOUT_MIN_HOST == "1.2.0"


def test_every_kind_of_refusal_has_a_corpus_case_both_languages_run():
    layout_cases = [case for case in CASES["manifests"]["cases"] if case["name"].startswith("mobile layout")]
    assert {case["code"] for case in layout_cases} == {None, "BAD_MOBILE"}
    paths = {message.split('"')[1] for case in layout_cases for message in case.get("messages", [])}
    assert {"mobile.layout", "mobile.slots", "mobile.slots.bg", "mobile.slots.grid", "mobile.slots.scan"} <= paths
    packages = [case for case in CASES["packages"]["cases"] if case["name"].startswith("mobile layout")]
    assert {case["code"] for case in packages} == {None, "MOBILE_CANNOT_PAINT"}


def test_an_older_host_would_refuse_the_block_so_the_engines_must_admit_none():
    document = {**build.hello_manifest(), "engines": {"automatos": "^1.1.0", "sdk": "^1.0.0"}, "mobile": _block(HUD)}
    assert [(error["code"], error["path"]) for error in validate_manifest(document).errors] == [("BAD_MOBILE", "engines.automatos")]


# ---------------------------------------------------------------------------
# The slot kinds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value, fits", [
    ("#030507", True), ("210 40% 2%", True), ("rgba(3,5,7,.9)", True), ("rgba(255,255,255,1)", True),
    ("rgba(0,0,0,0)", True), ("rgba(0,0,0,0.5)", True), ("rgba(0,0,0,1.0)", True), ("transparent", True),
    ("rgba(0, 0, 0, .5)", False), ("rgba(256,0,0,.5)", False), ("rgba(0,0,0,1.5)", False), ("rgba(0,0,0,.12345)", False),
    ("rgb(0,0,0)", False), ("Transparent", False), ("url(x.png)", False), ("var(--x)", False), ("#035", False), (0, False),
    (None, False),
])
def test_a_tint_is_a_colour_an_rgba_or_transparent(value, fits):
    assert is_tint(value) is fits


def test_a_number_slot_takes_a_whole_number_in_range_as_an_int():
    spec = contract_of("candy@1").slots["roundness"]
    assert [slot_value(spec, value) for value in (16, 28.0, 40, 15, 41, 28.5, "28", True)] == [16, 28, 40, None, None, None, None, None]
    assert isinstance(slot_value(spec, 28.0), int)


def test_a_number_slot_from_a_knob_takes_its_number_or_a_choice_key_that_writes_one():
    # Candy's roundness is a choice since kit 1.0.1: its keys are '16' to '40'.
    spec = contract_of("candy@1").slots["roundness"]
    assert [knob_slot_value(spec, value) for value in ("28", "16", "40", 32, 32.0)] == [28, 16, 40, 32, 32]
    assert [knob_slot_value(spec, value) for value in ("15", "41", "28px", "", 28.5, True, None)] == [None] * 7
    assert [knob_number(value) for value in ("-3", "007", 1e400)] == [-3, 7, None]


def test_a_choice_slot_takes_one_of_its_choices_from_a_value_or_a_knob():
    spec = contract_of("hud@1").slots["scan"]
    assert [slot_value(spec, value) for value in ("on", "off", "On", True, 1)] == ["on", "off", None, None, None]
    assert knob_slot_value(spec, "off") == "off" and knob_slot_value(spec, "paper") is None


# ---------------------------------------------------------------------------
# The look, painted as the web paints it
# ---------------------------------------------------------------------------


def test_paint_vars_keeps_the_named_variables_in_any_form_and_never_a_numeric_rules():
    table = {**TABLE, "knobs": {**TABLE["knobs"], "ground": {
        **GROUNDS, "paper": {"--hello-bg": "220 13% 9%", "--hello-line": "rgba(255,255,255,.1)", "--hello-glow": "#FFFFFF"},
    }}}
    painted = paint_vars(table, None, ["--hello-bg", "--hello-line", "--hello-glow", "--hello-accent"])
    assert painted == {"--hello-bg": "220 13% 9%", "--hello-line": "rgba(255,255,255,.1)", "--hello-accent": "#2F6FED"}
    assert paint_vars(table, NIGHT, ["--hello-bg"]) == {"--hello-bg": "#030507"}
    assert painted_look(table, {"accent": "red"}) == build.HELLO_DEFAULTS  # a look that doesn't fit: the defaults
    assert paint_vars({"knobs": "x"}, None, ["--hello-bg"]) == {} and painted_look(None, NIGHT) is None


def test_slots_follow_the_look_and_a_look_that_cant_fill_one_paints_the_defaults_whole():
    contract = contract_of("hud@1")
    night = resolve_slots(contract, _block(LOOKS), TABLE, NIGHT)
    assert night.unresolved == () and night.slots == {
        "bg": "#030507", "grid": 14, "scan": "on", "accent": "#1E7F3C", "accentInk": "#0B1A07", "glow": 0, "auto": "on",
    }
    # --hello-sky is painted for ink only, so a paper look can't fill bg: the defaults whole, which can't either.
    sky = _block({**LOOKS, "bg": {"look": "--hello-sky"}})
    assert resolve_slots(contract, sky, TABLE, NIGHT).slots["bg"] == "#0B2340"
    assert resolve_slots(contract, sky, TABLE, {"accent": "green", "ground": "paper", "radius": 3}).unresolved == ("bg",)
    scan_from_ground = resolve_slots(contract, _block({**LOOKS, "scan": {"look": "ground"}}), TABLE, NIGHT)
    assert scan_from_ground.unresolved == ("scan",), "ground's choices aren't scan's, so neither the look nor the defaults fill it"


def test_slots_left_out_take_the_contracts_defaults_and_the_tone_follows_the_rule():
    contract = contract_of("candy@1")
    resolved = resolve_slots(contract, _block(dict(contract.example), "candy@1"), None, None)
    assert resolved.unresolved == () and layout_tone(contract, resolved.slots) == "light"
    night = {**contract.example, "page": "#150B1F", "text": "#FBEFFA"}
    required = {name: value for name, value in night.items() if "default" not in contract.slots[name]}
    dark = resolve_slots(contract, _block(required, "candy@1"), None, None)
    assert dark.slots["roundness"] == 28 and dark.slots["gradient"] == "rich"
    assert layout_tone(contract, dark.slots) == "dark"
    assert layout_tone(contract_of("hud@1"), {"bg": "#FFFFFF"}) == "dark"  # HUD is always dark


# ---------------------------------------------------------------------------
# The reader's half: a slot from the look against the package's table
# ---------------------------------------------------------------------------


def test_a_package_whose_table_fills_every_look_slot_is_read():
    assert _package_errors(LOOKS) == []
    assert _package_errors(HUD, table=None) == []


def _cannot(errors: List[Mapping[str, str]]) -> List[tuple]:
    return [(error["code"], error["path"]) for error in errors]


def test_slots_from_the_look_with_no_table_are_refused_once():
    errors = _package_errors(LOOKS, table=None)
    assert _cannot(errors) == [("MOBILE_CANNOT_PAINT", "boot/prepaint.json")]
    assert '"bg", "accent", "accentInk", "glow"' in errors[0]["message"]


@pytest.mark.parametrize("slot, ref, says", [
    ("bg", "--hello-paper", "which boot/prepaint.json never writes"),
    ("bg", "--hello-glow", "which boot/prepaint.json never writes"),  # a numeric rule paints it from a number
    ("grid", "corners", "which is no knob of boot/prepaint.json"),
])
def test_a_look_slot_naming_what_the_table_lacks_is_refused_naming_the_slot(slot, ref, says):
    errors = _package_errors({**LOOKS, slot: {"look": ref}})
    assert _cannot(errors) == [("MOBILE_CANNOT_PAINT", f"mobile.slots.{slot}")]
    assert says in errors[0]["message"] and ref in errors[0]["message"]


@pytest.mark.parametrize("slots, table, slot", [
    ({**LOOKS, "bg": {"look": "--hello-sky"}}, TABLE, "bg"),
    (LOOKS, {**TABLE, "knobs": {**TABLE["knobs"], "ground": {"paper": {"--hello-bg": "12px"}, "ink": GROUNDS["ink"]}}}, "bg"),
    ({**LOOKS, "scan": {"look": "ground"}}, TABLE, "scan"),
], ids=["only-ink-writes-it", "a-length", "ground-is-no-scan"])
def test_defaults_that_cant_fill_a_look_slot_are_refused_naming_the_slot(slots, table, slot):
    errors = _package_errors(slots, table)
    assert _cannot(errors) == [("MOBILE_CANNOT_PAINT", f"mobile.slots.{slot}")]
    assert "default look" in errors[0]["message"] and "hud@1" in errors[0]["message"]
