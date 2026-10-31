"""PRD-261 — the reader refuses what the browser would leave out of a package's first paint and looks.

``services/themes/first_paint.py`` holds a package's ``boot/prepaint.json`` to what the browser keeps of it:

* a knob or pair cell's value the first paint's gate drops (``css_value.prepaint_drop_errors``, the gate
  boot-sanitize.ts runs): a gradient, ``color-mix()``, an ``_``; ``BAD_CSS_VALUE``, naming the variable, the
  look it paints for and why. The import grammar alone accepts all three, and so does the boot read
  (``prepaint_errors``), so a package imported before this check still boots and paints what the browser keeps;
* a ``looks/schema.json`` knob the kept table never paints: the browser's look spec (looks/spec.ts
  ``lookSpecOf``) has a choice knob for each table under ``knobs`` and a number knob for each numeric rule's
  knob only, so every look with any other knob fails, the presets included (``KNOB_NOT_PAINTED``);
* a knob the kept table paints that the schema lacks: the look spec needs it in every look and no look may
  carry it (``KNOB_NOT_IN_SCHEMA``);
* defaults that don't fit what the gate keeps: the browser drops the table whole (``PREPAINT_DEFAULTS_UNFIT``);
* a knob table, pair table or numeric rule whose knob name the gate refuses: dropped whole (``BAD_PREPAINT``).

The shape is a theme whose first paint lost its gradients and whose looks all failed on a hosted install,
while every check passed: its cards, band and aura were gradients in the table, and its roundness a number
knob no numeric rule painted. The corpus (``cases.json`` ``prepaintDrops`` and ``packages``) holds both
languages and the reader to the same verdicts; this file holds the rules' edges.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes import contract, css_value, mobile_tokens  # noqa: E402
from services.themes.first_paint import (  # noqa: E402
    KNOB_NOT_IN_SCHEMA, KNOB_NOT_PAINTED, PREPAINT_DEFAULTS_UNFIT, first_paint_errors, first_paint_table_errors, spec_knobs,
    unknown_knob_errors, unpainted_knob_errors,
)
from services.themes.look_schema import validate_look_schema  # noqa: E402
from services.themes.package import ThemePackageError, read_theme_package  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

PREPAINT, SCHEMA = "boot/prepaint.json", "looks/schema.json"
CARD_HI = "radial-gradient(circle at 88% 8%,rgba(255,255,255,.34),transparent 46%)"
BAND = "linear-gradient(120deg,#FFC6E0 0%,#F2C9FF 52%,#FFD9C2 100%)"
GLOW = {
    "var": "--gloss-glow", "knob": "glow", "min": 0, "max": 10, "integer": True, "scale": 100, "capTable": None,
    "color": {"knob": "palette", "rgb": {"blush": "255,120,180", "night": "180,140,255"}}, "template": "rgba-alpha",
}
# Two palettes, two gradients and a glow, as a glossy theme paints them: a highlight per gradient, a band per cell.
TABLE: Dict[str, Any] = {
    "knobs": {
        "palette": {"blush": {"--gloss-bg": "#FFF4F8"}, "night": {"--gloss-bg": "#1A1024"}},
        "gradient": {"soft": {"--gloss-card-hi": CARD_HI}, "vivid": {"--gloss-card-hi": CARD_HI, "--gloss-edge": "#FFFFFF"}},
    },
    "pairs": [{"a": "palette", "b": "gradient", "table": {
        "blush": {"soft": {"--gloss-band": BAND}, "vivid": {"--gloss-band": BAND}},
        "night": {"soft": {"--gloss-band": "#2A1838"}, "vivid": {"--gloss-band": BAND}},
    }}],
    "numeric": [GLOW],
    "defaults": {"palette": "blush", "gradient": "soft", "glow": 6},
}
SCHEMA_DOC: Dict[str, Any] = {
    "schema": contract.LOOK_SCHEMA_SCHEMA,
    "look": {"schema": "gloss-look/v1"},
    "knobs": {
        "palette": {"kind": "choice", "label": "Palette", "choices": ["blush", "night"], "default": "blush"},
        "gradient": {"kind": "choice", "label": "Gradient", "choices": ["soft", "vivid"], "default": "soft"},
        "roundness": {"kind": "number", "label": "Roundness", "min": 16, "max": 40, "integer": True, "default": 28},
        "glow": {"kind": "number", "label": "Glow", "min": 0, "max": 10, "integer": True, "default": 6},
    },
}


def _schema(document: Dict[str, Any] = SCHEMA_DOC):
    schema, errors = validate_look_schema(document)
    assert errors == [] and schema is not None
    return schema


def _files(table: Any) -> Dict[str, bytes]:
    return {PREPAINT: json.dumps(table).encode("utf-8")}


def _codes(errors: List[Dict[str, str]]) -> List[tuple]:
    return [(error["code"], error["path"]) for error in errors]


def test_every_gradient_cell_is_refused_with_its_variable_its_look_and_why():
    errors = css_value.prepaint_drop_errors(TABLE)
    assert _codes(errors) == [
        ("BAD_CSS_VALUE", f"{PREPAINT}#knobs.gradient.soft.--gloss-card-hi"),
        ("BAD_CSS_VALUE", f"{PREPAINT}#knobs.gradient.vivid.--gloss-card-hi"),
        ("BAD_CSS_VALUE", f"{PREPAINT}#pairs[0].table.blush.soft.--gloss-band"),
        ("BAD_CSS_VALUE", f"{PREPAINT}#pairs[0].table.blush.vivid.--gloss-band"),
        ("BAD_CSS_VALUE", f"{PREPAINT}#pairs[0].table.night.vivid.--gloss-band"),
    ]
    assert errors[0]["message"] == (
        f'{PREPAINT}#knobs.gradient.soft.--gloss-card-hi: --gloss-card-hi for gradient "soft" is a gradient, '
        "radial-gradient(): the browser's first paint drops it, and every value built on it."
    )
    assert '--gloss-band for palette "blush" and gradient "soft" is a gradient, linear-gradient()' in errors[2]["message"]


def test_the_import_grammar_and_the_boot_read_still_take_the_table():
    # Only the reader refuses it: serve.theme_prepaint re-checks with prepaint_errors, so an installed copy still boots.
    assert css_value.prepaint_errors(TABLE) == []


def test_the_phone_paint_drops_what_the_first_paint_drops():
    # mobile_tokens paints a look through css_value's gate, the one Python copy of boot-sanitize.ts isCssValue.
    kept = mobile_tokens.boot_table(TABLE)
    assert kept is not None
    assert kept.knobs["gradient"]["vivid"] == {"--gloss-edge": "#FFFFFF"}
    assert kept.pairs[0].table["night"] == {"soft": {"--gloss-band": "#2A1838"}, "vivid": {}}
    assert mobile_tokens.is_boot_value is css_value.is_boot_value


def test_the_gate_keeps_what_boot_sanitize_keeps_and_says_why_it_drops_the_rest():
    for kept in ("#FF6A1F", "rgba(244,238,232,.12)", "hsl(12 5% 7% / 0.14)", "var(--theme-font-display),sans-serif",
                 "calc(4px + 1rem)", "hue-rotate(18deg) saturate(1.2)"):
        assert css_value.boot_value_problem(kept) is None, kept
    assert css_value.boot_value_problem("conic-gradient(red, blue)") == "a gradient, conic-gradient()"
    assert css_value.boot_value_problem("opacity(0.5)") == "a call to opacity(), a function the first paint never runs"
    assert css_value.boot_value_problem("x" * (contract.PREPAINT_VALUE_MAX + 1)) == "over 200 characters"
    assert css_value.boot_value_problem("x" * contract.PREPAINT_VALUE_MAX) is None
    assert css_value.boot_value_problem("a(b") == "text whose brackets don't pair"
    assert css_value.boot_value_problem(7) == "not text"
    assert css_value.CSS_VALUE_MAX == contract.PREPAINT_VALUE_MAX


def test_a_number_knob_no_numeric_rule_paints_is_refused():
    errors = unpainted_knob_errors(_schema(), TABLE)
    assert _codes(errors) == [(KNOB_NOT_PAINTED, f"{SCHEMA}#knobs.roundness")]
    assert "its look spec leaves \"roundness\" out and every look fails, the presets included" in errors[0]["message"]


def test_a_knob_painted_only_by_a_numeric_rule_the_browser_drops_is_refused():
    dropped = {**GLOW, "integer": 1}  # sanitizeNumericRule keeps a rule whose integer is true or false only
    table = {**TABLE, "numeric": [dropped]}
    schema = _schema({**SCHEMA_DOC, "knobs": {k: v for k, v in SCHEMA_DOC["knobs"].items() if k != "roundness"}})
    assert _codes(unpainted_knob_errors(schema, table)) == [(KNOB_NOT_PAINTED, f"{SCHEMA}#knobs.glow")]


def test_a_table_the_browser_drops_whole_has_no_look_spec_to_hold_the_schema_to():
    unfit = {**TABLE, "defaults": {"palette": "dusk", "gradient": "soft", "glow": 6}}
    assert mobile_tokens.boot_table(unfit) is None
    assert unpainted_knob_errors(_schema(), unfit) == []


def test_first_paint_errors_need_a_table_the_grammar_takes():
    assert first_paint_errors(_schema(), {}) == []
    assert first_paint_errors(_schema(), {PREPAINT: b"{ not json"}) == []
    refused = {**TABLE, "--gloss-art": "url(https://example.com/x.png)"}
    assert css_value.prepaint_errors(refused) != [] and first_paint_errors(_schema(), _files(refused)) == []
    codes = [error["code"] for error in first_paint_errors(_schema(), _files(TABLE))]
    assert codes == ["BAD_CSS_VALUE"] * 5 + [KNOB_NOT_PAINTED]
    assert [error["code"] for error in first_paint_errors(None, _files(TABLE))] == ["BAD_CSS_VALUE"] * 5


def test_the_reader_refuses_a_hello_package_whose_table_leaves_radius_unpainted():
    table = {**build.hello_prepaint({"ground": {"paper": {"--hello-bg": "#f4efe6"}, "ink": {"--hello-bg": "#101418"}}}),
             "numeric": []}
    try:
        read_theme_package(build.build_package(replace={PREPAINT: json.dumps(table).encode("utf-8")}))
    except ThemePackageError as exc:
        assert _codes(exc.errors) == [(KNOB_NOT_PAINTED, f"{SCHEMA}#knobs.radius")]
    else:
        raise AssertionError("the reader accepted a table that never paints radius")


def test_the_reader_takes_a_hello_package_whose_table_paints_every_knob():
    table = build.hello_prepaint({"ground": {"paper": {"--hello-bg": "#f4efe6"}, "ink": {"--hello-bg": "#101418"}}})
    package = read_theme_package(build.build_package(replace={PREPAINT: json.dumps(table).encode("utf-8")}))
    assert PREPAINT in package.files


def test_a_knob_the_table_paints_that_the_schema_lacks_is_refused_where_the_table_paints_it():
    shade = {"light": {"--gloss-shade": "#FFFFFF"}}
    halo = {**GLOW, "var": "--gloss-halo", "knob": "halo"}
    table = {**TABLE, "knobs": {**TABLE["knobs"], "shade": shade}, "numeric": [GLOW, halo],
             "defaults": {**TABLE["defaults"], "shade": "light", "halo": 3}}
    assert spec_knobs(mobile_tokens.boot_table(table)) == ["palette", "gradient", "shade", "glow", "halo"]
    errors = unknown_knob_errors(_schema(), table)
    assert _codes(errors) == [(KNOB_NOT_IN_SCHEMA, f"{PREPAINT}#knobs.shade"), (KNOB_NOT_IN_SCHEMA, f"{PREPAINT}#numeric[1].knob")]
    assert 'its look spec needs "shade" in every look and no look may carry it' in errors[0]["message"]


def test_defaults_that_do_not_fit_what_the_gate_keeps_refuse_the_table_whole():
    spaced = {**GLOW, "color": {"knob": "palette", "rgb": {"blush": "255, 120, 180", "night": "180, 140, 255"}}}
    for table in ({**TABLE, "defaults": {**TABLE["defaults"], "palette": "dusk"}}, {**TABLE, "numeric": [spaced]}):
        assert css_value.prepaint_errors(table) == [] and mobile_tokens.boot_table(table) is None
        codes = _codes(first_paint_table_errors(_schema(), table))
        assert codes[-1] == (PREPAINT_DEFAULTS_UNFIT, f"{PREPAINT}#defaults")
        assert KNOB_NOT_PAINTED not in [code for code, _ in codes]


def test_a_knob_name_the_gate_refuses_drops_its_table_pair_or_rule():
    table = {
        **TABLE,
        "knobs": {**TABLE["knobs"], "gloss-tint": {"warm": {"--gloss-tint": "#F4E0C0"}}},
        "pairs": [{**TABLE["pairs"][0], "b": "2nd"}],
        "numeric": [{**GLOW, "capTable": ["palette", "x-y"], "caps": {}}],
    }
    assert _codes(css_value.prepaint_drop_errors(table)) == [
        ("BAD_CSS_VALUE", f"{PREPAINT}#knobs.gradient.soft.--gloss-card-hi"),
        ("BAD_CSS_VALUE", f"{PREPAINT}#knobs.gradient.vivid.--gloss-card-hi"),
        ("BAD_PREPAINT", f"{PREPAINT}#knobs.gloss-tint"),
        ("BAD_PREPAINT", f"{PREPAINT}#pairs[0].b"),
        ("BAD_PREPAINT", f"{PREPAINT}#numeric[0].capTable[1]"),
    ]
    assert all(css_value.is_knob_name(name) for name in ("palette", "glow", "my_knob", "Ab9"))
    assert not any(css_value.is_knob_name(name) for name in ("gloss-tint", "2nd", "x" * 33, None, 7))
    assert mobile_tokens.is_knob_name is css_value.is_knob_name


def test_the_reader_refuses_a_hello_table_with_defaults_it_cannot_fit_or_a_knob_the_schema_lacks():
    ground = {"paper": {"--hello-bg": "#f4efe6"}, "ink": {"--hello-bg": "#101418"}}
    unfit = build.hello_prepaint({"ground": ground}, {"ground": "dusk"})
    extra = build.hello_prepaint({"ground": ground, "shade": {"light": {"--hello-shade": "#ffffff"}}}, {"shade": "light"})
    for table, expected in ((unfit, (PREPAINT_DEFAULTS_UNFIT, f"{PREPAINT}#defaults")), (extra, (KNOB_NOT_IN_SCHEMA, f"{PREPAINT}#knobs.shade"))):
        try:
            read_theme_package(build.build_package(replace={PREPAINT: json.dumps(table).encode("utf-8")}))
        except ThemePackageError as exc:
            assert _codes(exc.errors) == [expected]
        else:
            raise AssertionError(f"the reader accepted {expected[0]}")
