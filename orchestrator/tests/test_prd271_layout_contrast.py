"""PRD-271 C04 — the review measures the pairs of the layout a ``mobile`` block names (``layout_contrast.py``).

Packages are the hello fixture read by ``read_theme_package`` and reviewed by ``run_review_stage`` (the LLM a
fake answering risk 0), as in ``test_prd261_review_stage.py``. A block whose slots are values is measured
once; one that takes slots from the look is measured for the table's defaults and then each preset, each
resolved as the phone's answer resolves it. A layout pair under AA is an error that blocks the listing,
named; the Studio skin is still measured beside it. The shared review corpus (``review-cases.json``) repeats
the verdicts for the kit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes.contract import LAYOUT_MIN_HOST  # noqa: E402
from services.themes.layout_contrast import measure_layout  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from services.themes.review_stage import member_view, run_review_stage  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

ENGINES = {"automatos": f"^{LAYOUT_MIN_HOST}", "sdk": "^1.0.0"}
SURFACE = {"theme": ["defineTheme"]}
SKIN = {"tokens": {"light": {"--background": "0 0% 100%", "--foreground": "#14181c"}}}
HUD = {"bg": "#030507", "accent": "#3FE0FF", "accentInk": "#021116"}
# hello's three look knobs painted: the green accent's ink reads at 3.57:1 on it, under AA for text.
TABLE = {
    "knobs": {
        "accent": {"blue": {"--hello-accent": "#2F6FED", "--hello-on-accent": "#FFFFFF"},
                   "green": {"--hello-accent": "#1E7F3C", "--hello-on-accent": "#0B1A07"}},
        "ground": {"paper": {"--hello-bg": "#14181C"}, "ink": {"--hello-bg": "#030507"}},
    },
    "pairs": [], "numeric": [build.HELLO_RADIUS_RULE], "defaults": dict(build.HELLO_DEFAULTS),
}
LOOKS = {"bg": {"look": "--hello-bg"}, "accent": {"look": "--hello-accent"}, "accentInk": {"look": "--hello-on-accent"},
         "glow": {"look": "radius"}}
NIGHT_INK = 'Preset "night", mobile.slots (hud@1): accentInk on accent is 3.57:1, under the AA minimum of 4.5:1 for text.'


def _reply(model, text):
    answer = json.dumps({"risk_score": 0, "findings": [], "summary": "risk 0"})
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=answer)], usage=SimpleNamespace(input_tokens=10, output_tokens=2))


def _review(slots, skin=SKIN, table=None):
    fields = {"uses": ["theme"], "engines": ENGINES, "mobile": {**skin, "layout": "hud@1", "slots": slots}}
    files = {build.MANIFEST: build.changed_manifest(fields)}
    files = {**files, "boot/prepaint.json": json.dumps(table).encode("utf-8")} if table is not None else files
    data = build.build_package(replace=files)
    return run_review_stage(read_theme_package(data), len(data), llm_call=_reply, surface=SURFACE)


def test_values_that_meet_aa_pass_measured_once_beside_the_skin():
    report = _review(HUD)
    contrast = report["checks"]["contrast"]
    assert (report["verdict"], contrast["level"], contrast["status"]) == ("pass", "pass", "pass")
    layout = contrast["layout"]
    assert (layout["contract"], [(subject["id"], subject["kind"]) for subject in layout["subjects"]]) == (
        "hud@1", [("mobile.slots", "slots")],
    )
    pairs = [(pair["foreground"], pair["background"], pair["kind"], pair["status"]) for pair in layout["subjects"][0]["pairs"]]
    assert pairs == [("text", "bg", "text", "pass"), ("text2", "bg", "text", "pass"), ("accentInk", "accent", "text", "pass"),
                     ("accent", "bg", "ui", "pass")]
    assert [scheme["id"] for scheme in contrast["schemes"]] == ["light"]  # the skin is still measured
    assert "hud@1" in contrast["reason"]


def test_a_layout_pair_under_aa_is_an_error_naming_it_that_blocks_the_listing():
    report = _review({**HUD, "accentInk": "#2AA8C0"})
    contrast = report["checks"]["contrast"]
    assert (report["verdict"], contrast["level"], contrast["status"]) == ("blocked", "error", "fail")
    assert contrast["problems"] == [
        "mobile.slots (hud@1): accentInk on accent is 1.79:1, under the AA minimum of 4.5:1 for text.",
    ]
    assert member_view(report)["checks"]["contrast"] == contrast  # the uploader sees why


def test_a_fixed_colour_of_the_layout_is_measured_against_the_theme_s_slots():
    # HUD's text is fixed (#E6F1F5): a light page under it fails, whatever the theme meant.
    contrast = _review({**HUD, "bg": "#F4F4F5"})["checks"]["contrast"]
    assert contrast["status"] == "fail"
    assert contrast["problems"][0].startswith("mobile.slots (hud@1): text on bg is 1.")


def test_slots_from_the_look_are_measured_for_the_defaults_and_each_preset():
    report = _review(LOOKS, table=TABLE)
    contrast = report["checks"]["contrast"]
    subjects = [(subject["id"], subject["kind"], subject["status"]) for subject in contrast["layout"]["subjects"]]
    assert subjects == [
        ("boot/prepaint.json#defaults", "defaults", "pass"), ("default", "preset", "pass"), ("night", "preset", "fail"),
    ]
    assert (report["verdict"], contrast["problems"]) == ("blocked", [NIGHT_INK])


def test_a_preset_that_cant_fill_a_slot_is_measured_as_the_defaults_the_phone_draws_and_said_so():
    # green's ink is a length here: members on the night preset get the whole default look on the phone.
    accents = {**TABLE["knobs"]["accent"], "green": {"--hello-accent": "#1E7F3C", "--hello-on-accent": "12px"}}
    contrast = _review(LOOKS, table={**TABLE, "knobs": {**TABLE["knobs"], "accent": accents}})["checks"]["contrast"]
    subjects = [(subject["id"], subject["status"], subject["fromDefaults"]) for subject in contrast["layout"]["subjects"]]
    assert subjects == [("boot/prepaint.json#defaults", "pass", False), ("default", "pass", False), ("night", "pass", True)]
    assert contrast["status"] == "pass" and "1 preset(s) can't fill every slot of hud@1" in contrast["reason"]


def test_a_skin_under_aa_still_blocks_when_the_layout_passes():
    skin = {"tokens": {"light": {"--background": "0 0% 100%", "--foreground": "0 0% 58%"}}}
    contrast = _review(HUD, skin=skin)["checks"]["contrast"]
    assert (contrast["status"], contrast["layout"]["status"]) == ("fail", "pass")
    assert contrast["problems"] == [
        "mobile.tokens.light: --foreground on --background is 3.03:1, under the AA minimum of 4.5:1 for text.",
    ]


def test_a_block_with_no_layout_is_measured_as_before():
    assert measure_layout(SKIN, {}) is None
    contrast = _review(HUD)["checks"]["contrast"]
    assert "layout" in contrast
    fields = {"uses": ["theme"], "engines": ENGINES, "mobile": SKIN}
    data = build.build_package(replace={build.MANIFEST: build.changed_manifest(fields)})
    plain = run_review_stage(read_theme_package(data), len(data), llm_call=_reply, surface=SURFACE)["checks"]["contrast"]
    assert "layout" not in plain and plain["status"] == "pass"
