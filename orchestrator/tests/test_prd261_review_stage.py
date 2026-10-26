"""PRD-261 C-02a — the automated review stage (``services/themes/review_stage.py``, ``sdk_uses.py``).

Packages are the hello fixture, read by ``read_theme_package`` and then changed one way each
(seeded bad packages), so each one fails at its own check. The LLM is a fake answering as the
Messages API would; the SDK's names are a small test surface, so these tests don't depend on the
generated ``data/sdk-surface.json`` (``test_prd261_sdk_surface_drift.py`` holds that to frontend/),
except E3d's storage tests and theme voice's voice tests, which review on this host's own list: they
prove the SDK's storage and voice areas are ones this host's review accepts. M-03's cases give hello a ``mobile`` block, whose colour pairs the
contrast check measures (an error under AA blocks the verdict).
The routes that run the stage are covered in ``test_prd261_theme_import.py`` (private upload) and
by C-01a's submit tests.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config  # noqa: E402
from services.themes import sdk_uses  # noqa: E402
from services.themes.contract import core_page_hashes  # noqa: E402
from services.themes.llm_stage import DID_NOT_RUN  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from services.themes.review_stage import REPORT_SCHEMA, member_view, run_review_stage, verdict_of  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

SURFACE = {
    "theme": ["defineTheme"],
    "chat": ["useChatController", "useStartChat"],
    "pages/agents": ["useAgents"],
}
CHAT_PAGE = "app/chat/page.tsx"
HELLO_ENTRY = (build.HELLO_DIR / "dist" / "entry.js").read_text(encoding="utf-8")
SDK_IMPORT = "import { defineTheme } from '@automatos/theme-sdk'"


def _reply(score: int):
    text = json.dumps({"risk_score": score, "findings": [], "summary": f"risk {score}"})
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=2))


def _llm(score: int = 0):
    return lambda model, text: _reply(score)


def _checked(entry: str = HELLO_ENTRY, **manifest):
    fields = {"uses": ["theme"], **manifest}
    replace = {build.MANIFEST: build.changed_manifest(fields), build.ENTRY: entry.encode("utf-8")}
    data = build.build_package(replace=replace)
    return read_theme_package(data), len(data)


def _review(entry: str = HELLO_ENTRY, score: int = 0, **manifest):
    checked, size = _checked(entry, **manifest)
    return run_review_stage(checked, size, llm_call=_llm(score), surface=SURFACE)


def _entry_importing(names: str) -> str:
    return HELLO_ENTRY.replace(SDK_IMPORT, f"import {{ {names} }} from '@automatos/theme-sdk'")


# ---------------------------------------------------------------------------
# The report and the verdict
# ---------------------------------------------------------------------------


def test_a_clean_package_passes_with_every_check_reported():
    report = _review()
    checks = report["checks"]

    assert (report["schema"], report["stage"], report["verdict"]) == (REPORT_SCHEMA, "C-02a", "pass")
    assert set(checks) == {"sizes", "cssScope", "looks", "contrast", "uses", "audioCapture", "clonedFrom", "llm"}
    assert {name: check["level"] for name, check in checks.items()} == {
        "sizes": "pass", "cssScope": "pass", "looks": "pass", "contrast": "not_measured",
        "uses": "pass", "audioCapture": "pass", "clonedFrom": "pass", "llm": "pass",
    }
    assert report["summary"] == {"pass": 7, "warning": 0, "error": 0, "not_measured": 1}
    assert checks["uses"]["usedAreas"] == ["theme"] and checks["llm"]["calls"] >= 1


def test_the_verdict_is_blocked_on_an_error_review_required_on_a_warning_and_never_not_run():
    assert verdict_of({"a": {"level": "pass"}, "b": {"level": "error"}, "c": {"level": "warning"}}) == "blocked"
    assert verdict_of({"a": {"level": "pass"}, "b": {"level": "warning"}}) == "review_required"
    assert verdict_of({"a": {"level": "pass"}, "b": {"level": "not_measured"}}) == "pass"


def test_contrast_is_listed_per_preset_as_not_measured_and_never_moves_the_verdict():
    contrast = _review()["checks"]["contrast"]
    assert contrast["level"] == "not_measured" and contrast["reason"]
    assert [preset["id"] for preset in contrast["presets"]] == ["default", "night"]


# M-03 (PB-35): a theme with a mobile block has its ten colour pairs measured (test_prd261_mobile_contrast.py
# holds the measuring itself), for the table's defaults and then each preset. hello's two presets painted for
# the phone: default (paper, blue) reads, night (ink, green) puts white text on a green at 3.41:1.
MOBILE_ENGINES = {"automatos": "^1.1.0", "sdk": "^1.0.0"}
WHITE, INK_TEXT = "0 0% 100%", "#14181c"
# The table paints Studio tokens as triplets only (the web reads them as hsl(var(--x))): #14181c and #2e9e5b.
INK_TRIPLET, GREEN = "210 16.67% 9.41%", "144.11 54.9% 40%"
LOOK_TABLE = {
    "knobs": {
        "ground": {
            "paper": {"--background": WHITE, "--foreground": INK_TRIPLET},
            "ink": {"--background": "220 13% 9%", "--foreground": "210 20% 96%"},
        },
        "accent": {
            "blue": {"--primary": "221.21 83.19% 53.33%", "--primary-foreground": WHITE},
            "green": {"--primary": GREEN, "--primary-foreground": WHITE},
        },
    },
    "defaults": {"accent": "blue", "ground": "paper", "radius": 6},
}


def _review_mobile(mobile, files=None):
    """The review of hello with a ``mobile`` block (and any files added), its engines at the block's lowest host."""
    fields = {"uses": ["theme"], "engines": MOBILE_ENGINES, "mobile": mobile}
    data = build.build_package(replace={build.MANIFEST: build.changed_manifest(fields), **(files or {})})
    return run_review_stage(read_theme_package(data), len(data), llm_call=_llm(), surface=SURFACE)


def test_a_mobile_pair_under_aa_makes_contrast_an_error_naming_it_and_blocks():
    report = _review_mobile({"tokens": {"light": {"--background": WHITE, "--foreground": "0 0% 58%"}}})
    contrast = report["checks"]["contrast"]

    assert (report["verdict"], contrast["level"], contrast["status"]) == ("blocked", "error", "fail")
    assert contrast["problems"] == [
        "mobile.tokens.light: --foreground on --background is 3.03:1, under the AA minimum of 4.5:1 for text.",
    ]
    assert report["summary"]["error"] == 1 and report["checks"]["llm"]["level"] == "pass"
    assert member_view(report)["checks"]["contrast"] == contrast  # the uploader sees why it is blocked


def test_a_mobile_block_whose_measured_pairs_meet_aa_passes_contrast():
    report = _review_mobile({"tokens": {"light": {"--background": WHITE, "--foreground": INK_TEXT, "--ring": "0 0% 56.5%"}}})
    contrast = report["checks"]["contrast"]

    assert (report["verdict"], contrast["level"], contrast["source"]) == ("pass", "pass", "tokens")
    assert [scheme["id"] for scheme in contrast["schemes"]] == ["light"] and "presets" not in contrast
    assert report["summary"] == {"pass": 8, "warning": 0, "error": 0, "not_measured": 0}


def test_from_looks_a_preset_under_aa_blocks_and_is_named():
    report = _review_mobile({"fromLooks": True}, {"boot/prepaint.json": json.dumps(LOOK_TABLE).encode("utf-8")})
    contrast = report["checks"]["contrast"]

    assert (report["verdict"], contrast["level"], contrast["source"]) == ("blocked", "error", "fromLooks")
    assert [(preset["id"], preset["status"]) for preset in contrast["presets"]] == [
        ("boot/prepaint.json#defaults", "pass"), ("default", "pass"), ("night", "fail"),
    ]
    assert contrast["problems"] == [
        'Preset "night": --primary-foreground on --primary is 3.41:1, under the AA minimum of 4.5:1 for text.',
    ]


def test_from_looks_defaults_under_aa_block_while_every_preset_passes():
    # The answer serves the table's defaults to every member with no stored look, so the review measures them too.
    table = {**LOOK_TABLE, "defaults": {"accent": "green", "ground": "paper", "radius": 6}}
    night = {
        "schema": "automatos-e2e-hello-look/v1", "id": "night", "name": "Night", "sub": "",
        "accent": "blue", "ground": "ink", "radius": 0,
    }
    report = _review_mobile({"fromLooks": True}, {
        "boot/prepaint.json": json.dumps(table).encode("utf-8"),
        "looks/presets/night.look.json": json.dumps(night).encode("utf-8"),
    })
    contrast = report["checks"]["contrast"]

    assert (report["verdict"], contrast["level"]) == ("blocked", "error")
    assert [(preset["id"], preset["status"]) for preset in contrast["presets"]] == [
        ("boot/prepaint.json#defaults", "fail"), ("default", "pass"), ("night", "pass"),
    ]
    assert contrast["problems"] == [
        "boot/prepaint.json#defaults: --primary-foreground on --primary is 3.41:1, under the AA minimum of 4.5:1 for text.",
    ]


# ---------------------------------------------------------------------------
# Seeded bad packages, each failing at its own check
# ---------------------------------------------------------------------------


def test_an_import_the_sdk_does_not_export_is_an_error_at_uses():
    report = _review(_entry_importing("defineTheme, stealSession as s"))
    uses = report["checks"]["uses"]

    assert report["verdict"] == "blocked" and uses["level"] == "error"
    assert uses["unknownNames"] == ["stealSession"] and "can't load" in uses["problems"][0]
    assert report["checks"]["llm"]["level"] == "pass"


def test_an_area_used_but_not_declared_is_a_warning_at_uses():
    report = _review(_entry_importing("defineTheme, useChatController"))
    uses = report["checks"]["uses"]

    assert report["verdict"] == "review_required" and uses["level"] == "warning"
    assert uses["undeclaredAreas"] == ["chat"] and uses["usedNames"] == ["defineTheme", "useChatController"]


def test_a_minified_import_is_read_too():
    minified = HELLO_ENTRY.replace(SDK_IMPORT, 'import{defineTheme as e,useAgents as t}from"@automatos/theme-sdk"')
    uses = _review(minified, uses=["theme", "pages-agents"])["checks"]["uses"]
    assert uses["level"] == "pass" and uses["usedAreas"] == ["pages-agents", "theme"]


def test_a_page_area_may_be_written_with_a_dash_or_a_dot_and_a_non_area_is_a_warning():
    entry = _entry_importing("defineTheme, useAgents")
    assert _review(entry, uses=["theme", "pages.agents"])["checks"]["uses"]["level"] == "pass"
    uses = _review(entry, uses=["theme", "pages-agents", "telepathy"])["checks"]["uses"]
    assert uses["level"] == "warning" and uses["notAreas"] == ["telepathy"]


def test_a_declared_area_the_code_never_uses_is_reported_not_warned():
    uses = _review(uses=["theme", "chat"])["checks"]["uses"]
    assert uses["level"] == "pass" and uses["declaredUnused"] == ["chat"]


def test_a_namespace_import_is_a_warning_and_its_names_are_found_in_the_text():
    entry = HELLO_ENTRY.replace(SDK_IMPORT, "import * as sdk from '@automatos/theme-sdk'\nconst { defineTheme } = sdk")
    entry += "\nexport const chat = () => sdk.useStartChat()\n"
    uses = _review(entry, uses=["theme", "chat"])["checks"]["uses"]

    assert uses["level"] == "warning" and uses["wideImport"] is True
    assert uses["usedAreas"] == ["chat", "theme"] and uses["unknownNames"] == []


STORAGE_IMPORT = "defineTheme, readThemeState, writeThemeState"
STORAGE_AREA = "storage"


def _review_on_this_host(entry: str, **manifest):
    """The review with no test surface: the SDK names are this host's ``data/sdk-surface.json``."""
    checked, size = _checked(entry, **manifest)
    return run_review_stage(checked, size, llm_call=_llm())


def test_an_sdk_storage_call_declared_in_uses_passes_on_this_hosts_sdk():
    report = _review_on_this_host(_entry_importing(STORAGE_IMPORT), uses=["theme", STORAGE_AREA])
    uses = report["checks"]["uses"]

    assert (report["verdict"], uses["level"], uses["unknownNames"]) == ("pass", "pass", [])
    assert STORAGE_AREA in uses["usedAreas"] and uses["undeclaredAreas"] == []
    assert {"readThemeState", "writeThemeState"} <= set(uses["usedNames"])


def test_an_sdk_storage_call_without_storage_in_uses_is_a_warning_listing_it():
    report = _review_on_this_host(_entry_importing(STORAGE_IMPORT))
    uses = report["checks"]["uses"]

    assert (report["verdict"], uses["level"], uses["unknownNames"]) == ("review_required", "warning", [])
    assert uses["undeclaredAreas"] == [STORAGE_AREA]
    assert uses["problems"] == [sdk_uses.UNDECLARED.format(areas=STORAGE_AREA)]


VOICE_IMPORT = "defineTheme, VoiceButton, readVoiceLevels, useVoice"
VOICE_AREA = "voice"
VOICE_NAMES = {"VoiceButton", "readVoiceLevels", "useVoice"}


def test_an_sdk_voice_call_declared_in_uses_passes_on_this_hosts_sdk():
    # Theme voice (TV-C5): the SDK 1.2.0 voice area is one this host's review accepts.
    report = _review_on_this_host(_entry_importing(VOICE_IMPORT), uses=["theme", VOICE_AREA])
    uses = report["checks"]["uses"]

    assert (report["verdict"], uses["level"], uses["unknownNames"]) == ("pass", "pass", [])
    assert VOICE_AREA in uses["usedAreas"] and uses["undeclaredAreas"] == []
    assert VOICE_NAMES <= set(uses["usedNames"])


def test_an_sdk_voice_call_without_voice_in_uses_is_a_warning_listing_it():
    report = _review_on_this_host(_entry_importing(VOICE_IMPORT))
    uses = report["checks"]["uses"]

    assert (report["verdict"], uses["level"], uses["unknownNames"]) == ("review_required", "warning", [])
    assert uses["undeclaredAreas"] == [VOICE_AREA]
    assert uses["problems"] == [sdk_uses.UNDECLARED.format(areas=VOICE_AREA)]


def test_without_the_sdk_name_list_the_uses_check_is_a_warning(monkeypatch):
    monkeypatch.setattr(sdk_uses, "sdk_surface", lambda: None)
    checked, size = _checked()
    uses = run_review_stage(checked, size, llm_call=_llm())["checks"]["uses"]
    assert (uses["level"], uses["status"]) == ("warning", "not_checked")


def test_a_clone_of_a_core_page_this_host_changed_is_a_warning_at_cloned_from():
    stale = {"/chat": {"mode": "clone", "clonedFrom": {"path": CHAT_PAGE, "sha": "0" * 64}}}
    report = _review(pages=stale)
    cloned = report["checks"]["clonedFrom"]

    assert report["verdict"] == "review_required" and cloned["level"] == "warning"
    assert cloned["clones"] == ["/chat"] and cloned["stale"] == [
        {"route": "/chat", "path": CHAT_PAGE, "sha": "0" * 64, "releaseSha": core_page_hashes()[CHAT_PAGE]},
    ]


def test_a_clone_of_the_current_core_page_passes():
    fresh = {"/chat": {"mode": "clone", "clonedFrom": {"path": CHAT_PAGE, "sha": core_page_hashes()[CHAT_PAGE]}}}
    cloned = _review(pages=fresh)["checks"]["clonedFrom"]
    assert (cloned["level"], cloned["stale"], cloned["clones"]) == ("pass", [], ["/chat"])


def test_a_high_llm_risk_blocks_and_a_middle_one_asks_for_review():
    blocked = _review(score=85)
    assert blocked["verdict"] == "blocked" and blocked["checks"]["llm"]["level"] == "error"
    review = _review(score=40)
    assert review["verdict"] == "review_required" and review["checks"]["llm"]["riskScore"] == 40


# ---------------------------------------------------------------------------
# No LLM key: review_required, never safe, never not_run
# ---------------------------------------------------------------------------


def test_with_no_llm_key_the_verdict_is_review_required_which_a_reviewer_may_approve(monkeypatch):
    monkeypatch.setattr(config, "ANTHROPIC_API_KEY", "")
    checked, size = _checked()
    report = run_review_stage(checked, size, surface=SURFACE)
    llm = report["checks"]["llm"]

    assert report["verdict"] == "review_required"
    assert report["verdict"] not in ("pass", "not_run", "blocked")  # C-03a's approve refuses only blocked and not_run
    assert (llm["status"], llm["level"], llm["calls"]) == ("did_not_run", "warning", 0)
    assert llm["reason"].startswith(DID_NOT_RUN)


def test_a_submission_reads_dist_and_source_and_a_private_upload_reads_dist():
    replace = {
        build.MANIFEST: build.changed_manifest({"uses": ["theme"]}),
        "source/src/index.tsx": b"export default {}\n",
    }
    data = build.build_package(replace=replace)
    checked = read_theme_package(data)

    submitted = run_review_stage(checked, len(data), llm_call=_llm(), surface=SURFACE)["checks"]["llm"]
    private = run_review_stage(checked, len(data), include_source=False, llm_call=_llm(), surface=SURFACE)["checks"]["llm"]
    assert submitted["read"] == "dist+source" and submitted["files"][-1] == "source/src/index.tsx"
    assert "dist/entry.js" in submitted["files"]  # source/ never replaces dist/
    assert private["read"] == "dist" and "dist/entry.js" in private["files"]
    assert not any(path.startswith("source/") for path in private["files"])


def test_the_member_view_keeps_the_verdict_and_plain_checks_never_the_llm_check():
    report = _review(score=40)
    view = member_view(report)
    assert view["verdict"] == report["verdict"] == "review_required" and view["schema"] == REPORT_SCHEMA
    assert set(view["checks"]) == set(report["checks"]) - {"llm"} and "summary" not in view
    assert "llm" in report["checks"] and report["checks"]["llm"]["riskScore"] == 40  # the stored report keeps it
    assert member_view({"contrast": "ok"}) == {"contrast": "ok"}  # an older shape keeps its keys
    assert member_view(None) is None


def test_a_token_budget_too_small_for_the_stage_makes_no_call():
    checked, size = _checked()
    asked = []

    def call(model, text):
        asked.append(text)
        return _reply(0)

    report = run_review_stage(checked, size, llm_call=call, surface=SURFACE, token_budget=1)
    llm = report["checks"]["llm"]
    assert asked == [] and report["verdict"] == "review_required"
    assert llm["reason"] == "LLM stage did not run: daily budget reached." and llm["level"] == "warning"
