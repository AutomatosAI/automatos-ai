"""PRD-261 B-03e1 — the shared review corpus (``tests/fixtures/themes/review-cases.json``), core side.

One corpus both reviews run, so the kit's ``automatos-theme check`` and this host's review stage
(``services/themes/review_stage.py``) cannot drift apart: the kit vendors the file byte for byte
and runs ``check`` over every case (B-03e2); this test runs ``run_review_stage`` over every case.

A case is the hello package (``fixtures/themes/hello``) changed as ``build.py`` changes it
(``manifestSet``/``manifestUnset`` on theme.json, ``replace`` setting a file's text, null dropping
it), with the level it expects from each check both sides make: ``sizes``, ``cssScope``,
``looks``, ``uses`` and ``clonedFrom``. The LLM is a fake answering risk 0. The SDK names are this
host's own (``data/sdk-surface.json``), and a fresh clone's sha is written ``{"$corePageSha": path}``
and read from ``data/core-pages.json``, so no edit of a core page changes the corpus.

``COVERED`` names, for each test of ``test_prd261_review_stage.py`` that proves a path of those
five checks, the cases that repeat it; ``OUTSIDE`` names every other test of that file and why it
has no case. A test added there fails here until it is placed in one of them.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.themes import sdk_uses  # noqa: E402
from services.themes.contract import core_page_hashes  # noqa: E402
from services.themes.package import read_theme_package  # noqa: E402
from services.themes.review_stage import LEVELS, READER_CHECKS, run_review_stage  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402

CORPUS_FILE = build.HERE / "review-cases.json"
CORPUS_TEXT = CORPUS_FILE.read_text(encoding="utf-8")
CORPUS = json.loads(CORPUS_TEXT)
CASES = CORPUS["cases"]
CORPUS_SCHEMA = "automatos-theme-review-corpus/v1"
CHECKS = (*READER_CHECKS, "uses", "clonedFrom")
CASE_FIELDS = ("name", "manifestSet", "manifestUnset", "replace", "host", "expected")
CORE_PAGE_SHA = "$corePageSha"
HOST_SURFACE, SURFACE_MISSING = "sdkSurface", "missing"
HOST_CONDITIONS = {HOST_SURFACE: (SURFACE_MISSING,)}
SHA256_TEXT = re.compile(r"[0-9a-f]{64}")
CANONICAL = "json.dumps(corpus, indent=2) + newline"
REVIEW_STAGE_TESTS = Path(__file__).with_name("test_prd261_review_stage.py")

COVERED = {
    "test_a_clean_package_passes_with_every_check_reported": ("hello declaring the theme area",),
    "test_an_import_the_sdk_does_not_export_is_an_error_at_uses": ("an import the SDK doesn't export",),
    "test_an_area_used_but_not_declared_is_a_warning_at_uses": ("an area the code uses that uses doesn't declare",),
    "test_a_minified_import_is_read_too": ("a minified import",),
    "test_a_page_area_may_be_written_with_a_dash_or_a_dot_and_a_non_area_is_a_warning": (
        "a page area written with a dot", "a uses entry that is no SDK area",
    ),
    "test_a_declared_area_the_code_never_uses_is_reported_not_warned": ("a declared area the code never uses",),
    "test_a_namespace_import_is_a_warning_and_its_names_are_found_in_the_text": ("a namespace import",),
    "test_an_sdk_storage_call_declared_in_uses_passes_on_this_hosts_sdk": ("SDK storage calls declared in uses",),
    "test_an_sdk_storage_call_without_storage_in_uses_is_a_warning_listing_it": (
        "SDK storage calls without storage in uses",
    ),
    "test_an_sdk_voice_call_declared_in_uses_passes_on_this_hosts_sdk": ("SDK voice calls declared in uses",),
    "test_an_sdk_voice_call_without_voice_in_uses_is_a_warning_listing_it": ("SDK voice calls without voice in uses",),
    "test_without_the_sdk_name_list_the_uses_check_is_a_warning": ("a host with no SDK name list",),
    "test_a_clone_of_a_core_page_this_host_changed_is_a_warning_at_cloned_from": (
        "a clone of a core page this host changed",
    ),
    "test_a_clone_of_the_current_core_page_passes": ("a clone of the current core page",),
}
VERDICT, CONTRAST, LLM, VIEW = "the verdict rule", "contrast (M-03)", "the LLM stage", "the uploader's view"
OUTSIDE = {
    "test_the_verdict_is_blocked_on_an_error_review_required_on_a_warning_and_never_not_run": VERDICT,
    "test_contrast_is_listed_per_preset_as_not_measured_and_never_moves_the_verdict": CONTRAST,
    "test_a_mobile_pair_under_aa_makes_contrast_an_error_naming_it_and_blocks": CONTRAST,
    "test_a_mobile_block_whose_measured_pairs_meet_aa_passes_contrast": CONTRAST,
    "test_from_looks_a_preset_under_aa_blocks_and_is_named": CONTRAST,
    "test_from_looks_defaults_under_aa_block_while_every_preset_passes": CONTRAST,
    "test_a_high_llm_risk_blocks_and_a_middle_one_asks_for_review": LLM,
    "test_with_no_llm_key_the_verdict_is_review_required_which_a_reviewer_may_approve": LLM,
    "test_a_submission_reads_dist_and_source_and_a_private_upload_reads_dist": LLM,
    "test_a_token_budget_too_small_for_the_stage_makes_no_call": LLM,
    "test_the_member_view_keeps_the_verdict_and_plain_checks_never_the_llm_check": VIEW,
}


def _fake_llm(model: str, text: str) -> SimpleNamespace:
    """The Messages API's answer of risk 0 (``test_prd261_review_stage.py`` ``_reply``)."""
    answer = json.dumps({"risk_score": 0, "findings": [], "summary": "risk 0"})
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=answer)],
                           usage=SimpleNamespace(input_tokens=10, output_tokens=2))


def _resolved(value: Any) -> Any:
    """``value`` with each ``{"$corePageSha": path}`` replaced by this release's sha256 of that core file."""
    if isinstance(value, dict) and set(value) == {CORE_PAGE_SHA}:
        return core_page_hashes()[value[CORE_PAGE_SHA]]
    if isinstance(value, dict):
        return {key: _resolved(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolved(item) for item in value]
    return value


def _package(case: Mapping[str, Any]) -> bytes:
    """The hello package with the case's changes (``test_prd261_theme_package.py`` ``_corpus_package``)."""
    texts = _resolved(case.get("replace", {}))
    replace: Dict[str, Any] = {path: None if text is None else text.encode("utf-8") for path, text in texts.items()}
    if "manifestSet" in case or "manifestUnset" in case:
        manifest = build.changed_manifest(_resolved(case.get("manifestSet")), case.get("manifestUnset", ()))
        replace = {build.MANIFEST: manifest, **replace}
    return build.build_package(replace=replace)


def _levels(case: Mapping[str, Any], monkeypatch: pytest.MonkeyPatch) -> Dict[str, str]:
    """The level each corpus check gives the case's package, on the host the case names."""
    if case.get("host", {}).get(HOST_SURFACE) == SURFACE_MISSING:
        monkeypatch.setattr(sdk_uses, "sdk_surface", lambda: None)
    data = _package(case)
    checks = run_review_stage(read_theme_package(data), len(data), llm_call=_fake_llm)["checks"]
    return {name: checks[name]["level"] for name in CORPUS["checks"]}


@pytest.mark.parametrize("case", CASES, ids=[case["name"] for case in CASES])
def test_every_case_gives_its_expected_level_at_each_check(case, monkeypatch):
    assert _levels(case, monkeypatch) == case["expected"]


# ---------------------------------------------------------------------------
# The file describes itself, in one stable form
# ---------------------------------------------------------------------------


def test_the_corpus_names_its_schema_package_checks_and_levels():
    assert (CORPUS["schema"], CORPUS["package"]) == (CORPUS_SCHEMA, build.HELLO_DIR.name)
    assert tuple(CORPUS["checks"]) == CHECKS and tuple(CORPUS["levels"]) == LEVELS
    assert CORPUS["_comment"].strip() and CASES


def test_every_case_has_a_unique_name_known_fields_and_each_checks_level_in_order():
    names = [case["name"] for case in CASES]
    assert len(names) == len(set(names)) and all(name.strip() for name in names)
    for case in CASES:
        assert set(case) <= set(CASE_FIELDS), case["name"]
        assert list(case["expected"]) == CORPUS["checks"], case["name"]
        assert set(case["expected"].values()) <= set(LEVELS), case["name"]
        for condition, value in case.get("host", {}).items():
            assert value in HOST_CONDITIONS[condition], case["name"]


def test_the_corpus_is_in_its_canonical_form():
    # The kit records the vendored file's sha256: whitespace or key order alone must never move it.
    assert CORPUS_TEXT == json.dumps(CORPUS, indent=2) + "\n", f"review-cases.json must be {CANONICAL}"


def test_the_corpus_holds_no_core_page_hash():
    # A literal release hash would turn a fresh clone stale with the next edit of that page.
    assert not set(SHA256_TEXT.findall(CORPUS_TEXT)) & set(core_page_hashes().values())


def test_uses_and_cloned_from_each_reach_every_level_they_can_give():
    reached = {check: {case["expected"][check] for case in CASES} for check in ("uses", "clonedFrom")}
    assert reached == {"uses": {"pass", "warning", "error"}, "clonedFrom": {"pass", "warning"}}


def test_every_review_stage_test_is_repeated_here_or_named_as_outside_the_corpus():
    tree = ast.parse(REVIEW_STAGE_TESTS.read_text(encoding="utf-8"))
    tests = {node.name for node in tree.body if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")}
    assert tests == set(COVERED) | set(OUTSIDE) and not set(COVERED) & set(OUTSIDE)
    names = {case["name"] for case in CASES}
    assert {name for cases in COVERED.values() for name in cases} <= names
