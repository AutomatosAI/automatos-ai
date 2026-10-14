"""PRD-259 S-001 — the skin format automatos-skin/v1, backend side.

Runs EVERY case of the shared corpus ``tests/fixtures/skins/cases.json``
(the frontend's ``lib/skins/__tests__/validator-parity.test.ts`` runs the
same file), so the two validators can't drift: same verdict, same codes.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from services.skins.color import format_triplet, normalise_color  # noqa: E402
from services.skins.contrast import contrast_ratio  # noqa: E402
from services.skins.schema import (  # noqa: E402
    ALLOWED_TOKENS,
    DERIVED_ALIASES,
    MASCOT_STATES,
    SKIN_ERROR_CODES,
    contrast_pairs_document,
)
from services.skins.validate import validate_skin  # noqa: E402

CORPUS_PATH = ORCH_ROOT / "tests" / "fixtures" / "skins" / "cases.json"
CORPUS = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
CASES = CORPUS["cases"]


def _codes(result: dict) -> list:
    return sorted({error["code"] for error in result.get("errors", [])})


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_every_corpus_case_gets_its_verdict_and_codes(case):
    result = validate_skin(copy.deepcopy(case["skin"]))
    assert result["ok"] is case["expect"]["ok"], result.get("errors")
    assert _codes(result) == case["expect"]["codes"]


def test_the_corpus_lists_are_this_module_s_lists():
    assert list(ALLOWED_TOKENS) == CORPUS["allowedTokens"]
    assert len(ALLOWED_TOKENS) == 47
    assert contrast_pairs_document() == CORPUS["contrastPairs"]


def test_the_corpus_is_big_enough_and_covers_every_code():
    assert len(CASES) >= 25
    covered = {code for case in CASES for code in case["expect"]["codes"]}
    assert covered == set(SKIN_ERROR_CODES)


def test_studio_s_own_tokens_pass_as_a_skin():
    studio = next(c for c in CASES if c["name"].startswith("Studio's own"))
    result = validate_skin(studio["skin"])
    assert result["ok"] is True
    assert set(result["skin"]["tokens"]["light"]) == set(ALLOWED_TOKENS)
    assert set(result["skin"]["tokens"]["dark"]) == set(ALLOWED_TOKENS)


def test_hex_normalises_to_the_triplet_form():
    assert normalise_color("#1a1814") == "40 13.04% 9.02%"
    assert normalise_color("#FFFFFF") == "0 0% 100%"
    assert normalise_color("38 50.0% 92%") == "38 50% 92%"
    assert normalise_color("hsl(38 50% 92%)") is None
    assert format_triplet(12.344, 50, 99.999) == "12.34 50% 100%"


def test_contrast_matches_the_measured_values():
    # Measured 14 Oct on globals.css: Studio's accent pair is 4.57:1, Classic's primary pair 3.31:1.
    assert round(contrast_ratio("38 50% 96%", "15 76% 44%"), 2) == 4.57
    assert round(contrast_ratio("0 0% 100%", "0 0% 0%"), 2) == 21.0


def test_a_valid_skin_comes_back_fresh_and_normalised():
    skin = {
        "schema": "automatos-skin/v1", "slug": "ink", "name": "Ink", "version": "1.0.0",
        "base": "classic", "tones": ["light"],
        "tokens": {"light": {"chart-1": "#c44a1a"}},
    }
    result = validate_skin(skin)
    assert result["ok"] is True
    assert result["skin"]["tokens"]["light"]["chart-1"] == normalise_color("#c44a1a")
    assert result["skin"] is not skin
    assert skin["tokens"]["light"]["chart-1"] == "#c44a1a"


def test_a_contrast_message_names_both_tokens_the_tone_and_the_ratio():
    case = next(c for c in CASES if c["name"].startswith("Classic's own primary"))
    errors = validate_skin(case["skin"])["errors"]
    message = next(e["message"] for e in errors if e["code"] == "CONTRAST")
    assert '"primary-foreground"' in message and '"primary"' in message
    assert "light" in message and "3.31:1" in message


def test_derived_aliases_and_mascot_states_are_the_prd_s():
    assert DERIVED_ALIASES == ("paper", "bg-1", "bg-2", "fg", "fg-2", "fg-3")
    assert len(MASCOT_STATES) == 12
