"""PRD-251B Wave 1, US-B104 — ``blocks.durations``: the lengths a video template offers.

Pinned:

* the contract: a video's ``durations`` is a strictly ascending list of whole seconds
  (1..600, at most 8); an image declares none; a template without the list is still
  valid (it offers its root duration alone, ``template_gallery.durations_of``);
* the starter loader carries ``durations`` into the seeded row (the RVW-7 rule: no key
  of the contract is dropped unread);
* every seeded video starter declares a valid list that holds its root length; no image
  starter declares any. The 15 s and 30 s cuts of UI story promo and App promo, and the
  ``cuts`` a shorter length needs, are pinned in test_prd251bw1_cuts.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.social_templates import (  # noqa: E402
    BLOCK_KEYS,
    SOCIAL_IMAGE,
    SOCIAL_VIDEO,
    SocialTemplateError,
    root_duration,
    validate_social_blocks,
)
from modules.documents import social_starters  # noqa: E402
from modules.socials.template_gallery import durations_of  # noqa: E402

ROOT_HTML = (
    '<!doctype html><html><head></head><body><div id="root" data-composition-id="main" data-width="1080" '
    'data-height="1920" data-duration="39.5"><h1>{{ headline }}</h1></div></body></html>'
)
IMAGE_HTML = ROOT_HTML.replace(' data-duration="39.5"', "").replace('data-height="1920"', 'data-height="1080"')
VIDEO = {"html": ROOT_HTML, "css": "", "variables_schema": {"headline": {"type": "text"}}, "sizes": ["1080x1920"]}
IMAGE = {"html": IMAGE_HTML, "css": "", "variables_schema": {"headline": {"type": "text"}}, "sizes": ["1080x1080"]}


def _errors(blocks, fmt):
    with pytest.raises(SocialTemplateError) as caught:
        validate_social_blocks(blocks, fmt)
    return {e["field"]: e["message"] for e in caught.value.errors}


def test_durations_is_part_of_the_contract_and_optional():
    assert "durations" in BLOCK_KEYS
    assert "durations" not in validate_social_blocks(VIDEO, SOCIAL_VIDEO)
    cuts = {"15": [[0, 15]], "30": [[0, 30]]}  # shorter than the 39.5 s timeline: each needs its cut
    checked = validate_social_blocks({**VIDEO, "durations": [15, 30, 40], "cuts": cuts}, SOCIAL_VIDEO)
    assert checked["durations"] == [15, 30, 40] and checked["cuts"] == cuts
    assert durations_of(VIDEO, SOCIAL_VIDEO) == [40] and durations_of(checked, SOCIAL_VIDEO) == [15, 30, 40]


@pytest.mark.parametrize("bad, field", [
    ([30, 15], "durations"), ([15, 15], "durations"), ([], "durations"), ("15", "durations"),
    ([0], "durations[0]"), ([True], "durations[0]"), ([15.5], "durations[0]"), ([601], "durations[0]"),
    (list(range(1, 10)), "durations"),
])
def test_a_video_refuses_a_malformed_list(bad, field):
    assert field in _errors({**VIDEO, "durations": bad}, SOCIAL_VIDEO)


def test_an_image_declares_no_durations():
    assert "durations" in _errors({**IMAGE, "durations": [15]}, SOCIAL_IMAGE)
    assert durations_of({**IMAGE, "durations": [15]}, SOCIAL_IMAGE) == []


def test_the_starter_loader_carries_durations_into_the_row(tmp_path, monkeypatch):
    (tmp_path / "cut.html").write_text(ROOT_HTML, encoding="utf-8")
    (tmp_path / "cut.json").write_text(json.dumps({
        "name": "Cut", "description": "", "format": SOCIAL_VIDEO, "category": "social",
        "variables_schema": {"headline": {"type": "text"}}, "sizes": ["1080x1920"], "durations": [15, 30],
        "cuts": {"15": [[0, 15]], "30": [[0, 30]]}, "sample_data": {"headline": "Hello"},
    }), encoding="utf-8")
    monkeypatch.setattr(social_starters, "STARTERS_DIR", tmp_path)
    social_starters._starters.cache_clear()
    try:
        starter = social_starters._load("cut")
    finally:
        social_starters._starters.cache_clear()
    assert starter["blocks"]["durations"] == [15, 30] and starter["blocks"]["cuts"] == {"15": [[0, 15]], "30": [[0, 30]]}


def test_every_seeded_video_starter_declares_its_lengths_and_no_image_does():
    for starter in social_starters.social_starters():
        blocks = starter["blocks"]
        if starter["format"] == SOCIAL_VIDEO:
            declared = blocks.get("durations")
            assert isinstance(declared, list) and declared, starter["slug"]
            assert declared == sorted(set(declared)) and all(isinstance(d, int) and d > 0 for d in declared), starter["slug"]
            root = root_duration(blocks["html"])
            assert int(round(root)) in declared or int(-(-root // 1)) in declared, (starter["slug"], root, declared)
        else:
            assert "durations" not in blocks, starter["slug"]
    social_starters._starters.cache_clear()
