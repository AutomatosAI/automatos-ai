"""F377 (night 11, 7 Oct) — the Data story is a numbers story any business can tell.

Night 11: a coffee roaster's Data story showed a trading screen (LONG / SHORT / STOP, odds,
a trade log, candles drawn from 60 OHLC bars), asked for 59 fields, drew empty photo slots
as pale-blue boxes and overlapped its own text ("Lantern Kitchen" over "96 kg").

Pinned (``modules/documents/templates/social/data-story.{html,json}``):

* **The story.** An opener (a label and a headline), one big number with what it counts, a
  ranked list of three to five items whose bars the script draws from their values, a closing
  line and the end card with the brand. Nothing of a trading screen, nor of Automatos.
* **Few, plain fields.** At most 20, each labelled in an owner's words with a limit; the
  values are numbers; only the story's own lines are required (11), and a default is never
  copy (an empty text or 0).
* **15, 30 and 40 s.** 40 s as authored (it was 40 s before, so a post that chose 40 s still
  renders); the 30 s cut keeps every scene, and the 15 s cut keeps the headline, the list and
  the end card: it needs 8 fields, and a 15 s post that fills only those renders.
* **Nothing overlaps, nothing pale.** No element opts out of the layout check's overlap
  rule, every page is a fitted flex column, and an empty photo leaves nothing behind.
* **The voice.** Eleven lines at 40 s (nine at 30 s, four at 15 s), each from the fields on
  screen as it is spoken.
"""
from __future__ import annotations

import json
import os
import re
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.social_cuts import cut_to_length, shown_variables  # noqa: E402
from core.social_templates import SOCIAL_VIDEO, placeholders, resolve_variables, validate_social_blocks  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import render  # noqa: E402

SEED = _ORCH / "modules" / "documents" / "templates" / "social" / "data-story.json"
STORY_FIELDS = [
    "headline", "stat_value", "stat_label", "ranking_title",
    "item_1_name", "item_1_value", "item_2_name", "item_2_value", "item_3_name", "item_3_value", "closing_line",
]
AT_15 = ["headline", "ranking_title", "item_1_name", "item_1_value", "item_2_name", "item_2_value", "item_3_name", "item_3_value"]
# What the trading reference printed, and the platform's own words: none of it is the post's.
NOT_THIS_STORY = re.compile(
    r"\b(?:long|short|stop|odds|ledger|candles?|trad(?:e|es|ing)|market|replay|signals?|automatos|agents?|"
    r"command cent(?:re|er)|deliverables?)\b",
    re.IGNORECASE,
)
MAX_FIELDS = 20


def _starter():
    starter = next(s for s in social_starters.social_starters() if s["slug"] == "data-story")
    social_starters._starters.cache_clear()
    return starter


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _visible_text(html):
    """The page's own words: its markup without the scripts, styles, tags and placeholders."""
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S | re.I)
    return re.sub(r"\{\{[^{}]*\}\}", " ", re.sub(r"<[^<>]*>", " ", body))


def _post(values, length):
    variables = {name: {"value": value} for name, value in values.items()}
    return SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=variables, length_seconds=length,
                           footage=None, music=None, voice=None, targets=[])


def test_it_tells_a_business_numbers_story_and_nothing_of_a_trading_screen():
    starter = _starter()
    blocks = starter["blocks"]
    assert validate_social_blocks(blocks, SOCIAL_VIDEO) == blocks
    seed = json.loads(SEED.read_text(encoding="utf-8"))
    found = [text for text in _strings({k: v for k, v in seed.items() if k != "audio_plan"}) if NOT_THIS_STORY.search(text)]
    assert found == []
    assert not NOT_THIS_STORY.search(_visible_text(blocks["html"]))
    for name in ("ranking_title", "item_1_name", "item_5_value", "stat_value", "closing_line"):
        assert "{{ " + name + " }}" in blocks["html"], name
    # The bars are the post's own values: each row carries its value, and the script divides by the largest.
    assert all(f'data-value="{{{{ item_{n}_value }}}}"' in blocks["html"] for n in range(1, 6))
    assert "values[i] / largest" in blocks["html"]


def test_its_fields_are_few_plain_and_only_the_story_is_required():
    schema = _starter()["blocks"]["variables_schema"]
    assert len(schema) <= MAX_FIELDS
    assert [name for name, spec in schema.items() if "default" not in spec] == STORY_FIELDS
    for name, spec in schema.items():
        # An owner's words: "Headline", "Top item 1: name"; never "Odds: outcome 1's share".
        assert re.fullmatch(r"(?:Top item \d: )?[^:]+", spec.get("label") or ""), name
        if spec["type"] == "text":
            assert 0 < spec["max_chars"] <= 60, name
        if "default" in spec:
            assert spec["default"] in ("", 0), (name, spec["default"])  # a default is never copy
    assert {schema[f"item_{n}_value"]["type"] for n in range(1, 6)} == {"number"}


def test_it_offers_15_30_and_40_seconds_and_the_15_second_cut_needs_its_own_fields_only():
    starter = _starter()
    blocks = starter["blocks"]
    assert blocks["durations"] == [15, 30, 40] and list(blocks["cuts"]) == ["15", "30"]
    at_15 = resolve_variables(cut_to_length(blocks, 15)["variables_schema"], {}).missing
    assert at_15 == AT_15
    assert {"stat_value", "stat_label", "closing_line"}.isdisjoint(shown_variables(blocks, 15))
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_VIDEO, blocks=blocks)
    filled = {name: starter["sample_data"][name] for name in AT_15}
    bundle = render.bundle_for(_post(filled, 15), template, {})
    assert [line["id"] for line in bundle["audio"]["voice"]["lines"]] == ["l01", "l04", "l05", "l11"]
    page = bundle["composition"]["html"]
    assert all(name in bundle["variables"] for name in placeholders(page) if not name.startswith(("brand.", "size.")))
    at_30 = render.bundle_for(_post(starter["sample_data"], 30), template, {})
    assert [line["id"] for line in at_30["audio"]["voice"]["lines"]] == ["l01", "l02", "l03", "l04", "l05", "l06", "l07", "l10", "l11"]
    assert resolve_variables(cut_to_length(blocks, 30)["variables_schema"], {}).missing == STORY_FIELDS
    whole = render.bundle_for(_post(starter["sample_data"], 40), template, {})
    assert len(whole["audio"]["voice"]["lines"]) == 11


def test_nothing_opts_out_of_the_overlap_check_and_an_empty_photo_draws_nothing():
    starter = _starter()
    html = starter["blocks"]["html"]
    assert "data-layout-allow-overlap" not in html
    assert html.count('class="page') == 5 and "function fit(page)" in html
    css = "".join(re.findall(r"<style>(.*?)</style>", html, re.S))
    for selector in (".photo", ".photo .push", ".photo img"):
        body = re.search(re.escape(selector) + r" \{([^{}]*)\}", css).group(1)
        assert "background" not in body, selector  # nothing pale where a photo was not given
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_VIDEO, blocks=starter["blocks"])
    page = render.bundle_for(_post(starter["sample_data"], 40), template, {})["composition"]["html"]
    assert "data-slot" not in page and "assets/slots/" not in page


def test_each_voice_line_speaks_the_fields_on_screen_as_it_is_spoken():
    blocks = _starter()["blocks"]
    lines = blocks["audio_plan"]["voice"]["lines"]
    assert [line["id"] for line in lines] == [f"l{n:02d}" for n in range(1, 12)]
    spoken = {name for line in lines for name in placeholders(line["text"])}
    assert spoken <= set(blocks["variables_schema"]) | {"brand.name"}
    assert {"headline", "stat_value", "ranking_title", "item_1_name", "item_3_value", "item_5_name", "closing_line"} <= spoken
