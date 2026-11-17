"""A template's fields are shown in the order the image shows them (16 Nov 2026).

The owner's screenshot of Photo highlights: "Handle, Label, Headline, Card 1 text, Card 2
text, Card 3 text, Card 1 title, Card 2 title, Card 3 title, Accent words". ``blocks`` is
JSONB, which stores an object's keys by length and then by name, so the author's order
("card 1 title, card 1 text, …") never reached the form. Pinned:

* ``ordered_schema``: each field at its first ``{{ name }}`` in the composition, then the
  ones the HTML never shows, in stored order; a brand or size placeholder is no field;
* the editor's gallery entry and the composer's template entry both carry that order, so a
  card's title sits next to its text for the person and for Auto.
"""
from __future__ import annotations

import os
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

import api.socials_compose as compose_api  # noqa: E402
from core.social_field_order import ordered_schema  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import template_gallery  # noqa: E402

HTML = (
    "<header>{{ brand.name }} {{ handle }}</header><p>{{ eyebrow }}</p><h1>{{ headline }}</h1>"
    "<h2>{{ card_1_title }}</h2><p>{{ card_1_text }}</p><h2>{{ card_2_title }}</h2><p>{{ card_2_text }}</p>"
    "<a>{{ cta }}</a><footer>{{ headline }} {{ size.width }}</footer>"
)
AUTHORED = ["eyebrow", "headline", "card_1_title", "card_1_text", "card_2_title", "card_2_text", "cta", "handle", "voice_line"]
ON_THE_IMAGE = ["handle", "eyebrow", "headline", "card_1_title", "card_1_text", "card_2_title", "card_2_text", "cta", "voice_line"]


def _as_jsonb_stores_it(names):
    """An object as it comes back from a JSONB column: keys by length, then by name."""
    return {name: {"type": "text", "label": name} for name in sorted(names, key=lambda n: (len(n), n))}


def _blocks():
    return {"html": HTML, "variables_schema": _as_jsonb_stores_it(AUTHORED), "sizes": ["1080x1350"]}


def test_jsonb_puts_every_cards_text_before_any_cards_title():
    # The defect, as the column stores it: this is the order the form showed.
    assert list(_blocks()["variables_schema"]) == [
        "cta", "handle", "eyebrow", "headline", "voice_line", "card_1_text", "card_2_text", "card_1_title", "card_2_title",
    ]


def test_the_fields_come_in_the_order_the_image_shows_them():
    blocks = _blocks()

    ordered = ordered_schema(blocks)

    # A field the HTML never shows (a spoken line) comes last; a brand or size placeholder is no field.
    assert list(ordered) == ON_THE_IMAGE
    assert ordered == blocks["variables_schema"]  # the same fields and specs, only their order changed


def test_blocks_without_fields_or_without_a_composition_are_no_surprise():
    assert ordered_schema(None) == {}
    assert ordered_schema({"html": HTML}) == {}
    assert ordered_schema({"html": HTML, "variables_schema": ["not", "a", "schema"]}) == {}
    stored = _as_jsonb_stores_it(["headline", "cta"])
    assert list(ordered_schema({"variables_schema": stored})) == ["cta", "headline"]  # nothing to order by: as stored


def test_the_editors_gallery_and_the_composer_both_carry_that_order():
    row = SimpleNamespace(
        id=uuid.uuid4(), name="Photo highlights", description=None, format="social_image", blocks=_blocks(),
        thumbnail_url=None, created_by="system", updated_at=None, sample_data={},
    )

    assert list(template_gallery.entry(row, None)["variables_schema"]) == ON_THE_IMAGE
    assert list(compose_api.template_entry(row)["variables_schema"]) == ON_THE_IMAGE


def test_every_starters_card_title_comes_before_its_text_once_stored():
    # Each starter, stored and read back: no numbered text or body field is shown before its own title.
    for starter in social_starters():
        blocks = {**starter["blocks"], "variables_schema": _as_jsonb_stores_it(starter["blocks"]["variables_schema"])}
        order = list(ordered_schema(blocks))
        for name in order:
            for suffix in ("_text", "_body"):
                title = name[: -len(suffix)] + "_title"
                if name.endswith(suffix) and title in order:
                    assert order.index(title) < order.index(name), (starter["slug"], title, name)
