"""PRD-251B (3 Oct 2026 pass) — photo cards for any business, not only Automatos's own.

Six image starters, each with a photo slot behind its words: Photo + headline, Offer,
Review, Highlights, Before / after (two photos) and Just the photo. Pins:

* they are seeded after the eight image families, at the feed post, the story, the
  square and the link card;
* each passes the template contract and the brand rule and shows the brand's mark, and
  every photo slot is an image slot a toolkit may fill, so the editor's AI-made, Upload
  and Library fill it and the gallery marks the template Photo (``image_slots``);
* their sample data fills every field the render needs, with the brand-neutral handle;
* an empty photo slot is taken out and the brand's backdrop shows; a filled one keeps
  its element and reaches the render as media;
* boot seeding reaches every workspace that has Socials on, and only those; one that
  fails is rolled back and the others still get theirs.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.media_render_bundle import build_bundle  # noqa: E402
from core.social_brand_rule import brand_literals  # noqa: E402
from core.social_templates import IMAGE_SLOT, SOCIAL_IMAGE, resolve_variables, slot_generatable, validate_social_blocks  # noqa: E402
from modules.documents import seed_templates  # noqa: E402
from modules.documents.social_starters import SOCIAL_IMAGE_STARTER_SLUGS, SOCIAL_PHOTO_STARTER_SLUGS, SOCIAL_STARTER_SLUGS, social_starters  # noqa: E402
from modules.socials import template_gallery  # noqa: E402
from modules.socials.settings import parse_workspace_socials  # noqa: E402

PHOTO_NAMES = {
    "photo-headline": "Photo + headline",
    "photo-offer": "Offer",
    "photo-review": "Review",
    "photo-highlights": "Highlights",
    "before-after": "Before / after",
    "photo-only": "Just the photo",
}
# The Instagram feed post, a story or reel, the square, and LinkedIn's and X's link card.
PHOTO_SIZES = ["1080x1350", "1080x1920", "1080x1080", "1200x628"]
KIT = {"name": "Corner Salon", "primary_color": "#c2410c", "secondary_color": "#1c1917"}
PHOTO_URL = "https://storage.example-ci.test/social-media/ws/post/upload-1.png"


def _photos():
    return [s for s in social_starters(SOCIAL_IMAGE) if s["slug"] in SOCIAL_PHOTO_STARTER_SLUGS]


def _values(starter):
    resolved = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"])
    return resolved.values


def test_the_photo_cards_are_seeded_after_the_eight_image_families():
    assert list(SOCIAL_PHOTO_STARTER_SLUGS) == list(PHOTO_NAMES)
    assert list(SOCIAL_STARTER_SLUGS[-len(SOCIAL_PHOTO_STARTER_SLUGS):]) == list(SOCIAL_PHOTO_STARTER_SLUGS)
    assert not set(SOCIAL_PHOTO_STARTER_SLUGS) & set(SOCIAL_IMAGE_STARTER_SLUGS)
    photos = _photos()
    assert [s["name"] for s in photos] == list(PHOTO_NAMES.values())
    for starter in photos:
        assert (starter["format"], starter["category"], starter["blocks"]["sizes"]) == (SOCIAL_IMAGE, "social", PHOTO_SIZES)
        assert starter["description"]


@pytest.mark.parametrize("slug", list(PHOTO_NAMES))
def test_each_photo_card_passes_the_contract_and_the_brand_rule_and_shows_the_mark(slug):
    starter = next(s for s in _photos() if s["slug"] == slug)
    blocks = starter["blocks"]
    assert validate_social_blocks(blocks, SOCIAL_IMAGE) == blocks
    assert brand_literals(blocks["html"], blocks.get("css") or "") == []
    # PRD-255 FR-9: a photo card's words sit on the brand's ink, a dark stage: it shows the
    # kit's logo for dark backgrounds (else the mark on a light chip), never the wordmark.
    assert "{{ brand.logo_on_dark }}" in blocks["html"] and "{{ brand.logo }}" not in blocks["html"]


@pytest.mark.parametrize("slug", list(PHOTO_NAMES))
def test_every_photo_slot_is_an_image_the_editor_fills(slug):
    blocks = next(s for s in _photos() if s["slug"] == slug)["blocks"]
    slots = blocks["slots"]
    assert slots and all(spec["kind"] == IMAGE_SLOT and slot_generatable(spec) for spec in slots.values())
    assert all(spec["label"] and spec["description"] for spec in slots.values())
    # The gallery's image slots: what AI-made, Upload and Library fill, and what marks the card Photo.
    assert sorted(template_gallery.image_slots(blocks)) == sorted(slots)
    assert template_gallery.image_slot_labels(blocks) == {name: slots[name]["label"] for name in template_gallery.image_slots(blocks)}


@pytest.mark.parametrize("slug", list(PHOTO_NAMES))
def test_the_sample_data_fills_every_field_the_render_needs(slug):
    starter = next(s for s in _photos() if s["slug"] == slug)
    schema = starter["blocks"]["variables_schema"]
    resolved = resolve_variables(schema, starter["sample_data"])
    assert resolved.missing == [] and resolved.invalid == []
    assert starter["sample_data"]["handle"] == "@yourbrand"
    assert all(spec.get("label") and spec.get("description") for spec in schema.values())


@pytest.mark.parametrize("slug", list(PHOTO_NAMES))
def test_an_empty_photo_slot_shows_the_backdrop_and_a_filled_one_the_photo(slug):
    starter = next(s for s in _photos() if s["slug"] == slug)
    blocks, slots = starter["blocks"], starter["blocks"]["slots"]

    empty = build_bundle(workspace_id="ws", reference=slug, blocks=blocks, values=_values(starter), brand_kit=KIT,
                         size="1080x1350", fmt=SOCIAL_IMAGE)
    html = empty["composition"]["html"]
    assert "data-slot=" not in html and 'class="backdrop"' in html and "media" not in empty

    filled = build_bundle(workspace_id="ws", reference=slug, blocks=blocks, values=_values(starter), brand_kit=KIT,
                          size="1080x1350", fmt=SOCIAL_IMAGE, slot_media={name: PHOTO_URL for name in slots})
    for name, spec in slots.items():
        assert f'data-slot="{name}"' in filled["composition"]["html"]
        assert {"path": spec["path"], "url": PHOTO_URL} in filled["media"]


# ---------------------------------------------------------------------------
# Boot seeding: every workspace that has Socials on
# ---------------------------------------------------------------------------


class _Workspaces:
    """The workspaces table as the boot seeding reads it: ids and settings."""

    def __init__(self, rows):
        self.rows, self.rollbacks = rows, 0

    def query(self, *_columns):
        return self

    def all(self):
        return self.rows

    def rollback(self):
        self.rollbacks += 1


def test_boot_seeding_reaches_every_workspace_with_socials_on_and_only_those(monkeypatch):
    on, off, odd, broken = (uuid.uuid4() for _ in range(4))
    db = _Workspaces([
        SimpleNamespace(id=on, settings={"socials": {"enabled": True}}),
        SimpleNamespace(id=off, settings={"socials": {"enabled": False}}),
        SimpleNamespace(id=odd, settings={"socials": {"enabled": "true"}}),  # a stray string is never on
        SimpleNamespace(id=broken, settings={"socials": {"enabled": True}}),
    ])
    seeded = []

    def seed(_db, workspace_id, *, commit=True):
        if workspace_id == broken:
            raise RuntimeError("the row lock timed out")
        seeded.append(workspace_id)
        return {"created": 6, "refreshed": 0}

    monkeypatch.setattr(seed_templates, "seed_social_starters", seed)

    totals = seed_templates.seed_social_starters_where_on(db, lambda settings: parse_workspace_socials(settings).enabled)

    assert seeded == [on]
    assert totals == {"workspaces": 2, "created": 6, "refreshed": 0}
    assert db.rollbacks == 1
