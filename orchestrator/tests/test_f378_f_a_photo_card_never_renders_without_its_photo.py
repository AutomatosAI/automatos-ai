"""F378 (night 11, 7 Oct): a photo card never renders without the photo it is made of.

B19: "Just the photo", saved with no photo, rendered a giant "Automatos A.I" wordmark
(6eff00bf); Before / after rendered two empty panels (B2/31). Pinned:

* a post's own render is refused while a photo the template marks required is empty,
  naming the photo spot to fill; each empty one of a before and after is named in turn;
* a filled photo (an upload, a Library pick or an AI-made one: the slots the render
  shows) renders;
* a preview and a template thumbnail still show the stand-in (no ``require_photos``);
* the post render asks for it (``api/socials.render_post``).
"""
from __future__ import annotations

import inspect
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

import api.socials as socials_api  # noqa: E402
from core.social_templates import empty_required_slots  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import render  # noqa: E402


def _starter(name):
    starter = next(s for s in social_starters() if s["name"] == name)
    template = SimpleNamespace(id=uuid.uuid4(), format=starter["format"], blocks=starter["blocks"], name=name)
    values = {key: {"value": value, "claim": False} for key, value in starter["sample_data"].items()}
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=values, length_seconds=None, targets=[])
    return post, template


def test_just_the_photo_without_its_photo_is_refused_naming_the_spot():
    post, template = _starter("Just the photo")
    with pytest.raises(render.NotRenderable, match=r"this template needs a photo: add one to 'Photo'"):
        render.bundle_for(post, template, {}, require_photos=True)


def test_each_empty_photo_of_a_before_and_after_is_named():
    post, template = _starter("Before / after")
    with pytest.raises(render.NotRenderable, match="'Before photo'"):
        render.bundle_for(post, template, {}, require_photos=True)
    with pytest.raises(render.NotRenderable, match="'After photo'"):
        render.bundle_for(post, template, {}, footage_slots=("before",), require_photos=True)
    assert empty_required_slots(template.blocks, ("before", "after")) == []


def test_a_filled_photo_renders():
    post, template = _starter("Just the photo")
    bundle = render.bundle_for(post, template, {}, footage_slots=("photo",), require_photos=True)
    assert 'data-slot="photo"' in bundle["composition"]["html"]


def test_a_preview_or_a_thumbnail_still_shows_the_stand_in():
    post, template = _starter("Just the photo")
    bundle = render.bundle_for(post, template, {})
    assert 'data-slot="photo"' not in bundle["composition"]["html"]


def test_an_optional_photo_never_stops_a_render():
    post, template = _starter("Offer")
    assert render.bundle_for(post, template, {}, require_photos=True)["composition"]["html"]


def test_the_post_render_requires_the_photos():
    assert "require_photos=True" in inspect.getsource(socials_api.render_post)
