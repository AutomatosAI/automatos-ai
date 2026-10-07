"""F378 (night 11, 7 Oct): the composer's picker sees what each template is for.

B19: "Just the photo" was picked for "I haven't got the photo yet", with no warning, and
rendered a giant wordmark. The picker had each template's name and fields only. Pinned:

* each template entry carries its description, its photo spots (label, required or not)
  and, when the template declares it, who it is made for (fix/n11-video's gate key);
* an archived template is never offered, as in the gallery;
* a brief saying there is no photo never gets a template whose photo is required, unless
  the owner chose it; the prompt says how to pick;
* a proposal whose template shows a photo says so: a required one is a warning and a
  question for the owner, an optional one a warning; a slot an AI tool fills needs neither;
* the seeded Just the photo and Before / after mark their photos required.
"""
from __future__ import annotations

import json
import os
import sys
import time
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

import sqlalchemy as sa  # noqa: E402

import api.socials_compose as compose_api  # noqa: E402
import tests.test_prd251w2_compose as compose_harness  # noqa: E402
from core.social_templates import validate_social_blocks  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import compose, compose_checks, compose_photos  # noqa: E402
from tests.test_prd251w2_compose import WS_A, _answer, _compose  # noqa: E402

# A linear read of a capped brief takes milliseconds; a polynomial one takes minutes.
ADVERSARIAL_SECONDS = 0.5

api = compose_harness.api
composer = compose_harness.composer

PHOTO_HTML = ('<!doctype html><html><head></head><body><div data-composition-id="main">'
              '<img data-slot="photo" src="assets/slots/photo.png" alt="" /><p>{{ headline }}</p></div></body></html>')
SCHEMA = {"headline": {"type": "text", "max_chars": 60}}


def _photo_blocks(required=True):
    slot = {"kind": "image", "path": "assets/slots/photo.png", "label": "Photo", "required": required}
    return {"html": PHOTO_HTML, "sizes": ["1080x1350"], "variables_schema": SCHEMA, "slots": {"photo": slot}}


def _insert(state, name, blocks, active=True, description=None):
    template_id = uuid.uuid4()
    state.api.session.execute(
        sa.text("INSERT INTO document_templates (id, workspace_id, name, description, format, data_schema, blocks, is_active) "
                "VALUES (:id, :ws, :name, :description, 'social_image', '{}', :blocks, :active)"),
        {"id": template_id.hex, "ws": WS_A.hex, "name": name, "description": description, "blocks": json.dumps(blocks),
         "active": active},
    )
    state.api.session.commit()
    return str(template_id)


def _offered(state):
    return {t["name"]: t for t in json.loads(state.model.asked[0][1]["content"])["templates"]}


def test_an_entry_carries_its_description_photo_spots_and_who_it_is_made_for():
    row = SimpleNamespace(id="t-1", name="Just the photo", description="The picture itself, edge to edge.",
                          format="social_image", blocks={**_photo_blocks(), "made_for": "software"})
    entry = compose_api.template_entry(row)
    assert entry["description"] == "The picture itself, edge to edge."
    assert entry["photo_slots"] == [{"slot": "photo", "label": "Photo", "required": True}]
    assert entry["made_for"] == "software"
    # F377: a template any business can use says so with None.
    assert compose_api.template_entry(SimpleNamespace(**{**vars(row), "blocks": _photo_blocks()}))["made_for"] is None


def test_the_slot_contract_takes_required_as_a_switch():
    assert validate_social_blocks(_photo_blocks(), "social_image")["slots"]["photo"]["required"] is True
    with pytest.raises(ValueError, match="slots.photo.required"):
        validate_social_blocks({**_photo_blocks(), "slots": {"photo": {**_photo_blocks()["slots"]["photo"], "required": "yes"}}},
                               "social_image")


@pytest.mark.parametrize("brief, none", [
    ("Harbour Blend is back. I haven't got the photo yet.", True),
    ("Harbour Blend is back, I don’t have a picture", True),
    ("No photo for this one, just words", True),
    ("Photo to come later this week", True),
    ("Here is the photo of Harbour Blend", False),
    ("Harbour Blend is back on Friday", False),
])
def test_a_brief_saying_there_is_no_photo_is_read_as_such(brief, none):
    assert compose_photos.brief_says_no_photo(brief) is none


@pytest.mark.parametrize("brief", [
    "haven't" + " " * 50_000 + "x",
    "no " * 20_000 + "x",
    "without \t" * 10_000 + "pictures later",
    "have " * 30_000,
])
def test_a_long_adversarial_brief_is_read_fast(brief):
    # CodeQL py/polynomial-redos: whitespace is collapsed and the brief capped before the pattern runs.
    started = time.perf_counter()
    compose_photos.brief_says_no_photo(brief)
    assert time.perf_counter() - started < ADVERSARIAL_SECONDS


def test_a_no_photo_brief_is_never_offered_a_template_whose_photo_is_required(composer):
    _insert(composer, "Just the photo", _photo_blocks(required=True))
    _insert(composer, "Offer", _photo_blocks(required=False))
    _compose(composer, [_answer(composer)], {"brief": "Harbour Blend is back. I haven't got the photo yet."})
    assert "Just the photo" not in _offered(composer) and "Offer" in _offered(composer)
    _compose(composer, [_answer(composer)], {"brief": "Harbour Blend is back, photo attached."})
    assert {"Just the photo", "Offer"} <= set(_offered(composer))


def test_the_owners_chosen_photo_template_is_kept_and_asked_for(composer):
    chosen = _insert(composer, "Just the photo", _photo_blocks(required=True))
    resp = _compose(composer, [_answer(composer, template_id=chosen, variables={"headline": "Back"})],
                    {"brief": "Harbour Blend is back. No photo yet.", "template_id": chosen})
    proposal = resp.json()
    assert proposal["template_id"] == chosen
    assert "A photo for 'Photo'" in proposal["questions"]


def test_an_archived_template_is_never_offered(composer):
    _insert(composer, "Old card", {"html": "<html></html>", "sizes": ["1080x1350"], "variables_schema": SCHEMA}, active=False)
    _compose(composer, [_answer(composer)], {"brief": "Harbour Blend is back."})
    assert "Old card" not in _offered(composer)


def _proposal_for(template, **ctx):
    context = compose.ComposeContext(brief="Harbour Blend is back.", format="image",
                                     channels=[{"toolkit": "instagram", "label": "Instagram"}], templates=[template],
                                     candidates=[], **ctx)
    raw = {"title": "Back", "format": "image", "template_id": template["id"], "variables": {"headline": "Back"},
           "copy": {"base": "Back.", "channels": {}}}
    return compose_checks.checked_proposal(raw, context)


def test_a_photo_template_says_the_photo_is_needed():
    entry = compose_api.template_entry(SimpleNamespace(id="t", name="Just the photo", description=None,
                                                       format="social_image", blocks=_photo_blocks(required=True)))
    needed = _proposal_for(entry)
    assert compose_photos.NEEDS_PHOTO.format(label="Photo") in needed["warnings"]
    assert needed["questions"] == ["A photo for 'Photo'"]
    optional = _proposal_for({**entry, "photo_slots": [{"slot": "photo", "label": "Photo", "required": False}]})
    assert compose_photos.SHOWS_PHOTO.format(label="Photo") in optional["warnings"] and optional["questions"] == []
    by_ai = _proposal_for(entry, visual_slots=({"slot": "photo", "kind": "image", "label": "Photo"},))
    assert by_ai["questions"] == [] and not any("photo" in w.lower() for w in by_ai["warnings"])


def test_the_prompt_says_how_to_pick():
    system = compose.build_messages(compose.ComposeContext(brief="b", format=None, channels=[], templates=[], candidates=[]))[0]
    assert compose.PICK_NOTE in system["content"]


@pytest.mark.parametrize("name, slots", [("Just the photo", ["photo"]), ("Before / after", ["before", "after"])])
def test_the_seeded_photo_only_cards_mark_their_photos_required(name, slots):
    starter = next(s for s in social_starters() if s["name"] == name)
    assert [slot["slot"] for slot in compose_photos.photo_slots(starter["blocks"]) if slot["required"]] == slots
