"""F379 / F381 (night 11, 7 Oct): Auto knows how Socials works today, and the real template names.

Night 11, Auto said rendering "happens later, usually after approval" and that it had no tool to
make a video (B10, B-I5-1), named templates that don't exist ("instagram carousel template",
"Instagram Image"), and said Socials had no photo upload (B-I4-2). Its always-on skill still
describes the retired automatos-social pipeline. Now a turn about social posts (or photos for a
post or an agent) gets the Socials note last, with this workspace's social template names; a
social template the owner names gets its fields for a post, never generate_document's note; the
template list names the social templates too; and a social template's schema gives the bare field
names a post's variables take.
"""
from __future__ import annotations

import asyncio
import contextlib
import uuid
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from consumers.chatbot import named_template_note as note_module
from consumers.chatbot.paperwork_to_the_team import team_note
from consumers.chatbot.socials_turn_note import SOCIALS_NOTE, about_socials, socials_note

WS = uuid.UUID("00000000-0000-0000-0000-0000000379c1")
QUOTE_CARD = NS(id=uuid.UUID("00000000-0000-0000-0000-0000000379c2"), name="Quote card", format="social_image",
                category="social", description="", data_schema={}, sample_data={}, version=1, blocks={
                    "variables_schema": {"quote": {"type": "text", "label": "Quote"},
                                         "attribution": {"type": "text", "label": "Who said it", "default": ""}}})
CAROUSEL = NS(id=uuid.uuid4(), name="Carousel", format="social_image", category="social", blocks={}, version=1)
UI_STORY = NS(id=uuid.uuid4(), name="UI story promo", format="social_video", category="social", blocks={}, version=1)
INVOICE = NS(id=uuid.uuid4(), name="Branded Invoice", format="pdf", category="invoice", blocks=None, version=1)
ROWS = [CAROUSEL, INVOICE, QUOTE_CARD, UI_STORY]


@pytest.mark.parametrize("texts", [
    ["Can you make me a carousel for the Harvest Club box?"],
    ["Draft three Instagram posts for next week."],
    ["Make a 15-second video of our top three wholesale cafés."],
    ["How do I give my agent two photos?"],
    ["Where do I add photos?"],
    ["Where do the photos go?", "Make me an Instagram post about the Guji."],
])
def test_a_turn_about_social_posts_or_their_photos_is_one(texts):
    assert about_socials(texts) is True


@pytest.mark.parametrize("texts", [
    ["Can you make me a wholesale price list for cafés, as a spreadsheet?"],
    ["Write a blog post about the Harvest Club."],
    ["Add photos to my Shopify products."],
    ["Send it to the post office by Friday."],
])
def test_other_turns_are_not(texts):
    assert about_socials(texts) is False


def test_the_note_tells_how_socials_works_and_names_the_workspaces_own_templates():
    note = socials_note(["Can you make me a carousel for the Harvest Club box?"], ROWS)

    assert note.startswith("Socials, as it works today.")
    assert "image: Carousel, Quote card; video: UI story promo" in note
    assert "renders as soon as it is saved" in note and "rendering never waits for approval" in note
    assert "A video is a post too: format video with a video template, and you make it." in note
    assert "social_image and social_video are the formats of templates, never of a post" in note
    assert "Look panel" in note and '"Where your picture goes"' in note and "never use Dropbox" in note
    assert "Branded Invoice" not in note


def test_a_social_template_the_owner_names_gets_its_fields_for_a_post():
    note = socials_note(["Make a Quote card for Rosa at Lantern Kitchen."], ROWS)

    assert f'The owner named the social template "Quote card" (template_id {QUOTE_CARD.id})' in note
    assert 'platform_create_social_post with template "Quote card"' in note
    assert "quote* (Quote), attribution (Who said it)" in note
    assert "generate_document" not in note


def test_a_turn_about_something_else_gets_no_socials_note():
    assert socials_note(["Can you make me an invoice on my Branded Invoice?"], ROWS) is None


class _ChatDb:
    def begin_nested(self):
        return contextlib.nullcontext()


@pytest.fixture
def workspace_templates(monkeypatch):
    from modules.documents import template_service

    class _Templates:
        def __init__(self, db):
            pass

        def list_templates(self, workspace_id, format=None, category=None):
            return [row for row in ROWS if format is None or row.format == format]

    monkeypatch.setattr(template_service, "DocumentTemplateService", _Templates)
    monkeypatch.setattr(note_module, "designer_note", lambda *a, **k: None)


def test_the_turn_gets_the_socials_note_and_paperwork_still_goes_to_the_team(workspace_templates):
    price_list = "Can you make me a wholesale price list for cafés, as a spreadsheet?"

    assert note_module.read_note(_ChatDb(), WS, ["Make me an Instagram carousel."]).startswith(SOCIALS_NOTE[:40])
    assert note_module.read_note(_ChatDb(), WS, [price_list]) == team_note(price_list)


def test_the_template_list_names_the_social_templates_in_one_line(workspace_templates):
    from modules.tools.discovery.template_tools import list_templates_answer

    every = list_templates_answer(object(), WS, {})
    named = list_templates_answer(object(), WS, {"name": "card"})

    assert every["count"] == 1 and every["templates"][0].startswith("Branded Invoice | pdf | ")
    assert every["social_templates"] == "image: Carousel, Quote card; video: UI story promo"
    assert named["social_templates"] == "image: Quote card" and named["count"] == 0
    assert "social_templates" not in list_templates_answer(object(), WS, {"format": "social_image"})


def test_a_social_templates_schema_gives_the_bare_names_a_post_takes():
    from modules.tools.discovery.template_tools import template_schema_answer

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = QUOTE_CARD
    schema = template_schema_answer(db, WS, {"template_id": str(QUOTE_CARD.id)})

    assert schema["post_variables"] == "quote* (Quote), attribution (Who said it)"
    assert "no \"data.\"" in schema["post_variables_note"]
    assert schema["data_fields"] == ["data.quote", "data.attribution"]


def test_the_handler_reads_the_same_schema():
    from modules.tools.discovery.handlers_documents import get_template_schema

    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = QUOTE_CARD
    schema = asyncio.run(get_template_schema(db, WS, {"template_id": str(QUOTE_CARD.id)}))
    assert schema["post_variables"].startswith("quote*")
