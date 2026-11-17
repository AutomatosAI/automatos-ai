"""PRD-251D US-D103 (E2): an empty caption is Auto's to write at a render, like an empty field.

The owner gave a brief and picked a look; the render wrote the words on the image and left the
caption empty, so the post went for approval with nothing to say on any channel. Pinned:

* ``empty_fields``: a caption holds words in its base text or in any channel's own text; it is
  Auto's to write when it is empty and the post has a brief to write it from; Auto's caption
  joins the edit only where the post's caption is empty, never over words, and never as an empty
  one; the history line names the fields, then the caption;
* the composer is asked once, for the post's own channels (every connected one when it has none),
  and the caption it wrote joins the edit, both for a chosen template's empty fields
  (``auto_fields``) and for "Let Auto pick" (``auto_template``); without the caption nothing
  changes in what is asked;
* the render route: a post with every field filled and an empty caption gets the caption alone
  (an optional field left empty stays empty); an empty field and an empty caption are written in
  one edit with one history line; a caption with words in one channel only is left as it is and
  nothing is asked; a post with no brief keeps its empty caption and nothing is asked; "Let Auto
  pick" gets its template, fields and caption in one edit, and is not asked again; the preview
  writes the caption too and keeps the status, and a preview of an approved post writes none, so
  its approval stands;
* a render a plan or an agent starts (``render_post`` itself) writes no caption.
"""
from __future__ import annotations

import asyncio
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
import api.socials_compose as compose_api  # noqa: E402
import api.socials_preview as preview_api  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from core.models.workspaces import Workspace  # noqa: E402
from modules.socials import compose, empty_fields, service  # noqa: E402
from tests.test_prd251w1_render_lifecycle import WS, _create, _post, _start, _template  # noqa: E402

env = render_harness.env

EMPTY = {"base": "", "channels": {}}
CAPTION = {"base": "Three weeks to go.", "channels": {"linkedin": "Three weeks to go. See you there."}}
BRIEF = "Three weeks to Web Summit: our stand opens on day one."
TITLE_CARD = {
    "id": "tpl-card", "name": "Title card", "format": "social_image", "sizes": ["1080x1350"], "durations": [],
    "variables_schema": {"headline": {"type": "text", "label": "Headline"}},
}


def _value(text):
    return {"value": text, "claim": False}


def _target(toolkit):
    return SimpleNamespace(toolkit=toolkit)


def _post_like(**over):
    post = dict(
        id=uuid.uuid4(), title="Web Summit countdown", brief="Three weeks to go.", copy=dict(EMPTY), format="image",
        template_id=uuid.uuid4(), length_seconds=None, variables={}, sources={}, media={}, targets=[],
    )
    post.update(over)
    return SimpleNamespace(**post)


def _proposal(copy=CAPTION):
    return {
        "template_id": TITLE_CARD["id"], "template": TITLE_CARD, "copy": copy,
        "variables": {"headline": _value("Three weeks to go")}, "sources": {}, "warnings": [], "questions": [],
    }


# ---------------------------------------------------------------------------
# empty_fields: what an empty caption is, and what Auto's caption replaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "copy, holds",
    [
        ({"base": "Three weeks to go."}, True),
        ({"base": "", "channels": {"linkedin": "See you there."}}, True),
        ({"base": "  ", "channels": {"linkedin": " ", "twitter": ""}}, False),
        ({"base": None, "channels": "not channels"}, False),
        ({}, False),
        (None, False),
        ("not a caption", False),
    ],
)
def test_a_caption_holds_words_in_its_base_or_in_any_channels_own_text(copy, holds):
    assert empty_fields.holds_a_caption(copy) is holds


@pytest.mark.parametrize(
    "over, wanted",
    [
        ({}, True),
        ({"copy": {"base": "Mine."}}, False),            # a caption with words is the person's
        ({"brief": None}, False),                         # no brief: nothing to write it from
        ({"brief": "   "}, False),
        ({"copy": None}, True),
        ({"status": "needs_approval"}, True),
        ({"status": "approved"}, False),                  # an approval a caption would void
        ({"status": "scheduled"}, False),
    ],
)
def test_the_caption_is_autos_to_write_when_it_is_empty_and_the_post_has_a_brief(over, wanted):
    assert empty_fields.caption_wanted(_post_like(**over)) is wanted


def test_autos_caption_joins_the_edit_only_where_the_caption_is_empty():
    assert empty_fields.caption_edit(_proposal(), _post_like()) == {"copy": CAPTION}
    # A caption with words, even one channel's alone, is never rewritten.
    worded = _post_like(copy={"base": "", "channels": {"twitter": "Mine."}})
    assert empty_fields.caption_edit(_proposal(), worded) == {}
    # Auto wrote no words either: nothing is saved as the caption.
    assert empty_fields.caption_edit(_proposal(copy=EMPTY), _post_like()) == {}


def test_only_the_caption_of_an_edit_is_kept_when_every_required_field_holds_a_value():
    edit = {"variables": {"eyebrow": _value("New")}, "sources": {}, "copy": CAPTION}
    assert empty_fields.only_caption(edit) == {"copy": CAPTION}
    assert empty_fields.only_caption({"variables": {}, "sources": {}}) == {}


def test_the_history_line_names_the_fields_and_then_the_caption():
    assert empty_fields.written_line(["Headline"], False) == "Auto wrote the fields left empty: Headline."
    assert empty_fields.written_line(["Headline", "Card 1 title"], True) == (
        "Auto wrote the fields left empty: Headline, Card 1 title. Auto wrote the caption."
    )
    assert empty_fields.written_line([], True) == "Auto wrote the caption."


# ---------------------------------------------------------------------------
# The composer is asked once, for the post's own channels
# ---------------------------------------------------------------------------


@pytest.fixture
def composer(monkeypatch):
    """The composer's two seams: the context it is given and its one call."""
    seen = SimpleNamespace(bodies=[], proposal=_proposal())

    def fake_context(db, workspace_id, body, *, given_fields=None):
        seen.bodies.append(body)
        return compose.ComposeContext(
            brief=body.brief, format=body.format, channels=[], templates=[TITLE_CARD], candidates=[],
            given_fields=dict(given_fields or {}),
        )

    async def fake_propose(context, llm_factory, timeout):
        return seen.proposal

    monkeypatch.setattr(compose_api, "compose_context", fake_context)
    monkeypatch.setattr(compose, "propose", fake_propose)
    monkeypatch.setattr(compose, "llm_factory", lambda workspace_id: (lambda: None))
    return seen


def test_the_fields_call_writes_the_caption_for_the_posts_own_channels(composer):
    post = _post_like(targets=[_target("linkedin"), _target("instagram"), _target("linkedin")])

    changes = asyncio.run(compose_api.auto_fields(None, WS, post, caption=True))

    (body,) = composer.bodies
    assert body.channels == ["instagram", "linkedin"] and body.template_id == post.template_id
    assert changes == {"variables": {"headline": _value("Three weeks to go")}, "sources": {}, "copy": CAPTION}


def test_a_post_with_no_channel_has_its_caption_written_for_every_connected_one(composer):
    asyncio.run(compose_api.auto_fields(None, WS, _post_like(), caption=True))
    assert composer.bodies[0].channels is None


def test_without_the_caption_the_fields_call_asks_and_writes_as_before(composer):
    changes = asyncio.run(compose_api.auto_fields(None, WS, _post_like(targets=[_target("linkedin")])))
    assert composer.bodies[0].channels is None and "copy" not in changes


def test_the_pick_writes_the_caption_with_the_template_and_its_fields(composer):
    post = _post_like(template_id=None, targets=[_target("twitter")])

    changes, name = asyncio.run(compose_api.auto_template(None, WS, post, caption=True))

    assert composer.bodies[0].channels == ["twitter"]
    assert changes == {
        "template_id": "tpl-card", "variables": {"headline": _value("Three weeks to go")}, "sources": {}, "copy": CAPTION,
    }
    assert name == "Title card"


def test_the_pick_never_rewrites_a_caption_that_holds_words(composer):
    post = _post_like(template_id=None, copy={"base": "Mine."})
    changes, _ = asyncio.run(compose_api.auto_template(None, WS, post, caption=True))
    assert "copy" not in changes


# ---------------------------------------------------------------------------
# The render route
# ---------------------------------------------------------------------------


@pytest.fixture
def auto(env, monkeypatch):
    """Auto's words and picks, recorded with whether the caption was asked for."""
    env.template = _template(env)
    env.asked = []

    async def fake_auto_fields(db, workspace_id, post, *, caption=False):
        env.asked.append(("fields", caption))
        written = {
            "variables": {"headline": _value("Three weeks to go"), "eyebrow": _value("New"), **(post.variables or {})},
            "sources": dict(post.sources or {}),
        }
        return {**written, "copy": CAPTION} if caption else written

    async def fake_auto_template(db, workspace_id, post, *, caption=False):
        env.asked.append(("pick", caption))
        edit = {"template_id": str(env.template), "variables": {"headline": _value("Three weeks to go")}, "sources": {}}
        return ({**edit, "copy": CAPTION} if caption else edit), "Story promo"

    monkeypatch.setattr(compose_api, "auto_fields", fake_auto_fields)
    monkeypatch.setattr(compose_api, "auto_template", fake_auto_template)
    return env


def _auto_line(row):
    return [e for e in row.review_log if e["action"] == service.ACTION_EDIT and e.get("agent") == service.AUTO]


def _briefed(env, **body):
    """The harness's post, with the brief the person gave: what Auto writes the caption from."""
    return _create(env, brief=BRIEF, **body)


def test_a_post_with_every_field_filled_and_an_empty_caption_gets_the_caption_alone(auto):
    post = _briefed(auto, template_id=str(auto.template), copy=EMPTY)  # the harness's post holds its headline

    _, job = _start(auto, post)

    row = _post(auto, post["id"])
    assert row.copy == CAPTION
    # Only the caption: the optional field Auto offered a value for stays empty.
    assert row.variables == {"headline": _value("Three weeks to go")}
    (line,) = _auto_line(row)
    assert line["comment"] == "Auto wrote the caption." and line["fields"] == ["copy"]
    assert auto.asked == [("fields", True)]
    # The caption is content: the hash moved, and the render is bound to the post as written.
    assert row.status == "rendering" and row.content_hash != post["content_hash"] and job.content_hash == row.content_hash


def test_an_empty_field_and_an_empty_caption_are_written_in_one_edit(auto):
    post = _briefed(auto, template_id=str(auto.template), variables={}, copy=EMPTY)

    _start(auto, post)

    row = _post(auto, post["id"])
    assert row.copy == CAPTION and row.variables["headline"] == _value("Three weeks to go")
    (line,) = _auto_line(row)
    assert line["comment"] == "Auto wrote the fields left empty: Headline, Eyebrow. Auto wrote the caption."
    assert line["fields"] == ["copy", "sources", "variables"]
    assert auto.asked == [("fields", True)]


def test_a_caption_with_words_in_one_channel_is_left_as_it_is_and_nothing_is_asked(auto):
    worded = {"base": "", "channels": {"linkedin": "See you there."}}
    post = _briefed(auto, template_id=str(auto.template), copy=worded)

    _start(auto, post)

    row = _post(auto, post["id"])
    assert auto.asked == [] and row.copy == worded and _auto_line(row) == []


def test_a_post_left_to_auto_with_an_empty_caption_gets_its_template_fields_and_caption_in_one_edit(auto):
    post = _briefed(auto, copy=EMPTY)  # a video, no template: left to Auto

    _, job = _start(auto, post)

    row = _post(auto, post["id"])
    assert row.template_id == auto.template and row.copy == CAPTION and row.status == "rendering"
    (line,) = _auto_line(row)
    assert line["comment"] == "Auto picked the template Story promo and wrote its fields and the caption."
    assert line["fields"] == ["copy", "sources", "template_id", "variables"]
    # One call: the render's own pick finds the template on the post and asks nothing again.
    assert auto.asked == [("pick", True)] and job.content_hash == row.content_hash


def test_a_post_with_no_brief_keeps_its_empty_caption_and_nothing_is_asked_for_it(auto):
    # Every field filled, no brief: Auto would have only the title to write a caption from.
    post = _create(auto, template_id=str(auto.template), copy=EMPTY)

    _start(auto, post)

    row = _post(auto, post["id"])
    assert auto.asked == [] and not empty_fields.holds_a_caption(row.copy) and _auto_line(row) == []


def test_the_preview_writes_the_caption_and_keeps_the_status(auto, monkeypatch):
    previews = []
    monkeypatch.setattr(preview_api, "_launch_preview", previews.append)
    post = _briefed(auto, template_id=str(auto.template), copy=EMPTY)

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render", json={"preview": True})

    assert resp.status_code == 202, resp.text
    row = _post(auto, post["id"])
    assert row.copy == CAPTION and row.status == "draft"
    assert len(previews) == 1 and previews[0].content_hash == row.content_hash


def test_a_preview_of_an_approved_post_writes_no_caption_and_keeps_the_approval(auto, monkeypatch):
    # A caption is content: written here, it would void the approval a preview never touches.
    previews = []
    monkeypatch.setattr(preview_api, "_launch_preview", previews.append)
    post = _briefed(auto, template_id=str(auto.template), copy=EMPTY)
    row = _post(auto, post["id"])
    row.status, row.approved_hash = service.APPROVED, row.content_hash  # what an approval stores
    auto.session.commit()

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render", json={"preview": True})

    assert resp.status_code == 202, resp.text
    row = _post(auto, post["id"])
    assert auto.asked == [] and not empty_fields.holds_a_caption(row.copy) and _auto_line(row) == []
    assert row.status == service.APPROVED and row.approved_hash == row.content_hash == post["content_hash"]
    assert len(previews) == 1


def test_a_render_a_plan_or_an_agent_starts_writes_no_caption(auto):
    workspace = auto.session.get(Workspace, WS)

    chosen = _briefed(auto, template_id=str(auto.template), copy=EMPTY)
    asyncio.run(socials_api.render_post(auto.session, workspace, _post(auto, chosen["id"]), "the plan"))
    row = _post(auto, chosen["id"])
    assert auto.asked == [] and not empty_fields.holds_a_caption(row.copy) and row.status == "rendering"

    left = _briefed(auto, copy=EMPTY)  # left to Auto: the plan's render picks the template, as it always has
    asyncio.run(socials_api.render_post(auto.session, workspace, _post(auto, left["id"]), "the agent"))
    row = _post(auto, left["id"])
    assert auto.asked == [("pick", False)] and not empty_fields.holds_a_caption(row.copy)
    assert row.template_id == auto.template
    (line,) = _auto_line(row)
    assert line["comment"] == "Auto picked the template Story promo and wrote its fields."
