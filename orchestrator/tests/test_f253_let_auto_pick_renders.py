"""F253 (TESTER, 3 Oct 2026) — "Let Auto pick" could not render.

A new image post on "Let Auto pick" (the editor's first template choice) was refused at
Render preview: "this post has no template to render: choose a social template first".
Every template has fields a render needs, so the render now asks Auto, the composer, first:

* ``left_to_auto``: a visual format with no template and no file of the person's own; a text
  post, a post with a template, one with no format and an upload or Library pick are not;
* ``auto_brief``: the post's brief and copy, or its title alone;
* ``auto_template``: the composer is asked for the post's format with no template chosen,
  with only the templates that offer a chosen length; its pick is the edit: the template,
  its fields (the person's own value wins) and the sources (the person's own stay). No
  template for the format is refused before the model is asked; a refused length, a model
  that does not answer and a proposal with no template are 422 saying why;
* the render route: a post left to Auto renders on Auto's pick, saved first with a line in
  the history, and a second render asks nothing again; the preview does the same and keeps
  the post's status; a post that cannot render is refused before the model is asked.
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

import api.socials_compose as compose_api  # noqa: E402
import api.socials_preview as preview_api  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from modules.socials import compose, service  # noqa: E402
from modules.socials.render import NotRenderable  # noqa: E402
from tests.test_prd251w1_render_lifecycle import WS, _create, _post, _start, _template  # noqa: E402

env = render_harness.env
TITLE_CARD = {"id": "tpl-title", "name": "Title card", "format": "social_image", "durations": []}
HEADLINE = {"headline": {"type": "text", "label": "Headline"}, "subline": {"type": "text", "default": ""}}


def _post_like(**over):
    post = dict(
        id=uuid.uuid4(), title="First Post", brief=None, copy={"base": "", "channels": {}}, format="image",
        template_id=None, length_seconds=None, variables={}, sources={}, media={},
    )
    post.update(over)
    return SimpleNamespace(**post)


def _proposal(template=TITLE_CARD, variables=None, sources=None):
    return {
        "title": "Our first post", "copy": {"base": "", "channels": {}}, "format": "image",
        "template_id": template["id"] if template else None,
        "template": {**template, "sizes": ["1080x1350"], "variables_schema": HEADLINE} if template else None,
        "variables": {"headline": {"value": "Our first post", "claim": False}} if variables is None else variables,
        "sources": sources or {}, "channels": [], "length_seconds": None, "visual_prompts": {}, "warnings": [],
    }


@pytest.fixture
def composer(monkeypatch):
    """The composer's two seams: its context (the templates it is given) and its one call."""
    seen = SimpleNamespace(bodies=[], contexts=[], templates=[TITLE_CARD], proposal=_proposal(), error=None, refuse=None)

    def fake_context(db, workspace_id, body):
        seen.bodies.append(body)
        if seen.refuse:
            raise seen.refuse
        return compose.ComposeContext(brief=body.brief, format=body.format, channels=[], templates=list(seen.templates), candidates=[])

    async def fake_propose(context, llm_factory, timeout):
        seen.contexts.append(context)
        if seen.error:
            raise seen.error
        return seen.proposal

    monkeypatch.setattr(compose_api, "compose_context", fake_context)
    monkeypatch.setattr(compose, "propose", fake_propose)
    monkeypatch.setattr(compose, "llm_factory", lambda workspace_id: (lambda: None))
    return seen


def _pick(post):
    return asyncio.run(compose_api.auto_template(None, WS, post))


# ---------------------------------------------------------------------------
# What is left to Auto, and what Auto writes from
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "over, left",
    [
        ({}, True),
        ({"format": "video", "length_seconds": 15}, True),
        ({"format": "carousel"}, True),
        ({"format": "text"}, False),
        ({"format": None}, False),
        ({"template_id": uuid.uuid4()}, False),
        ({"media": {"original": ["d-1"]}}, False),
    ],
)
def test_a_post_is_left_to_auto_with_a_visual_format_no_template_and_no_file_of_its_own(over, left):
    assert compose_api.left_to_auto(_post_like(**over)) is left


def test_auto_writes_from_the_brief_and_the_copy_or_the_title_alone():
    assert compose_api.auto_brief(_post_like()) == "First Post"
    assert compose_api.auto_brief(_post_like(brief=" Launch week ", copy={"base": "Ships Monday."})) == "Launch week\n\nShips Monday."
    assert compose_api.auto_brief(_post_like(copy={"base": "Ships Monday."})) == "Ships Monday."


# ---------------------------------------------------------------------------
# auto_template: the composer's pick, as the edit that records it
# ---------------------------------------------------------------------------


def test_the_composer_picks_for_the_posts_format_and_its_pick_is_the_edit(composer):
    changes, name = _pick(_post_like())

    (body,) = composer.bodies
    assert (body.brief, body.format, body.template_id, body.length_seconds) == ("First Post", "image", None, None)
    assert changes == {
        "template_id": "tpl-title",
        "variables": {"headline": {"value": "Our first post", "claim": False}},
        "sources": {},
    }
    assert name == "Title card"


def test_the_persons_own_values_win_and_their_sources_stay(composer):
    composer.proposal = _proposal(
        variables={"headline": {"value": "Auto's", "claim": False}, "subline": {"value": "Written by Auto", "claim": False}},
        sources={"members": {"kind": "metric", "ref": "m-1", "as_of": None}},
    )
    mine = {"headline": {"value": "Mine", "claim": False}, "stat_1": {"value": "9", "claim": True}}
    own_source = {"stat_1": {"kind": "report", "ref": "r-1", "as_of": None}}

    changes, _ = _pick(_post_like(variables=mine, sources=own_source))

    # headline is the template's and the person's: theirs wins. stat_1 is not the template's: dropped.
    assert changes["variables"] == {
        "headline": {"value": "Mine", "claim": False}, "subline": {"value": "Written by Auto", "claim": False},
    }
    assert changes["sources"] == {**composer.proposal["sources"], **own_source}


def test_a_chosen_length_keeps_only_the_templates_that_offer_it(composer):
    composer.templates = [
        {"id": "a", "name": "Story", "format": "social_video", "durations": [15, 30]},
        {"id": "b", "name": "Promo", "format": "social_video", "durations": [40]},
    ]
    composer.proposal = _proposal(template=composer.templates[0], variables={})
    _pick(_post_like(format="video", length_seconds=15))
    assert [t["id"] for t in composer.contexts[0].templates] == ["a"]


def test_no_template_for_the_format_is_refused_before_the_model_is_asked(composer):
    composer.templates = []
    with pytest.raises(NotRenderable, match="Auto found no template for this image post: add one in Templates"):
        _pick(_post_like())
    assert composer.contexts == []


def test_a_refused_length_says_why(composer):
    composer.refuse = compose_api.ChoiceRefused("length_not_declared", "length_seconds must be a length any video template declares ([15]), got 20.")
    with pytest.raises(NotRenderable, match="Auto could not pick a template: length_seconds must be a length"):
        _pick(_post_like(format="video", length_seconds=20))


@pytest.mark.parametrize(
    "error, says",
    [
        (compose.ComposeTimedOut("The model did not answer within 60 seconds. Try again."), "did not answer within 60 seconds"),
        (compose.ComposeFailed("The model's answer could not be read as a proposal."), "could not be read as a proposal"),
        (RuntimeError("connection reset"), "the model could not be reached"),
    ],
)
def test_a_model_that_gives_no_usable_answer_is_refused_saying_why(composer, error, says):
    composer.error = error
    with pytest.raises(NotRenderable, match=f"Auto could not pick a template: .*{says}"):
        _pick(_post_like())


def test_a_proposal_without_a_template_is_refused(composer):
    composer.proposal = _proposal(template=None, variables={})
    with pytest.raises(NotRenderable, match="Auto found no template for this image post"):
        _pick(_post_like())


# ---------------------------------------------------------------------------
# The render route
# ---------------------------------------------------------------------------


@pytest.fixture
def auto(env, monkeypatch):
    """Auto's pick, recorded: the harness's own video template, with a headline."""
    env.template = _template(env)
    env.asked = []

    async def fake_auto_template(db, workspace_id, post):
        env.asked.append(post.id)
        edit = {
            "template_id": str(env.template),
            "variables": {"headline": {"value": "Three weeks to go", "claim": False}},
            "sources": {},
        }
        return edit, "Story promo"

    monkeypatch.setattr(compose_api, "auto_template", fake_auto_template)
    return env


def _auto_line(row):
    return [e for e in row.review_log if e["action"] == service.ACTION_EDIT and e.get("agent") == service.AUTO]


def test_a_post_left_to_auto_renders_on_autos_pick_saved_first(auto):
    post = _create(auto)  # a video, no template: left to Auto

    _, job = _start(auto, post)

    row = _post(auto, post["id"])
    assert row.template_id == auto.template and row.status == "rendering"
    (line,) = _auto_line(row)
    assert line["comment"] == "Auto picked the template Story promo and wrote its fields."
    assert line["fields"] == ["sources", "template_id", "variables"]
    # The pick is content: the hash moved, and the render is bound to the post as picked.
    assert row.content_hash != post["content_hash"] and job.content_hash == row.content_hash
    assert auto.asked == [uuid.UUID(post["id"])]

    # A second render finds the template on the post: Auto is not asked again.
    service.fail_render(row, "tester", "The renderer was restarted.")
    auto.session.commit()
    _start(auto, post)
    assert auto.asked == [uuid.UUID(post["id"])] and len(_auto_line(_post(auto, post["id"]))) == 1


def test_the_preview_of_a_post_left_to_auto_uses_autos_pick_and_keeps_the_status(auto, monkeypatch):
    previews = []
    monkeypatch.setattr(preview_api, "_launch_preview", previews.append)
    post = _create(auto)

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render", json={"preview": True})

    assert resp.status_code == 202, resp.text
    row = _post(auto, post["id"])
    assert row.template_id == auto.template and row.status == "draft" and len(_auto_line(row)) == 1
    assert len(previews) == 1 and previews[0].content_hash == row.content_hash


def test_a_post_that_cannot_render_is_refused_before_auto_is_asked(auto):
    post = _create(auto)
    row = _post(auto, post["id"])
    service.start_render(row, "tester")
    auto.session.commit()

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render")

    assert resp.status_code == 409
    assert auto.asked == [] and _post(auto, post["id"]).template_id is None
