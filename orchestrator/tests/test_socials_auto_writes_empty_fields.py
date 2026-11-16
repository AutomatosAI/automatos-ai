"""A field left empty is Auto's to write; a field with a value stays as written (16 Nov 2026).

The owner: "when I select a template I am forced to fill in loads of fields, I might as well
write the post myself". Choosing a template handed over its whole form: a render with a
required field empty was refused ("fill in … before rendering"), and "Redraft with Auto"
replaced whatever had been typed. Pinned:

* ``empty_fields``: the fields that hold a value, the required ones that hold none (at the
  post's length), and the ones an edit newly writes; Auto's value goes only where the post
  holds none (a blank never hides it), a variable the template does not have stays, and an
  empty field's old source never rides on to Auto's value; a field stored blank on purpose
  is still not one the render refuses;
* the composer keeps a field the person wrote (``given_fields``): it wins over the model's
  value, is never asked for again, and is in the prompt as given; a source line or a handle
  the person typed is theirs, never taken out as invented;
* ``auto_fields``: the composer is asked with the post's own template and its own fields
  given, and its answer is the edit: the person's values win, only fields and sources change;
* the render route: a post with a chosen template and an empty required field renders on
  Auto's words, saved first with a line in the history; nothing is asked when nothing is
  empty; what Auto cannot write is asked by its label with what it wrote kept; the preview
  does the same; a post that cannot render is refused before the model is asked;
* a render started by anything but the person's route (a plan, an agent) asks nothing.
"""
from __future__ import annotations

import asyncio
import json
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
from modules.socials import compose, compose_checks, empty_fields, service  # noqa: E402
from modules.socials.render import NotRenderable  # noqa: E402
from tests.test_prd251w1_render_lifecycle import COMPOSITION, WS, _create, _post, _start, _template  # noqa: E402

env = render_harness.env

HIGHLIGHTS = {
    "eyebrow": {"type": "text", "label": "Label", "default": ""},
    "headline": {"type": "text", "label": "Headline", "max_chars": 60},
    "card_1_title": {"type": "text", "label": "Card 1 title", "max_chars": 32},
    "card_1_text": {"type": "text", "label": "Card 1 text", "default": ""},
    "handle": {"type": "text", "label": "Handle", "default": ""},
}
CARD = {"id": "tpl-card", "name": "Photo highlights", "format": "social_image", "sizes": ["1080x1350"],
        "variables_schema": HIGHLIGHTS, "durations": []}
FACT = {"id": "tpl-fact", "name": "Fact card", "format": "social_image", "sizes": ["1080x1350"], "durations": [],
        "variables_schema": {"fact": {"type": "text", "label": "The fact"},
                             "source_label": {"type": "text", "label": "Source line"}}}


def _post_like(**over):
    post = dict(
        id=uuid.uuid4(), title="Server keys", brief="We shipped server keys with scopes.",
        copy={"base": "", "channels": {}}, format="image", template_id=uuid.uuid4(), length_seconds=None,
        variables={}, sources={}, media={},
    )
    post.update(over)
    return SimpleNamespace(**post)


def _value(text):
    return {"value": text, "claim": False}


# ---------------------------------------------------------------------------
# empty_fields: what is the person's, and what is left to Auto
# ---------------------------------------------------------------------------


def test_a_field_holds_a_value_when_it_has_words_a_number_or_a_switch():
    post = _post_like(variables={
        "headline": _value("SERVER KEYS"), "eyebrow": _value("  "), "card_1_title": _value(None),
        "rating": _value(5), "muted": _value(False), "broken": "not a field",
    })
    assert empty_fields.own_values(post) == {"headline": "SERVER KEYS", "rating": 5, "muted": False}


def test_the_empty_required_fields_are_those_with_no_value_and_no_default():
    assert empty_fields.empty_required(CARD, _post_like()) == ["headline", "card_1_title"]
    typed = _post_like(variables={"headline": _value("SERVER KEYS")})
    assert empty_fields.empty_required(CARD, typed) == ["card_1_title"]
    full = _post_like(variables={"headline": _value("SERVER KEYS"), "card_1_title": _value("Scoped")})
    assert empty_fields.empty_required(CARD, full) == []


def test_a_field_the_posts_cut_never_shows_is_not_empty():
    story = {**CARD, "format": "social_video", "durations": [15, 30], "fields_cut_out": {"15": ["card_1_title"]}}
    assert empty_fields.empty_required(story, _post_like(length_seconds=15)) == ["headline"]
    assert empty_fields.empty_required(story, _post_like(length_seconds=30)) == ["headline", "card_1_title"]


def test_newly_written_names_only_the_fields_the_post_did_not_hold():
    post = _post_like(variables={"headline": _value("SERVER KEYS")})
    changes = {"variables": {"headline": _value("SERVER KEYS"), "card_1_title": _value("Scoped")}, "sources": {}}
    assert empty_fields.newly_written(changes, post) == ["card_1_title"]


def test_autos_value_goes_only_where_the_post_holds_none_and_the_rest_of_the_post_stays():
    post = _post_like(
        variables={
            "headline": _value("SERVER KEYS"),   # the person's: stays
            "card_1_title": _value("  "),        # stored blank: Auto's value is not hidden by it
            "slides": _value(5),                 # not the template's (a carousel's slide count): stays
        },
        sources={"headline": {"kind": "report", "ref": "r-1", "as_of": None},
                 "card_1_title": {"kind": "report", "ref": "old", "as_of": None}},
    )
    proposal = {
        "template": CARD,
        "variables": {"headline": _value("Auto's headline"), "card_1_title": _value("Scoped"), "card_1_text": _value("One key per job.")},
        "sources": {"card_1_text": {"kind": "metric", "ref": "m-1", "as_of": None}},
    }

    changes = empty_fields.fields_edit(proposal, post)

    assert changes["variables"] == {
        "headline": _value("SERVER KEYS"), "card_1_title": _value("Scoped"), "card_1_text": _value("One key per job."),
        "slides": _value(5),
    }
    # The person's source stays with the field that holds their value; the empty field's old
    # source does not ride on to the value Auto wrote for it.
    assert changes["sources"] == {
        "headline": {"kind": "report", "ref": "r-1", "as_of": None},
        "card_1_text": {"kind": "metric", "ref": "m-1", "as_of": None},
    }
    assert empty_fields.newly_written(changes, post) == ["card_1_title", "card_1_text"]


def test_a_field_stored_blank_is_empty_to_auto_and_still_a_value_to_the_render():
    blank = _post_like(variables={"headline": _value(""), "card_1_title": _value("Scoped")})
    assert empty_fields.empty_required(CARD, blank) == ["headline"]  # Auto is asked for it
    assert empty_fields.render_refuses(CARD, blank) == []            # and the render still takes it, as before
    assert empty_fields.render_refuses(CARD, _post_like()) == ["headline", "card_1_title"]


# ---------------------------------------------------------------------------
# The composer keeps what the person wrote
# ---------------------------------------------------------------------------


def _ctx(template=CARD, **over):
    given = dict(
        brief="We shipped server keys with scopes.", format="image", channels=[{"toolkit": "linkedin", "label": "LinkedIn"}],
        templates=[template], candidates=[], template_id=template["id"],
    )
    given.update(over)
    return compose.ComposeContext(**given)


def _answer(template=CARD, **variables):
    return {"title": "Server keys", "copy": {"base": "Shipped.", "channels": {"linkedin": "Shipped."}},
            "format": "image", "template_id": template["id"], "variables": variables, "sources": {}}


class _Model:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        return SimpleNamespace(content=json.dumps(self.answers.pop(0)))


def test_the_persons_own_field_wins_over_the_models_and_is_never_missing():
    ctx = _ctx(given_fields={"headline": "SERVER KEYS,|WITH SCOPES.", "not_a_field": "x"})

    proposal = compose_checks.checked_proposal(_answer(headline="Auto's headline", card_1_title="Scoped"), ctx)

    assert proposal["variables"]["headline"] == _value("SERVER KEYS,|WITH SCOPES.")
    assert proposal["variables"]["card_1_title"] == _value("Scoped")
    assert "not_a_field" not in proposal["variables"]
    assert not any("not_a_field" in warning or "No value yet" in warning for warning in proposal["warnings"])


def test_a_field_the_person_wrote_is_not_asked_of_the_model_again():
    # The model leaves the given headline out, as it is told to: no follow-up asks for it.
    model = _Model([_answer(card_1_title="Scoped")])
    ctx = _ctx(given_fields={"headline": "SERVER KEYS"})

    proposal = asyncio.run(compose.propose(ctx, lambda: model, 5.0))

    assert len(model.asked) == 1
    assert proposal["variables"]["headline"] == _value("SERVER KEYS") and proposal["questions"] == []


def test_the_prompt_gives_the_persons_fields_and_says_they_stay():
    system, user = compose.build_messages(_ctx(given_fields={"headline": "SERVER KEYS"}))
    assert compose.GIVEN_FIELDS_NOTE in system["content"]
    assert json.loads(user["content"])["given_fields"] == {"headline": "SERVER KEYS"}

    system, user = compose.build_messages(_ctx())
    assert compose.GIVEN_FIELDS_NOTE not in system["content"] and "given_fields" not in json.loads(user["content"])


def test_a_source_line_and_a_handle_the_person_typed_are_theirs():
    # Neither is in the brief or the brand kit: typed by the person, they are not an invention.
    typed = {"source_label": "Internal audit, Nov 2026"}
    proposal = compose_checks.checked_proposal(_answer(FACT, fact="Keys are scoped."), _ctx(FACT, given_fields=typed))
    assert proposal["variables"]["source_label"] == _value("Internal audit, Nov 2026")
    assert proposal["questions"] == []

    carded = compose_checks.checked_proposal(
        _answer(headline="Scoped keys", card_1_title="Scoped"), _ctx(given_fields={"handle": "@automatosai"}))
    assert carded["variables"]["handle"] == _value("@automatosai")


def test_without_the_persons_field_an_invented_source_line_is_still_left_for_them():
    proposal = compose_checks.checked_proposal(_answer(FACT, fact="Keys are scoped.", source_label="Internal audit"), _ctx(FACT))
    assert "source_label" not in proposal["variables"]


# ---------------------------------------------------------------------------
# auto_fields: the composer's answer, as the edit
# ---------------------------------------------------------------------------


@pytest.fixture
def composer(monkeypatch):
    """The composer's two seams: the context it is given and its one call."""
    seen = SimpleNamespace(bodies=[], given=[], proposal=None, error=None, refuse=None)

    def fake_context(db, workspace_id, body, *, given_fields=None):
        seen.bodies.append(body)
        seen.given.append(dict(given_fields or {}))
        if seen.refuse:
            raise seen.refuse
        return _ctx(given_fields=dict(given_fields or {}))

    async def fake_propose(context, llm_factory, timeout):
        if seen.error:
            raise seen.error
        return seen.proposal

    monkeypatch.setattr(compose_api, "compose_context", fake_context)
    monkeypatch.setattr(compose, "propose", fake_propose)
    monkeypatch.setattr(compose, "llm_factory", lambda workspace_id: (lambda: None))
    return seen


def _proposal(**variables):
    return {"template_id": CARD["id"], "template": CARD, "variables": {n: _value(v) for n, v in variables.items()},
            "sources": {}, "warnings": [], "questions": []}


def test_auto_is_asked_with_the_posts_template_and_its_own_fields_given(composer):
    composer.proposal = _proposal(headline="Auto's headline", card_1_title="Scoped", card_1_text="One key per job.")
    post = _post_like(variables={"headline": _value("SERVER KEYS"), "slides": _value(5)})

    changes = asyncio.run(compose_api.auto_fields(None, WS, post))

    (body,) = composer.bodies
    assert (body.brief, body.format, body.template_id) == ("We shipped server keys with scopes.", "image", post.template_id)
    assert composer.given == [{"headline": "SERVER KEYS", "slides": 5}]
    assert changes == {
        "variables": {"headline": _value("SERVER KEYS"), "slides": _value(5), "card_1_title": _value("Scoped"),
                      "card_1_text": _value("One key per job.")},
        "sources": {},
    }


@pytest.mark.parametrize(
    "error, says",
    [
        (compose.ComposeTimedOut("The model did not answer within 60 seconds. Try again."), "did not answer within 60 seconds"),
        (RuntimeError("connection reset"), "the model could not be reached"),
    ],
)
def test_a_model_that_gives_no_answer_is_refused_saying_auto_could_not_write(composer, error, says):
    composer.error = error
    with pytest.raises(NotRenderable, match=f"Auto could not write the words on the image: .*{says}"):
        asyncio.run(compose_api.auto_fields(None, WS, _post_like()))


def test_a_choice_the_workspace_cannot_honour_says_why(composer):
    composer.refuse = compose_api.ChoiceRefused("template_not_allowed", "template_id is not one of this workspace's social templates.")
    with pytest.raises(NotRenderable, match="Auto could not write the words on the image: template_id is not one"):
        asyncio.run(compose_api.auto_fields(None, WS, _post_like()))


# ---------------------------------------------------------------------------
# The render route
# ---------------------------------------------------------------------------

REVIEW = {
    **COMPOSITION,
    "html": COMPOSITION["html"].replace("<h1>{{ headline }}</h1>", "<h1>{{ headline }}</h1><p>{{ quote }}</p>"),
    "variables_schema": {"headline": {"type": "text"}, "quote": {"type": "text", "label": "Their words"}},
}


@pytest.fixture
def auto(env, monkeypatch):
    """Auto's words, recorded: what it writes for each post it is asked about."""
    env.asked, env.writes = [], {"headline": _value("Three weeks to go")}

    async def fake_auto_fields(db, workspace_id, post):
        env.asked.append(post.id)
        return {"variables": {**env.writes, **(post.variables or {})}, "sources": dict(post.sources or {})}

    monkeypatch.setattr(compose_api, "auto_fields", fake_auto_fields)
    return env


def _empty(env, blocks=COMPOSITION, **body):
    """A post with a chosen template and none of its fields filled in (or only ``variables``)."""
    return _create(env, **{"template_id": str(_template(env, blocks=blocks)), "variables": {}, **body})


def _auto_line(row):
    return [e for e in row.review_log if e["action"] == service.ACTION_EDIT and e.get("agent") == service.AUTO]


def test_a_post_with_an_empty_field_renders_on_autos_words_saved_first(auto):
    post = _empty(auto)

    _, job = _start(auto, post)

    row = _post(auto, post["id"])
    assert row.variables == {"headline": _value("Three weeks to go")} and row.status == "rendering"
    (line,) = _auto_line(row)
    assert line["comment"] == "Auto wrote the fields left empty: Headline."
    assert line["fields"] == ["sources", "variables"]
    # Auto's words are content: the hash moved, and the render is bound to the post as written.
    assert row.content_hash != post["content_hash"] and job.content_hash == row.content_hash
    assert auto.asked == [uuid.UUID(post["id"])]

    # A second render finds the field filled: Auto is not asked again.
    service.fail_render(row, "tester", "The renderer was restarted.")
    auto.session.commit()
    _start(auto, post)
    assert auto.asked == [uuid.UUID(post["id"])] and len(_auto_line(_post(auto, post["id"]))) == 1


def test_a_post_with_every_field_filled_never_asks_auto(auto):
    post = _create(auto, template_id=str(_template(auto)))  # the harness's post holds its headline

    _start(auto, post)

    assert auto.asked == [] and _auto_line(_post(auto, post["id"])) == []


def test_what_auto_cannot_write_is_asked_by_its_label_and_what_it_wrote_is_kept(auto):
    # The brief gives no customer's words: Auto writes the headline and leaves the quote.
    post = _empty(auto, blocks=REVIEW)

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render")

    assert resp.status_code == 422
    assert resp.json()["detail"] == (
        "Auto could not write these from the brief: Their words. Say them in the brief, or fill them in yourself."
    )
    row = _post(auto, post["id"])
    assert row.variables == {"headline": _value("Three weeks to go")} and row.status == "draft"
    assert auto.launched == []


def test_the_persons_own_field_is_kept_when_auto_writes_the_rest(auto):
    auto.writes = {"headline": _value("Auto's headline"), "quote": _value("It just works.")}
    post = _empty(auto, blocks=REVIEW, variables={"headline": _value("MY HEADLINE")})

    _start(auto, post)

    row = _post(auto, post["id"])
    assert row.variables == {"headline": _value("MY HEADLINE"), "quote": _value("It just works.")}
    assert _auto_line(row)[0]["comment"] == "Auto wrote the fields left empty: Their words."


def test_the_preview_writes_the_empty_fields_and_keeps_the_status(auto, monkeypatch):
    previews = []
    monkeypatch.setattr(preview_api, "_launch_preview", previews.append)
    post = _empty(auto)

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render", json={"preview": True})

    assert resp.status_code == 202, resp.text
    row = _post(auto, post["id"])
    assert row.variables == {"headline": _value("Three weeks to go")} and row.status == "draft"
    assert len(previews) == 1 and previews[0].content_hash == row.content_hash


def test_a_post_that_cannot_render_is_refused_before_auto_is_asked(auto):
    post = _empty(auto)
    row = _post(auto, post["id"])
    service.start_render(row, "tester")
    auto.session.commit()

    resp = auto.client.post(f"/api/socials/posts/{post['id']}/render")

    assert resp.status_code == 409
    assert auto.asked == [] and _post(auto, post["id"]).variables == {}


def test_a_render_started_by_a_plan_or_an_agent_asks_auto_nothing(auto):
    # ``render_post`` itself (the plan maker's and the agent tool's call) still refuses an empty
    # field: the composer already wrote what it could for those posts, and the owner is asked.
    post = _empty(auto)
    row = _post(auto, post["id"])
    workspace = auto.session.get(Workspace, WS)

    with pytest.raises(NotRenderable, match="fill in headline before rendering"):
        asyncio.run(socials_api.render_post(auto.session, workspace, row, "the plan"))

    assert auto.asked == []
