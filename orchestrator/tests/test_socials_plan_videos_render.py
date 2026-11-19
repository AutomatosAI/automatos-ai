"""19 Nov 2026: a plan's videos render with no field left to the owner.

No video post had ever rendered in the workspace. Traced on the VPS:

* the UI story promo and the Cinematic product promo failed the renderer's layout check
  (hyperframes 0.8.220, ``text_occluded``): the check exempts only a covering element marked
  ``data-layout-allow-occlusion``, and the marks sat on the text's own wrapper. The covers are
  the scene's backdrop (``.sky``, ``#dawn``) and the result cards' light sweep (``.sheen``);
* every end card's web address stayed empty: the composer was never given the brand kit's
  own website, and it may write no other;
* a claim with no sourced figure held the whole video (the Results count, the app's list
  line): each now has an empty default, so the line shows without the figure;
* a video's whole proposal ran into the 8,000-token default and came back cut, so unreadable:
  the composer now has 16,000 and 90 seconds;
* a post moved to another template kept the old one's words, which the new one refused (too
  long): such a value is no longer held as the owner's, and Auto writes the field again;
* a template the check refused was sent to the renderer again on every gallery list
  (51 refused renders in 30 hours): a refused composition now waits until it changes.
"""
from __future__ import annotations

import asyncio
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

from api import socials_compose as compose_api  # noqa: E402
from config import config  # noqa: E402
from core.llm import output_budget  # noqa: E402
from core.social_templates import resolve_variables  # noqa: E402
from modules.socials import compose, empty_fields, render, template_thumbnails  # noqa: E402

STARTERS = _ORCH / "modules" / "documents" / "templates" / "social"
MARK = "data-layout-allow-occlusion"


def _html(name: str) -> str:
    return (STARTERS / f"{name}.html").read_text(encoding="utf-8")


def _schema(name: str) -> dict:
    return json.loads((STARTERS / f"{name}.json").read_text(encoding="utf-8"))["variables_schema"]


def _tags(html: str, pattern: str) -> list:
    return re.findall(rf"<div [^>]*{pattern}[^>]*>", html)


# ── the layout check: the covers carry the mark ─────────────────────────────


def test_every_light_sweep_over_a_result_card_is_marked_as_covering_on_purpose():
    for name in ("ui-story-promo", "cinematic-product-promo"):
        sweeps = _tags(_html(name), r'class="sheen"')
        assert len(sweeps) == 4, name
        assert all(MARK in tag for tag in sweeps), name


def test_the_morning_scenes_backdrop_is_marked_as_covering_on_purpose():
    html = _html("ui-story-promo")
    for pattern in (r'class="sky"', r'id="dawn"'):
        (tag,) = _tags(html, pattern)
        assert MARK in tag, pattern


# ── a claim with no sourced figure never holds a video ──────────────────────


def test_the_results_count_and_the_apps_list_line_render_without_a_figure():
    fields = (("ui-story-promo", "outputs_count"), ("cinematic-product-promo", "outputs_count"), ("app-promo", "tracks_detail"))
    for name, field in fields:
        assert field not in resolve_variables(_schema(name), {}).missing, (name, field)
        spec = _schema(name)[field]
        assert spec["claim"] is True and spec["default"] == "", (name, field)
        assert "leave it empty" in spec["description"], (name, field)


def test_a_fact_cards_figure_is_still_required():
    assert "fact_value" in resolve_variables(_schema("fact-card"), {}).missing


# ── the end card's web address is the brand kit's own ───────────────────────


def _db_with_website(website):
    settings = {"brand_kit": {"company": {"website": website}}} if website is not None else {}
    return SimpleNamespace(get=lambda _model, _id: SimpleNamespace(settings=settings))


def test_the_brand_website_is_written_as_an_end_card_shows_it():
    for given, shown in (("https://automatos.app/", "automatos.app"), ("http://Example.com/shop/", "Example.com/shop"),
                         ("automatos.app", "automatos.app"), ("", ""), (None, "")):
        assert compose_api.brand_website(_db_with_website(given), uuid.uuid4()) == shown, given


def test_the_composer_is_given_the_brand_website_and_told_it_is_the_only_one():
    ctx = compose.ComposeContext(brief="Studio", format="video", channels=[], templates=[], candidates=[], website="automatos.app")
    system, material = compose.build_messages(ctx)
    assert json.loads(material["content"])["brand_website"] == "automatos.app"
    assert compose.WEBSITE_NOTE in system["content"]
    bare = compose.ComposeContext(brief="Studio", format="video", channels=[], templates=[], candidates=[])
    assert json.loads(compose.build_messages(bare)[1]["content"])["brand_website"] == ""


# ── a post moved to another template: the old words it refuses are Auto's to write ──

END_CARD = {
    "end_line": {"type": "text", "label": "End card: line 1", "max_chars": 26},
    "headline": {"type": "text", "label": "Headline", "max_chars": 40},
}


def _moved_post():
    # The App promo's end line (28 characters allowed there), kept when the post moved to the UI story promo.
    return SimpleNamespace(variables={"end_line": {"value": "Out and about, on your phone", "claim": False},
                                      "headline": {"value": "Work on the move", "claim": False}},
                           sources={}, length_seconds=None)


def test_a_value_the_chosen_template_refuses_is_not_held_and_counts_as_empty():
    post = _moved_post()
    assert set(empty_fields.own_values(post)) == {"end_line", "headline"}
    assert empty_fields.own_values(post, END_CARD) == {"headline": "Work on the move"}
    assert empty_fields.empty_required({"variables_schema": END_CARD}, post) == ["end_line"]


def test_auto_writes_over_the_refused_value_and_keeps_the_ones_that_fit():
    post = _moved_post()
    proposal = {"template": {"variables_schema": END_CARD},
                "variables": {"end_line": {"value": "Your day, from your phone"}, "headline": {"value": "Not this"}}}
    edit = empty_fields.fields_edit(proposal, post)
    assert edit["variables"]["end_line"]["value"] == "Your day, from your phone"
    assert edit["variables"]["headline"]["value"] == "Work on the move"
    assert empty_fields.newly_written(edit, post, END_CARD) == ["end_line"]


def test_a_refused_value_in_a_field_with_a_default_still_asks_auto_to_write():
    schema = {**END_CARD, "count": {"type": "text", "label": "The count", "max_chars": 6, "default": "", "claim": True}}
    post = _moved_post()
    post.variables = {**post.variables, "count": {"value": "1,691 practice questions", "claim": True}}
    assert empty_fields.empty_required({"variables_schema": schema}, post) == ["end_line", "count"]


def test_a_refused_value_auto_did_not_write_is_left_empty_for_the_render_to_ask_for():
    proposal = {"template": {"variables_schema": END_CARD}, "variables": {}}
    edit = empty_fields.fields_edit(proposal, _moved_post())
    assert "end_line" not in edit["variables"]
    assert edit["variables"]["headline"]["value"] == "Work on the move"


def test_auto_fields_gives_the_composer_only_the_values_the_chosen_template_takes(monkeypatch):
    seen = []

    def fake_context(_db, _workspace_id, _body, *, given_fields=None):
        return compose.ComposeContext(brief="b", format="video", channels=[], templates=[{"variables_schema": END_CARD}],
                                      candidates=[], given_fields=dict(given_fields or {}))

    async def fake_proposal(context, _workspace_id, _post_id, _failed):
        seen.append(dict(context.given_fields))
        return {"template": {"variables_schema": END_CARD}, "variables": {"end_line": {"value": "Your day, from your phone"}}}

    monkeypatch.setattr(compose_api, "compose_context", fake_context)
    monkeypatch.setattr(compose_api, "_auto_proposal", fake_proposal)
    post = _moved_post()
    post.__dict__.update(id=uuid.uuid4(), title="Go", brief="Go on the move", copy={}, format="video", template_id=uuid.uuid4())
    edit = asyncio.run(compose_api.auto_fields(None, uuid.uuid4(), post))
    assert seen == [{"headline": "Work on the move"}]
    assert edit["variables"]["end_line"]["value"] == "Your day, from your phone"


def test_a_videos_whole_proposal_has_room_to_be_written_and_the_time_to_finish():
    # 3 of 42 composer calls on 19 Nov stopped at the 8,000 settings default: a cut answer is unreadable.
    assert output_budget.DEFAULT_BUDGETS[compose.REQUEST_TYPE] == 16000
    assert config.SOCIALS_COMPOSE_TIMEOUT_SECONDS >= 90


# ── a refused thumbnail is not sent again until its composition changes ─────


def _template(blocks="<div>one</div>"):
    return SimpleNamespace(id=uuid.UUID(int=7), name="UI story promo", format="social_video", blocks=blocks, sample_data={})


def _refusing_renderer(monkeypatch, code):
    sent = []

    async def submit(_client, bundle, _deadline):
        sent.append(bundle)
        raise render.RenderFailure(code, "The composition failed its check.")

    async def renderer_ready(_client, _store):
        return None

    monkeypatch.setattr(template_thumbnails, "_refused", {})
    monkeypatch.setattr(template_thumbnails, "thumbnail_bundle", lambda *_args: {"bundle": True})
    monkeypatch.setattr(template_thumbnails, "_job", lambda *_args: SimpleNamespace())
    monkeypatch.setattr(render, "_submit", submit)
    monkeypatch.setattr(render, "ensure_renderer", renderer_ready)
    return sent


def _backfill(monkeypatch, template):
    workspace = SimpleNamespace(id=uuid.uuid4(), name="ws", settings={})
    monkeypatch.setattr(template_thumbnails, "_load", lambda _factory, _ws, _ids: (workspace, [template]))
    return asyncio.run(template_thumbnails.ensure_thumbnails(
        workspace.id, [template.id], brand_kit_of=lambda _settings: {}, client=object(), store=object(),
        session_factory=lambda: None,
    ))


def test_a_composition_the_check_refused_is_not_sent_again(monkeypatch):
    sent = _refusing_renderer(monkeypatch, template_thumbnails.CHECK_FAILED)
    template = _template()
    assert _backfill(monkeypatch, template) == [] and len(sent) == 1
    assert _backfill(monkeypatch, template) == [] and len(sent) == 1


def test_a_changed_composition_or_an_old_refusal_is_tried_again(monkeypatch):
    sent = _refusing_renderer(monkeypatch, template_thumbnails.CHECK_FAILED)
    _backfill(monkeypatch, _template())
    _backfill(monkeypatch, _template(blocks="<div>fixed</div>"))
    assert len(sent) == 2
    later = template_thumbnails.time.monotonic() + template_thumbnails.CHECK_REFUSED_RETRY_SECONDS + 1
    assert not template_thumbnails.refused_recently(_template(), later)


def test_a_failure_that_is_not_the_checks_is_tried_again_on_the_next_list(monkeypatch):
    sent = _refusing_renderer(monkeypatch, "render_timeout")
    template = _template()
    _backfill(monkeypatch, template)
    _backfill(monkeypatch, template)
    assert len(sent) == 2
