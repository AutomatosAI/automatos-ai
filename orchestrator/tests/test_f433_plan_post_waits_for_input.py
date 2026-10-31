"""P256-FIX-SOC-2 (F433, socials day 14 Oct §44a/§45c): a plan's post the render refuses waits for you.

A plan's stats post came back as a silent FAILED draft: the composer left the figures empty
(the topic gave none, and it never invents one), the render refused the missing fields, and
the plan maker turned that refusal into "A planned post was not made", linked to the plan,
with nothing in the post's history saying why. Now the refusal leaves the post a draft that
waits for the owner (``service.waits_for_input``). On the plan maker's harness
(``tests/test_prd251bw2_maker.py``: SQLite, the composer faked), with the REAL render route:

* (a) a stats template whose figures the proposal leaves empty: the slot ends ``waits_for_input``,
  the post stays a draft, its history names the fields in the owner's words and the composer's
  questions, the owner is told once, linked to the post, and no "not made" notice goes out;
* (b) a template whose required photo is empty waits the same way, saying the render's words;
* (c) a proposal that fills every field and binds a figure to the topic's knowledge fact is
  rendered once and made, the figure stored as a claim with its document;
* (d) any other failure of the render (P256-FIX-SOC-5 replaced "a failed slot, told as 'not made'",
  which was false for a made draft): the draft waits for the owner too, saying a fixed line and
  never the exception's text (``plan_refusals``; the platform's own refusals are
  ``tests/test_f437_a_a_plan_post_that_cannot_render_says_why.py``).
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials as socials_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.core import Document  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import plan_notify, plan_refusals, plan_store, render, service  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_maker import EVERY_DAY, NOW, _made, _make, _text_plan, _topic  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)

api = api_harness.api
KEY = "r1|2026-10-14|09:00"
SKIPPED_EVENT = "social_plan_slot_skipped"
QUESTION = "What were this month's two figures?"
KNOWLEDGE_FACT = {"text": "We roasted 1,240 bags in September.", "source": {"kind": "knowledge", "ref": "7", "label": "Roast log"}}


def starter_blocks(slug):
    """A starter template's checked composition (``modules/documents/templates/social``)."""
    blocks = next(s for s in social_starters.social_starters() if s["slug"] == slug)["blocks"]
    social_starters._starters.cache_clear()
    return blocks


STATS = starter_blocks("stats-card")  # required: headline, Figure 1 and 2 with what each means
PHOTO_ONLY = starter_blocks("photo-only")  # a required photo, no required field


def image_template(env, blocks):
    """A social image template of the workspace with ``blocks``."""
    template_id = uuid.uuid4()
    env.session.execute(
        sa.text(
            "INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks, is_active, version) "
            "VALUES (:id, :ws, 'Stats', 'social_image', '{}', :blocks, 1, 1)"
        ),
        {"id": template_id.hex, "ws": WS_A.hex, "blocks": json.dumps(blocks)},
    )
    env.session.commit()
    return template_id


def allow_templates(monkeypatch, *template_ids):
    """A plan row may name these image templates; the render reads no brand files; the
    waiting notices are kept, and returned."""
    monkeypatch.setattr(plan_store, "templates_of", lambda db, ws: {str(t): ("social_image", []) for t in template_ids})
    monkeypatch.setattr(socials_api, "_render_brand_kit", lambda settings: {})
    told = []
    monkeypatch.setattr(plan_notify, "notify_post_waiting",
                        lambda ws, post_id, title, labels, why: told.append((str(post_id), list(labels), why)))
    return told


def _image_plan(env, monkeypatch, blocks, facts_topic=None):
    template_id = image_template(env, blocks)
    env.waiting = allow_templates(monkeypatch, template_id)
    row = {"channels": ["linkedin"], "format": "image", "template_id": str(template_id), "days": EVERY_DAY, "time": "09:00"}
    plan = _text_plan(env, cadence=[row])
    if facts_topic is None:
        _topic(env, plan, "September in numbers")
    else:
        resp = env.client.post(f"/api/socials/plans/{plan['id']}/topics", json=facts_topic)
        assert resp.status_code == 201, resp.text
    return plan


def _proposing(monkeypatch, variables, *, bind=None, questions=()):
    """The composer's answer: ``variables``; ``bind`` a field to the topic's first fact."""
    def propose(db, plan, slot, topic, now, visual_slots=()):
        found = maker_mod.fact_candidates(topic, now)
        sources = {bind: {k: found[0][k] for k in ("kind", "ref", "as_of")}} if bind else {}
        return {"title": f"Post: {topic.title}", "copy": {"base": "Our month.", "channels": {}}, "variables": variables,
                "sources": sources, "questions": list(questions), "warnings": []}

    monkeypatch.setattr(maker_mod, "_propose", propose)


def _skipped(env):
    return [title for event, title in env.told.plan if event == SKIPPED_EVENT]


def test_a_a_stats_post_the_topic_gave_no_figures_waits_for_you(maker, monkeypatch):  # noqa: F811
    plan = _image_plan(maker, monkeypatch, STATS)
    _proposing(monkeypatch, {"headline": {"value": "Our roast in numbers"}}, questions=[QUESTION])

    assert _make(plan, KEY) == maker_mod.WAITS

    post = _made(maker, KEY)
    labels = ["Figure 1", "Figure 1 meaning", "Figure 2", "Figure 2 meaning"]
    entry = post.review_log[-1]
    assert post.status == "draft" and service.waits_for_input(post)
    assert (entry["action"], entry["fields"], entry["questions"]) == ("needs_input", labels, [QUESTION])
    assert entry["comment"] == f"Fill in {', '.join(labels)} before rendering."
    assert maker.waiting == [(str(post.id), labels, "fill in stat_1_value, stat_1_label, stat_2_value, stat_2_label before rendering")]
    assert _skipped(maker) == [] and maker.told.approval == []
    assert maker_mod.collect_due(NOW) == []  # the slot stays the post's own


def test_b_a_template_whose_required_photo_is_empty_waits_saying_the_renders_words(maker, monkeypatch):  # noqa: F811
    plan = _image_plan(maker, monkeypatch, PHOTO_ONLY)
    _proposing(monkeypatch, {})

    assert _make(plan, KEY) == maker_mod.WAITS

    post = _made(maker, KEY)
    entry = post.review_log[-1]
    assert post.status == "draft" and entry["action"] == "needs_input" and entry["fields"] == ["Photo"]
    assert entry["comment"] == render.NEEDS_PHOTO_MESSAGE.format(label="Photo")
    assert [(post_id, labels) for post_id, labels, _why in maker.waiting] == [(str(post.id), ["Photo"])]
    assert _skipped(maker) == []


@pytest.fixture
def documents(maker):  # noqa: F811
    """The knowledge base's documents table, with the roast log (document 7) of the workspace."""
    table = api_harness._sqlite_copy(Document.__table__, sa.MetaData())
    table.create(maker.session.get_bind())
    maker.session.execute(table.insert().values(id=7, workspace_id=WS_A, filename="7-roast-log.pdf",
                                                original_filename="roast-log.pdf", status="processed"))
    maker.session.commit()
    return maker


def test_c_a_post_with_every_field_filled_is_rendered_with_its_figure_bound(documents, monkeypatch):
    plan = _image_plan(documents, monkeypatch, STATS, facts_topic={"title": "September", "facts": [KNOWLEDGE_FACT]})
    filled = {"headline": "Our roast in numbers", "stat_1_value": "1,240", "stat_1_label": "bags roasted",
              "stat_2_value": "18", "stat_2_label": "origins"}
    _proposing(monkeypatch, {name: {"value": value} for name, value in filled.items()}, bind="stat_1_value")
    rendered = []

    async def render_post(db, workspace, post, actor):
        rendered.append(post.id)

    monkeypatch.setattr(socials_api, "render_post", render_post)

    assert _make(plan, KEY) == maker_mod.MADE

    post = _made(documents, KEY)
    assert rendered == [post.id] and documents.waiting == []
    assert post.sources == {"stat_1_value": {"kind": "document", "ref": "7", "as_of": NOW.isoformat()}}
    assert post.variables["stat_1_value"] == {"value": "1,240", "claim": True}  # F433: the template's claim, marked


def test_d_any_other_render_failure_waits_for_you_with_a_fixed_line(maker, monkeypatch):  # noqa: F811
    plan = _image_plan(maker, monkeypatch, STATS)
    _proposing(monkeypatch, {})

    async def renderer_down(db, workspace, post, actor):
        raise RuntimeError("the renderer is down")

    monkeypatch.setattr(socials_api, "render_post", renderer_down)

    assert _make(plan, KEY) == maker_mod.WAITS

    post = _made(maker, KEY)
    assert post.status == "draft" and service.waits_for_input(post)
    assert post.review_log[-1]["comment"] == plan_refusals.COULD_NOT_WRITE  # never "the renderer is down"
    assert _skipped(maker) == []  # "not made" was false: the draft is made
    assert maker.waiting == [(str(post.id), [], plan_refusals.COULD_NOT_WRITE)]
