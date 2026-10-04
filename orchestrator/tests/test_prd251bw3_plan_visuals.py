"""PRD-251B Wave 3, US-B305 with US-B207 — a plan's visual mix uses the AI-made path.

Pinned:

* each slot draws its visual from the plan's mix by its own key: always the same one for a
  slot, and the shares hold over many slots; templates alone never asks for anything;
* AI images or AI footage ask the template's own image or video slots an AI tool may fill
  (never one taking the workspace's own file), each with the composer's prompt for it, else
  the topic; the composer is asked for those prompts only, and its answer is trimmed to them;
* the library picks the image or video Deliverable that shares most words with the topic,
  as the editor's Library does (the post's media, no template); nothing fitting leaves the
  template's visuals;
* the plan maker writes the post with that visual, then renders it (a template) or sends it
  for approval (a library file);
* the plan's research reads the brand kit's style profile with the plan.
"""
from __future__ import annotations

import asyncio
import functools
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import sqlalchemy as sa

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_targets as socials_targets  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
import tests.test_prd251bw2_topics as topics_harness  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import compose, compose_checks, plan_visuals, plans  # noqa: E402
from modules.tools.discovery import handlers_socials  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_prd251_api import WS_A  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402

api = api_harness.api
bank = topics_harness.bank
NOW = datetime(2026, 10, 14, 7, 30, tzinfo=timezone.utc)
TEMPLATE_ID = uuid.uuid4()


def _data_story_blocks():
    starter = next(s for s in social_starters.social_starters() if s["slug"] == "data-story")
    social_starters._starters.cache_clear()
    return starter["blocks"]


BLOCKS = _data_story_blocks()


# ── the draw ───────────────────────────────────────────────────────────────


def test_a_slot_always_draws_the_same_visual_and_the_shares_hold():
    mix = {"templates": 50, "library": 10, "ai_images": 30, "ai_footage": 10}
    key = "r1|2026-10-14|09:00"
    assert len({plan_visuals.visual_for(mix, key) for _ in range(5)}) == 1
    drawn = [plan_visuals.visual_for(mix, f"r1|2026-{month:02d}-{day:02d}|09:00") for month in range(1, 13) for day in range(1, 29)]
    for visual, share in mix.items():
        assert abs(drawn.count(visual) / len(drawn) * 100 - share) < 6, visual
    assert {plan_visuals.visual_for({"templates": 100}, f"k{n}") for n in range(50)} == {"templates"}


def test_the_ai_visuals_ask_the_templates_own_slots_a_tool_may_fill():
    stills = plan_visuals.ai_slots(BLOCKS, "ai_images")
    assert [slot["slot"] for slot in stills] == ["still_1", "still_2", "still_3"]
    assert {slot["kind"] for slot in stills} == {"image"}
    footage = [slot["slot"] for slot in plan_visuals.ai_slots(BLOCKS, "ai_footage")]
    assert footage == ["hook", "tide", "end"]
    assert plan_visuals.ai_slots(BLOCKS, "templates") == [] and plan_visuals.ai_slots(None, "ai_images") == []
    app = next(s for s in social_starters.social_starters() if s["slug"] == "app-promo")
    assert "app_loop" not in [slot["slot"] for slot in plan_visuals.ai_slots(app["blocks"], "ai_footage")]  # its own file
    social_starters._starters.cache_clear()


def test_each_slot_takes_the_composers_prompt_else_the_topics():
    slots = plan_visuals.ai_slots(BLOCKS, "ai_images")
    asks = plan_visuals.footage_asks(slots, {"still_1": " A harbour at dawn "}, plan_visuals.topic_prompt("Lisbon", "Why visit"))
    assert asks == {"still_1": {"prompt": "A harbour at dawn"}, "still_2": {"prompt": "Lisbon Why visit"},
                    "still_3": {"prompt": "Lisbon Why visit"}}


def test_the_library_picks_what_shares_most_words_with_the_topic():
    items = [
        {"id": "newest", "title": "Team photo", "summary": None},
        {"id": "stand", "title": "Lisbon stand render", "summary": "Web Summit booth"},
        {"id": "stand-2", "title": "Booth plan", "summary": "Lisbon"},
    ]
    assert plan_visuals.best_fit(items, "Three weeks to Lisbon", "Visit our booth")["id"] == "stand"
    assert plan_visuals.best_fit(items, "Quarterly numbers", None) is None


# ── the composer ───────────────────────────────────────────────────────────


def _ctx(**overrides):
    base = dict(brief="Three weeks to Lisbon", format="video", channels=[], templates=[], candidates=[])
    return compose.ComposeContext(**{**base, **overrides})


def test_the_composer_is_asked_for_the_slots_prompts_only_when_there_are_slots():
    slots = tuple(plan_visuals.ai_slots(BLOCKS, "ai_images"))
    system, user = compose.build_messages(_ctx(visual_slots=slots))
    assert compose.VISUAL_PROMPTS_NOTE in system["content"]
    assert [slot["slot"] for slot in json.loads(user["content"])["visual_slots"]] == ["still_1", "still_2", "still_3"]
    bare_system, bare_user = compose.build_messages(_ctx())
    assert compose.VISUAL_PROMPTS_NOTE not in bare_system["content"] and "visual_slots" not in json.loads(bare_user["content"])


def test_the_answers_prompts_are_kept_for_the_asked_slots_only():
    ctx = _ctx(visual_slots=({"slot": "still_1", "kind": "image", "label": "Still"},))
    raw = {"visual_prompts": {"still_1": "  A harbour \n at dawn " + "x" * 700, "hook": "not asked", "still_2": 7}}
    prompts = compose_checks.checked_proposal(raw, ctx)["visual_prompts"]
    assert list(prompts) == ["still_1"] and prompts["still_1"].startswith("A harbour at dawn x")
    assert len(prompts["still_1"]) == compose_checks.VISUAL_PROMPT_MAX_CHARS
    assert compose_checks.checked_proposal(raw, _ctx())["visual_prompts"] == {}


# ── the plan maker ─────────────────────────────────────────────────────────


TOPIC = SimpleNamespace(title="Three weeks to Lisbon", angle="Visit the stand")


def _db(rows=()):
    db = MagicMock()
    db.get.return_value = SimpleNamespace(workspace_id=WS_A, format="social_video", blocks=BLOCKS)
    db.execute.return_value.mappings.return_value.all.return_value = list(rows)
    return db


def _plan(mix):
    return SimpleNamespace(workspace_id=WS_A, created_by="owner-1", make={"visual_mix": mix})


def _slot(fmt="video", template_id=TEMPLATE_ID, visual_source=None, visual_toolkit=None):
    return SimpleNamespace(key="r1|2026-10-14|09:00", format=fmt, template_id=str(template_id) if template_id else None,
                           channels=("linkedin",), length_seconds=None, kind=None,
                           visual_source=visual_source, visual_toolkit=visual_toolkit)  # PRD-251C: a row's own visual


def test_the_visual_changes_ask_ai_slots_or_set_a_library_file():
    proposal = {"template_id": str(TEMPLATE_ID), "visual_prompts": {"still_2": "Boats at the quay"}}
    asked = maker_mod._visual_changes(_db(), _plan({}), _slot(), TOPIC, proposal, "ai_images")
    assert asked["footage"]["still_2"] == {"prompt": "Boats at the quay"}
    assert asked["footage"]["still_1"] == {"prompt": "Three weeks to Lisbon Visit the stand"}
    library = maker_mod._visual_changes(_db([{"id": "d-1", "title": "Lisbon stand", "summary": ""}]), _plan({}), _slot(),
                                        TOPIC, proposal, "library")
    assert library == {"media": {"original": ["d-1"]}, "template_id": None, "length_seconds": None, "format": "video"}
    assert maker_mod._visual_changes(_db(), _plan({}), _slot(), TOPIC, proposal, "library") == {}  # nothing fits
    assert maker_mod._visual_changes(_db(), _plan({}), _slot(), TOPIC, proposal, "templates") == {}


def test_another_workspaces_template_gives_no_slots():
    db = _db()
    db.get.return_value = SimpleNamespace(workspace_id=uuid.uuid4(), format="social_video", blocks=BLOCKS)
    assert maker_mod.template_blocks(db, WS_A, TEMPLATE_ID) is None
    assert maker_mod.template_blocks(db, WS_A, None) is None


def test_the_library_reads_only_the_workspaces_live_files_of_the_kind():
    db = _db([])
    maker_mod.library_media(db, WS_A, "fact_card", TOPIC)
    (statement, params), _kwargs = db.execute.call_args
    assert params == {"workspace_id": WS_A, "kind": "image", "limit": maker_mod.LIBRARY_CANDIDATES}
    assert "deleted_at IS NULL" in str(statement) and "workspace_id = :workspace_id" in str(statement)


class _PostsApi:
    def __init__(self):
        self.edits, self.renders, self.submits = [], [], []

    async def edit_post(self, db, post, actor, changes, *, agent=None):
        self.edits.append(changes)
        if "template_id" in changes:
            post.template_id = changes["template_id"]

    async def render_post(self, db, workspace, post, actor):
        self.renders.append(post.id)

    def submit_post(self, db, post, actor, *, note=None):
        self.submits.append((post.id, note))


def _write(monkeypatch, mix, rows=(), visual_prompts=None):
    posts_api, asked = _PostsApi(), []

    def propose(db, plan, slot, topic, now, visual_slots=()):
        asked.append(tuple(s["slot"] for s in visual_slots))
        return {"title": "Lisbon", "copy": {"base": "See you there."}, "variables": {}, "sources": {},
                "template_id": str(TEMPLATE_ID), "visual_prompts": visual_prompts or {}}

    monkeypatch.setattr(maker_mod, "_posts_api", lambda: posts_api)
    monkeypatch.setattr(maker_mod, "_propose", propose)
    monkeypatch.setattr(maker_mod, "slot_targets", lambda db, ws, slot: [])
    monkeypatch.setattr(socials_targets, "set_post_targets", lambda db, post, actor, targets, agent=None: None)
    post = SimpleNamespace(id=uuid.uuid4(), template_id=TEMPLATE_ID)
    run = functools.partial(maker_mod.write, _db(rows), SimpleNamespace(id=WS_A), _plan(mix), _slot(), TOPIC, post, NOW)
    anyio.run(functools.partial(anyio.to_thread.run_sync, run))
    return posts_api, asked, post


def test_the_maker_writes_ai_images_into_the_slots_then_renders(monkeypatch):
    posts_api, asked, post = _write(monkeypatch, {"ai_images": 100}, visual_prompts={"still_1": "A harbour at dawn"})
    assert asked == [("still_1", "still_2", "still_3")]  # the composer wrote their prompts
    (changes,) = posts_api.edits
    assert changes["footage"]["still_1"] == {"prompt": "A harbour at dawn"} and set(changes["footage"]) == {"still_1", "still_2", "still_3"}
    assert posts_api.renders == [post.id] and posts_api.submits == []


def test_the_maker_sends_a_library_post_for_approval_without_a_render(monkeypatch):
    posts_api, asked, post = _write(monkeypatch, {"library": 100}, rows=[{"id": "d-9", "title": "Lisbon stand", "summary": ""}])
    assert asked == [()]
    (changes,) = posts_api.edits
    assert changes["media"] == {"original": ["d-9"]} and changes["template_id"] is None
    assert posts_api.renders == [] and [post_id for post_id, _note in posts_api.submits] == [post.id]


def test_templates_alone_change_nothing_about_the_visual(monkeypatch):
    posts_api, asked, post = _write(monkeypatch, {"templates": 100})
    (changes,) = posts_api.edits
    assert "footage" not in changes and "media" not in changes and asked == [()]
    assert posts_api.renders == [post.id]


def test_the_make_settings_carry_the_mix():
    assert plans.make_settings(_plan({"ai_footage": 40, "templates": 60}))["visual_mix"] == {"ai_footage": 40, "templates": 60}


# ── the research reads the style ───────────────────────────────────────────


def test_the_plans_research_reads_the_brand_kits_style(bank):
    plan = _create_plan(bank)
    bank.session.execute(sa.text("UPDATE workspaces SET settings = :s WHERE id = :id"), {
        "id": WS_A.hex, "s": json.dumps({"socials": {"enabled": True}, "brand_style": {"profile": {"mood": ["bold"]}}}),
    })
    bank.session.commit()
    read = asyncio.run(handlers_socials.get_social_plan(bank.session, WS_A, {"plan_id": plan["id"]}))
    assert read["success"] is True and read["brand_style"] == "Brand style (from the brand kit's references): mood: bold."
