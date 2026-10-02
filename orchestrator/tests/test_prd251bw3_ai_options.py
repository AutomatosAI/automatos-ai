"""PRD-251B Wave 3, US-B305 — AI-made visuals in the editor's Look.

On the S0.3b API harness (SQLite) with the data-story starter (three still slots, video
slots around them), the toolkit faked. Pinned:

* asking for options of an image slot answers 202 at once: the slot says ``making`` and
  four shots are planned through the workspace's AI images default (fal here), each its
  own file, the brand kit's style after the prompt and the liked reference with it;
* the background work records the four options (or why none were made) only while the
  slot still asks for that prompt; a pick makes the option the slot's file exactly as a
  render would (``done``), so the next render reuses it and makes nothing more;
* it is a render setting: an approved post keeps its approval and hash throughout;
* refused: a video slot or one taking the workspace's own file (422), a post with no
  template (422), no AI image toolkit (422, saying why), a post that can no longer change
  (409), an unknown option (404), and a viewer (403).
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_brand as socials_brand  # noqa: E402
import api.socials_media_tools as media_tools_api  # noqa: E402
import core.utils.background_tasks as background_tasks  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core.models.socials import SocialPost  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import ai_options, render  # noqa: E402
from modules.socials.recipes import footage as footage_recipes  # noqa: E402
from tests.test_prd251_api import WS_A, _approved, _create  # noqa: E402
from tests.test_prd251bw3_media_tools import BOTH, _caps  # noqa: E402

api = api_harness.api
PROMPT = "A harbour at first light, boats moored"
REFERENCE = "https://signed/documents/ref.png"


async def _nothing():
    return None


@pytest.fixture
def options(api, monkeypatch):
    starter = next(s for s in social_starters.social_starters() if s["slug"] == "data-story")
    template_id = uuid.uuid4()
    api.session.execute(
        sa.text("INSERT INTO document_templates (id, workspace_id, name, format, data_schema, blocks, is_active) "
                "VALUES (:id, :ws, 'Data story', 'social_video', '{}', :blocks, 1)"),
        {"id": template_id.hex, "ws": WS_A.hex, "blocks": json.dumps(starter["blocks"])},
    )
    api.session.commit()
    social_starters._starters.cache_clear()
    api.template_id, api.blocks, api.made, api.started = template_id, starter["blocks"], [], []
    monkeypatch.setattr(media_tools_api, "media_capabilities", lambda db, ws: BOTH)
    monkeypatch.setattr(socials_brand, "liked_reference_links", lambda settings: (REFERENCE,))
    monkeypatch.setattr(media_tools_api, "_make_options", lambda *args: (api.made.append(args), _nothing())[1])

    def launch(work, **kwargs):
        api.started.append(kwargs)
        work.close()

    monkeypatch.setattr(background_tasks, "launch_guarded", launch)
    return api


def _video_post(api, **body):
    return _create(api, template_id=str(api.template_id), format="video", **body)


def _ask(api, post_id, slot="still_1", prompt=PROMPT):
    return api.client.post(f"/api/socials/posts/{post_id}/ai-options", json={"slot": slot, "prompt": prompt})


def _footage(api, post_id):
    api.session.expire_all()
    return api.session.get(SocialPost, uuid.UUID(post_id)).footage


def _made(slot, n, toolkit="fal_ai"):
    name = f"footage-{slot}-{n:08x}.png"
    return footage_recipes.Made(slot=slot, path="assets/slots/still_1.png", key=f"k/{name}", name=name, toolkit=toolkit,
                                model="fal-ai/flux-pro/v1.1-ultra", bytes=1000 + n, sha256=f"{n:064x}", estimate_usd=0.06,
                                deliverable_id=f"d-{n}")


def test_asking_answers_at_once_and_plans_four_styled_shots(options):
    options.session.execute(sa.text("UPDATE workspaces SET settings = :s WHERE id = :id"), {
        "id": WS_A.hex, "s": json.dumps({"socials": {"enabled": True}, "brand_style": {"profile": {"mood": ["calm"]}}}),
    })
    options.session.commit()
    post = _video_post(options)
    resp = _ask(options, post["id"])
    assert resp.status_code == 202, resp.text
    assert resp.json()["footage"]["still_1"] == {"prompt": PROMPT, "options": [], "options_state": "making"}
    ((plan, workspace_id, post_id, title, slot, prompt),) = options.made
    assert (workspace_id, str(post_id), slot, prompt) == (WS_A, post["id"], "still_1", PROMPT)
    shots = [shot for shot, _route in plan.shots]
    assert [shot.slot for shot in shots] == [f"still_1_option_{n}" for n in range(1, ai_options.OPTIONS + 1)]
    assert {(shot.kind, shot.prompt, shot.references) for shot in shots} == {("image", PROMPT, (REFERENCE,))}
    assert shots[0].style == "Brand style (from the brand kit's references): mood: calm."
    assert {route.recipe.toolkit for _shot, route in plan.shots} == {"fal_ai"}
    assert options.started == [{"subsystem": "socials", "operation": "ai_options", "workspace_id": WS_A}]


def test_the_options_are_recorded_then_one_is_picked_as_the_slots_file(options, monkeypatch):
    post = _video_post(options)
    assert _ask(options, post["id"]).status_code == 202
    factory = sessionmaker(bind=options.session.get_bind())
    monkeypatch.setattr(media_tools_api, "SessionLocal", factory)

    async def generate(plan, **kwargs):
        return {shot.slot: _made(shot.slot, n) for n, (shot, _route) in enumerate(plan.shots, start=1)}

    monkeypatch.setattr(footage_recipes, "generate", generate)
    ((plan, *_rest),) = options.made
    records, error = asyncio.run(ai_options.make(plan, workspace_id=WS_A, post_id=uuid.UUID(post["id"]), title="T",
                                                 slot="still_1", prompt=PROMPT, session_factory=factory, store=None))
    assert error is None and len(records) == ai_options.OPTIONS
    media_tools_api._settle(WS_A, uuid.UUID(post["id"]), "still_1", PROMPT, records, None)
    asked = _footage(options, post["id"])["still_1"]
    assert asked["options_state"] == "ready" and [o["deliverable_id"] for o in asked["options"]] == ["d-1", "d-2", "d-3", "d-4"]

    chosen = asked["options"][2]["name"]
    picked = options.client.put(f"/api/socials/posts/{post['id']}/ai-options/still_1", json={"name": chosen})
    assert picked.status_code == 200, picked.text
    record = _footage(options, post["id"])["still_1"]
    assert (record["status"], record["name"], record["prompt"], record["toolkit"]) == ("done", chosen, PROMPT, "fal_ai")
    assert "options" not in record and "options_state" not in record

    template = SimpleNamespace(format="social_video", blocks=options.blocks)
    next_render = render.footage_plan_for(SimpleNamespace(footage={"still_1": record}, length_seconds=None), template, BOTH)
    assert next_render.shots == () and [kept.slot for kept in next_render.kept] == ["still_1"]


def test_no_options_records_why_and_a_stale_answer_changes_nothing(options, monkeypatch):
    post = _video_post(options)
    assert _ask(options, post["id"]).status_code == 202
    monkeypatch.setattr(media_tools_api, "SessionLocal", sessionmaker(bind=options.session.get_bind()))

    async def refused(plan, **kwargs):
        raise footage_recipes.FootageRefused("this post has spent $10.00 of its $10.00 media cap: nothing was submitted.")

    monkeypatch.setattr(footage_recipes, "generate", refused)
    ((plan, *_rest),) = options.made
    records, error = asyncio.run(ai_options.make(plan, workspace_id=WS_A, post_id=uuid.UUID(post["id"]), title="T",
                                                 slot="still_1", prompt=PROMPT, session_factory=None, store=None))
    assert records == [] and "media cap" in error
    media_tools_api._settle(WS_A, uuid.UUID(post["id"]), "still_1", PROMPT, records, error)
    failed = _footage(options, post["id"])["still_1"]
    assert (failed["options_state"], failed["options_error"]) == ("failed", error)

    assert _ask(options, post["id"], prompt="A new idea").status_code == 202
    media_tools_api._settle(WS_A, uuid.UUID(post["id"]), "still_1", PROMPT, [{"name": "late.png"}], None)
    assert _footage(options, post["id"])["still_1"] == {"prompt": "A new idea", "options": [], "options_state": "making"}


def test_options_are_a_render_setting_an_approval_survives(options):
    post = _approved(options, template_id=str(options.template_id), format="video")
    resp = _ask(options, post["id"])
    assert resp.status_code == 202
    body = resp.json()
    assert (body["status"], body["content_hash"], body["approved_hash"]) == ("approved", post["content_hash"], post["content_hash"])


@pytest.mark.parametrize("slot, reason", [
    ("hook", "is not one of this template's image slots"),
    ("nowhere", "is not one of this template's image slots"),
])
def test_only_an_image_slot_of_the_template_takes_options(options, slot, reason):
    post = _video_post(options)
    resp = _ask(options, post["id"], slot=slot)
    assert resp.status_code == 422 and reason in resp.text
    assert options.made == [] and _footage(options, post["id"]) in (None, {})


def test_a_post_without_a_template_or_without_an_ai_image_tool_is_422(options, monkeypatch):
    bare = _create(options)
    assert _ask(options, bare["id"]).status_code == 422
    monkeypatch.setattr(media_tools_api, "media_capabilities", lambda db, ws: _caps())
    resp = _ask(options, _video_post(options)["id"])
    assert resp.status_code == 422 and "No AI image can be made" in resp.text
    assert options.made == []


def test_a_post_that_can_no_longer_change_is_409(options):
    post = _video_post(options)
    options.session.execute(sa.text("UPDATE social_posts SET status = 'published' WHERE id = :id"), {"id": uuid.UUID(post["id"]).hex})
    options.session.commit()
    assert _ask(options, post["id"]).status_code == 409
    assert options.client.put(f"/api/socials/posts/{post['id']}/ai-options/still_1", json={"name": "x.png"}).status_code == 409


def test_an_unknown_option_is_404_and_a_viewer_is_403(options):
    post = _video_post(options)
    assert options.client.put(f"/api/socials/posts/{post['id']}/ai-options/still_1", json={"name": "x.png"}).status_code == 404
    options.role = "viewer"
    assert _ask(options, post["id"]).status_code == 403


def test_the_template_summary_names_its_image_slots(options):
    from modules.socials import template_gallery

    assert template_gallery.image_slots(options.blocks) == ["still_1", "still_2", "still_3"]
    assert template_gallery.image_slots({}) == [] and template_gallery.image_slots(None) == []


def test_an_editor_echoing_the_stored_options_keeps_them(options):
    post = _video_post(options)
    assert _ask(options, post["id"]).status_code == 202
    stored = _footage(options, post["id"])
    echoed = options.client.patch(f"/api/socials/posts/{post['id']}", json={"footage": stored})
    assert echoed.status_code == 200, echoed.text
    assert _footage(options, post["id"])["still_1"] == {"prompt": PROMPT, "options": [], "options_state": "making"}
