"""P256-FIX-SOC-5 (F437, socials day 14 Oct): a plan's post that cannot render says why, safely.

A plan's video draft never got media and recorded no reason: the render was refused before it
started, the plan maker's catch-all logged it, rolled back (the composer's edit was already
committed) and sent "A planned post was not made", which was false, without the reason. The post
stayed a draft with nothing in its history. Now the reason decides (``plan_refusals.held_as``,
``socials_plan_maker._held``). On the plan maker's harness, a weekly plan whose image row has a
template, the render route faked to refuse, pinned:

* missing fields keep the post a draft that waits for you (SOC-2's ``needs_input``), naming the
  fields; the owner is told, and the week's notice ends "1 waits for you";
* a renderer outage fails the post with a fixed, safe reason: the renderer's own words (an
  address) reach no history entry and no notice; a quota refusal fails it with the quota's text;
  the week's notice says "1 could not render";
* a composer failure leaves a waiting draft with a fixed line, never the exception's text;
* a render another writer already started is left alone;
* the readers show the reason: the post list's row, the ticket check, and the post's ``render``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials as socials_api  # noqa: E402
import tests.test_prd251_api as api_harness  # noqa: E402
from core import media_render_quota as render_quota  # noqa: E402
from modules.socials import plan_refusals, render, service  # noqa: E402
from modules.tools.discovery import social_post_list  # noqa: E402
from services import social_ticket_post  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_f433_plan_post_waits_for_input import STATS, allow_templates, image_template  # noqa: E402
from tests.test_prd251bw2_maker import _topic as _maker_topic  # noqa: E402
from tests.test_prd251bw2_maker import maker  # noqa: E402,F401  (the fixture)
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251cw2_batches import SUNDAY_EVENING, _posts, _tick, week  # noqa: E402,F401  (the fixture)

api = api_harness.api
FILL_IN = "fill in item_1_name, item_1_value before rendering"
OUTAGE = "The renderer is not healthy: ConnectError 10.0.0.5:8080"
QUOTA = "This workspace has used 10.0 of its 10 render minutes this month on the Basic plan. Rendering resumes on 1 November."
SECRET = "password=hunter2"


def _image_week(env, monkeypatch, name="Countdown"):
    """A weekly plan of one image post, on Tuesday 20 Oct, through a template; the waiting notices kept."""
    template_id = image_template(env, STATS)
    env.waiting = allow_templates(monkeypatch, template_id)
    row = {"channels": ["linkedin"], "format": "image", "template_id": str(template_id), "days": ["tue"], "time": "09:00"}
    plan = _create_plan(env, name=name, timezone="UTC", starts_on="2026-10-19", ends_on="2026-10-25", cadence=[row],
                        make={"rhythm": "weekly", "batch_day": "sun", "time": "17:00"})
    _maker_topic(env, plan, "Our roast in numbers")
    return plan


def _render_raises(monkeypatch, exc):
    async def render_post(db, workspace, post, actor):
        raise exc

    monkeypatch.setattr(socials_api, "render_post", render_post)


def _made_with(env, monkeypatch, exc):
    plan = _image_week(env, monkeypatch)
    _render_raises(monkeypatch, exc)
    outcomes = _tick(SUNDAY_EVENING)
    (post,) = _posts(env, plan)
    return post, outcomes


def _everything_said(env, post):
    """Every history entry and every notice, as text."""
    return json.dumps([post.review_log, env.waiting, env.told.plan, env.reviews], default=str)


def test_missing_fields_keep_the_post_waiting_and_say_what_to_fill(week, monkeypatch):  # noqa: F811
    post, outcomes = _made_with(week, monkeypatch, render.NotRenderable(FILL_IN))

    assert outcomes.get(maker_mod.WAITS) == 1
    assert post.status == "draft" and service.waits_for_input(post)
    assert (post.review_log[-1]["action"], post.review_log[-1]["comment"]) == (service.ACTION_NEEDS_INPUT, FILL_IN)
    assert week.waiting[-1] == (str(post.id), [], FILL_IN)  # SOC-2's notice, linked to the post
    assert not [event for event, _title in week.told.plan if event == "social_plan_slot_skipped"]
    (review,) = week.reviews
    assert review[1].endswith("; 1 waits for you")


def test_a_renderer_outage_fails_the_post_with_a_safe_reason(week, monkeypatch):  # noqa: F811
    post, outcomes = _made_with(week, monkeypatch, render.RendererUnavailable(OUTAGE))

    assert outcomes.get(maker_mod.FAILED) == 1
    assert post.status == "failed"
    assert (post.review_log[-1]["action"], post.review_log[-1]["comment"]) == (service.ACTION_RENDER_FAILED, render.UNREACHABLE_MESSAGE)
    assert "10.0.0.5" not in _everything_said(week, post) and "not healthy" not in _everything_said(week, post)
    assert week.waiting[-1] == (str(post.id), [], render.UNREACHABLE_MESSAGE)
    assert week.reviews == [("Your week is ready: ", "Countdown: 0 posts for the week of 19 Oct; 1 could not render")]


def test_quota_refusal_fails_with_the_quota_text(week, monkeypatch):  # noqa: F811
    post, _outcomes = _made_with(week, monkeypatch, render_quota.RenderQuotaExceeded(QUOTA))

    assert post.status == "failed" and post.review_log[-1]["comment"] == QUOTA


@pytest.mark.parametrize("fixed", [render.STORAGE_MESSAGE, render.NOT_CONFIGURED_MESSAGE, render.MEDIA_PROFILE_MESSAGE])
def test_a_renderers_fixed_messages_pass_and_anything_else_is_the_unreachable_one(fixed):
    assert plan_refusals.renderer_text(render.RendererUnavailable(fixed)) == fixed
    assert plan_refusals.renderer_text(render.RendererUnavailable(OUTAGE)) == render.UNREACHABLE_MESSAGE


def test_a_compose_failure_logs_a_fixed_line_without_internals(week, monkeypatch):  # noqa: F811
    plan = _image_week(week, monkeypatch)

    def propose(db, plan, slot, topic, now, visual_slots=()):
        raise RuntimeError(SECRET)

    monkeypatch.setattr(maker_mod, "_propose", propose)

    outcomes = _tick(SUNDAY_EVENING)

    (post,) = _posts(week, plan)
    assert outcomes.get(maker_mod.WAITS) == 1 and post.status == "draft" and service.waits_for_input(post)
    assert post.review_log[-1]["comment"] == plan_refusals.COULD_NOT_WRITE
    assert "hunter2" not in _everything_said(week, post)


def test_a_render_already_started_elsewhere_is_left_alone(week, monkeypatch):  # noqa: F811
    post, outcomes = _made_with(week, monkeypatch, service.IllegalTransition("rendering", "render"))

    assert outcomes.get(maker_mod.TAKEN) == 1 and post.status == "draft"
    assert post.review_log[-1]["action"] not in (service.ACTION_NEEDS_INPUT, service.ACTION_RENDER_FAILED)
    assert week.waiting == [] and week.told.plan == []


def test_the_readers_show_the_reason(week, monkeypatch):  # noqa: F811
    waiting, _ = _made_with(week, monkeypatch, render.NotRenderable(FILL_IN))
    assert social_post_list.what_it_needs(waiting.to_dict()) == [f"your input: {FILL_IN}"]
    assert waiting.to_dict()["render"] == {"state": "waiting", "started_at": None, "gives_up_at": None, "finished_at": None, "why": FILL_IN}

    failed_plan = _image_week(week, monkeypatch, name="Second plan")
    _render_raises(monkeypatch, render.RendererUnavailable(OUTAGE))
    _tick(SUNDAY_EVENING)
    (failed,) = _posts(week, failed_plan)

    assert social_post_list.what_it_needs(failed.to_dict()) == [f"its render failed: {render.UNREACHABLE_MESSAGE}"]
    assert social_ticket_post.render_state(failed) == (social_ticket_post.FAILED_RENDER, render.UNREACHABLE_MESSAGE)
    assert {key: failed.to_dict()["render"][key] for key in ("state", "started_at", "why")} == {
        "state": "failed", "started_at": None, "why": render.UNREACHABLE_MESSAGE,
    }
