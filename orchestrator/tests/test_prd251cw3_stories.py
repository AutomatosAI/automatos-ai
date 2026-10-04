"""PRD-251C Wave 3, US-C301 — stories.

Pinned:

* **The recipe.** Instagram's ``story``: the account, a STORIES container from the post's one
  file (an image as a JPEG, or a video), the wait until Instagram has processed it, the
  publish and the link; a story carries no caption.
* **Published (mocked Composio, the US-303 harness).** An image story sends only
  ``image_file``, as a JPEG; a video story sends only ``video_file``, unconverted; a story
  with no file fails saying it needs an image or a video, and publishes nothing.
* **A story row.** ``cadence[].kind`` is ``story`` on an image or a video row and refused
  elsewhere; its slots carry the kind, and the maker gives the slot's Instagram the story
  kind and leaves out a channel that posts no stories.
* **The safe zone.** A story's 9:16 render moves the template's page clear of Instagram's
  bars, scaled to the size; another size, or a post with no story, renders as authored.
"""
from __future__ import annotations

import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251_api as api_harness  # noqa: E402
import tests.test_prd251w3_publisher as harness  # noqa: E402
from core.media_render_bundle import story_safe_css  # noqa: E402
from core.models.socials import SocialCampaign  # noqa: E402
from core.social_templates import SOCIAL_IMAGE  # noqa: E402
from modules.documents import social_starters  # noqa: E402
from modules.socials import plans, render  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS, ChannelKind, SocialChannel  # noqa: E402
from modules.socials.media_urls import MediaFile  # noqa: E402
from services import socials_plan_maker as maker_mod  # noqa: E402
from tests.test_prd251bw2_plans import _create as _create_plan  # noqa: E402
from tests.test_prd251bw2_plans import bank  # noqa: E402,F401  (the fixture)
from tests.test_prd251w3_instagram import (  # noqa: E402
    ACCOUNT,
    INSTAGRAM,
    JPEG,
    LINK,
    MEDIA,
    PNG,
    REEL,
    ReadingExecutor,
    _receipt,
    _run,
    to_jpeg,
)
from tests.test_prd251w3_publisher import FakeExecutor, _approved_post, _target, _targets  # noqa: E402

env = harness.env
api = api_harness.api
STORY_STILL = MediaFile(aspect="9:16", deliverable_id=str(uuid.uuid4()), name="image-9x16.png", key="k/image-9x16.png",
                        content_type="image/png", bytes=len(PNG))
SEQUENCE = [
    "INSTAGRAM_GET_USER_INFO", "INSTAGRAM_POST_IG_USER_MEDIA", "INSTAGRAM_GET_IG_MEDIA", "INSTAGRAM_GET_IG_MEDIA",
    "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH", "INSTAGRAM_GET_IG_MEDIA",
]


# ── the recipe ─────────────────────────────────────────────────────────────


def test_the_story_recipe_makes_a_stories_container_from_one_file_waits_publishes_and_links():
    steps = SEEDED_CHANNELS["instagram"].kinds["story"]
    assert [(step.id, step.action) for step in steps] == [
        ("account", "INSTAGRAM_GET_USER_INFO"), ("container", "INSTAGRAM_POST_IG_USER_MEDIA"), ("ready", "INSTAGRAM_GET_IG_MEDIA"),
        ("publish", "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH"), ("link", "INSTAGRAM_GET_IG_MEDIA"),
    ]
    container = steps[1]
    assert container.params == {"ig_user_id": "$steps.account", "image_file": "$media.image", "video_file": "$media.video",
                                "media_type": "STORIES"}
    assert (container.files, container.jpeg) == (("image_file", "video_file"), ("image_file",))
    assert "caption" not in container.params


# ── published, with Composio mocked ────────────────────────────────────────


def test_an_image_story_sends_only_its_jpeg_to_a_stories_container(env):
    env.media = [STORY_STILL]
    post_id = _approved_post(env, _target("instagram", "story"))
    executor = ReadingExecutor(INSTAGRAM)

    ended, converted = _run(env, post_id, executor, to_jpeg)

    assert ended == "published" and converted == [PNG]
    assert executor.actions == SEQUENCE
    container = executor.calls[1]
    assert container.upload_params == ("image_file",)
    assert set(container.params) == {"ig_user_id", "image_file", "media_type"}
    assert (container.params["ig_user_id"], container.params["media_type"]) == (ACCOUNT, "STORIES")
    assert container.params["image_file"].suffix == ".jpg"
    assert executor.read["INSTAGRAM_POST_IG_USER_MEDIA", "image_file"] == JPEG
    assert _receipt(env, post_id, "story") == ("published", MEDIA, LINK)


def test_a_video_story_sends_only_its_video_unconverted(env):
    env.media = [REEL]
    post_id = _approved_post(env, _target("instagram", "story"))
    executor = FakeExecutor(INSTAGRAM)

    ended, converted = _run(env, post_id, executor, to_jpeg)

    assert ended == "published" and converted == []
    container = executor.calls[1]
    assert container.upload_params == ("video_file",)
    assert set(container.params) == {"ig_user_id", "video_file", "media_type"}
    assert container.params["video_file"].name == "video-9x16.mp4" and container.params["media_type"] == "STORIES"
    assert _receipt(env, post_id, "story") == ("published", MEDIA, LINK)


def test_a_story_with_no_file_fails_saying_so_and_publishes_nothing(env):
    env.media = []
    post_id = _approved_post(env, _target("instagram", "story"))
    executor = FakeExecutor(INSTAGRAM)

    ended, _ = _run(env, post_id, executor, to_jpeg)

    assert ended == "failed" and "INSTAGRAM_POST_IG_USER_MEDIA_PUBLISH" not in executor.actions
    error = _targets(env, post_id)["instagram", "story"]["error"]
    assert "needs an image or a video file for image_file or video_file" in error


# ── a story row ────────────────────────────────────────────────────────────


def _story_row(**over):
    return {"channels": ["instagram", "linkedin"], "format": "image", "kind": "story", "days": ["mon"], "time": "09:00", **over}


def test_a_story_row_is_kept_on_an_image_or_a_video_row_and_refused_elsewhere(bank):  # noqa: F811
    plan = _create_plan(bank, cadence=[_story_row(), _story_row(format="video")])
    assert [row["kind"] for row in plan["cadence"]] == ["story", "story"]
    for bad in (_story_row(format="carousel"), _story_row(format="text"), _story_row(kind="reel")):
        resp = bank.client.post("/api/socials/plans", json={**_plan_body(), "cadence": [bad]})
        assert resp.status_code == 422, resp.text
        assert "kind may be story, on an image or a video row" in resp.text


def _plan_body():
    return {"name": "Stories", "starts_on": "2026-10-12", "ends_on": "2026-11-08", "timezone": "UTC",
            "cadence": [_story_row()]}


def test_a_story_rows_slots_carry_the_kind(bank):  # noqa: F811
    plan = _create_plan(bank, timezone="UTC", cadence=[_story_row(), _story_row(kind=None, time="12:00")])
    stored = bank.session.get(SocialCampaign, uuid.UUID(plan["id"]))
    start = datetime(2026, 10, 12, tzinfo=timezone.utc)
    slots = plans.expand_slots(stored, start, datetime(2026, 10, 13, tzinfo=timezone.utc))
    assert [(slot.local_time, slot.kind) for slot in slots] == [("09:00", "story"), ("12:00", None)]
    assert slots[0].to_dict()["kind"] == "story"


def _channel(toolkit, *kinds):
    post_kinds = tuple(ChannelKind(kind=k, available=True, reason=None, needs_public_storage=False, steps=()) for k in kinds)
    return SocialChannel(toolkit=toolkit, label=toolkit, post_kinds=post_kinds, verified=True, setup_note=None)


def _slot(kind=None, fmt="image"):
    return plans.Slot(key="r1|2026-10-12|09:00", row_id="r1", channels=("instagram", "linkedin"), format=fmt, length_seconds=None,
                      template_id=None, local_date=date(2026, 10, 12), local_time="09:00",
                      at=datetime(2026, 10, 12, 9, tzinfo=timezone.utc), kind=kind)


def test_the_maker_posts_a_story_slot_as_instagrams_story_and_leaves_out_a_channel_without_stories(monkeypatch):
    channels = [_channel("instagram", "image", "reel", "story"), _channel("linkedin", "text", "image", "video")]
    monkeypatch.setattr(maker_mod, "social_channels", lambda db, ws: channels)
    assert maker_mod.slot_targets(None, uuid.uuid4(), _slot("story")) == [{"toolkit": "instagram", "post_kind": "story", "options": {}}]
    assert maker_mod.slot_targets(None, uuid.uuid4(), _slot("story", fmt="video")) == [
        {"toolkit": "instagram", "post_kind": "story", "options": {}},
    ]
    assert [t["post_kind"] for t in maker_mod.slot_targets(None, uuid.uuid4(), _slot())] == ["image", "image"]
    monkeypatch.setattr(maker_mod, "social_channels", lambda db, ws: [_channel("instagram", "image", "reel")])
    assert maker_mod.slot_targets(None, uuid.uuid4(), _slot("story")) == []  # no story action: the slot is skipped


# ── the safe zone ──────────────────────────────────────────────────────────


def _starter(slug):
    return next(s for s in social_starters.social_starters() if s["slug"] == slug)


def _post(starter, *kinds):
    values = {name: {"value": value} for name, value in starter["sample_data"].items()}
    targets = [SimpleNamespace(toolkit="instagram", post_kind=kind) for kind in kinds]
    return SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=values, length_seconds=None, footage=None,
                           targets=targets)


def test_the_safe_zone_is_scaled_to_the_size_and_only_at_9_16():
    assert story_safe_css(1080, 1920).strip().endswith(".page { top: 250px !important; bottom: 340px !important; }")
    assert ".page { top: 125px !important; bottom: 170px !important; }" in story_safe_css(540, 960)
    assert story_safe_css(1080, 1350) == "" and story_safe_css(1080, 1080) == "" and story_safe_css(0, 0) == ""


@pytest.mark.parametrize("slug", ["title-card", "photo-headline"])
def test_a_story_renders_its_9_16_size_inside_the_safe_zone_and_nothing_else_changes(slug):
    starter = _starter(slug)
    template = SimpleNamespace(id=uuid.uuid4(), format=SOCIAL_IMAGE, blocks=starter["blocks"])
    authored = starter["blocks"].get("css") or ""
    story = render.bundle_for(_post(starter, "story"), template, {}, size="1080x1920")
    assert story["composition"]["css"] == authored + story_safe_css(1080, 1920)
    assert render.bundle_for(_post(starter, "story"), template, {}, size="1080x1350")["composition"]["css"] == authored
    assert render.bundle_for(_post(starter, "image"), template, {}, size="1080x1920")["composition"]["css"] == authored
    assert ".page" in story["composition"]["html"]  # every still template lays its words out in a page
