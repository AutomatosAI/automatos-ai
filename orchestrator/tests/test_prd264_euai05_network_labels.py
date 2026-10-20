"""PRD-264 EUAI-05 (EU AI Act Art. 50) — each network's own AI label, set at publish.

With the executor mocked (the US-301 harness, ``tests/test_prd251w3_publisher.py``):

* TikTok's ``is_aigc`` is true whenever the post carries AI media (``ai_media.py``): a
  synthetic voice alone sets it; the person's own files leave it to their own choice.
* YouTube: Composio's ``YOUTUBE_UPLOAD_VIDEO`` takes no synthetic-media field (its schema,
  read 19 Oct 2026, has title, description, categoryId, privacyStatus, tags and
  videoFilePath only), so none is invented and the receipt asks for YouTube Studio's
  setting. LinkedIn, X and Instagram take no AI flag either: the caption carries it.
* The receipt states the AI label decision for every channel: the line each target's
  publish used is kept with its receipt, and ``GET /posts/{id}/ai-labels`` answers it for
  every target, before publish too. The route is a plain ``def`` in the manifest.

Each media mix's decision, and the render record it is read from, are in
``tests/test_prd264_euai05_ai_media.py``.
"""
from __future__ import annotations

import inspect
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.routing import APIRoute

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w3_publisher as harness  # noqa: E402
from api import socials_publish  # noqa: E402
from modules.socials import ai_media, service  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS  # noqa: E402
from tests.test_prd251w3_publisher import IMAGE, MANIFEST, VIDEO, WS, FakeExecutor, _approved_post, _publish, _target  # noqa: E402
from tests.test_prd251w3_tiktok import _tiktok, _upload_params  # noqa: E402
from tests.test_prd251w3_youtube import OPTIONS, YOUTUBE  # noqa: E402
from tests.test_prd251w3_youtube import _publish as _publish_youtube  # noqa: E402

env = harness.env

SHA = "c" * 64
RENDERED = {"9:16": [{"deliverable_id": "d-video-1", "name": "video-9x16.mp4", "sha256": SHA, "bytes": 99,
                      "content_type": "video/mp4"}]}
SPOKE_ONLY = {"footage": [], "stills": [], "own": [], "voice": "kokoro", "files": [SHA]}
OWN_FILES_ONLY = {"footage": [], "stills": [], "own": ["photo"], "voice": None, "files": [SHA]}
AI_FOOTAGE = {"footage": ["hook"], "stills": [], "own": [], "voice": "kokoro", "files": [SHA]}
YOUTUBE_SCHEMA = {"title", "description", "categoryId", "privacyStatus", "tags", "videoFilePath"}
STUDIO = "Set 'Altered or synthetic content' in YouTube Studio for this video."
WHAT = "AI footage and a synthetic voice (Kokoro)"


def _rendered(env, post_id, record):
    """The post as a render that recorded ``record`` left it, then approved again."""
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        post.format = "video"
        post.media = RENDERED
        entry = {"at": datetime.now(timezone.utc).isoformat(), "by": "member-1", "action": "render_done", "comment": None,
                 "report": {"ai_media": record}}
        post.review_log = [*(post.review_log or []), entry]
        post.content_hash = post.approved_hash = service.compute_content_hash(post)
        db.commit()


def _stored_label(env, post_id, toolkit, kind):
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        (target,) = [t for t in post.targets if (t.toolkit, t.post_kind) == (toolkit, kind)]
        return target.action_plan.get(ai_media.LABEL)


def _labels(env, post_id, workspace_id=WS):
    with env.factory() as db:
        return socials_publish.social_post_ai_labels(post_id, db=db, ctx=SimpleNamespace(workspace_id=workspace_id))


# ---------------------------------------------------------------------------
# TikTok
# ---------------------------------------------------------------------------


def test_a_synthetic_voice_alone_sets_tiktoks_ai_label(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video"))
    _rendered(env, post_id, SPOKE_ONLY)
    executor = FakeExecutor(_tiktok())
    assert _publish(env, post_id, executor) == "published"
    assert _upload_params(executor)["is_aigc"] is True
    assert _stored_label(env, post_id, "tiktok", "video") == (
        "TikTok's AI-generated label is on: the post carries a synthetic voice (Kokoro)."
    )


def test_ai_media_overrides_the_persons_off_choice(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video", is_aigc=False))
    _rendered(env, post_id, AI_FOOTAGE)
    executor = FakeExecutor(_tiktok())
    assert _publish(env, post_id, executor) == "published"
    assert _upload_params(executor)["is_aigc"] is True


def test_the_persons_own_files_leave_tiktoks_label_to_their_choice(env):
    env.media = [VIDEO]
    post_id = _approved_post(env, _target("tiktok", "video"))
    _rendered(env, post_id, OWN_FILES_ONLY)
    executor = FakeExecutor(_tiktok())
    assert _publish(env, post_id, executor) == "published"
    assert "is_aigc" not in _upload_params(executor)
    assert _stored_label(env, post_id, "tiktok", "video") == "No AI media in this post: TikTok's AI-generated label stays off."


# ---------------------------------------------------------------------------
# YouTube: no field to set, so none is invented; the receipt asks for Studio's
# ---------------------------------------------------------------------------


def test_youtube_sets_only_its_documented_fields_and_the_receipt_asks_for_studio(env):
    env.media = [VIDEO, IMAGE]
    post_id = _approved_post(env, _target("youtube", "video", **OPTIONS))
    _rendered(env, post_id, AI_FOOTAGE)
    executor = FakeExecutor(YOUTUBE)
    ended, _ = _publish_youtube(env, post_id, executor, lambda post, media: "https://public.example/still.png?sig=1")
    assert ended == "published"
    upload = next(call for call in executor.calls if call.action == "YOUTUBE_UPLOAD_VIDEO")
    assert set(upload.params) == YOUTUBE_SCHEMA  # no containsSyntheticMedia: Composio's action has none
    assert _stored_label(env, post_id, "youtube", "video") == (
        f"This video carries {WHAT}, and YouTube's upload takes no AI label. {STUDIO}"
    )


def test_the_seeded_youtube_upload_maps_no_undocumented_field():
    (upload,) = [s for s in SEEDED_CHANNELS["youtube"].kinds["video"] if s.action == "YOUTUBE_UPLOAD_VIDEO"]
    assert set(upload.params) == YOUTUBE_SCHEMA


# ---------------------------------------------------------------------------
# The receipt: every channel's decision
# ---------------------------------------------------------------------------


def test_every_seeded_channel_has_its_ai_label_and_only_tiktok_sets_a_flag():
    assert set(SEEDED_CHANNELS) == set(ai_media.CHANNEL_AI_LABEL)
    reads_generated = {
        toolkit for toolkit, adapter in SEEDED_CHANNELS.items()
        if any("$generated" in json.dumps(dict(step.params)) for steps in adapter.kinds.values() for step in steps)
    }
    flags = {toolkit for toolkit, how in ai_media.CHANNEL_AI_LABEL.items() if how == ai_media.NETWORK_FLAG}
    assert reads_generated == flags == {"tiktok"}


AI = ai_media.AiMedia(footage=("hook",), voice="kokoro")
NONE = ai_media.AiMedia()


@pytest.mark.parametrize("toolkit, ai, options, line", [
    ("tiktok", AI, {}, f"TikTok's AI-generated label is on: the post carries {WHAT}."),
    ("tiktok", NONE, {}, "No AI media in this post: TikTok's AI-generated label stays off."),
    ("tiktok", NONE, {"is_aigc": True}, "TikTok's AI-generated label is on: you chose it. The post carries no AI media."),
    ("youtube", AI, {}, f"This video carries {WHAT}, and YouTube's upload takes no AI label. {STUDIO}"),
    ("youtube", NONE, {}, ai_media.NO_AI_MEDIA),
    ("linkedin", AI, {}, f"This post carries {WHAT}, and LinkedIn takes no AI label through its API, so the caption "
                         "carries the disclosure: start it with \"AI-generated.\""),
    ("twitter", AI, {}, f"This post carries {WHAT}, and X takes no AI label through its API, so the caption "
                        "carries the disclosure: start it with \"AI-generated.\""),
    ("instagram", AI, {}, f"This post carries {WHAT}, and Instagram takes no AI label through its API, so the caption "
                          "carries the disclosure: start it with \"AI-generated.\""),
    ("instagram", NONE, {}, ai_media.NO_AI_MEDIA),
    ("mastodon", AI, {}, f"This post carries {WHAT}, and Mastodon takes no AI label through its API, so the caption "
                         "carries the disclosure: start it with \"AI-generated.\""),
])
def test_the_receipt_line_for_each_channel(toolkit, ai, options, line):
    assert ai_media.label_line(toolkit, ai, options) == line


def test_the_route_states_every_channels_decision_before_and_after_publish(env):
    env.media = [VIDEO]
    post_id = _approved_post(
        env, _target("tiktok", "video"), _target("youtube", "video", **OPTIONS), _target("linkedin", "text"),
        _target("twitter", "text"), _target("instagram", "image"),
    )
    _rendered(env, post_id, AI_FOOTAGE)

    answer = _labels(env, post_id)
    assert answer["ai_media"]["carries"] is True and answer["ai_media"]["recorded"] is True
    lines = {(t["toolkit"], t["post_kind"]): t for t in answer["targets"]}
    assert set(lines) == {("tiktok", "video"), ("youtube", "video"), ("linkedin", "text"), ("twitter", "text"),
                          ("instagram", "image")}
    for (toolkit, _), target in lines.items():
        assert target["ai_label"] == ai_media.label_line(toolkit, ai_media.AiMedia(footage=("hook",), voice="kokoro"), {})
        assert target["at_publish"] is False

    # Once a publish ran a target, its receipt keeps the line that publish used.
    with env.factory() as db:
        post = service.get_post(db, WS, post_id)
        tiktok = next(t for t in post.targets if t.toolkit == "tiktok")
        tiktok.action_plan = {**tiktok.action_plan, ai_media.LABEL: "TikTok's AI-generated label is on: kept."}
        db.commit()
    kept = next(t for t in _labels(env, post_id)["targets"] if t["toolkit"] == "tiktok")
    assert kept == {**kept, "ai_label": "TikTok's AI-generated label is on: kept.", "at_publish": True}


def test_the_route_answers_only_in_the_callers_workspace(env):
    post_id = _approved_post(env, _target("linkedin", "text"))
    with pytest.raises(HTTPException) as refused:
        _labels(env, post_id, workspace_id=uuid.uuid4())
    assert refused.value.status_code == 404


def test_the_route_is_a_plain_def_in_the_manifest():
    routes = {route.path: route for route in socials_publish.router.routes if isinstance(route, APIRoute)}
    route = routes["/posts/{post_id}/ai-labels"]
    assert route.methods == {"GET"}
    assert not inspect.iscoroutinefunction(route.endpoint)  # F105
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert {"method": "GET", "path": "/api/socials/posts/{post_id}/ai-labels"} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
