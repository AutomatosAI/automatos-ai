"""PRD-264 EUAI-05 (EU AI Act Art. 50) — whether a post carries AI media, from its render record.

``modules/socials/ai_media.py`` decides it: AI footage and AI stills (a slot a generation
toolkit filled), and a synthetic voice (a render that spoke its script, with Kokoro, the
default, or a voice toolkit). The person's own uploads and Library files, the template's
own visuals and music, are no AI media. Pinned here:

* each media mix gives its decision;
* a finished render records what it was made from in its ``render_done`` entry
  (``report.ai_media``: the slots, the voice that spoke, the sha256 of each file it made),
  over the real render path (the US-104 harness: media-render over httpx.MockTransport);
* the decision is read from the record whose files the post still carries, so editing the
  voice or a slot's prompt after the render does not change it; media replaced by an edit
  leave the record behind; a render older than the record is judged from the post's fields;
* the rule moved out of the service (replace, don't shim), and the voice stays outside
  the content hash;
* ``GET /api/socials/posts/{id}/ai-labels`` answers the decision, in the caller's workspace only.

Each network's own label (TikTok's ``is_aigc``, YouTube, the receipt) is in
``tests/test_prd264_euai05_network_labels.py``.
"""
from __future__ import annotations

import os
import sys
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

import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from modules.socials import ai_media, service  # noqa: E402
from tests.test_prd251w1_render_lifecycle import (  # noqa: E402
    MP4_SHA, WS_OTHER, FakeStore, Renderer, _carousel_blocks, _create, _ctx, _post, _renderable, _run, _start, _template,
)

env = render_harness.env

SHA_RENDER = "a" * 64
SHA_OLDER = "b" * 64
RENDERED = {"9:16": [{"deliverable_id": "d-video-1", "name": "video-9x16.mp4", "sha256": SHA_RENDER, "bytes": 10}]}
SPOKEN = {"audio": {"voice": {"lines": [{"id": "l01", "at": 0.3, "text": "Three weeks to go."}]}}}
SILENT = {"audio": {"music": {"track": "calm"}}}
KOKORO_EMMA = {"toolkit": "kokoro", "voice_id": "bf_emma", "name": "Emma"}
FISH = {"toolkit": "fish_audio", "voice_id": "v-1", "name": "Narrator"}


def _done(toolkit, content_type="video/mp4"):
    return {"prompt": "A harvest table", "status": "done", "toolkit": toolkit, "content_type": content_type,
            "deliverable_id": "d-1", "name": "slot.mp4"}


def _post_of(**fields):
    base = {"format": "video", "voice": None, "footage": None, "media": RENDERED, "review_log": [], "targets": []}
    return SimpleNamespace(**{**base, **fields})


def _render_done(record):
    return {"at": "2026-10-19T09:00:00+00:00", "by": "member-1", "action": "render_done", "comment": None,
            "report": {"check": {"ok": True}, "ai_media": record}}


def _rendered(footage=None, voice=None, bundle=SPOKEN, **fields):
    """A post as its render left it: the render record of ``bundle`` in its review_log."""
    made = _post_of(footage=footage, voice=voice, **fields)
    record = ai_media.render_record(made, bundle, made.media)
    return _post_of(footage=footage, voice=voice, review_log=[_render_done(record)], **fields)


def _expect(*, footage=(), stills=(), voice=None, own=(), recorded=True):
    carries = bool(footage or stills or voice)
    return {"carries": carries, "footage": list(footage), "stills": list(stills), "voice": voice, "own": list(own),
            "recorded": recorded}


# ---------------------------------------------------------------------------
# Each media mix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("footage, voice, bundle, expected", [
    # The person's own upload and Library file, the template's visuals and music: no AI media.
    ({"photo": _done("upload", "image/png"), "logo": _done("library", "image/png")}, None, SILENT,
     _expect(own=("logo", "photo"))),
    ({"hook": _done("fal_ai")}, None, SILENT, _expect(footage=("hook",))),
    ({"still": _done("kieai", "image/png")}, None, SILENT, _expect(stills=("still",))),
    # The template's own voice is Kokoro, synthetic; so is a chosen Kokoro voice, and any voice toolkit.
    (None, None, SPOKEN, _expect(voice="kokoro")),
    (None, KOKORO_EMMA, SPOKEN, _expect(voice="kokoro")),
    (None, FISH, SPOKEN, _expect(voice="fish_audio")),
    ({"photo": _done("upload", "image/png")}, None, SPOKEN, _expect(voice="kokoro", own=("photo",))),
    ({"hook": _done("fal_ai"), "still": _done("kieai", "image/png")}, FISH, SPOKEN,
     _expect(footage=("hook",), stills=("still",), voice="fish_audio")),
    # A slot asked for but never made is nothing yet; a render that spoke nothing has no voice.
    ({"hook": {"prompt": "A harvest table"}}, FISH, SILENT, _expect()),
])
def test_each_media_mix_gives_its_decision(footage, voice, bundle, expected):
    assert ai_media.for_post(_rendered(footage, voice, bundle)).to_dict() == expected


def test_the_record_holds_the_slots_the_voice_and_the_rendered_files_only():
    post = _post_of(footage={"hook": _done("fal_ai"), "photo": _done("upload", "image/png")}, voice=FISH)
    assert ai_media.render_record(post, SPOKEN, RENDERED) == {
        "footage": ["hook"], "stills": [], "own": ["photo"], "voice": "fish_audio", "files": [SHA_RENDER],
    }
    report = {"check": {"ok": True}}
    marked = ai_media.with_record(report, post, SILENT, RENDERED)
    assert marked["check"] == {"ok": True} and marked["ai_media"]["voice"] is None
    assert report == {"check": {"ok": True}}  # a new copy: the renderer's report is untouched


# ---------------------------------------------------------------------------
# The decision can't drift from what was rendered
# ---------------------------------------------------------------------------


def test_a_voice_chosen_after_the_render_does_not_change_the_decision():
    silent = _rendered(bundle=SILENT, format="image")
    edited = _post_of(format="image", voice=FISH, review_log=silent.review_log)
    assert ai_media.for_post(edited).carries is False  # the files were rendered without speech

    spoken = _rendered(voice=None)
    assert ai_media.for_post(_post_of(voice=FISH, review_log=spoken.review_log)).voice == "kokoro"


def test_a_new_prompt_after_the_render_keeps_the_footage_the_files_show():
    rendered = _rendered(footage={"hook": _done("fal_ai")}, bundle=SILENT)
    asked_again = _post_of(footage={"hook": {"prompt": "Something else"}}, review_log=rendered.review_log)
    assert ai_media.for_post(asked_again).footage == ("hook",)


def test_media_replaced_by_an_edit_leave_the_render_record_behind():
    rendered = _rendered(footage={"hook": _done("fal_ai")})
    library = _post_of(media={"9:16": ["d-library-1"]}, review_log=rendered.review_log)
    assert ai_media.for_post(library).to_dict() == _expect(recorded=False)


def test_the_newest_record_of_the_files_the_post_carries_decides():
    older = ai_media.render_record(_post_of(footage={"hook": _done("fal_ai")}), SPOKEN, {"9:16": [{"sha256": SHA_OLDER}]})
    newer = ai_media.render_record(_post_of(), SILENT, RENDERED)
    post = _post_of(review_log=[_render_done(older), {"action": "submit"}, _render_done(newer)])
    assert ai_media.for_post(post).to_dict() == _expect()
    assert ai_media.for_post(_post_of(media={"9:16": [{"sha256": SHA_OLDER}]}, review_log=post.review_log)).carries


def test_a_post_without_a_record_is_judged_from_its_fields():
    # Rendered before the record existed: a rendered video speaks with its voice.
    before = _post_of(footage={"photo": _done("upload", "image/png")}, review_log=[{"action": "render_done", "report": {}}])
    assert ai_media.for_post(before).to_dict() == _expect(voice="kokoro", own=("photo",), recorded=False)
    # Never rendered: the person's own files, no speech.
    assert ai_media.for_post(_post_of(media={"1:1": ["d-library-1"]})).carries is False
    assert ai_media.for_post(_post_of(format="image", footage={"still": _done("kieai", "image/png")})).stills == ("still",)


def test_the_rule_moved_out_of_the_service_and_the_voice_stays_outside_the_hash():
    assert not hasattr(service, "footage_generated")
    post = _post_of(copy={}, variables={}, sources={}, template_id=None, title="T", length_seconds=None,
                    targets=[SimpleNamespace(toolkit="tiktok", post_kind="video", options={})])
    before = service.compute_content_hash(post)
    assert service.compute_content_hash(SimpleNamespace(**{**vars(post), "voice": FISH})) == before
    with_footage = SimpleNamespace(**{**vars(post), "footage": {"hook": _done("fal_ai")}})
    assert service.compute_content_hash(with_footage) != before  # AI footage stays approved content, as before


def test_the_ai_media_in_words():
    assert ai_media.describe(ai_media.AiMedia()) == "no AI media"
    assert ai_media.describe(ai_media.AiMedia(stills=("a",))) == "an AI still"
    assert ai_media.describe(ai_media.AiMedia(footage=("h",), stills=("a", "b"), voice="fish_audio")) == (
        "AI footage, AI stills and a synthetic voice (Fish Audio)"
    )


# ---------------------------------------------------------------------------
# The real render writes the record; the route answers it
# ---------------------------------------------------------------------------


def test_a_finished_render_records_its_synthetic_voice_and_files(env):
    post = _renderable(env)  # the template speaks one line with Kokoro, its own voice
    _, job = _start(env, post)
    assert _run(job, Renderer(), FakeStore(), env.factory) is True

    done = _post(env, post["id"]).review_log[-1]
    assert done["action"] == "render_done" and done["report"]["check"] == {"ok": True, "errors": 0}
    assert done["report"]["ai_media"] == {"footage": [], "stills": [], "own": [], "voice": "kokoro", "files": [MP4_SHA]}

    env.session.expire_all()
    resp = env.client.get(f"/api/socials/posts/{post['id']}/ai-labels")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"ai_media": _expect(voice="kokoro"), "targets": []}


def test_a_template_render_that_speaks_nothing_carries_no_ai_media(env):
    post = _create(
        env,
        template_id=str(_template(env, blocks=_carousel_blocks(), fmt="social_image")),
        format="carousel",
        variables={"headline": {"value": "4 SIGNS|YOUR AGENT|NEEDS A|HARNESS", "claim": False},
                   "point_1_title": {"value": "It forgets what it just did", "claim": False},
                   "point_2_title": {"value": "It picks the wrong tool", "claim": False},
                   "closing_title": {"value": "Build the harness first.", "claim": False}},
    )
    _, job = _start(env, post)
    pngs = {f"render-{i:02d}.png": b"\x89PNG\r\n\x1a\n" + bytes([i]) * 64 for i in range(1, 5)}
    outputs = [
        {"name": name, "kind": "still", "index": i, "at": at, "aspect": "4:5", "width": 1080, "height": 1350, "bytes": 72}
        for i, (name, at) in enumerate(zip(sorted(pngs), (0.5, 1.5, 2.5, 7.5)), start=1)
    ]
    assert _run(job, Renderer(outputs=outputs, files=pngs), FakeStore(), env.factory) is True

    row = _post(env, post["id"])
    record = row.review_log[-1]["report"]["ai_media"]
    assert record["voice"] is None and record["footage"] == record["stills"] == []
    assert record["files"] == sorted(r["sha256"] for r in row.media["4:5"])
    assert ai_media.for_post(row).to_dict() == _expect()


def test_the_ai_labels_route_answers_only_in_the_callers_workspace(env):
    post = _renderable(env)
    env.ctx = _ctx(WS_OTHER)
    assert env.client.get(f"/api/socials/posts/{post['id']}/ai-labels").status_code == 404
