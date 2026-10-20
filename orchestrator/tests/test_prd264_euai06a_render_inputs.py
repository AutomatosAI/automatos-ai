"""PRD-264 EUAI-06a (EU AI Act Art. 50(2)) — the render job says which of its inputs are AI-made.

media-render marks every file it makes with an AI element (IPTC DigitalSourceType in XMP,
``services/media-render/media_render/provenance.py``). It learns what is AI-made from the
bundle's ``ai_inputs``, which the orchestrator sets from the rule the post's network labels
use (``modules/socials/ai_media.py``, EUAI-05), so a post and its files agree. Pinned here:

* each mix of inputs gives its flags: a slot the render fills is the person's own file
  (an upload or a Library file) or AI footage / an AI still; a render that speaks its
  script speaks with a synthetic voice; the template's visuals are always in it;
* every bundle ``render.bundle_for`` builds carries them (a post's render, a preview, a
  thumbnail), the crops of the person's own still carry "nothing AI-made", and a render
  no slot file joins (a document's social render, ``with_template_inputs``) carries the
  template and, when it speaks, the synthetic voice
  (test_prd264_euai06a_document_render_inputs.py drives the document path);
* a post's render sends them to media-render over the real render path (the US-104
  harness: media-render over httpx.MockTransport);
* media-render's own parser takes what the orchestrator sends.

The read-back of every format runs in the media-render image (services/media-render/tests).
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_MEDIA_RENDER = _ORCH.parent / "services" / "media-render"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
# media-render's bundle parser (standard library only), found LAST so no orchestrator module is shadowed.
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from media_render.provenance import AiInputs, parse_ai_inputs  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from core.media_render_bundle import AI_INPUTS, with_template_inputs  # noqa: E402
from modules.socials import ai_media, render, upload_crops  # noqa: E402
from tests.test_prd251w1_render_lifecycle import FakeStore, Renderer, _renderable, _run, _start  # noqa: E402

env = render_harness.env

SPOKEN = {"audio": {"voice": {"lines": [{"id": "l01", "at": 0.3, "text": "Three weeks to go."}]}}}
SILENT = {"audio": {"music": {"track": "calm"}}}
IMAGE_SLOTS = {"photo": {"kind": "image", "path": "assets/slots/photo.png"}, "before": {"kind": "image"}}
VIDEO_SLOTS = {"hook": {"kind": "video", "path": "assets/slots/hook.mp4"}}
NOTHING_AI = {"footage": False, "stills": False, "voice": False, "uploads": False, "template": True}


def _done(toolkit, content_type="video/mp4", name="slot.mp4"):
    return {"prompt": "A harvest table", "status": "done", "toolkit": toolkit, "content_type": content_type,
            "deliverable_id": "d-1", "name": name}


def _flags(**changes):
    return {**NOTHING_AI, **changes}


# ---------------------------------------------------------------------------
# Each mix of inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("footage, shown, slots, speaks, expected", [
    # The template alone, silent: nothing AI-made.
    (None, (), {}, False, _flags()),
    # The template's script, spoken: Kokoro (or a voice toolkit) is a synthetic voice.
    (None, (), {}, True, _flags(voice=True)),
    # The person's own upload and Library file in the slots it shows.
    ({"photo": _done("upload", "image/png"), "before": _done("library", "image/png")}, ("photo", "before"),
     IMAGE_SLOTS, False, _flags(uploads=True)),
    # An AI pick, done; an AI shot still to be made, told by its template slot.
    ({"photo": _done("kieai", "image/png")}, ("photo",), IMAGE_SLOTS, False, _flags(stills=True)),
    ({"photo": {"prompt": "A harvest table"}}, ("photo",), IMAGE_SLOTS, False, _flags(stills=True)),
    ({"hook": _done("fal_ai")}, ("hook",), VIDEO_SLOTS, True, _flags(footage=True, voice=True)),
    ({"hook": {"prompt": "A harvest table"}}, ("hook",), VIDEO_SLOTS, False, _flags(footage=True)),
    # Mixed: the person's own photo beside an AI still.
    ({"photo": _done("upload", "image/png"), "before": _done("kieai", "image/png")}, ("photo", "before"),
     IMAGE_SLOTS, False, _flags(stills=True, uploads=True)),
    # A slot the render does not show is not in it.
    ({"hook": _done("fal_ai"), "photo": _done("upload", "image/png")}, (), VIDEO_SLOTS, False, _flags()),
])
def test_each_mix_of_inputs_gives_its_flags(footage, shown, slots, speaks, expected):
    post = SimpleNamespace(footage=footage)
    assert ai_media.render_inputs(post, shown, slots, speaks=speaks) == expected


def test_the_flags_join_a_new_copy_of_the_bundle_and_speech_is_read_from_its_script():
    post = SimpleNamespace(footage={"hook": _done("fal_ai")})
    spoken = ai_media.with_inputs(SPOKEN, post, ("hook",), VIDEO_SLOTS)
    assert spoken[ai_media.INPUTS] == _flags(footage=True, voice=True)
    assert spoken["audio"] == SPOKEN["audio"] and ai_media.INPUTS not in SPOKEN  # the bundle given is untouched
    assert ai_media.with_inputs(SILENT, post, (), VIDEO_SLOTS)[ai_media.INPUTS] == _flags()


def test_a_render_no_slot_file_joins_declares_the_template_and_its_spoken_voice():
    spoken, silent = with_template_inputs(SPOKEN), with_template_inputs(SILENT)
    assert spoken[AI_INPUTS] == _flags(voice=True) and silent[AI_INPUTS] == _flags()
    assert AI_INPUTS not in SPOKEN and spoken["audio"] == SPOKEN["audio"]  # the bundle given is untouched
    assert ai_media.INPUTS == AI_INPUTS, "one key on both sides of the contract"


def test_media_render_takes_the_flags_the_orchestrator_sends():
    """The contract: media-render's own parser reads every flag the orchestrator sets."""
    sent = ai_media.render_inputs(SimpleNamespace(footage={"hook": _done("fal_ai")}), ("hook",), VIDEO_SLOTS, speaks=True)
    assert parse_ai_inputs(sent) == AiInputs(footage=True, stills=False, voice=True, uploads=False, template=True, declared=True)
    assert parse_ai_inputs(ai_media.own_file_inputs()) == AiInputs(uploads=True, declared=True)


# ---------------------------------------------------------------------------
# Every bundle carries them
# ---------------------------------------------------------------------------


def _starter(name):
    starter = next(s for s in social_starters() if s["name"] == name)
    template = SimpleNamespace(id=uuid.uuid4(), format=starter["format"], blocks=starter["blocks"], name=starter["name"])
    values = {key: {"value": value, "claim": False} for key, value in starter["sample_data"].items()}
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), variables=values, length_seconds=None, targets=[])
    return post, template


def test_a_photo_cards_bundle_says_which_photo_is_the_persons_and_which_ai_made():
    post, template = _starter("Before / after")
    post.footage = {"before": _done("upload", "image/png", "before.png"), "after": _done("kieai", "image/png", "after.png")}
    bundle = render.bundle_for(post, template, {}, footage_slots=("before", "after"), require_photos=True)
    assert bundle[ai_media.INPUTS] == _flags(stills=True, uploads=True)


def test_a_video_with_ai_footage_and_its_voice_says_both():
    post, template = _starter("Cinematic product promo")
    assert template.blocks["slots"]["hook"]["kind"] == "video"
    post.footage = {"hook": {"prompt": "A harvest table"}}  # to be generated as the render starts
    bundle = render.bundle_for(post, template, {}, footage_slots=("hook",))
    assert render.voice_script(bundle), "the seeded video speaks its script"
    assert bundle[ai_media.INPUTS] == _flags(footage=True, voice=True)
    preview = render.bundle_for(post, template, {})  # a preview plays the template's own motion graphics
    assert preview[ai_media.INPUTS] == _flags(voice=True)


def test_the_crops_of_the_persons_own_still_carry_nothing_ai_made():
    post = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.uuid4(), targets=[])
    bundles = upload_crops.crop_bundles(post)
    assert bundles and all(bundle[ai_media.INPUTS] == _flags(uploads=True) for bundle in bundles)


# ---------------------------------------------------------------------------
# Over the real render path
# ---------------------------------------------------------------------------


def test_a_posts_render_sends_the_flags_to_media_render(env):
    """The harness template speaks one Kokoro line and fills no slot: a synthetic voice over
    the template, and media-render's parser reads the flags as declared."""
    post = _renderable(env)
    _, job = _start(env, post)
    renderer = Renderer()
    assert _run(job, renderer, FakeStore(), env.factory) is True
    (sent,) = renderer.bundles
    assert sent[ai_media.INPUTS] == _flags(voice=True)
    assert parse_ai_inputs(sent[ai_media.INPUTS]).declared is True


# ---------------------------------------------------------------------------
# The writer is in the image, and the on-demand lane reads the mark back
# ---------------------------------------------------------------------------


def test_the_media_render_image_carries_exiftool_the_marks_writer():
    """D14: exiftool, from Debian, writes and reads the mark; its version is recorded at build."""
    dockerfile = (_MEDIA_RENDER / "Dockerfile").read_text()
    assert "libimage-exiftool-perl" in dockerfile
    assert '"exiftool", "-ver"' in (_MEDIA_RENDER / "media_render" / "versions.py").read_text()


def test_the_media_render_lane_reads_every_formats_mark_back():
    """media-render.yml (on demand) runs tests/ in the image, so every format's read-back runs
    there, and it reads the fixture MP4's mark back with the image's exiftool."""
    assert (_MEDIA_RENDER / "tests" / "test_provenance.py").is_file()
    assert (_MEDIA_RENDER / "tests" / "test_provenance_render.py").is_file()
    workflow = yaml.safe_load((_ORCH.parent / ".github" / "workflows" / "media-render.yml").read_text())
    commands = "\n".join(step.get("run", "") for step in workflow["jobs"]["media-render"]["steps"])
    assert '--entrypoint python "$TEST_IMAGE" -m pytest' in commands
    assert "--entrypoint exiftool" in commands and "XMP-iptcExt:DigitalSourceType /out/fixture.mp4" in commands
