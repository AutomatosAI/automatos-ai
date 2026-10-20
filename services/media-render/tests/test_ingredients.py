"""Which inputs a signed file keeps as C2PA ingredients (PRD-264 EUAI-06b, FR-10).

Only AI-made inputs, as the job's ``ai_inputs`` (EUAI-06a) declares them: AI footage and AI
stills, and a voice file when the voice is synthetic. A person's own upload never becomes an
ingredient, whatever manifest it carries: a camera's or a phone's can hold a place, EXIF data
or who took it, and an ingredient copies it into every file made from it. ``ai_inputs``
declares the slots together, so a render that shows an upload keeps none of its slot files.
The SFX are never AI-made, and a job without ``ai_inputs`` declares nothing.

The c2pa cases run inside the image, with its own ffmpeg, exiftool and c2pa-python, and a
throwaway signing certificate (conftest).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict

from helpers import STORAGE, VOICES, bundle
from media_render import content_credentials, provenance
from media_render.bundle import parse_bundle
from media_render.content_credentials import VISUAL_PREFIXES, ContentSigner
from media_render.jobs import Job
from media_render.pipeline import ANY_INPUT, RenderPipeline, ai_made_inputs
from media_render.provenance import COMPOSITE, TRAINED
from media_render.tts import Speaker
from test_content_credentials import _jpeg, _mp4

JOB_ID = "d" * 32
SIGNED_BY = "render " + JOB_ID
FOOTAGE = "assets/slots/hook.mp4"
UPLOAD = "assets/slots/own-photo.jpg"
VOICE_FILE = "assets/voice/l01.wav"
SFX = "assets/sfx/click.wav"
# How a test makes a slot file of each type (the image's own ffmpeg).
MAKERS = {".mp4": _mp4, ".jpg": _jpeg}


def _inputs(**flags: bool) -> Dict[str, bool]:
    return {**{"footage": False, "stills": False, "voice": False, "uploads": False, "template": True}, **flags}


def _media(*paths: str):
    return [{"path": path, "url": f"{STORAGE}{Path(path).name}"} for path in paths]


def _body(*paths: str, **overrides: Any) -> Dict[str, Any]:
    return bundle(media=_media(*paths), **overrides)


def _voiced(*paths: str, **overrides: Any) -> Dict[str, Any]:
    """A bundle whose one voice line is a file, with an SFX cue, over the slot ``paths``."""
    audio = {"voice": {"lines": [{"id": "l01", "at": 0.2, "path": VOICE_FILE}]}, "sfx": [{"path": SFX, "at": 1.0}]}
    return _body(*paths, VOICE_FILE, SFX, audio=audio, **overrides)


def _parsed(settings, body: Dict[str, Any]):
    return parse_bundle(body, settings, {}, VOICES)


# ── which inputs are AI-made ─────────────────────────────────────────────


def test_a_render_that_shows_an_upload_keeps_no_slot_file(settings):
    parsed = _parsed(settings, _body(FOOTAGE, UPLOAD, ai_inputs=_inputs(footage=True, uploads=True)))
    assert ai_made_inputs(parsed) == ()


def test_ai_footage_and_ai_stills_with_no_upload_are_ai_made(settings):
    footage = _parsed(settings, _body(FOOTAGE, ai_inputs=_inputs(footage=True)))
    assert ai_made_inputs(footage) == (FOOTAGE,)
    still = "assets/slots/still.png"
    stills = _parsed(settings, _body(still, ai_inputs=_inputs(stills=True)))
    assert ai_made_inputs(stills) == (still,)


def test_a_voice_file_is_ai_made_only_with_a_synthetic_voice_and_an_sfx_never(settings):
    spoken = _parsed(settings, _voiced(ai_inputs=_inputs(voice=True)))
    assert ai_made_inputs(spoken) == (VOICE_FILE,), "the voice file, and never the SFX"
    recorded = _parsed(settings, _voiced(ai_inputs=_inputs(uploads=True)))
    assert ai_made_inputs(recorded) == ()


def test_the_voice_stays_ai_made_beside_an_upload(settings):
    parsed = _parsed(settings, _voiced(UPLOAD, ai_inputs=_inputs(voice=True, uploads=True)))
    assert ai_made_inputs(parsed) == (VOICE_FILE,)


def test_a_job_without_ai_inputs_declares_no_input_ai_made(settings):
    assert ai_made_inputs(_parsed(settings, _voiced(FOOTAGE))) == ()


# ── the signed file's ingredients ────────────────────────────────────────


def _job(settings, tmp_path: Path, body: Dict[str, Any]) -> Job:
    return Job(id=JOB_ID, bundle=_parsed(settings, body), dir=tmp_path / "job", status="preparing", created_at=0.0)


def _signed(settings, signer: ContentSigner, path: Path) -> None:
    """Make ``path`` (an MP4 or a JPEG), mark it and sign it: a file carrying a manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    MAKERS[path.suffix](settings, path)
    instance_id = provenance.mark(path, TRAINED, exiftool_bin=settings.exiftool_bin, timeout_seconds=60)
    record = signer.sign(path, TRAINED, job=SIGNED_BY, instance_id=instance_id)
    assert record.manifest_id is not None, f"{path.name} was meant to carry a manifest"


def _ingredients(settings, signer: ContentSigner, job: Job, prefixes):
    pipeline = RenderPipeline(settings, Speaker(settings), signer)
    return asyncio.run(pipeline._ingredients(job, COMPOSITE, prefixes))


def test_an_upload_carrying_a_manifest_is_never_an_ingredient(settings, tmp_path):
    """A person's photo carrying its own manifest (a phone's, here a stand-in signed with the
    test certificate) beside AI footage that carries one: the render shows an upload, so
    neither becomes an ingredient of the video or of a picture."""
    signer = ContentSigner(settings)
    job = _job(settings, tmp_path, _body(FOOTAGE, UPLOAD, ai_inputs=_inputs(footage=True, uploads=True)))
    _signed(settings, signer, job.project_dir / FOOTAGE)
    _signed(settings, signer, job.project_dir / UPLOAD)
    upload = job.project_dir / UPLOAD
    assert content_credentials.read_store(upload, "image/jpeg") is not None, "the upload carries its manifest"

    assert _ingredients(settings, signer, job, ANY_INPUT) == ()
    assert _ingredients(settings, signer, job, VISUAL_PREFIXES) == ()


def test_ai_footage_carrying_a_manifest_is_kept_when_no_upload_is_shown(settings, tmp_path):
    signer = ContentSigner(settings)
    job = _job(settings, tmp_path, _body(FOOTAGE, ai_inputs=_inputs(footage=True)))
    _signed(settings, signer, job.project_dir / FOOTAGE)

    found = _ingredients(settings, signer, job, ANY_INPUT)
    assert [ingredient.path for ingredient in found] == [job.project_dir / FOOTAGE]
