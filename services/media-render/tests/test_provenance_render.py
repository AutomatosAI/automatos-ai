"""The real pipeline marks what it makes (PRD-264 EUAI-06a): every output with an AI element
carries the IPTC DigitalSourceType after its last encode, read back with the image's
exiftool, and its packet holds nothing but the mark.

One job per kind, through the real stage, Kokoro, mix, `hyperframes check` and render:
- a video: its Kokoro line's WAV, the mix (``mix.wav``) and ``render.mp4``;
- a still (``render.png``), and one made only from the template, which gets no mark;
- a preview: its frames (``preview-NN.png``) and its reel (``preview.mp4``);
- POST /tts: the Kokoro WAV it hands out.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN, VOICES
from media_render.bundle import parse_bundle
from media_render.composition import MIX_PATH
from media_render.fixture import fixture_bundle
from media_render.jobs import Job, JobStore
from media_render.pipeline import RenderPipeline, RenderResult
from media_render.provenance import COMPOSITE, TRAINED
from media_render.server import TOKEN_HEADER, create_app
from media_render.tts import Speaker
from provenance_checks import assert_only_the_mark, packet_of, source_type_of

AUTH = {TOKEN_HEADER: TOKEN}
WORKSPACE = "ws-canary-7f3a"
REFERENCE = "post-canary-91"
HEADLINE = "Canary headline 4411"


def _inputs(**flags: bool) -> Dict[str, bool]:
    return {**{"footage": False, "stills": False, "voice": False, "uploads": False, "template": True}, **flags}


def _body(**overrides: Any) -> Dict[str, Any]:
    """The committed fixture, under a workspace, reference and headline no mark may carry."""
    return {**fixture_bundle(), "workspace_id": WORKSPACE, "reference": REFERENCE, "variables": {"headline": HEADLINE}, **overrides}


def _silent(**overrides: Any) -> Dict[str, Any]:
    return {key: value for key, value in _body(**overrides).items() if key != "audio"}


def _absent(settings, job: Job) -> Tuple[str, ...]:
    return (WORKSPACE, REFERENCE, HEADLINE, job.id, str(job.dir), settings.work_dir)


def _run(settings, body: Dict[str, Any], inspect=None) -> Tuple[Job, RenderResult]:
    """Prepare, check and render ``body`` with the real pipeline; ``inspect(job)`` runs between
    the check and the render, while the staged composition and the spoken lines are still there."""
    job = JobStore(Path(settings.work_dir)).create(parse_bundle(body, settings, {}, VOICES))
    pipeline = RenderPipeline(settings, Speaker(settings))

    async def go() -> RenderResult:
        await pipeline.start()
        try:
            outcome = await pipeline.prepare_and_check(job)
            assert outcome.ok, json.dumps(outcome.report, indent=2)[:4000]
            if inspect is not None:
                inspect(job)
            return await pipeline.render(job)
        finally:
            await pipeline.close()

    return job, asyncio.run(go())


def _assert_marked(settings, job: Job, path: Path, expected: str) -> None:
    assert source_type_of(settings, path) == expected, path.name
    assert_only_the_mark(packet_of(settings, path), expected, absent=_absent(settings, job))


def test_a_video_marks_its_kokoro_line_its_mix_and_its_mp4(settings):
    """AI footage and a Kokoro voice over the template: the line and the mix are AI-made
    audio through and through; the video mixes them with the template's visuals."""
    seen: Dict[str, Path] = {}

    def inspect(job: Job) -> None:
        line, mix = job.audio_dir / "vo" / "l01.wav", job.project_dir / MIX_PATH
        _assert_marked(settings, job, line, TRAINED)
        _assert_marked(settings, job, mix, TRAINED)
        seen["mix"] = mix

    job, result = _run(settings, _body(ai_inputs=_inputs(footage=True, voice=True)), inspect)
    assert seen, "the check passed and the mix was read back"
    (output,) = result.outputs
    video = job.output_dir / output["name"]
    assert output["name"] == "render.mp4" and output["bytes"] == video.stat().st_size, "the size is the marked file's"
    assert output["probe"]["video_codec"] == "h264" and abs(output["duration"] - 3.0) <= 0.1
    _assert_marked(settings, job, video, COMPOSITE)


def test_a_still_with_an_ai_picture_is_marked_and_a_template_only_still_is_not(settings):
    job, result = _run(settings, _silent(still={"at": [1.0]}, ai_inputs=_inputs(stills=True)))
    assert [output["name"] for output in result.outputs] == ["render.png"]
    still = job.output_dir / "render.png"
    assert result.outputs[0]["bytes"] == still.stat().st_size
    _assert_marked(settings, job, still, COMPOSITE)

    plain_job, plain = _run(settings, _silent(still={"at": [0.5, 2.0]}, ai_inputs=_inputs(uploads=True)))
    assert [output["name"] for output in plain.outputs] == ["render-01.png", "render-02.png"]
    for output in plain.outputs:
        path = plain_job.output_dir / output["name"]
        assert source_type_of(settings, path) is None and packet_of(settings, path) == b"", output["name"]


def test_a_preview_marks_each_frame_and_its_reel(settings):
    job, result = _run(settings, _body(preview={"at": [0.5, 2.0]}, ai_inputs=_inputs(footage=True, voice=True)))
    assert [output["name"] for output in result.outputs] == ["preview-01.png", "preview-02.png", "preview.mp4"]
    for output in result.outputs:
        path = job.output_dir / output["name"]
        assert output["bytes"] == path.stat().st_size, output["name"]
        _assert_marked(settings, job, path, COMPOSITE)


def test_tts_hands_out_kokoro_wavs_marked_ai_made(settings, tmp_path):
    async def go() -> Dict[str, Any]:
        async with TestClient(TestServer(create_app(settings))) as client:
            request = {"lines": [{"id": "l01", "text": "Orders. Stock. Customers."}], "include_audio": True}
            response = await client.post("/tts", data=json.dumps(request), headers=AUTH)
            body = await response.json()
            assert response.status == 200, body
            return body

    wav = tmp_path / "l01.wav"
    wav.write_bytes(base64.b64decode(asyncio.run(go())["lines"][0]["audio_base64"]))
    assert source_type_of(settings, wav) == TRAINED
    assert_only_the_mark(packet_of(settings, wav), TRAINED, absent=(str(tmp_path), settings.work_dir))
