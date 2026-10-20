"""The real pipeline marks what it makes (PRD-264 EUAI-06a): every output with an AI element
carries the IPTC DigitalSourceType after its last encode, read back with the image's
exiftool, and its packet holds nothing but the mark.

One job per kind, through the real stage, Kokoro, mix, `hyperframes check` and render:
- a video: the mix (``mix.wav``) and ``render.mp4``; its Kokoro line's WAV, which never
  leaves the service, is left unmarked (no exiftool in the timed voice step);
- a still (``render.png``), and one made only from the template, which gets no mark;
- a preview: its frames (``preview-NN.png``) and its reel (``preview.mp4``);
- POST /tts: each Kokoro WAV it hands out, under its own random instance id.

PRD-264 EUAI-06b: each marked file also carries a signed C2PA manifest, read back with
c2pa: ``c2pa.created`` with the same digital source type, the XMP's instance id, nothing
that names the job; the video keeps the signed mix as an ingredient; each output names its
SHA-256 and manifest id. A render whose signing fails still ships every file XMP-marked.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from aiohttp.test_utils import TestClient, TestServer

from c2pa_checks import active_manifest, assert_signed, ingredient_labels
from helpers import TOKEN, VOICES
from media_render.bundle import parse_bundle
from media_render.composition import MIX_PATH
from media_render.content_credentials import ContentSigner
from media_render.fixture import fixture_bundle
from media_render.jobs import Job, JobStore
from media_render.pipeline import RenderPipeline, RenderResult
from media_render.provenance import COMPOSITE, INSTANCE_PREFIX, TRAINED
from media_render.server import TOKEN_HEADER, create_app
from media_render.tts import Speaker
from media_render.signing_keys import SigningKeyError
from provenance_checks import assert_only_the_mark, instance_id_of, packet_of, source_type_of

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


def _run(settings, body: Dict[str, Any], inspect=None, signer: Optional[ContentSigner] = None) -> Tuple[Job, RenderResult]:
    """Prepare, check and render ``body`` with the real pipeline; ``inspect(job)`` runs between
    the check and the render, while the staged composition and the spoken lines are still there."""
    job = JobStore(Path(settings.work_dir)).create(parse_bundle(body, settings, {}, VOICES))
    pipeline = RenderPipeline(settings, Speaker(settings), signer)

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


def _assert_marked(settings, job: Job, path: Path, expected: str, output: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The XMP mark (EUAI-06a) and the signed manifest (EUAI-06b), under one instance id; the
    output's record, when given, is the file's. The manifest store."""
    assert source_type_of(settings, path) == expected, path.name
    packet = packet_of(settings, path)
    assert_only_the_mark(packet, expected, absent=_absent(settings, job))
    instance_id = f"{INSTANCE_PREFIX}{instance_id_of(packet)}"
    store = assert_signed(path, expected, instance_id=instance_id, absent=_absent(settings, job))
    if output is not None:
        assert output["digital_source_type"] == expected, output["name"]
        assert output["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), output["name"]
        assert output["manifest_id"] == store["active_manifest"], output["name"]
    return store


def test_a_video_marks_its_mix_and_its_mp4_and_not_its_inner_kokoro_line(settings):
    """AI footage and a Kokoro voice over the template: the mix is AI-made audio through
    and through; the video mixes it with the template's visuals. The line's own WAV stays
    inside the service, so it carries no mark of its own: the mix and the video carry it."""
    seen: Dict[str, str] = {}

    def inspect(job: Job) -> None:
        line, mix = job.audio_dir / "vo" / "l01.wav", job.project_dir / MIX_PATH
        assert line.is_file(), "the spoken line is still held between the check and the render"
        assert source_type_of(settings, line) is None, "the inner line WAV was marked in the voice step"
        seen["mix"] = _assert_marked(settings, job, mix, TRAINED)["active_manifest"]

    job, result = _run(settings, _body(ai_inputs=_inputs(footage=True, voice=True)), inspect)
    assert seen, "the check passed and the mix was read back"
    (output,) = result.outputs
    video = job.output_dir / output["name"]
    assert output["name"] == "render.mp4" and output["bytes"] == video.stat().st_size, "the size is the marked file's"
    assert output["probe"]["video_codec"] == "h264" and abs(output["duration"] - 3.0) <= 0.1
    store = _assert_marked(settings, job, video, COMPOSITE, output)
    assert ingredient_labels(active_manifest(store)) == [seen["mix"]], "the video keeps the signed mix as its ingredient"


def test_a_still_with_an_ai_picture_is_marked_and_a_template_only_still_is_not(settings):
    job, result = _run(settings, _silent(still={"at": [1.0]}, ai_inputs=_inputs(stills=True)))
    assert [output["name"] for output in result.outputs] == ["render.png"]
    still = job.output_dir / "render.png"
    assert result.outputs[0]["bytes"] == still.stat().st_size
    _assert_marked(settings, job, still, COMPOSITE, result.outputs[0])

    plain_job, plain = _run(settings, _silent(still={"at": [0.5, 2.0]}, ai_inputs=_inputs(uploads=True)))
    assert [output["name"] for output in plain.outputs] == ["render-01.png", "render-02.png"]
    for output in plain.outputs:
        path = plain_job.output_dir / output["name"]
        assert source_type_of(settings, path) is None and packet_of(settings, path) == b"", output["name"]
        assert "sha256" not in output and "manifest_id" not in output, "an unmarked file has no provenance record"


def test_a_preview_marks_each_frame_and_its_reel(settings):
    job, result = _run(settings, _body(preview={"at": [0.5, 2.0]}, ai_inputs=_inputs(footage=True, voice=True)))
    assert [output["name"] for output in result.outputs] == ["preview-01.png", "preview-02.png", "preview.mp4"]
    for output in result.outputs:
        path = job.output_dir / output["name"]
        assert output["bytes"] == path.stat().st_size, output["name"]
        _assert_marked(settings, job, path, COMPOSITE, output)


def test_tts_hands_out_kokoro_wavs_marked_ai_made_each_with_its_own_instance_id(settings, tmp_path):
    """The real /tts with the real Kokoro: every WAV it returns is marked after it is spoken,
    and two lines never share an instance id (FR-16)."""

    async def go() -> Dict[str, Any]:
        async with TestClient(TestServer(create_app(settings))) as client:
            lines = [{"id": "l01", "text": "Orders. Stock. Customers."}, {"id": "l02", "text": "Orders. Stock. Customers."}]
            response = await client.post("/tts", data=json.dumps({"lines": lines, "include_audio": True}), headers=AUTH)
            body = await response.json()
            assert response.status == 200, body
            return body

    ids = set()
    for line in asyncio.run(go())["lines"]:
        wav = tmp_path / f"{line['id']}.wav"
        wav.write_bytes(base64.b64decode(line["audio_base64"]))
        assert source_type_of(settings, wav) == TRAINED, line["id"]
        packet = packet_of(settings, wav)
        assert_only_the_mark(packet, TRAINED, absent=(str(tmp_path), settings.work_dir))
        ids.add(instance_id_of(packet))
    assert len(ids) == 2, "two identical lines share an instance id"


def test_a_render_whose_signing_fails_still_ships_every_file_marked(settings, caplog):
    """EUAI-06b: no signing certificate, so nothing is signed; the still ships XMP-marked, its
    record names its SHA-256 and no manifest, and the failure is logged with the job."""

    def no_material(_settings):
        raise SigningKeyError("simulated: no signing certificate")

    signer = ContentSigner(settings, load=no_material)
    job, result = _run(settings, _silent(still={"at": [1.0]}, ai_inputs=_inputs(stills=True)), signer=signer)
    (output,) = result.outputs
    still = job.output_dir / output["name"]
    assert source_type_of(settings, still) == COMPOSITE
    assert_only_the_mark(packet_of(settings, still), COMPOSITE, absent=_absent(settings, job))
    assert output["manifest_id"] is None and output["sha256"] == hashlib.sha256(still.read_bytes()).hexdigest()
    assert signer.counts() == {"signed": 0, "failed": 1}
    assert any(job.id in entry.getMessage() for entry in caplog.records if entry.name == "media_render.content_credentials")
