"""The IPTC "AI-generated" mark (PRD-264 EUAI-06a, EU AI Act Art. 50(2)), file by file.

Inside the image, with its own ffmpeg and exiftool: one file of every format media-render
hands out (MP4, PNG, JPEG, WAV) is made, marked by ``provenance.mark`` and read back with
exiftool, the writer (D14), and its packet is held to the allowlist. The decision (which
files are AI-made, and how) and the job's ``ai_inputs`` contract are pinned here too.
POST /tts marks each WAV it returns, under its own instance id, and answers 500 with no
WAV when one cannot be marked; without ``include_audio`` no WAV leaves and none is marked.
The real pipeline's outputs are read back in test_provenance_render.py.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN, VOICES, bundle
from media_render import provenance
from media_render.bundle import AudioPlan, MusicCue, SfxCue, VoiceLine, VoicePlan, parse_bundle
from media_render.jpeg import to_jpeg
from media_render.provenance import COMPOSITE, TRAINED, AiInputs, ProvenanceError
from media_render.server import TOKEN_HEADER, create_app
from media_render.tts import Speaker
from media_render.validate import BundleError
from provenance_checks import assert_only_the_mark, ffmpeg, instance_id_of, packet_of, seconds_of, source_type_of
from test_server import fake_synthesize

AUTH = {TOKEN_HEADER: TOKEN}
DECLARED = {"footage": True, "stills": False, "voice": True, "uploads": False, "template": True}
# What a job knows that no mark may carry (Guidelines ¶94).
CANARIES = ("ws-canary-7f3a", "post-canary-91", "a harvest table at golden hour")


def _mark(settings, path: Path, kind) -> Optional[str]:
    return provenance.mark(path, kind, exiftool_bin=settings.exiftool_bin, timeout_seconds=60)


def _mp4(settings, path: Path) -> Path:
    ffmpeg(settings, "-f", "lavfi", "-i", "testsrc=size=64x64:rate=10", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
           "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path))
    return path


def _png(settings, path: Path) -> Path:
    ffmpeg(settings, "-f", "lavfi", "-i", "color=c=0x14171c:s=32x16", "-frames:v", "1", str(path))
    return path


def _wav(settings, path: Path) -> Path:
    ffmpeg(settings, "-f", "lavfi", "-i", "sine=frequency=220:sample_rate=48000", "-t", "0.5", "-ac", "2", "-c:a", "pcm_s16le", str(path))
    return path


def _with_personal_xmp(settings, path: Path) -> Path:
    """The file as if something upstream had written personal XMP into it."""
    subprocess.run(
        [settings.exiftool_bin, "-q", "-overwrite_original", f"-XMP-dc:Creator={CANARIES[0]}",
         f"-XMP-dc:Description={CANARIES[2]}", f"-XMP-xmp:Label={CANARIES[1]}", str(path)],
        check=True, capture_output=True, timeout=60,
    )
    assert CANARIES[0].encode() in packet_of(settings, path), "the fixture should start with personal XMP"
    return path


# ── each format, marked and read back ─────────────────────────────────────


@pytest.mark.parametrize("kind", [TRAINED, COMPOSITE])
def test_an_mp4_is_marked_and_reads_back_with_exiftool(settings, tmp_path, kind):
    mp4 = _mp4(settings, tmp_path / "render.mp4")
    assert source_type_of(settings, mp4) is None
    instance_id = _mark(settings, mp4, kind)
    assert source_type_of(settings, mp4) == kind
    assert f"{provenance.INSTANCE_PREFIX}{instance_id_of(packet_of(settings, mp4))}" == instance_id
    assert_only_the_mark(packet_of(settings, mp4), kind, absent=(str(tmp_path),))
    assert abs(seconds_of(settings, mp4) - 1.0) < 0.2, "the marked MP4 still plays"


def test_a_png_is_marked_and_reads_back_with_exiftool(settings, tmp_path):
    png = _png(settings, tmp_path / "render.png")
    _mark(settings, png, COMPOSITE)
    assert source_type_of(settings, png) == COMPOSITE
    assert_only_the_mark(packet_of(settings, png), COMPOSITE, absent=(str(tmp_path),))


def test_marking_clears_whatever_xmp_the_file_carried(settings, tmp_path):
    """The packet holds the mark and nothing else, even over a file that carried personal XMP."""
    for made in (_png(settings, tmp_path / "still.png"), _mp4(settings, tmp_path / "video.mp4")):
        _mark(settings, _with_personal_xmp(settings, made), TRAINED)
        assert_only_the_mark(packet_of(settings, made), TRAINED, absent=CANARIES)


def test_a_wav_carries_the_mark_as_its_xmp_chunk_and_still_plays(settings, tmp_path):
    wav = _wav(settings, tmp_path / "mix.wav")
    before = wav.stat().st_size
    _mark(settings, wav, TRAINED)
    data = wav.read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    assert struct.unpack("<I", data[4:8])[0] == len(data) - 8, "the RIFF size covers the new chunk"
    assert data[before:before + 4] == b"_PMX", "the packet is the last chunk"
    assert source_type_of(settings, wav) == TRAINED
    assert_only_the_mark(packet_of(settings, wav), TRAINED, absent=(str(tmp_path),))
    assert abs(seconds_of(settings, wav) - 0.5) < 0.05, "the marked WAV still decodes"
    with pytest.raises(ProvenanceError, match="already"):
        _mark(settings, wav, TRAINED)


def test_a_wav_whose_last_chunk_lacks_its_pad_byte_is_padded_then_marked(settings, tmp_path):
    fmt = struct.pack("<HHIIHH", 1, 1, 8000, 8000, 1, 8)  # PCM, mono, 8 kHz, 8-bit
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", 3) + b"\x80\x81\x82"
    wav = tmp_path / "odd.wav"
    wav.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    _mark(settings, wav, COMPOSITE)
    data = wav.read_bytes()
    assert data[8 + len(body)] == 0 and data[9 + len(body):13 + len(body)] == b"_PMX"
    assert struct.unpack("<I", data[4:8])[0] == len(data) - 8
    assert source_type_of(settings, wav) == COMPOSITE


def test_what_is_not_a_riff_wave_file_is_refused(settings, tmp_path):
    fake = tmp_path / "fake.wav"
    fake.write_bytes(b"not a wave file at all")
    with pytest.raises(ProvenanceError, match="RIFF WAVE"):
        _mark(settings, fake, TRAINED)


def test_no_mark_writes_nothing_and_an_unknown_term_is_refused(settings, tmp_path):
    png = _png(settings, tmp_path / "plain.png")
    before = png.read_bytes()
    assert _mark(settings, png, None) is None
    assert png.read_bytes() == before and packet_of(settings, png) == b""
    with pytest.raises(ValueError):
        _mark(settings, png, provenance.SOURCE_TYPE_BASE + "digitalCapture")


def test_a_jpeg_keeps_the_mark_of_the_still_it_was_made_from(settings, tmp_path):
    marked = _png(settings, tmp_path / "render.png")
    _mark(settings, marked, COMPOSITE)
    plain = _png(settings, tmp_path / "plain.png").read_bytes()
    kwargs = {"exiftool_bin": settings.exiftool_bin, "timeout_seconds": 60}
    for source, expected in ((marked.read_bytes(), COMPOSITE), (plain, None)):
        jpeg = to_jpeg(source, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=30)
        out = tmp_path / "out.jpg"
        out.write_bytes(provenance.carry_mark(source, jpeg, **kwargs).data)
        assert out.read_bytes()[:3] == b"\xff\xd8\xff"
        assert source_type_of(settings, out) == expected
        if expected:
            assert_only_the_mark(packet_of(settings, out), expected, absent=(str(tmp_path),))


def test_post_jpeg_answers_a_jpeg_carrying_the_stills_mark(settings, tmp_path):
    png = _png(settings, tmp_path / "render.png")
    _mark(settings, png, TRAINED)

    async def go():
        app = create_app(settings, pipeline=SimpleNamespace(), speaker=SimpleNamespace(), library={})
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/jpeg", data=png.read_bytes(), headers=AUTH)
            assert response.status == 200 and response.content_type == "image/jpeg"
            return await response.read()

    out = tmp_path / "answer.jpg"
    out.write_bytes(asyncio.run(go()))
    assert source_type_of(settings, out) == TRAINED


# ── POST /tts: a voice WAV is marked where it leaves the service ──────────

TTS_LINES = [{"id": "l01", "text": "Three weeks to go."}, {"id": "l02", "text": "Three weeks to go."}]


def _post_tts(settings, *, include_audio: bool):
    """POST /tts through the app with a stub Kokoro (real WAVs, no model): the status and the body."""

    async def go():
        speaker = Speaker(settings, synthesize=fake_synthesize)
        app = create_app(settings, pipeline=SimpleNamespace(), speaker=speaker, library={})
        async with TestClient(TestServer(app)) as client:
            body = {"lines": TTS_LINES, "include_audio": include_audio}
            response = await client.post("/tts", data=json.dumps(body), headers=AUTH)
            return response.status, await response.json()

    return asyncio.run(go())


def test_post_tts_marks_every_wav_it_returns_each_with_a_random_instance_id(settings, tmp_path):
    status, body = _post_tts(settings, include_audio=True)
    assert status == 200, body
    ids = set()
    for line in body["lines"]:
        wav = tmp_path / f"{line['id']}.wav"
        wav.write_bytes(base64.b64decode(line["audio_base64"]))
        assert source_type_of(settings, wav) == TRAINED, line["id"]
        packet = packet_of(settings, wav)
        assert_only_the_mark(packet, TRAINED, absent=(str(tmp_path), settings.work_dir))
        ids.add(instance_id_of(packet))
    assert len(ids) == 2 and all(found.version == 4 for found in ids), "each WAV gets its own random instance id"


def test_post_tts_answers_500_and_no_wav_when_exiftool_cannot_mark(settings, tmp_path):
    """EUAI-06a: an unmarked AI WAV is never handed out; the failure is a clear error."""
    broken = replace(settings, exiftool_bin=str(tmp_path / "no-such-exiftool"))
    status, body = _post_tts(broken, include_audio=True)
    assert status == 500 and body["error"] == "voice_unmarked", body
    assert "lines" not in body and "audio_base64" not in json.dumps(body)


def test_post_tts_returns_no_wav_when_any_line_cannot_be_marked(settings, monkeypatch):
    """The first line marks, the second fails: neither goes out, marked or not."""
    real_mark = provenance.mark
    calls = []

    def second_fails(path, kind, **kwargs):
        calls.append(path.name)
        if len(calls) > 1:
            raise ProvenanceError("exiftool could not make the XMP packet: simulated")
        return real_mark(path, kind, **kwargs)

    monkeypatch.setattr(provenance, "mark", second_fails)
    status, body = _post_tts(settings, include_audio=True)
    assert calls == ["l01.wav", "l02.wav"]
    assert status == 500 and body["error"] == "voice_unmarked", body
    assert "lines" not in body and "audio_base64" not in json.dumps(body)


def test_post_tts_without_audio_marks_nothing_since_no_wav_leaves(settings, tmp_path):
    """Durations and segments only: no WAV leaves, so no exiftool runs (a broken one is never reached)."""
    broken = replace(settings, exiftool_bin=str(tmp_path / "no-such-exiftool"))
    status, body = _post_tts(broken, include_audio=False)
    assert status == 200, body
    assert [line["id"] for line in body["lines"]] == ["l01", "l02"]
    assert all("audio_base64" not in line for line in body["lines"])


# ── FR-16: one random instance id per file ────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jpeg(settings, path: Path) -> Path:
    path.write_bytes(to_jpeg(_png(settings, path.with_suffix(".png")).read_bytes(), ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=30))
    return path


@pytest.mark.parametrize("make, name", [(_mp4, "render.mp4"), (_png, "render.png"), (_jpeg, "image.jpg"), (_wav, "mix.wav")])
def test_the_same_file_marked_twice_gives_two_hashes_and_two_random_instance_ids(settings, tmp_path, make, name):
    """FR-16: two renders of identical input never share a SHA-256, so a predictable file
    can't be confirmed by hashing guesses against the public detection route (EUAI-10)."""
    first = make(settings, tmp_path / name)
    second = tmp_path / f"again-{name}"
    shutil.copyfile(first, second)
    assert _sha256(first) == _sha256(second), "the two files start identical"
    marked = [_mark(settings, path, COMPOSITE) for path in (first, second)]
    assert _sha256(first) != _sha256(second)
    ids = [instance_id_of(packet_of(settings, path)) for path in (first, second)]
    assert all(found.version == 4 for found in ids) and ids[0] != ids[1]
    assert [f"{provenance.INSTANCE_PREFIX}{found}" for found in ids] == marked
    for path in (first, second):
        assert source_type_of(settings, path) == COMPOSITE
        assert_only_the_mark(packet_of(settings, path), COMPOSITE, absent=(str(tmp_path),))


def test_a_jpeg_made_from_a_marked_still_gets_its_own_instance_id(settings, tmp_path):
    still = _png(settings, tmp_path / "render.png")
    _mark(settings, still, TRAINED)
    jpeg = to_jpeg(still.read_bytes(), ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=30)
    kwargs = {"exiftool_bin": settings.exiftool_bin, "timeout_seconds": 60}
    outs = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    for out in outs:
        out.write_bytes(provenance.carry_mark(still.read_bytes(), jpeg, **kwargs).data)
    ids = {instance_id_of(packet_of(settings, path)) for path in (still, *outs)}
    assert len(ids) == 3 and _sha256(outs[0]) != _sha256(outs[1])


# ── the decision: which files are AI-made, and how ────────────────────────


def _voice(*lines: VoiceLine) -> VoicePlan:
    return VoicePlan(voice="af_heart", speed=0.95, lang="en-us", lines=tuple(lines))


KOKORO_LINE = VoiceLine(id="l01", at=0.3, text="Three weeks to go.")
FILE_LINE = VoiceLine(id="l02", at=1.0, path="assets/voice/l02.wav")
MUSIC = MusicCue(track="deep-house-003", path=Path("/opt/media-render/music/x.mp3"), start=32.0, fade_in=0.02, fade_out=1.7)
SFX = SfxCue(path="assets/sfx/click.ogg", at=0.5, volume=0.5)


def _job(inputs: AiInputs, plan: Optional[AudioPlan] = None) -> SimpleNamespace:
    return SimpleNamespace(ai_inputs=inputs, audio=plan or AudioPlan())


def _declared(**flags) -> AiInputs:
    return AiInputs(**{**{"footage": False, "stills": False, "voice": False, "uploads": False, "template": True}, **flags},
                    declared=True)


def test_a_render_made_only_from_the_template_and_own_content_gets_no_mark():
    job = _job(_declared(uploads=True), AudioPlan(music=MUSIC, sfx=(SFX,)))
    assert (provenance.video_type(job), provenance.image_type(job), provenance.audio_type(job)) == (None, None, None)


def test_ai_footage_over_the_template_is_a_composite_and_all_ai_is_trained():
    over_template = _job(_declared(footage=True))
    assert provenance.image_type(over_template) == COMPOSITE
    all_ai = _job(_declared(stills=True, template=False))
    assert provenance.image_type(all_ai) == TRAINED
    with_own = _job(_declared(stills=True, uploads=True, template=False))
    assert provenance.image_type(with_own) == COMPOSITE


def test_the_audio_decides_the_mix_and_joins_the_visuals_in_the_video():
    kokoro_only = _job(_declared(), AudioPlan(voice=_voice(KOKORO_LINE)))
    assert provenance.audio_type(kokoro_only) == TRAINED
    assert provenance.video_type(kokoro_only) == COMPOSITE, "the voice over the template's visuals"
    assert provenance.image_type(kokoro_only) is None, "a still or preview has no sound"
    over_music = _job(_declared(), AudioPlan(voice=_voice(KOKORO_LINE), music=MUSIC))
    assert provenance.audio_type(over_music) == COMPOSITE
    all_ai = _job(_declared(footage=True, voice=True, template=False), AudioPlan(voice=_voice(KOKORO_LINE)))
    assert provenance.video_type(all_ai) == TRAINED


def test_a_voice_file_is_ai_made_when_the_job_says_its_voice_is():
    toolkit = _job(_declared(voice=True), AudioPlan(voice=_voice(FILE_LINE)))
    assert provenance.audio_type(toolkit) == TRAINED
    own_recording = _job(_declared(voice=False), AudioPlan(voice=_voice(FILE_LINE)))
    assert provenance.audio_type(own_recording) is None
    # A line Kokoro speaks here is AI audio whatever the job says.
    spoken = _job(_declared(voice=False), AudioPlan(voice=_voice(KOKORO_LINE, FILE_LINE)))
    assert provenance.audio_type(spoken) == COMPOSITE


def test_a_job_without_ai_inputs_marks_only_the_kokoro_voice_and_says_so(caplog):
    older = _job(provenance.UNDECLARED, AudioPlan(voice=_voice(KOKORO_LINE, FILE_LINE)))
    assert provenance.audio_type(older) == COMPOSITE, "Kokoro's line is AI; the unflagged file is not"
    assert provenance.video_type(older) == COMPOSITE
    assert provenance.image_type(older) is None, "no AI visuals without the flags"
    assert provenance.audio_type(_job(provenance.UNDECLARED, AudioPlan(voice=_voice(FILE_LINE)))) is None
    with caplog.at_level(logging.WARNING, logger="media_render.provenance"):
        provenance.warn_if_undeclared("job-1", provenance.UNDECLARED)
        provenance.warn_if_undeclared("job-2", _declared())
    warned = [record.getMessage() for record in caplog.records if record.name == "media_render.provenance"]
    assert len(warned) == 1 and "job-1" in warned[0] and "ai_inputs" in warned[0]


# ── the contract: the bundle's ai_inputs ──────────────────────────────────


def test_the_bundle_takes_ai_inputs_and_an_older_bundle_without_them(settings):
    parsed = parse_bundle(bundle(ai_inputs=DECLARED), settings, {}, VOICES)
    assert parsed.ai_inputs == AiInputs(footage=True, stills=False, voice=True, uploads=False, template=True, declared=True)
    assert parse_bundle(bundle(), settings, {}, VOICES).ai_inputs == provenance.UNDECLARED


@pytest.mark.parametrize(
    "raw, message",
    [
        ({**DECLARED, "voice": "yes"}, "ai_inputs.voice must be true or false"),
        ({key: value for key, value in DECLARED.items() if key != "template"}, "missing template"),
        ({**DECLARED, "workspace": "ws-a"}, "unknown field"),
        (["footage"], "must be an object"),
    ],
)
def test_malformed_ai_inputs_are_refused(settings, raw, message):
    with pytest.raises(BundleError, match=message):
        parse_bundle(bundle(ai_inputs=raw), settings, {}, VOICES)
