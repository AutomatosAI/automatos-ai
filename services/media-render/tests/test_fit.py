"""Script windows (US-111, fit.py): every voice line fits the time its script gives it.

The planning is pure; the stretch is the image's real ffmpeg; the pipeline's
voice step runs for real with a stub Kokoro (tone WAVs of known lengths) and a
toolkit's voice file staged as media, so a line from either source is fitted
the same way.
"""

from __future__ import annotations

import asyncio
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from helpers import STORAGE, bundle, page
from media_render import fit, probe
from media_render.bundle import parse_bundle
from media_render.config import ConfigError, load_settings
from media_render.jobs import Job
from media_render.kokoro_tts import SpokenLine
from media_render.pipeline import RenderPipeline, voice_findings
from media_render.tts import Speaker

GAP = 0.1


def _timed(*lines):
    return [fit.Timed(line_id, at, seconds) for line_id, at, seconds in lines]


# ── planning (pure) ─────────────────────────────────────────────────────────
def test_a_window_runs_to_the_next_line_and_the_last_to_the_end():
    lines = _timed(("l02", 2.4, 1.0), ("l01", 0.3, 2.0), ("l03", 5.0, 1.0))
    assert fit.window_ends(lines, 8.0) == {"l01": 2.4, "l02": 5.0, "l03": 8.0}


def test_a_line_inside_its_window_is_left_as_it_is():
    assert fit.tempo_to_fit(0.3, 2.0, 2.4, GAP) is None
    assert fit.tempo_to_fit(0.3, 2.1, 2.4, GAP) is None  # ends exactly as the window closes
    assert fit.tempo_to_fit(0.3, None, 2.4, GAP) is None  # length unknown: the voice check says so


def test_a_line_past_its_window_ends_the_gap_before_it_closes():
    tempo = fit.tempo_to_fit(0.3, 2.091, 2.2, GAP)
    assert tempo == pytest.approx(2.091 / 1.8, abs=1e-4)
    # rounded UP, so the fitted line never ends past its target
    assert 2.091 / tempo <= 1.8
    assert fit.tempo_to_fit(2.0, 1.0, 2.05, GAP) == math.inf  # no room at all


def test_the_plan_fits_only_what_the_cap_allows():
    lines = _timed(("l01", 0.3, 2.0), ("l02", 2.0, 2.2), ("l03", 4.0, 0.5))
    plan = fit.plan_fit(lines, 5.0, max_tempo=1.25, gap=GAP)
    # l01: 2.0 s into 1.6 s is 1.25x, the cap; l02: 2.2 s into 1.9 s is 1.16x; l03 fits.
    assert plan == {"l01": 1.25, "l02": pytest.approx(2.2 / 1.9, abs=1e-4)}
    too_long = _timed(("l01", 0.3, 2.2), ("l02", 2.0, 0.5))
    assert fit.plan_fit(too_long, 5.0, max_tempo=1.25, gap=GAP) == {}


def test_the_last_line_fits_before_the_end_of_the_composition():
    assert fit.plan_fit(_timed(("l01", 3.0, 2.3)), 5.0, max_tempo=1.25, gap=GAP) == {"l01": pytest.approx(2.3 / 1.9, abs=1e-4)}


def test_segments_scale_with_the_tempo():
    assert fit.scaled_segments(((0.0, 0.5), (0.6, 1.25)), 1.25) == ((0.0, 0.4), (0.48, 1.0))


def test_the_cap_and_the_gap_are_config():
    settings = load_settings({})
    assert (settings.voice_max_tempo, settings.voice_fit_gap_seconds) == (1.25, 0.1)
    tuned = load_settings({"MEDIA_RENDER_VOICE_MAX_TEMPO": "1.4", "MEDIA_RENDER_VOICE_FIT_GAP_SECONDS": "0"})
    assert (tuned.voice_max_tempo, tuned.voice_fit_gap_seconds) == (1.4, 0.0)
    for name, raw in (
        ("MEDIA_RENDER_VOICE_MAX_TEMPO", "0.9"),
        ("MEDIA_RENDER_VOICE_MAX_TEMPO", "2.5"),
        ("MEDIA_RENDER_VOICE_MAX_TEMPO", "fast"),
        ("MEDIA_RENDER_VOICE_FIT_GAP_SECONDS", "-0.1"),
        ("MEDIA_RENDER_VOICE_FIT_GAP_SECONDS", "1"),
    ):
        with pytest.raises(ConfigError, match=name):
            load_settings({name: raw})


# ── the stretch (the image's ffmpeg) ────────────────────────────────────────
def _tone(settings, path: Path, seconds: float, rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [settings.ffmpeg_bin, "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}:sample_rate={rate}",
         "-c:a", "pcm_s16le", str(path)],
        check=True,
    )
    return path


def _length(settings, path: Path) -> float:
    return probe.duration_seconds(probe.probe(path, ffprobe_bin=settings.ffprobe_bin, timeout_seconds=30))


def test_a_stretched_line_lasts_its_length_over_the_tempo(settings, tmp_path):
    source = _tone(settings, tmp_path / "l01.wav", 2.0)
    target = tmp_path / "fit" / "l01.wav"
    fit.stretch(source, target, 1.25, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=60)
    assert _length(settings, target) == pytest.approx(1.6, abs=0.03)


def test_a_stretch_that_fails_says_so(settings, tmp_path):
    with pytest.raises(fit.FitError, match="missing.wav"):
        fit.stretch(tmp_path / "missing.wav", tmp_path / "out.wav", 1.1, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=60)


# ── the pipeline's voice step, for real ─────────────────────────────────────
def _speaker(settings, lengths):
    """Kokoro stood in by tones: each line's text says how long it lasts."""

    def synthesize(text, output, _settings, *, voice=None, speed=None, lang=None):
        seconds = lengths[text]
        _tone(settings, output, seconds)
        return SpokenLine(path=output, seconds=seconds, sample_rate=24000, segments=((0.0, seconds),))

    return Speaker(settings, synthesize=synthesize)


def _job(settings, tmp_path, body):
    parsed = parse_bundle(body, settings, {})
    return Job(id="c" * 32, bundle=parsed, dir=tmp_path / "job", status="preparing", created_at=0.0)


def _script(*lines, duration=5, media=None):
    body = bundle(composition={"html": page(duration=duration)})
    body["audio"] = {"voice": {"lines": [dict(line) for line in lines]}}
    if media:
        body["media"] = media
    return body


def test_a_line_past_its_window_is_fitted_and_the_report_says_so(settings, tmp_path):
    lengths = {"first": 2.0, "second": 0.8}
    body = _script({"id": "l01", "at": 0.3, "text": "first"}, {"id": "l02", "at": 2.0, "text": "second"}, duration=4)
    job = _job(settings, tmp_path, body)
    pipeline = RenderPipeline(settings, _speaker(settings, lengths))

    lines = {line.id: line for line in asyncio.run(pipeline._voice(job))}

    first, second = lines["l01"], lines["l02"]
    assert (first.tempo, first.window_end) == (1.25, 2.0)
    assert first.path == job.audio_dir / "fit" / "l01.wav"
    assert first.seconds == pytest.approx(1.6, abs=0.03)
    assert first.at + first.seconds <= 2.0 - GAP + 0.03
    assert first.segments == ((0.0, 1.6),)
    assert (second.tempo, second.window_end, second.seconds) == (None, 4.0, 0.8)
    assert voice_findings(list(lines.values()), 4.0, max_tempo=1.25) == ()
    report = first.report()
    assert report["tempo"] == 1.25 and report["window_end"] == 2.0


def test_a_voice_toolkits_file_is_fitted_like_a_kokoro_line(settings, tmp_path):
    """Whatever the source: a file from a voice toolkit, already in our storage, fits its window too."""
    body = _script(
        {"id": "l01", "at": 0.3, "text": "first"},
        {"id": "l02", "at": 2.0, "path": "assets/voice/l02.mp3"},
        duration=4,
        media=[{"path": "assets/voice/l02.mp3", "url": f"{STORAGE}voice-l02.mp3?X-Amz-Signature=abc"}],
    )
    job = _job(settings, tmp_path, body)
    staged = job.project_dir / "assets" / "voice" / "l02.mp3"
    staged.parent.mkdir(parents=True)
    wav = _tone(settings, tmp_path / "tone.wav", 2.2)
    subprocess.run([settings.ffmpeg_bin, "-v", "error", "-y", "-i", str(wav), str(staged)], check=True)
    pipeline = RenderPipeline(settings, _speaker(settings, {"first": 1.2}))

    lines = {line.id: line for line in asyncio.run(pipeline._voice(job))}

    toolkit = lines["l02"]
    assert toolkit.source == "file" and toolkit.window_end == 4.0
    # an MP3's measured length carries its encoder padding: a few hundredths either way
    assert toolkit.tempo == pytest.approx(2.2 / 1.9, abs=0.03)
    assert toolkit.path == job.audio_dir / "fit" / "l02.wav"
    assert toolkit.seconds == pytest.approx(1.9, abs=0.05)
    assert lines["l01"].tempo is None
    assert voice_findings(list(lines.values()), 4.0, max_tempo=1.25) == ()


def test_a_line_too_long_for_its_window_even_at_the_cap_is_refused_with_its_window(settings, tmp_path):
    body = _script({"id": "l01", "at": 0.3, "text": "first"}, {"id": "l02", "at": 2.0, "text": "second"}, duration=4)
    job = _job(settings, tmp_path, body)
    pipeline = RenderPipeline(settings, _speaker(settings, {"first": 2.4, "second": 0.8}))

    lines = asyncio.run(pipeline._voice(job))

    assert all(line.tempo is None for line in lines)
    (finding,) = voice_findings(lines, 4.0, max_tempo=settings.voice_max_tempo)
    assert finding["code"] == "voice_lines_overlap" and finding["line"] == "l02"
    assert "it lasts 2.40 s and its window is 1.70 s, more than 1.25x speed can fit" in finding["message"]


def test_a_raised_cap_fits_what_the_default_refuses(settings, tmp_path):
    body = _script({"id": "l01", "at": 0.3, "text": "first"}, {"id": "l02", "at": 2.0, "text": "second"}, duration=4)
    job = _job(settings, tmp_path, body)
    pipeline = RenderPipeline(replace(settings, voice_max_tempo=1.5), _speaker(settings, {"first": 2.4, "second": 0.8}))

    lines = {line.id: line for line in asyncio.run(pipeline._voice(job))}

    assert lines["l01"].tempo == pytest.approx(1.5, abs=1e-4)
    assert voice_findings(list(lines.values()), 4.0, max_tempo=1.5) == ()
