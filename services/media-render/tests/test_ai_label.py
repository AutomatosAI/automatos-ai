"""The visible "AI-generated" label on a video (PRD-264 EUAI-08a), inside the image.

- the bundle's ``ai_label`` is one line of text, a positive ``seconds`` and ``interval``;
  anything else is refused before anything runs;
- the label shows for the first ``seconds`` and again from the start of every ``interval``,
  read back from the pixels the image's ffmpeg drew (a black video: white text in the top
  left while it shows, black when it does not);
- its text reaches ffmpeg through a file, so quotes, colons and per cent signs are text;
- the real pipeline burns it into ``render.mp4`` before the AI mark, and the mark is still
  read back after it; a bundle without one gets none;
- the image's ffmpeg has ``drawtext`` and the label's font is where the config says.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict

import pytest

from helpers import VOICES, bundle
from media_render import boot
from media_render.ai_label import (
    LABELLED_NAME, TEXT_FILE, WORK_DIR, AiLabel, LabelError, burn, drawtext_filter, parse_ai_label, shows_at,
)
from media_render.bundle import parse_bundle
from media_render.fixture import fixture_bundle
from media_render.jobs import JobStore
from media_render.pipeline import RenderPipeline
from media_render.provenance import COMPOSITE
from media_render.tts import Speaker
from media_render.validate import BundleError
from provenance_checks import ffmpeg, seconds_of, source_type_of

LABEL = {"text": "AI-generated", "seconds": 3, "interval": 30}
# Brightest pixel of the top-left label area: white text over black, or black alone.
SHOWN, HIDDEN = 200, 40
TOP_LEFT = "crop=iw/2:ih/3:0:0,format=gray"


def _brightest(settings, video: Path, at: float) -> int:
    """The brightest grey level of the video's top-left area at ``at`` seconds."""
    raw = subprocess.run(
        [settings.ffmpeg_bin, "-v", "error", "-ss", f"{at:g}", "-i", str(video), "-frames:v", "1", "-vf", TOP_LEFT,
         "-f", "rawvideo", "-"],
        capture_output=True, check=True, timeout=120,
    ).stdout
    assert raw, f"no frame at {at:g} s"
    return max(raw)


def _black_video(settings, path: Path, seconds: int = 3) -> Path:
    ffmpeg(
        settings, "-f", "lavfi", "-i", f"color=c=black:s=640x360:d={seconds}:r=10",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )
    return path


def _burn(settings, video: Path, label: AiLabel, work: Path) -> float:
    return burn(
        video, label, work_dir=work, width=640, height=360, ffmpeg_bin=settings.ffmpeg_bin,
        font_path=settings.ai_label_font_path, timeout_seconds=settings.label_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def test_a_label_is_one_line_with_a_positive_time_and_interval():
    assert parse_ai_label(None) is None
    assert parse_ai_label(LABEL) == AiLabel(text="AI-generated", seconds=3.0, interval=30.0)
    assert parse_ai_label({**LABEL, "text": "  Gerado por IA "}).text == "Gerado por IA"
    for bad in (
        {"text": "AI-generated", "seconds": 3},
        {**LABEL, "colour": "red"},
        {**LABEL, "text": ""},
        {**LABEL, "text": "A" * 41},
        {**LABEL, "text": "AI-\ngenerated"},
        {**LABEL, "seconds": 0},
        {**LABEL, "interval": -30},
        {**LABEL, "seconds": True},
        "AI-generated",
    ):
        with pytest.raises(BundleError):
            parse_ai_label(bad)


def test_the_bundle_takes_the_label_and_refuses_a_bad_one(settings):
    assert parse_bundle(bundle(), settings, {}, VOICES).ai_label is None
    assert parse_bundle(bundle(ai_label=LABEL), settings, {}, VOICES).ai_label == parse_ai_label(LABEL)
    with pytest.raises(BundleError):
        parse_bundle(bundle(ai_label={**LABEL, "seconds": "3"}), settings, {}, VOICES)


def test_the_label_shows_at_the_start_and_again_every_interval():
    label = parse_ai_label(LABEL)
    assert [shows_at(label, t) for t in (0, 2.9, 3.0, 29.9, 30.0, 32.5, 33.0, 61.0)] == [
        True, True, False, False, True, True, False, True,
    ]
    always = AiLabel(text="AI-generated", seconds=5, interval=5)
    assert all(shows_at(always, t) for t in (0, 4.99, 5, 12.3))


def test_the_filter_reads_the_text_from_its_file_with_expansion_off(settings):
    vf = drawtext_filter(parse_ai_label(LABEL), width=1080, height=1920, font_path=settings.ai_label_font_path)
    assert vf.startswith(f"drawtext=fontfile={settings.ai_label_font_path}:textfile={TEXT_FILE}:expansion=none:")
    assert ":fontsize=49:" in vf and ":x=43:y=43:" in vf  # 4.5 % and 4 % of the shorter side
    assert vf.endswith(":enable='lt(mod(t,30),3)'")
    assert "AI-generated" not in vf, "the text never enters the filter"


# ---------------------------------------------------------------------------
# The pixels
# ---------------------------------------------------------------------------


def test_the_label_is_drawn_while_it_shows_and_gone_between(settings, tmp_path):
    video = _black_video(settings, tmp_path / "render.mp4")
    work = tmp_path / WORK_DIR
    took = _burn(settings, video, AiLabel(text="AI-generated", seconds=1, interval=2), work)
    assert took >= 0 and not (work / LABELLED_NAME).exists(), "the labelled copy replaced the video"
    assert _brightest(settings, video, 0.5) >= SHOWN
    assert _brightest(settings, video, 1.5) <= HIDDEN
    assert _brightest(settings, video, 2.5) >= SHOWN
    assert abs(seconds_of(settings, video) - 3.0) <= 0.2


def test_quotes_colons_and_per_cent_signs_are_text(settings, tmp_path):
    video = _black_video(settings, tmp_path / "render.mp4", seconds=1)
    text = "Off: it's 50%, [AI]; a,b='c'"
    _burn(settings, video, parse_ai_label({**LABEL, "text": text, "seconds": 1}), tmp_path / WORK_DIR)
    assert (tmp_path / WORK_DIR / TEXT_FILE).read_text(encoding="utf-8") == text
    assert _brightest(settings, video, 0.5) >= SHOWN


def test_a_burn_that_fails_says_so_and_leaves_the_video(settings, tmp_path):
    video = tmp_path / "render.mp4"
    video.write_bytes(b"not a video")
    with pytest.raises(LabelError):
        _burn(settings, video, parse_ai_label(LABEL), tmp_path / WORK_DIR)
    assert video.read_bytes() == b"not a video"


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------


def _render(settings, body: Dict[str, Any]):
    job = JobStore(Path(settings.work_dir)).create(parse_bundle(body, settings, {}, VOICES))
    pipeline = RenderPipeline(settings, Speaker(settings))

    async def go():
        await pipeline.start()
        try:
            outcome = await pipeline.prepare_and_check(job)
            assert outcome.ok, json.dumps(outcome.report, indent=2)[:4000]
            return await pipeline.render(job)
        finally:
            await pipeline.close()

    return job, asyncio.run(go())


def test_the_pipeline_burns_the_label_in_before_the_mark(settings):
    inputs = {"footage": False, "stills": False, "voice": True, "uploads": False, "template": True}
    body = {**fixture_bundle(), "ai_inputs": inputs, "ai_label": {"text": "AI-generated", "seconds": 1, "interval": 2}}
    job, result = _render(settings, body)
    (output,) = result.outputs
    video = job.output_dir / output["name"]
    assert output["name"] == "render.mp4" and output["bytes"] == video.stat().st_size
    assert "label_seconds" in result.timings
    assert source_type_of(settings, video) == COMPOSITE, "the mark follows the label's encode"
    assert _brightest(settings, video, 0.5) >= SHOWN
    assert _brightest(settings, video, 1.5) <= HIDDEN

    plain_job, plain = _render(settings, {**fixture_bundle(), "ai_inputs": inputs})
    assert "label_seconds" not in plain.timings
    assert _brightest(settings, plain_job.output_dir / "render.mp4", 0.5) <= HIDDEN


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------


def test_the_images_ffmpeg_draws_text_and_the_labels_font_is_there(settings, tmp_path):
    filters = subprocess.run([settings.ffmpeg_bin, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=60)
    assert " drawtext " in filters.stdout
    assert Path(settings.ai_label_font_path).is_file()
    assert boot.label_font_problems(settings) == []
    missing = replace(settings, ai_label_font_path=str(tmp_path / "no-font.ttf"))
    (problem,) = boot.label_font_problems(missing)
    assert "MEDIA_RENDER_AI_LABEL_FONT" in problem and problem in boot.boot_problems(missing, {})
