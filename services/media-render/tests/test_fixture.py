"""The fixture composition follows the Hyperframes rules the references learned
(PRD-251A stage 7), and the commands built around it are the reference ones.
The render itself is the CI job's own step (it is the timed proof).
"""

from __future__ import annotations

import re
from pathlib import Path

from media_render import audio, hyperframes
from media_render.config import load_settings
from media_render.fixture import AUDIO_PLAN, COMPOSITION_DIR, load_audio_plan, stage_project

HTML = (COMPOSITION_DIR / "index.html").read_text()


def test_one_root_composition_of_three_seconds_at_1080x1920():
    roots = re.findall(r"<div[^>]*data-composition-id=\"([^\"]+)\"[^>]*>", HTML)
    assert roots == ["main"]
    root = re.search(r"<div[^>]*data-composition-id=\"main\"[^>]*>", HTML).group(0)
    for attribute in ('data-duration="3"', 'data-width="1080"', 'data-height="1920"'):
        assert attribute in root


def test_one_paused_timeline_registered_last_on_main():
    assert HTML.count("gsap.timeline(") == 1
    assert "gsap.timeline({ paused: true })" in HTML
    script = HTML[HTML.rindex("<script>") : HTML.rindex("</script>")]
    assert script.rstrip().endswith('window.__timelines["main"] = tl;')


def test_the_composition_is_deterministic():
    for banned in ("Math.random", "Date.now", "repeat: -1", "repeat:-1", "fetch("):
        assert banned not in HTML, banned


def test_one_premixed_audio_track_and_a_local_gsap():
    assert HTML.count("<audio") == 1
    assert 'src="assets/audio/mix.wav"' in HTML
    assert '<script src="assets/vendor/gsap.min.js"></script>' in HTML
    assert "http" not in HTML, "the fixture must render without the network"


def test_colours_come_from_brand_tokens_with_fallbacks():
    colours = re.findall(r"#[0-9a-fA-F]{3,8}\b", HTML)
    in_fallbacks = re.findall(r"var\(--brand-[a-z]+, (#[0-9a-fA-F]{3,8})\)", HTML)
    assert colours and sorted(colours) == sorted(in_fallbacks)


def test_the_audio_plan_fits_the_composition():
    plan = load_audio_plan(AUDIO_PLAN)
    assert plan["duration"] == 3.0
    assert [line["id"] for line in plan["lines"]] == ["l01"]


def test_voice_placement_is_the_reference_voice_graph(tmp_path):
    argv = audio.voice_placement_argv("ffmpeg", [(tmp_path / "l01.wav", 0.3)], 3.0, tmp_path / "mix.wav")
    graph = argv[argv.index("-filter_complex") + 1]
    assert "[0:a]aformat=sample_rates=48000:channel_layouts=stereo,adelay=300|300[v0]" in graph
    assert "amix=inputs=1:normalize=0:duration=longest,apad=whole_dur=3.0,atrim=0:3.0[out]" in graph
    assert argv[-5:] == ["-map", "[out]", "-c:a", "pcm_s16le", str(tmp_path / "mix.wav")]


def test_a_line_outside_the_bed_is_refused(tmp_path):
    for start in (-0.1, 3.0):
        try:
            audio.voice_placement_argv("ffmpeg", [(tmp_path / "l.wav", start)], 3.0, tmp_path / "m.wav")
        except ValueError:
            continue
        raise AssertionError(f"a line at {start}s was placed on a 3s bed")


def test_render_uses_delivery_quality_at_30_fps_without_a_gpu(tmp_path):
    argv = hyperframes.render_argv(load_settings(), tmp_path, tmp_path / "out.mp4")
    assert argv[:3] == ["hyperframes", "render", str(tmp_path)]
    assert argv[argv.index("--quality") + 1] == "delivery"
    assert argv[argv.index("--fps") + 1] == "30"
    assert "--no-browser-gpu" in argv


def test_the_cli_runs_in_the_project_directory(tmp_path):
    run = hyperframes.run_cli(["pwd"], tmp_path, 30, capture=True)
    assert run.ok and Path(run.stdout.strip()).resolve() == tmp_path.resolve()


def test_staging_copies_the_composition_and_gsap(tmp_path):
    project = stage_project(tmp_path / "project", load_settings())
    assert (project / "index.html").read_text() == HTML
    assert (project / "assets" / "vendor" / "gsap.min.js").stat().st_size > 10_000
