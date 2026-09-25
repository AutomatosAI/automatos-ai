"""The fixture composition follows the Hyperframes rules the references learned
(PRD-251A stage 7), and the commands built around it are the reference ones.
The render itself is the CI job's own step: it posts the fixture bundle to
POST /render and asserts the MP4 (it is the timed, end-to-end proof).
"""

from __future__ import annotations

import re
from pathlib import Path

from media_render import hyperframes
from media_render.config import load_settings
from media_render.fixture import COMPOSITION, fixture_bundle

HTML = COMPOSITION.read_text()


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


def test_the_words_on_screen_are_template_variables():
    assert '<h1 id="title">{{ headline }}</h1>' in HTML
    assert fixture_bundle()["variables"] == {"headline": "Rendered on brand"}


def test_the_fixture_bundle_carries_one_kokoro_line_inside_the_composition():
    bundle = fixture_bundle()
    assert bundle["composition"]["html"] == HTML
    voice = bundle["audio"]["voice"]
    assert (voice["voice"], voice["speed"]) == ("af_heart", 0.95)
    assert [(line["id"], line["at"]) for line in voice["lines"]] == [("l01", 0.3)]


def test_render_uses_delivery_quality_at_30_fps_without_a_gpu(tmp_path):
    argv = hyperframes.render_argv(load_settings(), tmp_path, tmp_path / "out.mp4")
    assert argv[:3] == ["hyperframes", "render", str(tmp_path)]
    assert argv[argv.index("--quality") + 1] == "delivery"
    assert argv[argv.index("--fps") + 1] == "30"
    assert "--no-browser-gpu" in argv


def test_the_check_reports_json_without_a_gpu(tmp_path):
    assert hyperframes.check_argv(load_settings(), tmp_path) == [
        "hyperframes", "check", str(tmp_path), "--json", "--no-browser-gpu",
    ]


def test_the_cli_runs_in_the_project_directory(tmp_path):
    run = hyperframes.run_cli(["pwd"], tmp_path, 30, capture=True)
    assert run.ok and Path(run.stdout.strip()).resolve() == tmp_path.resolve()


def test_a_cli_run_past_its_timeout_is_stopped(tmp_path):
    run = hyperframes.run_cli(["sleep", "30"], tmp_path, 1, capture=True)
    assert run.timed_out and not run.ok and run.seconds < 10


def test_a_preview_snapshots_exactly_the_moments_asked_for_without_a_gpu(tmp_path):
    argv = hyperframes.snapshot_argv(load_settings(), tmp_path, tmp_path / "shots", (0.5, 2.0))
    assert argv[:3] == ["hyperframes", "snapshot", str(tmp_path)]
    assert argv[argv.index("--output") + 1] == str(tmp_path / "shots")
    assert argv[argv.index("--at") + 1] == "0.5,2"
    assert argv[argv.index("--describe") + 1] == "false"
    assert "--no-end" in argv and "--no-browser-gpu" in argv
