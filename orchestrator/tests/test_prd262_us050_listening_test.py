"""PRD-262 US-050 (D4): the fp16 voice listening test, on demand in media-render.yml.

Gerard picks the renderer's fp16 Kokoro model by ear. ``media-render.yml``,
dispatched with ``listening_test`` true, speaks the script-window step's script
with today's fp32 model and both fp16 candidates (the model-files-v1.1 export,
called with zero pauses, and the model-files-v1.0 release's fp16), each file
sha256-checked before it loads. It keeps the three WAVs as one artifact and writes
each model's voice_seconds to the job summary. The shipped model does not change.

The synthesis runs only in that job, inside the image. This suite pins the job and
the helper's pure parts (services/media-render/ci/listening_test.py): the
candidates and their pins, the sha256 refusal, the create() call each model gets,
the joined sample and the summary table. Pure file reads and numpy: no Docker, no
model, no database.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MEDIA_RENDER = ROOT / "services" / "media-render"
HELPER = MEDIA_RENDER / "ci" / "listening_test.py"
WORKFLOW = ROOT / ".github" / "workflows" / "media-render.yml"
DEFAULTS = SimpleNamespace(kokoro_voice="af_heart", kokoro_speed=0.95, kokoro_lang="en-us")
ZERO_PAUSES = {"sentence_pause": 0.0, "clause_pause": 0.0}
WAVS = ("fp32.wav", "fp16-v1.1.wav", "fp16-v1.0.wav")


def _helper():
    spec = importlib.util.spec_from_file_location("media_render_listening_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _job():
    return _workflow()["jobs"]["media-render-listening-test"]


def _commands(job) -> str:
    return "\n".join(step.get("run", "") for step in job["steps"])


def _script(helper):
    return helper.script_from_bundle(json.loads((MEDIA_RENDER / "fixtures" / "script.bundle.json").read_text()), DEFAULTS)


def _report(label: str, voice_seconds: float, line_seconds) -> dict:
    return {
        "label": label, "title": label, "file": f"model-files-v1.0/{label}.onnx", "bytes": 163_527_961,
        "sha256": "ab" * 32, "pauses": ZERO_PAUSES if label == "fp16-v1.1" else {},
        "script": {"lines": ["l01", "l02"], "voice": "af_heart", "speed": 0.95, "lang": "en-us"},
        "load_seconds": 1.0, "voice_seconds": voice_seconds, "warm_seconds": 0.5, "line_seconds": line_seconds,
        "sample_rate": 24000,
    }


# ---------------------------------------------------------------------------
# The workflow: an input, off by default, that adds one job
# ---------------------------------------------------------------------------


def test_the_listening_test_is_a_dispatch_input_off_by_default():
    on = _workflow().get("on", _workflow().get(True))
    assert list(on) == ["workflow_dispatch"], "the listening test adds no trigger: media-render.yml stays on demand"
    inputs = on["workflow_dispatch"]["inputs"]
    assert list(inputs) == ["listening_test"]
    assert inputs["listening_test"]["type"] == "boolean"
    assert inputs["listening_test"]["default"] is False


def test_the_listening_job_runs_only_when_asked_and_the_other_jobs_as_before():
    jobs = _workflow()["jobs"]
    job = jobs["media-render-listening-test"]
    assert job["if"].replace(" ", "") == "${{inputs.listening_test}}"
    assert "needs" not in job
    for other in ("media-render", "media-render-templates"):
        assert "if" not in jobs[other], f"{other} must still run on every dispatch"


def test_the_job_measures_the_image_railway_builds_with_the_helper_mounted_not_built_in():
    commands = _commands(_job())
    assert 'docker build -t "$IMAGE" services/media-render/' in commands
    assert "--build-arg" not in commands and "--target" not in commands
    # ci/ stays out of the build context, so the image is the one Railway builds.
    assert re.search(r"^ci/$", (MEDIA_RENDER / ".dockerignore").read_text(), re.M)
    assert '"$GITHUB_WORKSPACE/services/media-render/ci:/ci:ro"' in commands


def test_the_job_speaks_the_script_window_steps_script_once_per_model_in_a_fresh_container():
    commands = _commands(_job())
    script_window = _commands(_workflow()["jobs"]["media-render"])
    assert '"$IMAGE" fixture-bundle script > "$out/bundle.json"' in script_window
    assert '"$IMAGE" fixture-bundle script > "$out/bundle.json"' in commands
    assert "listening_test.py downloads" in commands and 'curl -fsSL --retry 3 -o "$models/$label.onnx"' in commands
    assert "labels=$(python3 services/media-render/ci/listening_test.py labels)" in commands
    assert re.search(r'for model in \$labels; do\s.*docker run --rm', commands, re.S)
    assert "--entrypoint python \"$IMAGE\" /ci/listening_test.py speak" in commands
    assert '--model "$model"' in commands and '"$RUNNER_TEMP/kokoro:/models:ro"' in commands


def test_the_summary_carries_voice_seconds_and_one_artifact_holds_the_three_wavs():
    job = _job()
    commands = _commands(job)
    assert 'listening_test.py summary --out "$out"' in commands
    assert '>> "$GITHUB_STEP_SUMMARY"' in commands
    uploads = [step for step in job["steps"] if "upload-artifact" in str(step.get("uses"))]
    assert len(uploads) == 1
    upload = uploads[0]["with"]
    assert upload["name"] == "media-render-listening-test"
    assert [Path(line.strip()).name for line in upload["path"].strip().splitlines()] == list(WAVS)
    assert upload["if-no-files-found"] == "error"


# ---------------------------------------------------------------------------
# The helper: the candidates, their pins and the call each one gets
# ---------------------------------------------------------------------------


def test_the_candidates_are_todays_fp32_and_the_two_fp16_exports():
    helper = _helper()
    assert [c.label for c in helper.CANDIDATES] == ["fp32", "fp16-v1.1", "fp16-v1.0"]
    fp32, v11, v10 = helper.CANDIDATES
    assert (fp32.release, fp32.file, fp32.in_image) == ("model-files-v1.0", "kokoro-v1.0.onnx", True)
    assert (v11.release, v11.file, v11.in_image) == ("model-files-v1.1", "kokoro-v1.0.fp16.onnx", False)
    assert (v10.release, v10.file, v10.in_image) == ("model-files-v1.0", "kokoro-v1.0.fp16.onnx", False)
    assert all(re.fullmatch(r"[0-9a-f]{64}", c.sha256) for c in helper.CANDIDATES)
    assert len({c.sha256 for c in helper.CANDIDATES}) == 3


def test_fp32_is_checked_against_the_images_own_pin():
    sums = (MEDIA_RENDER / "kokoro" / "SHA256SUMS").read_text().split("\n")
    pinned = dict(reversed(line.split("  ", 1)) for line in sums if line.strip())
    assert _helper().BY_LABEL["fp32"].sha256 == pinned["kokoro-v1.0.onnx"]


def test_the_job_fetches_only_the_two_fp16_files_from_their_releases(capsys):
    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download"
    assert _helper().main(["downloads"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"fp16-v1.1 {base}/model-files-v1.1/kokoro-v1.0.fp16.onnx",
        f"fp16-v1.0 {base}/model-files-v1.0/kokoro-v1.0.fp16.onnx",
    ]


def test_the_job_speaks_with_every_model_in_order(capsys):
    assert _helper().main(["labels"]) == 0
    assert capsys.readouterr().out.split() == ["fp32", "fp16-v1.1", "fp16-v1.0"]


def test_only_the_v1_1_export_is_called_with_zero_pauses():
    """The v1.1 graph reports durations, so kokoro-onnx 0.6.1 would top up a pause at every
    mark; zero keeps today's timing. The others get the call renders make today."""
    helper = _helper()
    assert {c.label: dict(c.pauses) for c in helper.CANDIDATES} == {
        "fp32": {}, "fp16-v1.1": ZERO_PAUSES, "fp16-v1.0": {},
    }


def test_a_model_file_whose_sha256_differs_is_refused(tmp_path):
    helper = _helper()
    model = tmp_path / "fp16-v1.1.onnx"
    model.write_bytes(b"not the model")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        helper.check_sha256(model, helper.BY_LABEL["fp16-v1.1"].sha256)
    assert helper.check_sha256(model, hashlib.sha256(b"not the model").hexdigest())
    with pytest.raises(ValueError, match="no model file at"):
        helper.check_sha256(tmp_path / "fp16-v1.0.onnx", helper.BY_LABEL["fp16-v1.0"].sha256)


def test_the_script_is_the_fixture_script_as_a_render_reads_it():
    script = _script(_helper())
    assert [line_id for line_id, _ in script.lines] == ["l01", "l02", "l03", "l04"]
    assert script.lines[0][1] == "Rendered on brand, in three seconds."
    assert (script.voice, script.speed, script.lang) == ("af_heart", 0.95, "en-us")


class _Model:
    def __init__(self, samples):
        self.samples, self.calls = samples, []

    def create(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return self.samples, 24000


def test_each_line_is_spoken_with_the_scripts_voice_and_the_models_pauses():
    helper = _helper()
    script = _script(helper)
    model = _Model(numpy.full(2400, 0.2, dtype=numpy.float32))
    parts, rate = helper.speak_lines(model, script, ZERO_PAUSES)
    assert rate == 24000 and len(parts) == 4
    assert [text for text, _ in model.calls] == [text for _, text in script.lines]
    assert all(kw == {"voice": "af_heart", "speed": 0.95, "lang": "en-us", **ZERO_PAUSES} for _, kw in model.calls)


def test_a_silent_or_broken_line_fails_the_listening_test():
    helper = _helper()
    script = _script(helper)
    with pytest.raises(ValueError, match="line l01 came out silent"):
        helper.speak_lines(_Model(numpy.zeros(2400, dtype=numpy.float32)), script, {})
    with pytest.raises(ValueError, match="line l01"):
        helper.speak_lines(_Model(numpy.full(2400, numpy.nan, dtype=numpy.float32)), script, {})


def test_the_lines_are_joined_with_a_fixed_gap():
    helper = _helper()
    joined = helper.join_lines([numpy.ones(100), numpy.ones(50)], 1000, gap_seconds=0.5)
    assert len(joined) == 100 + 500 + 50
    assert joined.dtype == numpy.float32
    assert not joined[100:600].any() and joined[:100].all() and joined[600:].all()


# ---------------------------------------------------------------------------
# The job summary
# ---------------------------------------------------------------------------


def test_the_summary_has_a_row_per_model_with_voice_seconds_against_fp32(tmp_path):
    helper = _helper()
    timings = {"fp32": (2.0, [1.0, 1.5]), "fp16-v1.1": (3.0, [1.0, 1.5]), "fp16-v1.0": (2.5, [1.25, 1.5])}
    for label, (voice_seconds, lines) in timings.items():
        (tmp_path / f"{label}.json").write_text(json.dumps(_report(label, voice_seconds, lines)))
    table = helper.summary(helper.load_reports(tmp_path))
    rows = [line for line in table.split("\n") if line.startswith("| ") and "**" in line]
    assert [row.split(" | ")[0] for row in rows] == ["| fp32", "| fp16-v1.1", "| fp16-v1.0"]
    assert "**2.000** | 1.00x" in rows[0] and "**3.000** | 1.50x" in rows[1] and "**2.500** | 1.25x" in rows[2]
    assert "2.500 (+0.000)" in rows[1] and "2.750 (+0.250)" in rows[2]
    assert "sentence_pause=0, clause_pause=0" in rows[1] and "defaults (today's call)" in rows[2]
    assert all(wav in table for wav in WAVS)


def test_the_summary_refuses_a_missing_model(tmp_path):
    helper = _helper()
    (tmp_path / "fp32.json").write_text(json.dumps(_report("fp32", 2.0, [1.0])))
    with pytest.raises(ValueError, match="no report for fp16-v1.1, fp16-v1.0"):
        helper.load_reports(tmp_path)
    assert helper.main(["summary", "--out", str(tmp_path)]) == 1
