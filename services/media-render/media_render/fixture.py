"""Render the committed fixture: the image's end-to-end proof (US-101).

Stages the 3-second composition in a scratch project, speaks its script line
with Kokoro, places the line on a bed the length of the composition, records
``hyperframes check`` (not a gate yet: US-102 makes /render refuse on it), and
renders the MP4. The MP4 and a timed report land in the output directory.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import audio, hyperframes
from .config import Settings
from .kokoro_tts import synthesize_line
from .versions import read_versions

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
COMPOSITION_DIR = FIXTURES / "fixture"
AUDIO_PLAN = FIXTURES / "fixture.audio.json"

OUTPUT_NAME = "fixture.mp4"
REPORT_NAME = "fixture.json"
CHECK_NAME = "check.json"

# Where the composition expects its staged files (fixtures/fixture/index.html).
GSAP_TARGET = Path("assets/vendor/gsap.min.js")
VOICE_DIR = Path("assets/vo")
MIX_TARGET = Path("assets/audio/mix.wav")

_LINE_ID = re.compile(r"^[a-z0-9_-]+$")


class FixtureError(RuntimeError):
    pass


def load_audio_plan(path: Path = AUDIO_PLAN) -> Dict[str, Any]:
    plan = json.loads(path.read_text())
    duration = float(plan["duration"])
    if duration <= 0:
        raise FixtureError("the audio plan needs a positive duration")
    lines = plan["lines"]
    if not lines:
        raise FixtureError("the audio plan needs at least one line")
    for line in lines:
        if not _LINE_ID.match(line["id"]):
            raise FixtureError(f"line id {line['id']!r} must be lowercase letters, digits, - or _")
        if not line["text"].strip():
            raise FixtureError(f"line {line['id']} has no text")
        if not 0 <= float(line["at"]) < duration:
            raise FixtureError(f"line {line['id']} starts outside the {duration}s composition")
    return {"duration": duration, "lines": lines}


def stage_project(target: Path, settings: Settings) -> Path:
    """Copy the composition into ``target`` with GSAP staged beside it."""
    shutil.copytree(COMPOSITION_DIR, target)
    gsap = target / GSAP_TARGET
    gsap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(settings.gsap_path, gsap)
    return target


def _check_ok(stdout: str) -> Optional[bool]:
    try:
        report = json.loads(stdout)
    except ValueError:
        return None
    return report.get("ok") if isinstance(report, dict) else None


def _place_voice(settings: Settings, plan: Dict[str, Any], spoken: List[Path], project: Path) -> float:
    mix = project / MIX_TARGET
    mix.parent.mkdir(parents=True, exist_ok=True)
    lines = [(path, float(line["at"])) for path, line in zip(spoken, plan["lines"])]
    argv = audio.voice_placement_argv(settings.ffmpeg_bin, lines, plan["duration"], mix)
    started = time.monotonic()
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=settings.mix_timeout_seconds, check=False)
    if proc.returncode != 0 or not mix.is_file():
        raise FixtureError(f"placing the voice failed (ffmpeg exit {proc.returncode}): {proc.stderr[-2000:]}")
    return round(time.monotonic() - started, 3)


def render_fixture(out_dir: Path, settings: Settings) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = load_audio_plan()
    timings: Dict[str, float] = {}
    started = time.monotonic()
    output = out_dir / OUTPUT_NAME
    with tempfile.TemporaryDirectory(prefix="media-render-fixture-") as scratch:
        project = stage_project(Path(scratch) / "project", settings)

        tts_started = time.monotonic()
        spoken = [
            synthesize_line(line["text"], project / VOICE_DIR / f"{line['id']}.wav", settings)
            for line in plan["lines"]
        ]
        timings["tts_seconds"] = round(time.monotonic() - tts_started, 3)
        timings["mix_seconds"] = _place_voice(settings, plan, [line.path for line in spoken], project)

        check = hyperframes.run_cli(
            hyperframes.check_argv(settings, project), project, settings.check_timeout_seconds, capture=True
        )
        (out_dir / CHECK_NAME).write_text(check.stdout or json.dumps({"stderr": check.stderr[-4000:]}))
        timings["check_seconds"] = check.seconds

        render = hyperframes.run_cli(
            hyperframes.render_argv(settings, project, output), project, settings.render_timeout_seconds, capture=False
        )
        timings["render_seconds"] = render.seconds
        if not render.ok or not output.is_file():
            reason = "timed out" if render.timed_out else f"exit {render.returncode}"
            raise FixtureError(f"hyperframes render failed ({reason})")
    timings["total_seconds"] = round(time.monotonic() - started, 3)

    report = {
        "output": OUTPUT_NAME,
        "bytes": output.stat().st_size,
        "duration": plan["duration"],
        "voice": [
            {"id": line["id"], "text": line["text"], "seconds": said.seconds, "sample_rate": said.sample_rate}
            for line, said in zip(plan["lines"], spoken)
        ],
        "check": {"exit_code": check.returncode, "timed_out": check.timed_out, "ok": _check_ok(check.stdout)},
        "timings": timings,
        "versions": read_versions(settings.versions_path),
    }
    (out_dir / REPORT_NAME).write_text(json.dumps(report, indent=2) + "\n")
    return report
