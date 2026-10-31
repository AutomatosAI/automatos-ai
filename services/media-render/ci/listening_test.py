#!/usr/bin/env python3
"""The fp16 voice listening test (PRD-262 US-050, D4): one script, three Kokoro models.

media-render speaks with Kokoro v1.0 in fp32 (325.5 MB). D4 moves it to an fp16
export once Gerard has picked one by ear. This speaks the fixture script that the
script-window step renders (``fixture-bundle script``: four lines, af_heart, speed
0.95) with each model:

- ``fp32``: today's model, the file the image already holds (checked at build),
  called as ``kokoro_tts.synthesize_line`` calls it;
- ``fp16-v1.1``: the model-files-v1.1 export (163.5 MB). Its graph also outputs
  phoneme durations, and kokoro-onnx 0.6.1 then tops up the pause after every
  sentence and clause mark (0.25 s / 0.1 s by default), line ends included. It is
  called with ``sentence_pause=0, clause_pause=0``, which keeps today's timing;
- ``fp16-v1.0``: the model-files-v1.0 release's fp16 (177.5 MB). It has today's
  graph, so it is called as today.

Every model file is checked against its pinned sha256 before it loads. The job
runs ``speak`` once per model, each in a fresh container, so each model's first
call pays its own imports and load, as a render's first voice stage does.
``voice_seconds`` is that load plus the four lines. ``warm_seconds`` is the same
four lines spoken again, with the model already loaded. Each model's lines go into
one WAV, ``LINE_GAP_SECONDS`` apart, for listening.

Commands:
  labels                         the models' labels, in order (runner, stdlib)
  downloads                      "label url" for each file to fetch (runner, stdlib)
  speak --model L --models DIR --bundle FILE --out DIR
                                 speak the script with one model (INSIDE the image)
  summary --out DIR              the job summary's table, in Markdown (runner, stdlib)

ci/ is not in the image (.dockerignore). listening_test.sh, which media-render.yml
runs and a machine that builds locally runs the same way, mounts it read-only, so
the image it measures is exactly the one Railway builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

RELEASES = "https://github.com/thewh1teagle/kokoro-onnx/releases/download"
# The silence between the script's lines in each sample, so they are heard apart.
LINE_GAP_SECONDS = 0.5
# A line whose loudest sample is this quiet came out silent (tests/test_kokoro.py's bar).
SILENT_PEAK = 0.01
CHUNK_BYTES = 1 << 20
MEGABYTE = 1_000_000


class Candidate(NamedTuple):
    """One model under test: where it comes from, its pinned bytes, and how it is called."""

    label: str
    title: str
    release: str
    file: str
    sha256: str
    # Extra keyword arguments for Kokoro.create(). Empty: the call renders make today.
    pauses: Tuple[Tuple[str, float], ...] = ()
    # Today's fp32 model is already in the image; the others are fetched by the job.
    in_image: bool = False

    @property
    def url(self) -> str:
        return f"{RELEASES}/{self.release}/{self.file}"


# fp32: kokoro/SHA256SUMS, the image's own pin. fp16-v1.1: GitHub's asset digest,
# matched by a download on 18 Oct 2026; upstream re-uploaded the v1.1 files on
# 18 Aug 2026, so a later re-upload fails here, not silently. fp16-v1.0: the
# model-files-v1.0 assets carry no GitHub digest, so this is the sha256 of a
# download on 18 Oct 2026 (177,464,787 bytes, the asset's listed size).
CANDIDATES: Tuple[Candidate, ...] = (
    Candidate(
        "fp32", "fp32 (today)", "model-files-v1.0", "kokoro-v1.0.onnx",
        "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
        in_image=True,
    ),
    Candidate(
        "fp16-v1.1", "fp16, v1.1 export", "model-files-v1.1", "kokoro-v1.0.fp16.onnx",
        "f3a290d384fbb27966d462905c71a46cef9e5fd00516b40df32a0b4afe77ac96",
        pauses=(("sentence_pause", 0.0), ("clause_pause", 0.0)),
    ),
    Candidate(
        "fp16-v1.0", "fp16, v1.0 release", "model-files-v1.0", "kokoro-v1.0.fp16.onnx",
        "c1610a859f3bdea01107e73e50100685af38fff88f5cd8e5c56df109ec880204",
    ),
)
BY_LABEL: Dict[str, Candidate] = {candidate.label: candidate for candidate in CANDIDATES}


class Script(NamedTuple):
    """What to speak and how: a render bundle's ``audio.voice`` block."""

    lines: Tuple[Tuple[str, str], ...]
    voice: str
    speed: float
    lang: str


def script_from_bundle(bundle: Mapping[str, Any], defaults: Any) -> Script:
    """The bundle's lines, voice, speed and language, with the service's defaults where it names none."""
    voice = bundle["audio"]["voice"]
    lines = tuple((line["id"], line["text"].strip()) for line in voice["lines"])
    if not lines:
        raise ValueError("the bundle has no voice lines to speak")
    return Script(
        lines=lines,
        voice=voice.get("voice", defaults.kokoro_voice),
        speed=float(voice.get("speed", defaults.kokoro_speed)),
        lang=voice.get("lang", defaults.kokoro_lang),
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_sha256(path: Path, expected: str) -> str:
    """Refuse a model file that is missing, or whose bytes are not the pinned ones."""
    if not path.is_file():
        raise ValueError(f"no model file at {path}")
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def speak_lines(model: Any, script: Script, pauses: Mapping[str, float]) -> Tuple[List[Any], int]:
    """Each line through ``model.create`` as a render speaks it, plus ``pauses``; refuses silence."""
    import numpy

    parts: List[Any] = []
    rate = 0
    for line_id, text in script.lines:
        samples, rate = model.create(text, voice=script.voice, speed=script.speed, lang=script.lang, **pauses)
        samples = numpy.asarray(samples, dtype=numpy.float32)
        if not numpy.isfinite(samples).all() or float(numpy.abs(samples).max(initial=0.0)) <= SILENT_PEAK:
            raise ValueError(f"line {line_id} came out silent or not a number")
        parts.append(samples)
    return parts, rate


def join_lines(parts: Sequence[Any], sample_rate: int, gap_seconds: float = LINE_GAP_SECONDS) -> Any:
    """The lines one after another, ``gap_seconds`` of silence between them."""
    import numpy

    silence = numpy.zeros(int(round(gap_seconds * sample_rate)), dtype=numpy.float32)
    joined: List[Any] = []
    for index, part in enumerate(parts):
        if index:
            joined.append(silence)
        joined.append(numpy.asarray(part, dtype=numpy.float32))
    return numpy.concatenate(joined)


def _model_path(chosen: Candidate, models: Path, settings: Any) -> Path:
    return Path(settings.kokoro_model_path) if chosen.in_image else models / f"{chosen.label}.onnx"


def _report(chosen: Candidate, path: Path, script: Script, timings: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "label": chosen.label,
        "title": chosen.title,
        "file": f"{chosen.release}/{chosen.file}",
        "bytes": path.stat().st_size,
        "sha256": chosen.sha256,
        "pauses": dict(chosen.pauses),
        # A laptop's arm64 run times fp16 on other kernels than Railway's x86_64.
        "machine": platform.machine(),
        "script": {
            "lines": [line_id for line_id, _ in script.lines],
            "voice": script.voice,
            "speed": script.speed,
            "lang": script.lang,
        },
        **timings,
    }


def speak(label: str, models: Path, bundle: Path, out: Path) -> Dict[str, Any]:
    """Speak the script with one model into ``out/<label>.wav`` and ``out/<label>.json`` (inside the image)."""
    import soundfile
    from media_render.config import load_settings
    # The loader a render's first line goes through, so the load is timed as a render pays it.
    from media_render.kokoro_tts import _model as load_model

    settings = load_settings()
    chosen = BY_LABEL[label]
    path = _model_path(chosen, models, settings)
    check_sha256(path, chosen.sha256)
    script = script_from_bundle(json.loads(bundle.read_text()), settings)
    pauses = dict(chosen.pauses)
    started = time.monotonic()
    model = load_model(str(path), settings.kokoro_voices_path, settings.espeak_data_path)
    loaded = time.monotonic()
    parts, rate = speak_lines(model, script, pauses)
    spoken = time.monotonic()
    speak_lines(model, script, pauses)
    timings = {
        "load_seconds": round(loaded - started, 3),
        "voice_seconds": round(spoken - started, 3),
        "warm_seconds": round(time.monotonic() - spoken, 3),
        "line_seconds": [round(len(part) / rate, 3) for part in parts],
        "sample_rate": rate,
    }
    soundfile.write(str(out / f"{label}.wav"), join_lines(parts, rate), rate)
    report = _report(chosen, path, script, timings)
    (out / f"{label}.json").write_text(json.dumps(report, indent=1))
    return report


def load_reports(out: Path) -> List[Dict[str, Any]]:
    """Every model's report, in CANDIDATES order; one missing is an error."""
    missing = [c.label for c in CANDIDATES if not (out / f"{c.label}.json").is_file()]
    if missing:
        raise ValueError(f"no report for {', '.join(missing)} in {out}")
    return [json.loads((out / f"{c.label}.json").read_text()) for c in CANDIDATES]


def _pauses_cell(pauses: Mapping[str, float]) -> str:
    return ", ".join(f"{name}={value:g}" for name, value in pauses.items()) or "defaults (today's call)"


def _row(report: Mapping[str, Any], fp32: Mapping[str, Any]) -> str:
    spoken = sum(report["line_seconds"])
    delta = spoken - sum(fp32["line_seconds"])
    return (
        f"| {report['title']} | `{report['file']}` | {report['bytes'] / MEGABYTE:.1f} "
        f"| `{report['sha256'][:12]}` ok | {_pauses_cell(report['pauses'])} "
        f"| {report['load_seconds']:.3f} | **{report['voice_seconds']:.3f}** "
        f"| {report['voice_seconds'] / fp32['voice_seconds']:.2f}x | {report['warm_seconds']:.3f} "
        f"| {spoken:.3f} ({delta:+.3f}) |"
    )


def summary(reports: Sequence[Mapping[str, Any]]) -> str:
    """The job summary: one row per model, voice_seconds against fp32's."""
    fp32 = next(report for report in reports if report["label"] == "fp32")
    script = fp32["script"]
    return "\n".join([
        "### US-050 listening test (PRD-262 D4): fp32 against the two fp16 Kokoro models",
        "",
        f"The script-window step's script (`fixture-bundle script`): lines {', '.join(script['lines'])}, "
        f"voice {script['voice']}, speed {script['speed']:g}, {script['lang']}. Spoken on {fp32['machine']} "
        "(Railway runs x86_64: timings from another machine are not production's).",
        "",
        "| Model | Release file | MB | sha256 | `create()` pauses | Load s | voice_seconds | vs fp32 "
        "| Warm s | Spoken s (vs fp32) |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
        *(_row(report, fp32) for report in reports),
        "",
        "voice_seconds: the first call in a fresh container, the model load plus the four lines, as a "
        "render's first voice stage. Warm: the same four lines again. Report only; nothing here fails on time.",
        "",
        f"Listen: the `media-render-listening-test` artifact holds {', '.join(r['label'] + '.wav' for r in reports)}, "
        f"each the four lines {LINE_GAP_SECONDS:g} s apart.",
    ])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="listening_test.py", description="The fp16 voice listening test (PRD-262 US-050, D4)."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("labels", help="the models' labels, in order")
    commands.add_parser("downloads", help='"label url" for each model file the job fetches')
    one = commands.add_parser("speak", help="speak the script with one model (inside the image)")
    one.add_argument("--model", required=True, choices=sorted(BY_LABEL))
    one.add_argument("--models", type=Path, required=True, help="where the fetched files are, as <label>.onnx")
    one.add_argument("--bundle", type=Path, required=True, help="the render bundle whose voice lines are spoken")
    one.add_argument("--out", type=Path, required=True)
    table = commands.add_parser("summary", help="the job summary's table, from each model's report")
    table.add_argument("--out", type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> str:
    if args.command == "labels":
        return "\n".join(c.label for c in CANDIDATES)
    if args.command == "downloads":
        return "\n".join(f"{c.label} {c.url}" for c in CANDIDATES if not c.in_image)
    if args.command == "speak":
        return json.dumps(speak(args.model, args.models, args.bundle, args.out), indent=1)
    return summary(load_reports(args.out))


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = _run(args)
    except ValueError as exc:
        sys.stderr.write(f"::error::{exc}\n")
        return 1
    sys.stdout.write(output + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
