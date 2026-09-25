"""The render pipeline: stage, fetch, speak, mix and check; then render.

POST /render runs ``prepare_and_check`` before it answers. The composition is
staged with its inline files and GSAP, the media are fetched from our storage,
the Kokoro lines are spoken, the mix is made, and `hyperframes check` runs on
the result. Any error refuses the job: 422, nothing rendered. A job that passes
waits for a render slot, then ``render`` runs `hyperframes render` and probes
the file. The renderer assembles; it never calls a generation provider (D3).
"""

from __future__ import annotations

import asyncio
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

import aiohttp
from yarl import URL

from . import audio, hyperframes, probe
from .bundle import Bundle, MediaInput
from .check_report import CheckReportError, parse_check_output
from .composition import GSAP_PATH, MIX_PATH
from .config import Settings
from .jobs import Job
from .media_urls import redact, url_allowed
from .tts import Speaker

OUTPUT_NAME = "render.mp4"
CHUNK_BYTES = 1 << 16
# How far one voice line may run into the next, or past the end, before it counts.
VOICE_TOLERANCE_SECONDS = 0.05


class PipelineError(RuntimeError):
    """The service could not do its part; the job fails with ``status``."""

    def __init__(self, code: str, message: str, *, status: int = 500, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.detail = detail


class MediaFetchError(PipelineError):
    def __init__(self, message: str) -> None:
        super().__init__("media_fetch_failed", message, status=502)


@dataclass(frozen=True)
class CheckOutcome:
    ok: bool
    report: Mapping[str, Any]
    findings: Tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class RenderResult:
    outputs: Tuple[Mapping[str, Any], ...]
    timings: Mapping[str, float]


class Pipeline(Protocol):
    async def prepare_and_check(self, job: Job) -> CheckOutcome: ...

    async def render(self, job: Job) -> RenderResult: ...


@dataclass(frozen=True)
class PlacedLine:
    id: str
    source: str
    at: float
    path: Path
    seconds: Optional[float]
    segments: Tuple[Tuple[float, float], ...] = ()

    def report(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"id": self.id, "source": self.source, "at": self.at, "seconds": self.seconds}
        if self.segments:
            entry["segments"] = [{"start": start, "end": end} for start, end in self.segments]
        return entry


@contextmanager
def _timed(timings: Dict[str, float], name: str) -> Iterator[None]:
    started = time.monotonic()
    try:
        yield
    finally:
        timings[name] = round(time.monotonic() - started, 3)


def stage_project(bundle: Bundle, project_dir: Path, gsap_path: str) -> None:
    """The composition directory: index.html, the inline files, GSAP, and room for the mix."""
    project_dir.mkdir(parents=True)
    (project_dir / "index.html").write_text(bundle.composition.html, encoding="utf-8")
    for item in bundle.files:
        target = project_dir / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.data)
    gsap = project_dir / GSAP_PATH
    gsap.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gsap_path, gsap)
    (project_dir / MIX_PATH).parent.mkdir(parents=True, exist_ok=True)


async def _download(session: aiohttp.ClientSession, item: MediaInput, target: Path, settings: Settings) -> None:
    where = f"{item.path} ({redact(item.url)})"
    timeout = aiohttp.ClientTimeout(total=settings.fetch_timeout_seconds)
    # encoded=True: a presigned URL is sent exactly as signed, never re-quoted.
    async with session.get(URL(item.url, encoded=True), allow_redirects=False, timeout=timeout) as response:
        if response.status != 200:
            raise MediaFetchError(f"{where}: storage answered {response.status}")
        if response.content_length is not None and response.content_length > settings.max_media_bytes:
            raise MediaFetchError(f"{where}: larger than the {settings.max_media_bytes}-byte limit")
        size = 0
        with target.open("wb") as handle:
            async for chunk in response.content.iter_chunked(CHUNK_BYTES):
                size += len(chunk)
                if size > settings.max_media_bytes:
                    raise MediaFetchError(f"{where}: larger than the {settings.max_media_bytes}-byte limit")
                handle.write(chunk)
    if size == 0:
        raise MediaFetchError(f"{where}: storage returned an empty file")


async def fetch_media(
    session: aiohttp.ClientSession, media: Sequence[MediaInput], project_dir: Path, settings: Settings
) -> None:
    """Copy each media input from our storage into the composition, one at a time."""
    for item in media:
        if not url_allowed(item.url, settings.media_url_prefixes):
            # parse_bundle refused these already; never fetch one that slipped past.
            raise MediaFetchError(f"{item.path}: {redact(item.url)} is not on the storage allowlist")
        target = project_dir / item.path
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            await _download(session, item, target, settings)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise MediaFetchError(f"{item.path} ({redact(item.url)}): {type(exc).__name__}") from None


def voice_findings(lines: Sequence[PlacedLine], duration: float) -> Tuple[Dict[str, Any], ...]:
    """Lines that run past the end, or into each other, are refused before the check."""

    def finding(code: str, line: PlacedLine, message: str) -> Dict[str, Any]:
        return {"section": "audio", "severity": "error", "code": code, "line": line.id, "message": message}

    findings = []
    ordered = sorted(lines, key=lambda line: line.at)
    for line in ordered:
        if line.seconds is None:
            findings.append(finding("voice_line_unreadable", line, f"line {line.id}: its audio file has no duration"))
        elif line.at + line.seconds > duration + VOICE_TOLERANCE_SECONDS:
            end = line.at + line.seconds
            message = f"line {line.id} runs from {line.at:g} s to {end:.2f} s, past the end of the {duration:g} s composition"
            findings.append(finding("voice_line_overruns", line, message))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.seconds is not None and previous.at + previous.seconds > current.at + VOICE_TOLERANCE_SECONDS:
            message = f"line {previous.id} is still speaking at {current.at:g} s, when line {current.id} starts"
            findings.append(finding("voice_lines_overlap", current, message))
    return tuple(findings)


def mix_plan(bundle: Bundle, lines: Sequence[PlacedLine], project_dir: Path) -> audio.MixPlan:
    plan = bundle.audio
    music = plan.music
    return audio.MixPlan(
        duration=bundle.composition.duration,
        voice=tuple(audio.VoiceTrack(path=line.path, at=line.at) for line in lines),
        music=audio.MusicTrack(music.path, music.start, music.fade_in, music.fade_out) if music else None,
        sfx=tuple(audio.SfxTrack(project_dir / cue.path, cue.at, cue.volume) for cue in plan.sfx),
    )


def _refused(findings: Sequence[Dict[str, Any]], report: Dict[str, Any]) -> CheckOutcome:
    summary = {"ok": False, "errors": len(findings), "warnings": 0, "sections": {"audio": {"ok": False}}}
    return CheckOutcome(ok=False, findings=tuple(findings), report={**report, "check": summary, "findings": list(findings)})


class RenderPipeline:
    def __init__(self, settings: Settings, speaker: Speaker) -> None:
        self._settings = settings
        self._speaker = speaker
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        # trust_env=False: storage is reached directly, never through a proxy from the environment.
        self._session = aiohttp.ClientSession(trust_env=False)

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def _voice(self, job: Job) -> List[PlacedLine]:
        plan = job.bundle.audio.voice
        if plan is None:
            return []
        texts = [(line.id, line.text) for line in plan.lines if line.text is not None]
        try:
            spoken = await self._speaker.speak(
                texts, job.audio_dir / "vo", voice=plan.voice, speed=plan.speed, lang=plan.lang
            )
        except ValueError as exc:  # Kokoro's refusal: an unknown voice, a line with nothing to say
            raise PipelineError("voice_refused", f"Kokoro could not speak the script: {exc}", status=400) from None
        by_id = {line_id: said for (line_id, _), said in zip(texts, spoken)}
        placed = []
        for line in plan.lines:
            if line.text is not None:
                said = by_id[line.id]
                placed.append(PlacedLine(line.id, "kokoro", line.at, said.path, said.seconds, said.segments))
                continue
            path = job.project_dir / line.path
            try:
                info = await asyncio.to_thread(
                    probe.probe, path, ffprobe_bin=self._settings.ffprobe_bin, timeout_seconds=self._settings.probe_timeout_seconds
                )
                seconds = probe.duration_seconds(info)
            except probe.ProbeError:
                seconds = None
            placed.append(PlacedLine(line.id, "file", line.at, path, seconds))
        return placed

    async def prepare_and_check(self, job: Job) -> CheckOutcome:
        settings, bundle = self._settings, job.bundle
        timings: Dict[str, float] = {}
        with _timed(timings, "stage_seconds"):
            await asyncio.to_thread(stage_project, bundle, job.project_dir, settings.gsap_path)
        if bundle.media:
            if self._session is None:
                raise PipelineError("not_started", "the media fetcher is not running")
            with _timed(timings, "fetch_seconds"):
                await fetch_media(self._session, bundle.media, job.project_dir, settings)
        with _timed(timings, "voice_seconds"):
            lines = await self._voice(job)
        report: Dict[str, Any] = {"voice": [line.report() for line in lines], "timings": timings}
        findings = voice_findings(lines, bundle.composition.duration)
        if findings:
            return _refused(findings, report)
        try:
            with _timed(timings, "mix_seconds"):
                report["audio"] = await asyncio.to_thread(
                    audio.run_mix,
                    mix_plan(bundle, lines, job.project_dir),
                    job.project_dir / MIX_PATH,
                    ffmpeg_bin=settings.ffmpeg_bin,
                    timeout_seconds=settings.mix_timeout_seconds,
                )
        except audio.MixError as exc:
            raise PipelineError("mix_failed", "the mix could not be made", detail=str(exc)) from None

        run = await asyncio.to_thread(
            hyperframes.run_cli,
            hyperframes.check_argv(settings, job.project_dir),
            job.project_dir,
            settings.check_timeout_seconds,
            capture=True,
        )
        timings["check_seconds"] = run.seconds
        if run.timed_out:
            raise PipelineError("check_timed_out", f"hyperframes check ran past {settings.check_timeout_seconds} s")
        try:
            result = parse_check_output(run.stdout, job.project_dir)
        except CheckReportError as exc:
            raise PipelineError("check_unavailable", str(exc), detail=run.tail()) from None
        report.update(
            check=dict(result.summary),
            lint=dict(result.summary.get("sections", {}).get("lint", {})),
            findings=list(result.findings),
        )
        return CheckOutcome(ok=result.ok, report=report, findings=result.findings)

    async def render(self, job: Job) -> RenderResult:
        settings, composition = self._settings, job.bundle.composition
        output = job.output_dir / OUTPUT_NAME
        job.output_dir.mkdir(parents=True, exist_ok=True)
        run = await asyncio.to_thread(
            hyperframes.run_cli,
            hyperframes.render_argv(settings, job.project_dir, output),
            job.project_dir,
            settings.render_timeout_seconds,
            capture=True,
        )
        if not run.ok or not output.is_file():
            if run.timed_out:
                raise PipelineError("render_timed_out", f"the render ran past {settings.render_timeout_seconds} s")
            raise PipelineError("render_failed", f"hyperframes render exited {run.returncode}", detail=run.tail())
        started = time.monotonic()
        try:
            facts = probe.output_facts(
                await asyncio.to_thread(
                    probe.probe, output, ffprobe_bin=settings.ffprobe_bin, timeout_seconds=settings.probe_timeout_seconds
                )
            )
        except probe.ProbeError as exc:
            raise PipelineError("render_unreadable", str(exc)) from None
        entry = {
            "name": OUTPUT_NAME,
            "aspect": composition.aspect,
            "width": composition.width,
            "height": composition.height,
            "bytes": output.stat().st_size,
            "duration": facts["duration"],
            "probe": facts,
        }
        timings = {"render_seconds": run.seconds, "probe_seconds": round(time.monotonic() - started, 3)}
        return RenderResult(outputs=(entry,), timings=timings)
