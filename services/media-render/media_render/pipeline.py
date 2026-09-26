"""The render pipeline: stage, fetch, speak, fit, mix and check; then render.

POST /render runs ``prepare_and_check`` before it answers. The composition is
staged with its inline files and GSAP, the media are fetched from our storage,
the Kokoro lines are spoken, every voice line (Kokoro's, or a voice toolkit's
file) is fitted into its script window (fit.py), the mix is made, and
`hyperframes check` runs on the result. Any error refuses the job: 422, nothing
rendered. A job that passes waits for a render slot, then ``render`` runs
`hyperframes render` and probes the file. The renderer assembles; it never
calls a generation provider (D3).

The report names the library track the mix plays, with its licence and
attribution line (``music_report``, S1.6), so the post it lands in can carry
a CC BY track's credit.

A bundle with a ``preview`` is checked the same way, and then its slot takes
PNG snapshots of the composition at the moments asked for instead of the full
render (US-106): each scaled to ``MEDIA_RENDER_PREVIEW_WIDTH``, and a short
reel of them, held ``1 / MEDIA_RENDER_PREVIEW_REEL_FPS`` seconds each.

A bundle with a ``still`` is an image (US-107): checked the same way, then its
render is a PNG snapshot of the composition at each moment asked for, at the
composition's own size and flattened to opaque RGB: ``render.png`` for one,
``render-01.png``, ``render-02.png``, … for a carousel's slides.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Protocol, Sequence, Tuple

import aiohttp
from yarl import URL

from . import audio, fit, hyperframes, probe
from .bundle import Bundle, MediaInput
from .check_report import CheckReportError, parse_check_output
from .composition import GSAP_PATH, MIX_PATH
from .config import Settings
from .hyperframes import LOG_TAIL_CHARS
from .jobs import Job
from .media_urls import redact, url_allowed
from .tts import Speaker

OUTPUT_NAME = "render.mp4"
STILL_NAME = "render.png"
STILL_NAMES = "render-{:02d}.png"
PREVIEW_FRAME = "preview-{:02d}.png"
PREVIEW_FRAMES = "preview-%02d.png"
PREVIEW_REEL = "preview.mp4"
SNAPSHOT_DIR = "snapshots"
FIT_DIR = "fit"
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
    # The script window's end (fit.py): the next line's start, or the composition's end.
    window_end: Optional[float] = None
    # Set when the line was sped up to fit its window.
    tempo: Optional[float] = None

    def report(self) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"id": self.id, "source": self.source, "at": self.at, "seconds": self.seconds}
        if self.window_end is not None:
            entry["window_end"] = self.window_end
        if self.tempo is not None:
            entry["tempo"] = self.tempo
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


def voice_findings(
    lines: Sequence[PlacedLine], duration: float, *, max_tempo: Optional[float] = None
) -> Tuple[Dict[str, Any], ...]:
    """Lines that run past the end, or into each other, are refused before the check.

    Run after the fit (fit.py): a line still too long here could not be sped up
    into its window within ``max_tempo``, and the finding says so.
    """

    def finding(code: str, line: PlacedLine, message: str) -> Dict[str, Any]:
        return {"section": "audio", "severity": "error", "code": code, "line": line.id, "message": message}

    def too_long(line: PlacedLine, window_end: float) -> str:
        if max_tempo is None:
            return ""
        window = window_end - line.at
        return (
            f": it lasts {line.seconds:.2f} s and its window is {window:.2f} s, more than {max_tempo:g}x "
            "speed can fit; shorten the line"
        )

    findings = []
    ordered = sorted(lines, key=lambda line: line.at)
    for line in ordered:
        if line.seconds is None:
            findings.append(finding("voice_line_unreadable", line, f"line {line.id}: its audio file has no duration"))
        elif line.at + line.seconds > duration + VOICE_TOLERANCE_SECONDS:
            end = line.at + line.seconds
            message = f"line {line.id} runs from {line.at:g} s to {end:.2f} s, past the end of the {duration:g} s composition"
            findings.append(finding("voice_line_overruns", line, message + too_long(line, duration)))
    for previous, current in zip(ordered, ordered[1:]):
        if previous.seconds is not None and previous.at + previous.seconds > current.at + VOICE_TOLERANCE_SECONDS:
            message = f"line {previous.id} is still speaking at {current.at:g} s, when line {current.id} starts"
            findings.append(finding("voice_lines_overlap", current, message + too_long(previous, current.at)))
    return tuple(findings)


def music_report(bundle: Bundle) -> Optional[Dict[str, Any]]:
    """The library track the mix plays, the window it plays (track seconds), its
    licence and its attribution line (S1.6): a CC BY track's credit reaches the
    post's copy from here. ``None`` when the plan has no music."""
    cue = bundle.audio.music
    if cue is None:
        return None
    return {**cue.about, "track": cue.track, "start": cue.start, "end": round(cue.start + bundle.composition.duration, 3)}


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
            placed.append(PlacedLine(line.id, "file", line.at, path, await self._seconds(path)))
        return await self._fit(job, placed)

    async def _seconds(self, path: Path) -> Optional[float]:
        """How long an audio file lasts, or ``None`` when ffprobe cannot tell."""
        settings = self._settings
        try:
            info = await asyncio.to_thread(
                probe.probe, path, ffprobe_bin=settings.ffprobe_bin, timeout_seconds=settings.probe_timeout_seconds
            )
        except probe.ProbeError:
            return None
        return probe.duration_seconds(info)

    async def _fit(self, job: Job, lines: List[PlacedLine]) -> List[PlacedLine]:
        """Every line in its script window: one that runs past it is sped up, within the cap (fit.py)."""
        settings, duration = self._settings, job.bundle.composition.duration
        timed = [fit.Timed(line.id, line.at, line.seconds) for line in lines]
        ends = fit.window_ends(timed, duration)
        plan = fit.plan_fit(timed, duration, max_tempo=settings.voice_max_tempo, gap=settings.voice_fit_gap_seconds)
        fitted = []
        for line in lines:
            tempo = plan.get(line.id)
            if tempo is None:
                fitted.append(replace(line, window_end=ends[line.id]))
                continue
            target = job.audio_dir / FIT_DIR / f"{line.id}.wav"
            try:
                await asyncio.to_thread(
                    fit.stretch, line.path, target, tempo,
                    ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=settings.mix_timeout_seconds,
                )
            except fit.FitError as exc:
                raise PipelineError("voice_fit_failed", f"line {line.id} could not be fitted to its window", detail=str(exc)) from None
            seconds = await self._seconds(target)
            fitted.append(
                replace(
                    line,
                    path=target,
                    seconds=seconds if seconds is not None else round(line.seconds / tempo, 3),
                    segments=fit.scaled_segments(line.segments, tempo),
                    window_end=ends[line.id],
                    tempo=tempo,
                )
            )
        return fitted

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
        music = music_report(bundle)
        if music is not None:
            report["music"] = music
        findings = voice_findings(lines, bundle.composition.duration, max_tempo=settings.voice_max_tempo)
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
        if job.bundle.preview is not None:
            return await self.preview(job)
        if job.bundle.still is not None:
            return await self.still(job)
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

    def _ffmpeg(self, argv: Sequence[str], what: str, *, code: str = "preview_failed") -> None:
        try:
            proc = subprocess.run(
                [self._settings.ffmpeg_bin, "-v", "error", "-y", *argv],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self._settings.preview_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise PipelineError(code, f"{what} ran past {self._settings.preview_timeout_seconds} s") from None
        if proc.returncode != 0:
            raise PipelineError(code, f"{what} failed", detail=proc.stderr.strip()[-LOG_TAIL_CHARS:])

    async def _snapshots(self, job: Job, at: Sequence[float], kind: str) -> Tuple[List[Path], float]:
        """``hyperframes snapshot`` of the checked composition at ``at``: the frames, in time order, and the seconds it took."""
        settings = self._settings
        shots = job.dir / SNAPSHOT_DIR
        run = await asyncio.to_thread(
            hyperframes.run_cli,
            hyperframes.snapshot_argv(settings, job.project_dir, shots, at),
            job.project_dir,
            settings.render_timeout_seconds,
            capture=True,
        )
        frames = sorted(shots.glob("frame-*.png")) if shots.is_dir() else []
        if not run.ok or len(frames) != len(at):
            if run.timed_out:
                raise PipelineError(f"{kind}_timed_out", f"the snapshots ran past {settings.render_timeout_seconds} s")
            message = f"hyperframes snapshot took {len(frames)} of {len(at)} frames (exit {run.returncode})"
            raise PipelineError(f"{kind}_failed", message, detail=run.tail())
        return frames, run.seconds

    async def still(self, job: Job) -> RenderResult:
        """The image: a full-size, opaque PNG of the checked composition at each moment (a carousel's slides)."""
        settings, composition = self._settings, job.bundle.composition
        at = job.bundle.still.at
        job.output_dir.mkdir(parents=True, exist_ok=True)
        frames, seconds = await self._snapshots(job, at, "still")
        started = time.monotonic()
        outputs: List[Dict[str, Any]] = []
        for index, (frame, moment) in enumerate(zip(frames, at), start=1):
            name = STILL_NAME if len(at) == 1 else STILL_NAMES.format(index)
            target = job.output_dir / name
            flatten = ["-i", str(frame), "-frames:v", "1", "-pix_fmt", "rgb24", str(target)]
            await asyncio.to_thread(self._ffmpeg, flatten, f"flattening {frame.name}", code="still_failed")
            try:
                info = await asyncio.to_thread(
                    probe.probe, target, ffprobe_bin=settings.ffprobe_bin, timeout_seconds=settings.probe_timeout_seconds
                )
            except probe.ProbeError as exc:
                raise PipelineError("still_failed", str(exc)) from None
            width, height = probe.image_size(info)
            if (width, height) != (composition.width, composition.height):
                message = f"{name} is {width}x{height}, not the composition's {composition.width}x{composition.height}"
                raise PipelineError("still_failed", message)
            outputs.append(
                {
                    "name": name,
                    "kind": "still",
                    "index": index,
                    "at": moment,
                    "aspect": composition.aspect,
                    "width": width,
                    "height": height,
                    "bytes": target.stat().st_size,
                }
            )
        shutil.rmtree(job.dir / SNAPSHOT_DIR, ignore_errors=True)
        timings = {"still_seconds": seconds, "encode_seconds": round(time.monotonic() - started, 3)}
        return RenderResult(outputs=tuple(outputs), timings=timings)

    async def preview(self, job: Job) -> RenderResult:
        """Snapshots of the checked composition at the preview's moments, small, and a short reel of them."""
        settings, composition = self._settings, job.bundle.composition
        at = job.bundle.preview.at
        shots = job.dir / SNAPSHOT_DIR
        job.output_dir.mkdir(parents=True, exist_ok=True)
        frames, snapshot_seconds = await self._snapshots(job, at, "preview")
        started = time.monotonic()
        width = settings.preview_width
        height = int(round(composition.height * width / composition.width / 2)) * 2
        outputs: List[Dict[str, Any]] = []
        for index, (frame, moment) in enumerate(zip(frames, at), start=1):
            name = PREVIEW_FRAME.format(index)
            target = job.output_dir / name
            scale = f"scale={width}:{height}:flags=area"
            await asyncio.to_thread(self._ffmpeg, ["-i", str(frame), "-vf", scale, "-frames:v", "1", str(target)], f"scaling {frame.name}")
            outputs.append(
                {"name": name, "kind": "frame", "at": moment, "width": width, "height": height, "bytes": target.stat().st_size}
            )
        reel = job.output_dir / PREVIEW_REEL
        reel_argv = [
            "-framerate", str(settings.preview_reel_fps), "-start_number", "1", "-i", str(job.output_dir / PREVIEW_FRAMES),
            "-vf", "format=yuv420p", "-r", str(settings.render_fps), "-c:v", "libx264", "-movflags", "+faststart", str(reel),
        ]
        await asyncio.to_thread(self._ffmpeg, reel_argv, "joining the frames into the reel")
        try:
            facts = probe.output_facts(
                await asyncio.to_thread(
                    probe.probe, reel, ffprobe_bin=settings.ffprobe_bin, timeout_seconds=settings.probe_timeout_seconds
                )
            )
        except probe.ProbeError as exc:
            raise PipelineError("preview_failed", str(exc)) from None
        outputs.append(
            {
                "name": PREVIEW_REEL,
                "kind": "reel",
                "width": width,
                "height": height,
                "bytes": reel.stat().st_size,
                "duration": facts["duration"],
                "probe": facts,
            }
        )
        shutil.rmtree(shots, ignore_errors=True)
        timings = {"preview_seconds": snapshot_seconds, "encode_seconds": round(time.monotonic() - started, 3)}
        return RenderResult(outputs=tuple(outputs), timings=timings)
