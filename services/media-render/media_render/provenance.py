"""The "AI-generated" mark on every file this service makes with an AI element (PRD-264 EUAI-06a).

EU AI Act Art. 50(2): generated audio, images and video are marked in a machine-readable
format. The mark is one XMP property, IPTC's ``Iptc4xmpExt:DigitalSourceType``:

- ``trainedAlgorithmicMedia`` when every visual and audio element of the file is AI-made;
- ``compositeWithTrainedAlgorithmicMedia`` when AI-made and other elements are mixed;
- no mark at all when the file was made only from templates and existing content.

**What is AI-made.** The render job says it (``ai_inputs`` in the bundle): AI footage,
AI stills and a synthetic voice, and the person's own uploads and the template, which are
not. The orchestrator sets it from the rule the post's network labels use
(``orchestrator/modules/socials/ai_media.py``), so a post and its file agree. What this
service does itself counts too: a line Kokoro speaks here is AI audio, whatever the job
says. A job without ``ai_inputs`` (an older caller) is logged, and then only the Kokoro
voice is marked as AI. The composition is always one visual element; music and SFX come
from the library and the workspace, and are not AI-made.

**Which elements a file holds.** A video holds the visuals and the mix; a still, a
preview frame and the preview reel (no sound) only the visuals; the mix only the audio.
A Kokoro line's own WAV is AI-made through and through. It is marked only where it
leaves the service, in a POST /tts answer (tts.mark_audio); a render's lines stay
inside it and reach the outside only through the mix and the video, which are marked.

**The writer and the reader** is exiftool (D14), installed in the image. ffmpeg writes no
XMP and its encodes drop metadata, so a file is marked after its last encode. MP4, PNG
and JPEG are written in place by exiftool. exiftool reads WAV but does not write it, so a
WAV's packet is made by exiftool (an XMP file from scratch) and stored as the RIFF
``_PMX`` chunk, where XMP lives in a WAV (Adobe XMP specification, part 3) and where
exiftool reads it back. No MP4 box is ever written by hand.

**One instance id per file** (FR-16). Beside the mark, every packet carries
``xmpMM:InstanceID``, a fresh random UUID (``xmp.iid:<uuid4>``), so two files made from
identical input never share a SHA-256: a predictable file can't be confirmed by hashing
guesses against the public detection route (EUAI-10). It is random, so it says nothing
about who made the file or from what.

**Nothing personal.** The packet carries those two properties and nothing else: the write
clears whatever XMP a file carried first, so no user, workspace, prompt or file path can
reach it (Guidelines ¶94).
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from . import validate
from .validate import BundleError

logger = logging.getLogger(__name__)

SOURCE_TYPE_BASE = "http://cv.iptc.org/newscodes/digitalsourcetype/"
TRAINED = SOURCE_TYPE_BASE + "trainedAlgorithmicMedia"
COMPOSITE = SOURCE_TYPE_BASE + "compositeWithTrainedAlgorithmicMedia"
AI_SOURCE_TYPES = frozenset({TRAINED, COMPOSITE})

# exiftool's name for the property; ``#=`` writes the IPTC URI as it is (no print conversion),
# and ``-n`` reads it back the same way.
XMP_TAG = "XMP-iptcExt:DigitalSourceType"
READ_KEY = "DigitalSourceType"
INSTANCE_TAG = "XMP-xmpMM:InstanceID"  # FR-16: random per file, never derived from the job
INSTANCE_PREFIX = "xmp.iid:"
INPUT_FLAGS = ("footage", "stills", "voice", "uploads", "template")

WAV_SUFFIX = ".wav"
RIFF, WAVE, XMP_CHUNK = b"RIFF", b"WAVE", b"_PMX"
RIFF_HEADER = struct.Struct("<4sI4s")
CHUNK_HEADER = struct.Struct("<4sI")
RIFF_SIZE = struct.Struct("<I")
RIFF_SIZE_OFFSET = 4
RIFF_SIZE_LIMIT = 0xFFFFFFFF
PACKET_NAME = "mark.xmp"
SOURCE_NAME = "source"  # no extension: exiftool tells the image's type by its bytes
JPEG_NAME = "image.jpg"
ERROR_TAIL_CHARS = 500


class ProvenanceError(RuntimeError):
    """exiftool could not mark or read a file. A file it could not mark never ships."""


@dataclass(frozen=True)
class AiInputs:
    """Which of a render's inputs are AI-made, as its job declared them (``declared``)."""

    footage: bool = False
    stills: bool = False
    voice: bool = False
    uploads: bool = False
    template: bool = True
    declared: bool = False


UNDECLARED = AiInputs()


@dataclass(frozen=True)
class Elements:
    """Whether a file holds AI-made elements, other elements, or both."""

    ai: bool = False
    other: bool = False

    def __or__(self, more: "Elements") -> "Elements":
        return Elements(ai=self.ai or more.ai, other=self.other or more.other)


def parse_ai_inputs(raw: Any) -> AiInputs:
    """The bundle's ``ai_inputs``: each of :data:`INPUT_FLAGS` true or false. Absent: undeclared."""
    if raw is None:
        return UNDECLARED
    body = validate.mapping(raw, "ai_inputs")
    validate.keys(body, "ai_inputs", required=INPUT_FLAGS)
    wrong = [name for name in INPUT_FLAGS if not isinstance(body[name], bool)]
    if wrong:
        raise BundleError(f"ai_inputs.{wrong[0]} must be true or false")
    return AiInputs(**{name: body[name] for name in INPUT_FLAGS}, declared=True)


def warn_if_undeclared(job_id: str, inputs: AiInputs) -> None:
    """A job from an older caller names no AI inputs: say so, once per job."""
    if not inputs.declared:
        logger.warning(
            "render %s carries no ai_inputs (an older caller): only the voice Kokoro speaks here is marked as AI",
            job_id,
        )


def source_type(elements: Elements) -> Optional[str]:
    """The IPTC term for a file holding ``elements``; ``None`` when nothing in it is AI-made."""
    if not elements.ai:
        return None
    return COMPOSITE if elements.other else TRAINED


def visual_elements(inputs: AiInputs) -> Elements:
    return Elements(ai=inputs.footage or inputs.stills, other=inputs.template or inputs.uploads)


def audio_elements(inputs: AiInputs, plan: Any) -> Elements:
    """The mix's elements: Kokoro's lines (AI-made here), a voice file (AI-made when the job
    says the voice is), the library music and the SFX (not AI-made)."""
    lines = plan.voice.lines if plan.voice is not None else ()
    spoken = any(line.text is not None for line in lines)
    filed = any(line.path is not None for line in lines)
    return Elements(
        ai=spoken or (filed and inputs.voice),
        other=plan.music is not None or bool(plan.sfx) or (filed and not inputs.voice),
    )


def video_type(bundle: Any) -> Optional[str]:
    """``render.mp4``: the composition's visuals and its mix."""
    return source_type(visual_elements(bundle.ai_inputs) | audio_elements(bundle.ai_inputs, bundle.audio))


def image_type(bundle: Any) -> Optional[str]:
    """A still, a preview frame, the preview reel: the visuals only (they carry no sound)."""
    return source_type(visual_elements(bundle.ai_inputs))


def audio_type(bundle: Any) -> Optional[str]:
    """``mix.wav``: the audio only."""
    return source_type(audio_elements(bundle.ai_inputs, bundle.audio))


def _run(argv: List[str], timeout_seconds: int, what: str) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        raise ProvenanceError(f"exiftool did not {what} within {timeout_seconds} s") from None
    except OSError as exc:
        raise ProvenanceError(f"exiftool could not run to {what}: {exc}") from None
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-ERROR_TAIL_CHARS:]
        raise ProvenanceError(f"exiftool could not {what}: {detail or f'exit {proc.returncode}'}")
    return proc


def new_instance_id() -> str:
    """A fresh ``xmpMM:InstanceID`` (FR-16): ``xmp.iid:`` and a random UUID."""
    return f"{INSTANCE_PREFIX}{uuid.uuid4()}"


def _tag_args(kind: str, instance_id: str) -> List[str]:
    return [f"-{XMP_TAG}#={kind}", f"-{INSTANCE_TAG}={instance_id}"]


def write_argv(exiftool_bin: str, kind: str, instance_id: str, path: Path) -> List[str]:
    """Clear the file's XMP, then write the mark and its instance id: the deletion runs first, so they stay."""
    return [exiftool_bin, "-q", "-overwrite_original", "-XMP:all=", *_tag_args(kind, instance_id), str(path)]


def packet(kind: str, instance_id: str, exiftool_bin: str, timeout_seconds: int) -> bytes:
    """The XMP packet holding the mark and its instance id, made by exiftool from scratch (one per WAV)."""
    with tempfile.TemporaryDirectory(prefix="media-render-xmp-") as scratch:
        target = Path(scratch) / PACKET_NAME
        _run([exiftool_bin, "-q", *_tag_args(kind, instance_id), "-o", str(target)], timeout_seconds, "make the XMP packet")
        if not target.is_file():
            raise ProvenanceError("exiftool made no XMP packet")
        return target.read_bytes()


def _riff_end(handle: Any, length: int) -> int:
    """Where the WAV's last chunk ends: its length, or one byte past it when the writer left
    off the last chunk's pad byte. Refuses anything but a well-formed, unmarked RIFF WAVE."""
    header = handle.read(RIFF_HEADER.size)
    if len(header) != RIFF_HEADER.size:
        raise ProvenanceError("the WAV is too short to be a RIFF file")
    riff, _, form = RIFF_HEADER.unpack(header)
    if riff != RIFF or form != WAVE:
        raise ProvenanceError("the WAV is not a RIFF WAVE file")
    offset = RIFF_HEADER.size
    while offset + CHUNK_HEADER.size <= length:
        handle.seek(offset)
        chunk_id, size = CHUNK_HEADER.unpack(handle.read(CHUNK_HEADER.size))
        if chunk_id == XMP_CHUNK:
            raise ProvenanceError("the WAV carries an XMP packet already")
        offset += CHUNK_HEADER.size + size + (size & 1)
    if offset not in (length, length + 1):
        raise ProvenanceError("the WAV's chunks do not end where the file does")
    return offset


def _mark_wav(path: Path, data: bytes) -> None:
    """Append the packet as the RIFF ``_PMX`` chunk and grow the RIFF size to match."""
    length = path.stat().st_size
    with path.open("r+b") as handle:
        end = _riff_end(handle, length)
        chunk = CHUNK_HEADER.pack(XMP_CHUNK, len(data)) + data + b"\x00" * (len(data) & 1)
        riff_size = end + len(chunk) - RIFF_SIZE_OFFSET - RIFF_SIZE.size  # the file's length less 8
        if riff_size > RIFF_SIZE_LIMIT:
            raise ProvenanceError("the WAV is too large to carry its mark")
        handle.seek(length)
        handle.write(b"\x00" * (end - length) + chunk)
        handle.seek(RIFF_SIZE_OFFSET)
        handle.write(RIFF_SIZE.pack(riff_size))


def mark(path: Path, kind: Optional[str], *, exiftool_bin: str, timeout_seconds: int) -> Optional[str]:
    """Write the mark ``kind`` (one of :data:`AI_SOURCE_TYPES`) and a fresh instance id into
    ``path``, after its last encode, and answer the instance id. ``None`` writes nothing and
    answers ``None``. Blocking: run it in a thread."""
    if kind is None:
        return None
    if kind not in AI_SOURCE_TYPES:
        raise ValueError(f"{kind!r} is not an AI digital source type")
    instance_id = new_instance_id()
    if path.suffix.lower() == WAV_SUFFIX:
        _mark_wav(path, packet(kind, instance_id, exiftool_bin, timeout_seconds))
    else:
        _run(write_argv(exiftool_bin, kind, instance_id, path), timeout_seconds, f"mark {path.name}")
    return instance_id


def read_source_type(path: Path, *, exiftool_bin: str, timeout_seconds: int) -> Optional[str]:
    """The file's ``DigitalSourceType`` as exiftool reads it, or ``None`` when it has none."""
    proc = _run([exiftool_bin, "-j", "-n", f"-{XMP_TAG}", str(path)], timeout_seconds, f"read {path.name}")
    try:
        found = json.loads(proc.stdout)
    except ValueError:
        raise ProvenanceError(f"exiftool printed unreadable JSON for {path.name}") from None
    record = found[0] if isinstance(found, list) and found and isinstance(found[0], dict) else {}
    value = record.get(READ_KEY)
    return value if isinstance(value, str) else None


def read_packet(path: Path, *, exiftool_bin: str, timeout_seconds: int) -> bytes:
    """The file's whole XMP packet, as exiftool extracts it (empty when it has none)."""
    return _run([exiftool_bin, "-b", "-XMP", str(path)], timeout_seconds, f"read the XMP of {path.name}").stdout


def carry_mark(source: bytes, jpeg: bytes, *, exiftool_bin: str, timeout_seconds: int) -> bytes:
    """``jpeg``, converted from ``source``, carrying the AI mark ``source`` carried (a rendered
    still's), since the conversion drops it, under its own fresh instance id. A source with
    no AI mark gives the JPEG none."""
    with tempfile.TemporaryDirectory(prefix="media-render-jpeg-") as scratch:
        original = Path(scratch) / SOURCE_NAME
        original.write_bytes(source)
        kind = read_source_type(original, exiftool_bin=exiftool_bin, timeout_seconds=timeout_seconds)
        if kind not in AI_SOURCE_TYPES:
            return jpeg
        target = Path(scratch) / JPEG_NAME
        target.write_bytes(jpeg)
        mark(target, kind, exiftool_bin=exiftool_bin, timeout_seconds=timeout_seconds)
        return target.read_bytes()
