"""A still as JPEG (PRD-251 Wave 3, US-303).

Instagram takes JPEG images only, and the renderer's stills are PNG. The
orchestrator has no image library (and adds none), so it sends the still here,
where ffmpeg already is: ``POST /jpeg`` with the image's bytes answers the same
picture as a JPEG. A transparent pixel is laid over white, the way a PNG shows on
a page, since JPEG has no transparency.
"""

from __future__ import annotations

import subprocess
from typing import List

JPEG_CONTENT_TYPE = "image/jpeg"
# ffmpeg's mjpeg quality scale: 2 is the best short of lossless.
JPEG_QUALITY = "2"
# Transparency laid over white, then 4:2:0 as every platform decodes it.
FLATTEN = "split[fg][bg];[bg]drawbox=t=fill:c=white[white];[white][fg]overlay=format=auto,format=yuvj420p"
ERROR_TAIL_CHARS = 500


class JpegError(RuntimeError):
    """ffmpeg could not read the image, or did not answer in time."""


def jpeg_argv(ffmpeg_bin: str) -> List[str]:
    """Read one image from stdin, write it as a JPEG to stdout."""
    return [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-f", "image2pipe", "-i", "pipe:0",
        "-frames:v", "1", "-filter_complex", FLATTEN, "-c:v", "mjpeg", "-q:v", JPEG_QUALITY,
        "-f", "image2pipe", "pipe:1",
    ]


def to_jpeg(image: bytes, *, ffmpeg_bin: str, timeout_seconds: int) -> bytes:
    """``image`` (a PNG, a WebP, a JPEG) as JPEG bytes. Blocking: run it in a thread."""
    if not image:
        raise JpegError("there is no image to convert")
    try:
        proc = subprocess.run(jpeg_argv(ffmpeg_bin), input=image, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        raise JpegError(f"ffmpeg did not convert the image within {timeout_seconds} seconds") from None
    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-ERROR_TAIL_CHARS:]
        raise JpegError(f"ffmpeg could not convert the image: {detail or 'no output'}")
    return proc.stdout
