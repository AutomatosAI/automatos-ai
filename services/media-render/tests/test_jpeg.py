"""POST /jpeg (PRD-251 Wave 3, US-303): a PNG still answered as a JPEG, with the
real ffmpeg of the image. Instagram takes JPEG only; the orchestrator has no image
library, so the conversion lives here."""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN
from media_render.jpeg import JpegError, jpeg_argv, to_jpeg
from media_render.server import TOKEN_HEADER, create_app

AUTH = {TOKEN_HEADER: TOKEN}
JPEG_MAGIC = b"\xff\xd8\xff"
PNG_MAGIC = b"\x89PNG"


def _png(settings, size="32x16", colour="0xFF000080"):
    """A PNG made by ffmpeg itself (half-transparent red): no image library needed."""
    argv = [settings.ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"color=c={colour}:s={size},format=rgba",
            "-frames:v", "1", "-f", "image2pipe", "-c:v", "png", "pipe:1"]
    return subprocess.run(argv, capture_output=True, check=True).stdout


def _size(settings, jpeg):
    argv = [settings.ffprobe_bin, "-v", "error", "-show_entries", "stream=width,height,codec_name", "-of", "csv=p=0", "-i", "pipe:0"]
    return subprocess.run(argv, input=jpeg, capture_output=True, check=True).stdout.decode().strip()


def test_a_png_becomes_a_jpeg_of_the_same_size(settings):
    png = _png(settings)
    assert png.startswith(PNG_MAGIC)
    jpeg = to_jpeg(png, ffmpeg_bin=settings.ffmpeg_bin, timeout_seconds=30)
    assert jpeg.startswith(JPEG_MAGIC)
    assert _size(settings, jpeg) == "mjpeg,32,16"


def test_what_is_not_an_image_is_refused():
    try:
        to_jpeg(b"not an image", ffmpeg_bin="ffmpeg", timeout_seconds=30)
    except JpegError as exc:
        assert "could not convert" in str(exc)
    else:
        raise AssertionError("a non-image was converted")
    try:
        to_jpeg(b"", ffmpeg_bin="ffmpeg", timeout_seconds=30)
    except JpegError as exc:
        assert "no image" in str(exc)


def test_the_argv_reads_stdin_and_writes_one_jpeg_to_stdout():
    argv = jpeg_argv("ffmpeg")
    assert argv[0] == "ffmpeg" and "pipe:0" in argv and argv[-1] == "pipe:1"
    assert argv[argv.index("-c:v") + 1] == "mjpeg" and argv[argv.index("-frames:v") + 1] == "1"


def test_post_jpeg_needs_the_token_and_answers_the_jpeg(settings):
    async def go():
        app = create_app(settings, pipeline=SimpleNamespace(), speaker=SimpleNamespace(), library={})
        async with TestClient(TestServer(app)) as client:
            png = _png(settings)
            assert (await client.post("/jpeg", data=png)).status == 401
            response = await client.post("/jpeg", data=png, headers=AUTH)
            assert response.status == 200 and response.content_type == "image/jpeg"
            assert (await response.read()).startswith(JPEG_MAGIC)
            bad = await client.post("/jpeg", data=b"nope", headers=AUTH)
            assert bad.status == 400 and (await bad.json())["error"] == "not_an_image"

    asyncio.run(go())
