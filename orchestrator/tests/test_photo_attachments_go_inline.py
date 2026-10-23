"""A photo from the app reaches the model as inline bytes it reads (23 Oct 2026).

The resolver sent an image of 500 KB or more as a signed link, which failed every way a
camera photo goes: the local edition's ``http://localhost`` store no provider can fetch,
the link expired after 15 minutes of a tool loop, Bedrock and Gemini left a linked image
out, and a 48 MP photo, a bmp or a tiff went to a provider that refuses it. Every image
now goes inline: a small jpeg, png, gif or webp as it is, anything else a JPEG at most
2576 px on its longer side, turned upright. A photo sent with no caption carries no blank
text part (Anthropic answers 400 to one), and a WAV is no longer labelled a WebP image.
"""
from __future__ import annotations

import asyncio
import base64
import io
import random
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PIL import Image

from modules.attachments.resolver import (
    READ_EDGE_PX,
    SEND_AS_IS_BYTES,
    AttachmentResolver,
    inject_parts_into_last_user_message,
)
from modules.attachments.store import MediaType
from modules.attachments.validation import ValidationError, detect_mime, validate_upload

MODEL = "claude-sonnet-4-6"  # a vision model the resolver knows without a database
EXIF_ORIENTATION = 0x0112
TURN_90_CLOCKWISE = 6
IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}}
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
WEBP = b"RIFF\x24\x00\x00\x00WEBPVP8 \x10\x00\x00\x00"


def _encoded(image: Image.Image, fmt: str, **options) -> bytes:
    out = io.BytesIO()
    image.save(out, fmt, **options)
    return out.getvalue()


def _noise_jpeg(size: tuple[int, int], seed: int, **options) -> bytes:
    """A JPEG of noise: incompressible, so well over 500 KB at a camera's size."""
    noise = random.Random(seed).randbytes(size[0] * size[1] * 3)
    return _encoded(Image.frombytes("RGB", size, noise), "JPEG", quality=95, **options)


class _Store:
    """An attachment store holding one image. It has no ``sign_url``: a resolver that
    tried to send a link would fail the image."""

    def __init__(self, data: bytes, mime: str):
        self.ref = SimpleNamespace(attachment_id=uuid4(), workspace_id=uuid4(), media_type=MediaType.IMAGE,
                                   mime=mime, filename="photo", size_bytes=len(data), s3_key="k")
        self._data = data

    async def get(self, attachment_id, workspace_id):
        return self.ref

    async def open(self, attachment_id, workspace_id):
        return self._data


def _resolved(data: bytes, mime: str):
    store = _Store(data, mime)
    return asyncio.run(AttachmentResolver(store=store).resolve(
        attachment_ids=[store.ref.attachment_id], workspace_id=uuid4(), model_id=MODEL))


def _image_of(part: dict) -> tuple[str, Image.Image, bytes]:
    """A data-URL part's media type, its image, and its bytes."""
    head, _, data = part["image_url"]["url"].partition(";base64,")
    raw = base64.b64decode(data)
    return head.removeprefix("data:"), Image.open(io.BytesIO(raw)), raw


def test_a_camera_photo_goes_inline_as_a_jpeg_within_the_vision_limit():
    photo = _noise_jpeg((3000, 2000), seed=0)
    assert len(photo) >= SEND_AS_IS_BYTES                     # the size that went as a signed link
    parts, failures = _resolved(photo, "image/jpeg")
    assert failures == []
    [part] = parts
    media_type, image, _ = _image_of(part)
    assert part["type"] == "image_url" and media_type == "image/jpeg"
    assert max(image.size) == READ_EDGE_PX


def test_a_photo_is_turned_upright_as_its_exif_says():
    # A phone writes a portrait photo landscape, with EXIF orientation 6 (turn 90 degrees clockwise).
    exif = Image.Exif()
    exif[EXIF_ORIENTATION] = TURN_90_CLOCKWISE
    photo = _noise_jpeg((1200, 800), seed=1, exif=exif)
    assert len(photo) >= SEND_AS_IS_BYTES
    [part], _ = _resolved(photo, "image/jpeg")
    assert _image_of(part)[1].size == (800, 1200)


def test_a_small_png_goes_as_it_is():
    png = _encoded(Image.new("RGB", (40, 30), (200, 10, 10)), "PNG")
    [part], failures = _resolved(png, "image/png")
    media_type, _, raw = _image_of(part)
    assert (media_type, raw, failures) == ("image/png", png, [])


@pytest.mark.parametrize("fmt, mime", [("BMP", "image/bmp"), ("TIFF", "image/tiff")])
def test_a_bmp_or_tiff_goes_as_a_jpeg(fmt, mime):
    picture = _encoded(Image.new("RGB", (40, 30), (10, 200, 10)), fmt)
    [part], failures = _resolved(picture, mime)
    media_type, image, _ = _image_of(part)
    assert (media_type, image.size, failures) == ("image/jpeg", (40, 30), [])


def test_an_svg_is_reported_as_an_image_that_could_not_be_loaded():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"/>'
    parts, failures = _resolved(svg, "image/svg+xml")
    assert [failure["reason"] for failure in failures] == ["image could not be loaded"]
    assert len(parts) == 1 and parts[0]["text"].startswith("[ATTACHMENT UNAVAILABLE]")


@pytest.mark.parametrize("caption", ["", "   "])
def test_a_photo_sent_with_no_caption_carries_no_blank_text_part(caption):
    messages = [{"role": "system", "content": "You are Auto."}, {"role": "user", "content": caption}]
    assert inject_parts_into_last_user_message(messages, [IMAGE_PART])[-1]["content"] == [IMAGE_PART]


def test_a_caption_still_comes_before_the_photo():
    messages = [{"role": "user", "content": "What is this?"}]
    assert inject_parts_into_last_user_message(messages, [IMAGE_PART])[-1]["content"] == [
        {"type": "text", "text": "What is this?"}, IMAGE_PART]


def test_a_wav_is_not_labelled_a_webp_image():
    assert detect_mime(WAV, "voice.wav") != "image/webp"
    with pytest.raises(ValidationError, match="not allowed"):
        validate_upload(WAV, "voice.wav")


def test_a_webp_is_still_a_webp():
    assert detect_mime(WEBP, "photo.webp") == "image/webp"
