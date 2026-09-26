"""F179 (C) — a public image link never renders as a page.

GET /api/generated-images/{id} needs no sign-in, and it served whatever type was
stored, inline, so anything stored as a page or an SVG ran as a page on the
backend's origin. The store kept any type it was given.

Every caller of save_image and what it saves: chat replies' data URLs, blog
covers (the upload and the tool) and workspace_get_public_url save PNG, JPEG,
GIF or WebP; Composio outputs add SVG. PRD-251 W1 adds no caller. The store
keeps those types and refuses the rest. Every response carries nosniff and a
sandboxed CSP that loads nothing; a raster image shows inline, and anything
else, an older object too, downloads.
"""
from __future__ import annotations

import asyncio
import base64
import io
import os
import re
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from botocore.exceptions import ClientError  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.generated_images as generated_images  # noqa: E402
import core.services.image_store as image_store  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(120))
CSP = "default-src 'none'; sandbox"


class Body(io.BytesIO):
    """botocore's StreamingBody, as the store reads it."""

    def iter_chunks(self, chunk_size=1024):
        while chunk := self.read(chunk_size):
            yield chunk


class Bucket:
    """The S3 calls the store makes to save, find and serve an image."""

    def __init__(self):
        self.objects: dict = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.objects[Key] = (bytes(Body), ContentType)

    def get_object(self, *, Bucket, Key, Range=None):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "NoSuchKey"}}, "GetObject")
        data, content_type = self.objects[Key]
        if Range:
            first, last = (int(n) for n in re.fullmatch(r"bytes=(\d+)-(\d+)", Range).groups())
            if first >= len(data):
                raise ClientError({"Error": {"Code": "InvalidRange", "Message": "InvalidRange"}}, "GetObject")
            part = data[first:last + 1]
            return {"Body": Body(part), "ContentType": content_type, "ContentLength": len(part),
                    "ContentRange": f"bytes {first}-{last}/{len(data)}"}
        return {"Body": Body(data), "ContentType": content_type, "ContentLength": len(data)}

    def head_object(self, *, Bucket, Key):
        return {"ContentLength": len(self.objects[Key][0])}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        return {"Contents": [{"Key": k} for k in sorted(self.objects) if k.startswith(Prefix)], "IsTruncated": False}


@pytest.fixture
def bucket(monkeypatch):
    stub = Bucket()
    monkeypatch.setattr(image_store, "get_s3_client", lambda *a, **k: stub)
    monkeypatch.setattr(image_store, "ensure_bucket", lambda *a, **k: True)
    return stub


@pytest.fixture
def store(bucket, monkeypatch):
    store = image_store.S3ImageStore()
    monkeypatch.setattr(generated_images, "get_image_store", lambda: store)
    return store


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(generated_images.router)
    return TestClient(app)


def _save(store, data: bytes, mime: str) -> str:
    return asyncio.run(store.save_image(base64.b64encode(data).decode("ascii"), mime, workspace_id="ws-1"))


def _older_object(bucket, data: bytes, content_type: str) -> str:
    """An object stored before the store refused types, with its pointer."""
    image_id = str(uuid.uuid4())
    key = f"generated-images/ws-1/{image_id}.png"
    bucket.objects[key] = (data, content_type)
    bucket.objects[f"generated-image-pointers/{image_id}"] = (key.encode(), "text/plain; charset=utf-8")
    return image_id


# ── the store ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mime", ["text/html", "application/pdf", "text/csv", "video/mp4",
                                  "application/octet-stream", "image/x-icon"])
def test_a_type_no_caller_saves_is_refused_and_nothing_is_stored(store, bucket, mime):
    with pytest.raises(ValueError, match="not an image type the public store keeps"):
        _save(store, b"<script>alert(1)</script>", mime)
    assert bucket.objects == {}


@pytest.mark.parametrize("mime,stored,ext", [
    ("image/png", "image/png", "png"),
    ("image/jpeg", "image/jpeg", "jpg"),
    ("image/jpg", "image/jpeg", "jpg"),          # the chat and blog data URLs' spelling
    ("image/gif", "image/gif", "gif"),
    ("image/webp", "image/webp", "webp"),
    ("image/svg+xml", "image/svg+xml", "svg"),   # Composio outputs
])
def test_every_type_a_caller_saves_is_kept(store, bucket, mime, stored, ext):
    image_id = _save(store, PNG, mime)
    assert bucket.objects[f"generated-images/ws-1/{image_id}.{ext}"] == (PNG, stored)


# ── the link ────────────────────────────────────────────────────────────────

def _serving(response):
    return (response.headers.get("content-disposition"), response.headers.get("x-content-type-options"),
            response.headers.get("content-security-policy"))


def test_a_raster_image_shows_inline_unsniffed_and_sandboxed(client, store):
    response = client.get(f"/api/generated-images/{_save(store, PNG, 'image/png')}")
    assert response.status_code == 200 and response.content == PNG
    assert response.headers["content-type"] == "image/png"
    assert _serving(response) == ("inline", "nosniff", CSP)


def test_an_svg_downloads_instead_of_rendering(client, store):
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    response = client.get(f"/api/generated-images/{_save(store, svg, 'image/svg+xml')}")
    assert response.status_code == 200
    assert _serving(response) == ("attachment", "nosniff", CSP)


@pytest.mark.parametrize("content_type", ["text/html", "text/html; charset=utf-8", "application/pdf",
                                          "application/octet-stream"])
def test_an_older_object_that_is_not_a_raster_image_downloads(client, bucket, store, content_type):
    image_id = _older_object(bucket, b"<html><script>alert(1)</script></html>", content_type)
    response = client.get(f"/api/generated-images/{image_id}")
    assert response.status_code == 200
    assert _serving(response) == ("attachment", "nosniff", CSP)


def test_a_range_of_an_image_carries_the_same_headers(client, store):
    image_id = _save(store, PNG, "image/png")
    response = client.get(f"/api/generated-images/{image_id}", headers={"Range": "bytes=0-7"})
    assert response.status_code == 206 and response.content == PNG[:8]
    assert _serving(response) == ("inline", "nosniff", CSP)


def test_an_older_object_with_no_type_is_served_as_an_image_never_sniffed(client, bucket, store):
    """Review LOW: the store names a type-less object image/png; nosniff keeps its bytes from being read as a page."""
    image_id = _older_object(bucket, b"<html><script>alert(1)</script></html>", None)
    response = client.get(f"/api/generated-images/{image_id}")
    assert response.headers["content-type"] == "image/png"
    assert _serving(response) == ("inline", "nosniff", CSP)


@pytest.mark.parametrize("case", ["unknown id", "range past the end", "storage down"])
def test_an_error_carries_the_same_headers(client, bucket, store, case):
    """Review LOW: every response, errors too."""
    image_id = _save(store, PNG, "image/png")
    if case == "unknown id":
        response = client.get(f"/api/generated-images/{uuid.uuid4()}")
    elif case == "range past the end":
        response = client.get(f"/api/generated-images/{image_id}", headers={"Range": f"bytes={len(PNG) + 5}-{len(PNG) + 9}"})
    else:
        def down(**kw):
            raise RuntimeError("storage down")

        bucket.get_object = down
        response = client.get(f"/api/generated-images/{image_id}")
    assert response.status_code == {"unknown id": 404, "range past the end": 416, "storage down": 503}[case]
    assert (response.headers.get("x-content-type-options"), response.headers.get("content-security-policy")) == (
        "nosniff", CSP)
