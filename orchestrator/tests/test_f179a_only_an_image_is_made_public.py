"""F179 (A) — workspace_get_public_url publishes an image, never an ordinary private file.

The tool downloaded any workspace path and uploaded it to the public image store:
a customer CSV, a page, an SVG. On the local edition that meant every file under
the operator's Development folder, mounted as projects/. It now publishes only a
raster image (png, jpeg, gif, webp), recognised by the file's first bytes, never
by its name. Anything else is refused and never stored.
"""
from __future__ import annotations

import asyncio
import base64
from uuid import uuid4

import pytest

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
GIF = b"GIF89a" + b"\x00" * 16
WEBP = b"RIFF" + b"\x10\x00\x00\x00" + b"WEBPVP8 " + b"\x00" * 8


class _Worker:
    def __init__(self, content):
        self.content = content

    async def download_file(self, path):
        return {"success": True, "content": self.content, "content_type": "application/octet-stream"}


class _Store:
    def __init__(self):
        self.saved = []

    async def save_image(self, b64, mime_type="image/png", workspace_id=None):
        self.saved.append((base64.b64decode(b64), mime_type))
        return "11111111-2222-4333-8444-555555555555"


def _publish(monkeypatch, path, content):
    from modules.tools.execution import exec_workspace

    store = _Store()
    monkeypatch.setattr("core.services.image_store.get_image_store", lambda: store)
    reply = asyncio.run(exec_workspace._get_public_url(_Worker(content), path, uuid4(), "t-1"))
    return reply, store


@pytest.mark.parametrize("path,content,mime", [
    ("content/social/instagram/post.png", PNG, "image/png"),
    ("content/social/cover.jpg", JPEG, "image/jpeg"),
    ("content/social/loop.gif", GIF, "image/gif"),
    ("content/social/card.webp", WEBP, "image/webp"),
    ("exports/render.dat", PNG, "image/png"),              # the bytes decide, not the name
])
def test_an_image_is_published_as_what_its_bytes_are(monkeypatch, path, content, mime):
    reply, store = _publish(monkeypatch, path, content)
    assert reply["success"] is True and reply["content_type"] == mime
    assert store.saved == [(content, mime)]


@pytest.mark.parametrize("path,content", [
    ("customers/export.csv", b"email,name,spend\nami@cafe.test,Ami,1200\n"),
    ("brand/logo.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'),
    ("content/social/post.png", b"<html><script>alert(1)</script></html>"),     # a page named .png
    ("projects/automatos-ai/orchestrator/notes.md", b"# private notes"),
])
def test_anything_but_an_image_is_refused_and_never_stored(monkeypatch, path, content):
    from modules.tools.execution.exec_workspace import ONLY_IMAGES_ARE_PUBLIC

    reply, store = _publish(monkeypatch, path, content)
    assert reply == {"success": False, "error": ONLY_IMAGES_ARE_PUBLIC, "tool": "workspace_get_public_url"}
    assert store.saved == []
