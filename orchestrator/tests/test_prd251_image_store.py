"""PRD-251 S0.4c — the image-store public lookup and Range streaming.

Before: GET /api/generated-images/{id} resolved an id by listing the prefix and
then ONE ``list_objects_v2`` page of 1000 keys across every workspace (past
1000 objects lookups 404'd), matched the id as a SUBSTRING of the key (so a
fragment like ``a`` returned some image), and read the whole object into
memory.

Pinned here against a stubbed S3 that paginates like the real one (1000 keys a
page, ``IsTruncated`` + ``NextContinuationToken``) and records every call:

* with 1,500 stored objects the 1,400th image resolves (every page is walked);
* a new save writes a pointer, and resolving a new id is one GET, no listing;
  a legacy id's first lookup writes its pointer, so the next is one GET too;
* ``Range: bytes=0-99`` → 206, exactly 100 bytes, ``Content-Range: bytes
  0-99/<size>``; no Range → 200 with the whole body, streamed in chunks (the
  image body is never ``read()`` whole);
* only canonical uuid ids reach the bucket, and they match a key exactly.
"""
from __future__ import annotations

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
from fastapi.responses import StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import api.generated_images as generated_images  # noqa: E402
import core.services.image_store as image_store  # noqa: E402

WORKSPACES = ("0b8c9c1e-1111-4a4a-9b9b-000000000001", "0b8c9c1e-2222-4a4a-9b9b-000000000002", "default")
IMAGE = bytes(range(256)) * 4  # 1,024 bytes, every byte value
POINTER_PREFIX = "generated-image-pointers/"


class _Body:
    """botocore's StreamingBody surface: read, iter_chunks, close — each recorded."""

    def __init__(self, key: str, data: bytes, log: list):
        self._key, self._stream, self._log = key, io.BytesIO(data), log
        self.closed = False

    def read(self, amt=None):
        self._log.append((self._key, "read", amt))
        return self._stream.read() if amt is None else self._stream.read(amt)

    def iter_chunks(self, chunk_size=1024):
        self._log.append((self._key, "iter_chunks", chunk_size))
        while True:
            chunk = self._stream.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def close(self):
        self.closed = True


def _client_error(code: str, operation: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, operation)


class StubS3:
    """An in-memory bucket with S3's listing semantics (sorted keys, 1000 a page)."""

    PAGE_LIMIT = 1000

    def __init__(self):
        self.objects: dict = {}
        self.calls: list = []
        self.body_log: list = []
        self.fail_with: str | None = None

    def _record(self, op: str, **kwargs):
        self.calls.append((op, kwargs))
        if self.fail_with:
            raise _client_error(self.fail_with, op)

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self._record("put_object", Key=Key)
        self.objects[Key] = (bytes(Body), ContentType)
        return {}

    def get_object(self, *, Bucket, Key, Range=None):
        self._record("get_object", Key=Key, Range=Range)
        if Key not in self.objects:
            raise _client_error("NoSuchKey", "GetObject")
        data, content_type = self.objects[Key]
        size = len(data)
        if Range is None:
            return {"Body": _Body(Key, data, self.body_log), "ContentType": content_type, "ContentLength": size}
        start, end = re.fullmatch(r"bytes=(\d*)-(\d*)", Range).groups()
        if start == "":
            first, last = max(size - int(end), 0), size - 1
        else:
            first, last = int(start), min(int(end), size - 1) if end else size - 1
        if first >= size:
            raise _client_error("InvalidRange", "GetObject")
        part = data[first:last + 1]
        return {"Body": _Body(Key, part, self.body_log), "ContentType": content_type,
                "ContentLength": len(part), "ContentRange": f"bytes {first}-{last}/{size}"}

    def head_object(self, *, Bucket, Key):
        self._record("head_object", Key=Key)
        return {"ContentLength": len(self.objects[Key][0])}

    def list_objects_v2(self, *, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        self._record("list_objects_v2", Prefix=Prefix, MaxKeys=MaxKeys, ContinuationToken=ContinuationToken)
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken) if ContinuationToken else 0
        stop = start + min(MaxKeys, self.PAGE_LIMIT)
        page = {"Contents": [{"Key": k} for k in keys[start:stop]], "IsTruncated": stop < len(keys)}
        if stop < len(keys):
            page["NextContinuationToken"] = str(stop)
        return page

    def ops(self, name: str) -> list:
        return [kwargs for op, kwargs in self.calls if op == name]


@pytest.fixture
def s3(monkeypatch):
    stub = StubS3()
    monkeypatch.setattr(image_store, "get_s3_client", lambda *args, **kwargs: stub)
    monkeypatch.setattr(image_store, "ensure_bucket", lambda *args, **kwargs: True)
    monkeypatch.setattr(image_store, "_image_store", None)
    return stub


@pytest.fixture
def store(s3):
    return image_store.S3ImageStore()


@pytest.fixture
def client(store, monkeypatch):
    monkeypatch.setattr(generated_images, "get_image_store", lambda: store)
    app = FastAPI()
    app.include_router(generated_images.router)
    return TestClient(app)


def _legacy_objects(s3: StubS3, count: int) -> list:
    """Images saved before pointers existed, spread over workspaces; sorted keys."""
    for i in range(count):
        image_id = str(uuid.uuid4())
        s3.objects[f"generated-images/{WORKSPACES[i % 3]}/{image_id}.png"] = (f"image-{i}".encode(), "image/png")
    return sorted(s3.objects)


async def _save(store, data: bytes = IMAGE, workspace_id: str = WORKSPACES[0]) -> str:
    return await store.save_image(base64.b64encode(data).decode("ascii"), "image/png", workspace_id=workspace_id)


# ── the lookup ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_1400th_of_1500_stored_images_resolves(store, s3):
    keys = _legacy_objects(s3, 1500)
    target = keys[1399]
    target_id = target.rsplit("/", 1)[-1].split(".", 1)[0]
    s3.calls.clear()

    stream = await store.open_image(target_id)

    assert stream is not None
    assert b"".join(stream.body) == s3.objects[target][0]
    pages = s3.ops("list_objects_v2")
    assert [p["ContinuationToken"] for p in pages] == [None, "1000"]  # the second page was read
    assert all(p["MaxKeys"] <= 1000 and p["Prefix"] == "generated-images/" for p in pages)


@pytest.mark.asyncio
async def test_a_new_save_writes_the_pointer_and_resolving_is_one_get_and_no_list(store, s3):
    _legacy_objects(s3, 1500)
    image_id = await _save(store)
    key = f"generated-images/{WORKSPACES[0]}/{image_id}.png"

    assert s3.objects[key] == (IMAGE, "image/png")
    assert s3.objects[POINTER_PREFIX + image_id][0] == key.encode()
    assert [c["Key"] for c in s3.ops("put_object")] == [key, POINTER_PREFIX + image_id]  # image first

    s3.calls.clear()
    assert await store.resolve_key(image_id) == key
    assert s3.calls == [("get_object", {"Key": POINTER_PREFIX + image_id, "Range": None})]

    s3.calls.clear()
    stream = await store.open_image(image_id)
    assert b"".join(stream.body) == IMAGE
    assert s3.ops("list_objects_v2") == []
    assert [c["Key"] for c in s3.ops("get_object")] == [POINTER_PREFIX + image_id, key]


@pytest.mark.asyncio
async def test_a_legacy_ids_first_lookup_writes_its_pointer(store, s3):
    keys = _legacy_objects(s3, 1500)
    target = keys[1234]
    target_id = target.rsplit("/", 1)[-1].split(".", 1)[0]

    assert await store.resolve_key(target_id) == target
    assert s3.objects[POINTER_PREFIX + target_id][0] == target.encode()

    s3.calls.clear()
    assert await store.resolve_key(target_id) == target
    assert s3.calls == [("get_object", {"Key": POINTER_PREFIX + target_id, "Range": None})]


@pytest.mark.asyncio
async def test_the_callers_workspace_is_searched_first_for_a_legacy_id(store, s3):
    keys = _legacy_objects(s3, 1500)
    target = next(k for k in keys if k.startswith(f"generated-images/{WORKSPACES[1]}/"))
    target_id = target.rsplit("/", 1)[-1].split(".", 1)[0]
    s3.calls.clear()

    assert await store.resolve_key(target_id, workspace_id=WORKSPACES[1]) == target
    assert s3.ops("list_objects_v2") == [{
        "Prefix": f"generated-images/{WORKSPACES[1]}/{target_id}.", "MaxKeys": 5, "ContinuationToken": None,
    }]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["a", "0b8c9c1e", "../generated-images", "", "not-a-uuid"])
async def test_only_a_canonical_uuid_reaches_the_bucket(store, s3, bad_id):
    _legacy_objects(s3, 10)
    s3.calls.clear()
    assert await store.open_image(bad_id) is None
    assert s3.calls == []


@pytest.mark.asyncio
async def test_an_id_matches_its_key_exactly_never_as_a_substring(store, s3):
    keys = _legacy_objects(s3, 30)
    real_id = keys[0].rsplit("/", 1)[-1].split(".", 1)[0]
    # A workspace id is a canonical uuid inside every key of that workspace: the
    # old `image_id in key` match served one of its images for it.
    assert await store.resolve_key(WORKSPACES[0]) is None
    assert await store.resolve_key(WORKSPACES[1]) is None
    assert await store.resolve_key(real_id) == keys[0]


@pytest.mark.asyncio
async def test_a_pointer_naming_another_id_is_ignored(store, s3):
    keys = _legacy_objects(s3, 5)
    target_id = keys[2].rsplit("/", 1)[-1].split(".", 1)[0]
    s3.objects[POINTER_PREFIX + target_id] = (keys[0].encode(), "text/plain")
    assert await store.resolve_key(target_id) == keys[2]


# ── the route: streaming and Range ───────────────────────────────────────────

def test_range_0_99_returns_206_with_exactly_100_bytes(client, store, s3):
    import asyncio

    image_id = asyncio.run(_save(store))
    response = client.get(f"/api/generated-images/{image_id}", headers={"Range": "bytes=0-99"})

    assert response.status_code == 206
    assert response.content == IMAGE[:100]
    assert len(response.content) == 100
    assert response.headers["content-range"] == f"bytes 0-99/{len(IMAGE)}"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-length"] == "100"
    assert s3.ops("get_object")[-1]["Range"] == "bytes=0-99"


def test_no_range_returns_200_with_the_whole_body_streamed_in_chunks(client, store, s3, monkeypatch):
    import asyncio

    monkeypatch.setattr(image_store.config, "GENERATED_IMAGE_STREAM_CHUNK_BYTES", 100, raising=False)
    image_id = asyncio.run(_save(store))
    key = f"generated-images/{WORKSPACES[0]}/{image_id}.png"
    s3.body_log.clear()

    response = client.get(f"/api/generated-images/{image_id}")

    assert response.status_code == 200
    assert response.content == IMAGE
    assert response.headers["content-length"] == str(len(IMAGE))
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["content-type"] == "image/png"
    assert "content-range" not in response.headers
    assert s3.ops("get_object")[-1]["Range"] is None
    # Streamed: the image body is iterated in 100-byte chunks and never read() whole.
    image_reads = [(op, arg) for k, op, arg in s3.body_log if k == key]
    assert image_reads == [("iter_chunks", 100)]


@pytest.mark.asyncio
async def test_the_route_returns_a_streaming_response(store, s3, monkeypatch):
    monkeypatch.setattr(generated_images, "get_image_store", lambda: store)
    image_id = await _save(store)
    response = await generated_images.get_generated_image(image_id, range_header=None)
    assert isinstance(response, StreamingResponse)
    assert response.status_code == 200


def test_a_suffix_range_serves_the_tail(client, store):
    import asyncio

    image_id = asyncio.run(_save(store))
    response = client.get(f"/api/generated-images/{image_id}", headers={"Range": "bytes=-24"})
    assert response.status_code == 206
    assert response.content == IMAGE[-24:]
    assert response.headers["content-range"] == f"bytes {len(IMAGE) - 24}-{len(IMAGE) - 1}/{len(IMAGE)}"


def test_a_range_past_the_end_is_416_with_the_size(client, store):
    import asyncio

    image_id = asyncio.run(_save(store))
    response = client.get(f"/api/generated-images/{image_id}", headers={"Range": f"bytes={len(IMAGE)}-"})
    assert response.status_code == 416
    assert response.headers["content-range"] == f"bytes */{len(IMAGE)}"


@pytest.mark.parametrize("header", ["bytes=0-1,5-6", "items=0-9", "bytes=9-2", "bytes=-", "garbage"])
def test_an_unusable_range_is_ignored_and_the_whole_body_served(client, store, header):
    import asyncio

    image_id = asyncio.run(_save(store))
    response = client.get(f"/api/generated-images/{image_id}", headers={"Range": header})
    assert response.status_code == 200
    assert response.content == IMAGE


def test_unknown_and_malformed_ids_are_404(client, s3):
    _legacy_objects(s3, 3)
    assert client.get(f"/api/generated-images/{uuid.uuid4()}").status_code == 404
    s3.calls.clear()
    assert client.get("/api/generated-images/a").status_code == 404
    assert s3.calls == []


def test_a_storage_failure_is_503_not_a_misleading_404(client, s3):
    s3.fail_with = "AccessDenied"
    response = client.get(f"/api/generated-images/{uuid.uuid4()}")
    assert response.status_code == 503
    assert response.json()["detail"] == "Image storage unavailable"


# ── the header parser and the config ─────────────────────────────────────────

@pytest.mark.parametrize("header,expected", [
    ("bytes=0-99", "bytes=0-99"),
    ("bytes=100-", "bytes=100-"),
    ("bytes=-50", "bytes=-50"),
    (" bytes=5-5 ", "bytes=5-5"),
    (None, None),
    ("", None),
    ("bytes=-", None),
    ("bytes=10-2", None),
    ("bytes=0-1,4-5", None),
    ("bits=0-9", None),
])
def test_parse_byte_range(header, expected):
    assert image_store.parse_byte_range(header) == expected


def test_the_stream_chunk_size_is_config():
    assert isinstance(image_store.config.GENERATED_IMAGE_STREAM_CHUNK_BYTES, int)
    assert image_store.config.GENERATED_IMAGE_STREAM_CHUNK_BYTES > 0
