"""PRD-264 EUAI-06b / EUAI-10 — every media-render file the orchestrator keeps gets its provenance row.

media-render marks and signs each file with an AI element and names its mark: each marked
output's ``digital_source_type``, ``sha256`` and ``manifest_id``, and a marked ``/jpeg``
answer's ``X-Digital-Source-Type``, ``X-Content-SHA256`` and ``X-C2PA-Manifest-Id`` headers.
It keeps no database, so the orchestrator writes the row
(``services.content_provenance.record_produced_file``) for every file it keeps
(``core/media_render_provenance.py``). Pinned here:

* a Socials render records each marked file it stores (a video, each carousel slide) as
  its workspace's, with the service as generator and the manifest id; an unmarked one
  gets no row;
* a file whose bytes are not the ones media-render marked fails the render, and nothing
  is stored or recorded;
* ``POST /jpeg``'s mark comes back with the JPEG, checked against its bytes, and the
  publish records the JPEG it posts;
* ``generate_document``'s social render (``render_to_file``) records its file too;
* a rendered video's hash then answers yes on the public check
  (``GET /api/public/provenance/{sha256}``);
* no hash reaches a log line (FR-17).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from pathlib import Path

import httpx
import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import core.media_render_client as media_render_client  # noqa: E402
import services.content_provenance as content_provenance  # noqa: E402
import tests.test_prd251w1_render_lifecycle as render_harness  # noqa: E402
from core.media_render_client import ConvertedJpeg, MediaRenderClient, MediaRenderError  # noqa: E402
from core.media_render_provenance import (  # noqa: E402
    MANIFEST_HEADER, MEDIA_RENDER_GENERATOR, SHA256_HEADER, SOURCE_TYPE_HEADER, MarkedFile, record_marked,
)
from modules.socials.media_urls import MediaFile  # noqa: E402
from modules.socials.publish_steps import Stager  # noqa: E402
from tests.test_prd251w1_render_lifecycle import (  # noqa: E402
    JOB_ID, MP4, MP4_SHA, OUTPUT, RENDER_URL, WS, FakeStore, Renderer, _carousel_blocks, _create, _post, _renderable,
    _run, _start, _template,
)

env = render_harness.env

TRAINED = "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
COMPOSITE = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
MANIFEST = "urn:c2pa:prd264-euai06b-test-manifest"
MARKED_OUTPUT = {**OUTPUT, "digital_source_type": COMPOSITE, "sha256": MP4_SHA, "manifest_id": MANIFEST}
PNG = b"\x89PNG\r\n\x1a\n-still"
JPEG = b"\xff\xd8\xff-marked-jpeg"
JPEG_SHA = hashlib.sha256(JPEG).hexdigest()


@pytest.fixture
def rows(monkeypatch):
    """Every provenance row written, as (workspace, sha256, kind, generator, manifest id)."""
    recorded = []

    def record(workspace_id, sha256, kind, generator, manifest_id=None):
        recorded.append((workspace_id, sha256, kind, generator, manifest_id))

    monkeypatch.setattr(content_provenance, "record_produced_file", record)
    return recorded


# ---------------------------------------------------------------------------
# A Socials render
# ---------------------------------------------------------------------------


def test_a_rendered_video_s_marked_file_gets_its_row_as_it_is_stored(env, rows):
    post = _renderable(env)
    _, job = _start(env, post)
    store = FakeStore()

    assert _run(job, Renderer(outputs=(MARKED_OUTPUT,)), store, env.factory) is True

    assert rows == [(WS, MP4_SHA, "video", MEDIA_RENDER_GENERATOR, MANIFEST)]
    assert list(store.objects) == [f"social-media/{WS}/{post['id']}/video-9x16.mp4"]
    assert _post(env, post["id"]).status == "needs_approval"


def test_an_unsigned_file_still_gets_its_row_without_a_manifest_id(env, rows):
    """Signing failed, so the file ships with its XMP mark alone: it is still detectable."""
    _, job = _start(env, _renderable(env))

    assert _run(job, Renderer(outputs=({**MARKED_OUTPUT, "manifest_id": None},)), FakeStore(), env.factory) is True

    assert rows == [(WS, MP4_SHA, "video", MEDIA_RENDER_GENERATOR, None)]


def test_a_file_with_no_ai_element_gets_no_row(env, rows):
    _, job = _start(env, _renderable(env))

    assert _run(job, Renderer(outputs=(OUTPUT,)), FakeStore(), env.factory) is True

    assert rows == []


def test_a_file_that_is_not_the_one_media_render_marked_fails_the_render_and_nothing_is_kept(env, rows):
    post = _renderable(env)
    _, job = _start(env, post)
    store = FakeStore()
    other = {**MARKED_OUTPUT, "sha256": "c" * 64}

    assert _run(job, Renderer(outputs=(other,)), store, env.factory) is False

    row = _post(env, post["id"])
    assert row.status == "failed" and row.review_log[-1]["report"]["code"] == "bad_response"
    assert store.objects == {} and rows == []


def test_each_marked_carousel_slide_gets_its_own_image_row(env, rows):
    post = _create(
        env,
        template_id=str(_template(env, blocks=_carousel_blocks(), fmt="social_image")),
        format="carousel",
        variables={
            "headline": {"value": "4 SIGNS|YOUR AGENT|NEEDS A|HARNESS", "claim": False},
            "point_1_title": {"value": "It forgets what it just did", "claim": False},
            "point_2_title": {"value": "It picks the wrong tool", "claim": False},
            "closing_title": {"value": "Build the harness first.", "claim": False},
        },
    )
    _, job = _start(env, post)
    pngs = {f"render-{i:02d}.png": b"\x89PNG\r\n\x1a\n" + bytes([i]) * 64 for i in range(1, 5)}
    outputs = [
        {"name": name, "kind": "still", "index": i, "at": at, "aspect": "4:5", "width": 1080, "height": 1350,
         "bytes": 72, "digital_source_type": TRAINED, "sha256": hashlib.sha256(pngs[name]).hexdigest(),
         "manifest_id": f"{MANIFEST}-{i}"}
        for i, (name, at) in enumerate(zip(sorted(pngs), (0.5, 1.5, 2.5, 7.5)), start=1)
    ]

    assert _run(job, Renderer(outputs=outputs, files=pngs), FakeStore(), env.factory) is True

    assert rows == [
        (WS, hashlib.sha256(pngs[f"render-{i:02d}.png"]).hexdigest(), "image", MEDIA_RENDER_GENERATOR, f"{MANIFEST}-{i}")
        for i in range(1, 5)
    ]


# ---------------------------------------------------------------------------
# POST /jpeg, and the publish that posts it
# ---------------------------------------------------------------------------


def _jpeg_client(monkeypatch, headers):
    def answer(request):
        return httpx.Response(200, content=JPEG, headers={"Content-Type": "image/jpeg", **headers})

    monkeypatch.setattr(media_render_client.config, "SOCIALS_RENDER_URL", RENDER_URL)
    return MediaRenderClient(http=httpx.AsyncClient(transport=httpx.MockTransport(answer)))


def test_a_marked_jpeg_comes_back_with_its_mark(monkeypatch):
    headers = {SOURCE_TYPE_HEADER: TRAINED, SHA256_HEADER: JPEG_SHA, MANIFEST_HEADER: MANIFEST}

    answer = asyncio.run(_jpeg_client(monkeypatch, headers).to_jpeg(PNG))

    assert answer == ConvertedJpeg(JPEG, MarkedFile(TRAINED, JPEG_SHA, MANIFEST))


def test_a_jpeg_whose_mark_names_other_bytes_is_refused(monkeypatch):
    headers = {SOURCE_TYPE_HEADER: TRAINED, SHA256_HEADER: "d" * 64, MANIFEST_HEADER: MANIFEST}

    with pytest.raises(MediaRenderError) as refused:
        asyncio.run(_jpeg_client(monkeypatch, headers).to_jpeg(PNG))

    assert refused.value.code == media_render_client.BAD_RESPONSE


class _Bucket:
    """The documents bucket: download_file writes the still's bytes."""

    def download_file(self, bucket, key, filename):
        Path(filename).write_bytes(PNG)


def test_the_publish_records_the_marked_jpeg_it_posts(tmp_path, rows):
    async def convert(image):
        return ConvertedJpeg(JPEG, MarkedFile(TRAINED, JPEG_SHA, MANIFEST))

    still = MediaFile(aspect="4:5", deliverable_id="d-still-1", name="carousel-4x5-01.png", key="k/carousel-4x5-01.png",
                      content_type="image/png", bytes=len(PNG))
    stager = Stager(tmp_path, WS, client=_Bucket(), converter=convert)

    staged = asyncio.run(stager.stage(still, jpeg=True))

    assert staged.name == "carousel-4x5-01.jpg" and staged.read_bytes() == JPEG
    assert rows == [(WS, JPEG_SHA, "image", MEDIA_RENDER_GENERATOR, MANIFEST)]


# ---------------------------------------------------------------------------
# generate_document's social render
# ---------------------------------------------------------------------------


def _render_to_file(monkeypatch, tmp_path, output):
    def answer(request):
        path = request.url.path
        if request.method == "POST" and path == "/render":
            return httpx.Response(202, json={"id": JOB_ID, "status": "rendering", "outputs": [], "report": {}})
        if path == f"/render/{JOB_ID}":
            return httpx.Response(200, json={"id": JOB_ID, "status": "done", "outputs": [output], "report": {}})
        return httpx.Response(200, content=MP4)

    monkeypatch.setattr(media_render_client.config, "SOCIALS_RENDER_URL", RENDER_URL)
    client = MediaRenderClient(http=httpx.AsyncClient(transport=httpx.MockTransport(answer)))
    target = tmp_path / "render.mp4"
    run = client.render_to_file({"workspace_id": str(WS)}, target, workspace_id=WS, max_wait_seconds=10, poll_seconds=0)
    return asyncio.run(run), target


def test_a_document_s_social_render_records_its_marked_file(monkeypatch, tmp_path, rows):
    finished, target = _render_to_file(monkeypatch, tmp_path, MARKED_OUTPUT)

    assert finished["status"] == "done" and target.read_bytes() == MP4
    assert rows == [(WS, MP4_SHA, "video", MEDIA_RENDER_GENERATOR, MANIFEST)]


def test_a_document_s_social_render_of_other_bytes_is_refused_and_not_recorded(monkeypatch, tmp_path, rows):
    with pytest.raises(MediaRenderError) as refused:
        _render_to_file(monkeypatch, tmp_path, {**MARKED_OUTPUT, "sha256": "e" * 64})

    assert refused.value.code == media_render_client.BAD_RESPONSE and rows == []


# ---------------------------------------------------------------------------
# The public check answers for a rendered file; no hash is logged
# ---------------------------------------------------------------------------


@pytest.fixture
def public_check(db_session, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api import provenance
    from api.widgets.rate_limit import RateLimitStore
    from core.database.database import get_db
    from tests.test_prd264_content_provenance import _FakeRedis

    redis = _FakeRedis()
    monkeypatch.setattr(provenance, "_rate_store", RateLimitStore(redis_factory=lambda: redis))
    app = FastAPI()
    app.include_router(provenance.router)
    app.dependency_overrides[get_db] = lambda: db_session
    return TestClient(app)


def test_a_rendered_video_s_hash_answers_yes_on_the_public_check(env, rows, db_session, seed_workspace, public_check):
    _, job = _start(env, _renderable(env))
    assert _run(job, Renderer(outputs=(MARKED_OUTPUT,)), FakeStore(), env.factory) is True
    assert public_check.get(f"/api/public/provenance/{MP4_SHA}").json() == {"made_by_automatos_ai": False}

    # The row the render wrote, as record_produced_file writes it (its own session, here the test's).
    ((workspace_id, sha256, kind, generator, manifest_id),) = rows
    seed_workspace(workspace_id)
    assert content_provenance.record_marked_file(db_session, workspace_id, sha256, kind, generator,
                                                 manifest_id=manifest_id) is True

    answer = public_check.get(f"/api/public/provenance/{MP4_SHA}")
    assert answer.status_code == 200
    body = answer.json()
    assert body["made_by_automatos_ai"] is True and body["kind"] == "video"


def test_a_marked_file_of_no_known_kind_is_logged_without_its_hash(rows, caplog):
    secret = "f" * 64
    with caplog.at_level(logging.ERROR, logger="core.media_render_provenance"):
        assert record_marked(WS, MarkedFile(TRAINED, secret, MANIFEST), "render.webm") is False

    assert rows == []
    assert ".webm" in caplog.text and secret not in caplog.text
