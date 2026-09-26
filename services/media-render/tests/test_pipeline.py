"""The real pipeline inside the image (US-102): the real `hyperframes check`
refusing a lint error, the storage allowlist holding before any fetch, the
media fetcher, the voice-timing refusals, and Kokoro through POST /tts.
"""

from __future__ import annotations

import asyncio
import json
import struct
import subprocess
from pathlib import Path
from typing import List

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN, bundle
from media_render.bundle import MediaInput, parse_bundle
from media_render.fixture import COMPOSITION, fixture_bundle
from media_render.media_urls import parse_prefixes
from media_render.pipeline import MediaFetchError, PlacedLine, fetch_media, stage_project, voice_findings
from media_render.server import TOKEN_HEADER, create_app

AUTH = {TOKEN_HEADER: TOKEN}
CLIP = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096


def storage_app(requests: List[str]) -> web.Application:
    """Our storage, as far as the renderer can tell: it records every request it gets."""

    async def serve(request: web.Request) -> web.StreamResponse:
        requests.append(request.path)
        name = request.match_info["name"]
        if name == "hook.mp4":
            return web.Response(body=CLIP, content_type="video/mp4")
        if name == "moved.mp4":
            raise web.HTTPFound("http://127.0.0.1:9/elsewhere.mp4")
        return web.Response(status=403, text="expired")

    app = web.Application()
    app.router.add_get("/automatos/{name}", serve)
    app.router.add_get("/other/{name}", serve)
    return app


def test_a_lint_error_is_refused_with_422_by_the_real_check_and_nothing_renders(settings):
    html = COMPOSITION.read_text().replace(
        'window.__timelines["main"] = tl;', 'const jitter = Math.random();\n      window.__timelines["main"] = tl;'
    )
    body = {"workspace_id": "ws-lint", "composition": {"html": html}, "variables": {"headline": "Lint me"}}

    async def go():
        async with TestClient(TestServer(create_app(settings))) as client:
            response = await client.post("/render", data=json.dumps(body), headers=AUTH)
            refusal = await response.json()
            print(json.dumps(refusal, indent=2)[:4000])
            assert response.status == 422, refusal
            assert refusal["error"] == "check_failed"
            codes = {(finding["section"], finding["code"]) for finding in refusal["findings"]}
            assert ("lint", "non_deterministic_code") in codes
            assert refusal["report"]["check"]["ok"] is False
            assert refusal["report"]["lint"]["errors"] >= 1
            job = await (await client.get(f"/render/{refusal['id']}", headers=AUTH)).json()
            assert job["status"] == "rejected" and job["outputs"] == []
            assert not (Path(settings.work_dir) / refusal["id"]).exists()

    asyncio.run(go())


def test_a_media_url_off_the_allowlist_is_refused_before_any_fetch(settings):
    async def go():
        requests: List[str] = []
        async with TestServer(storage_app(requests)) as storage:
            base = f"http://127.0.0.1:{storage.port}"
            allowed = replace_prefixes(settings, f"{base}/automatos/")
            media = [
                {"path": "assets/cine/hook.mp4", "url": f"{base}/automatos/hook.mp4?X-Amz-Signature=abc"},
                {"path": "assets/cine/other.mp4", "url": f"{base}/other/hook.mp4?X-Amz-Signature=abc"},
            ]
            async with TestClient(TestServer(create_app(allowed))) as client:
                response = await client.post("/render", data=json.dumps(bundle(media=media)), headers=AUTH)
                refusal = await response.json()
                assert response.status == 400, refusal
                assert "storage allowlist" in refusal["message"] and "X-Amz-Signature" not in refusal["message"]
            assert requests == [], "a refused bundle must not fetch anything, not even its allowed media"

    asyncio.run(go())


def replace_prefixes(settings, prefix):
    from dataclasses import replace

    return replace(settings, media_url_prefixes=parse_prefixes(prefix))


def test_the_fetcher_copies_allowed_media_and_refuses_errors_and_redirects(settings, tmp_path):
    async def go():
        requests: List[str] = []
        async with TestServer(storage_app(requests)) as storage:
            base = f"http://127.0.0.1:{storage.port}"
            allowed = replace_prefixes(settings, f"{base}/automatos/")
            project = tmp_path / "project"
            async with aiohttp.ClientSession() as session:
                await fetch_media(session, [MediaInput("assets/cine/hook.mp4", f"{base}/automatos/hook.mp4?sig=1")], project, allowed)
                assert (project / "assets/cine/hook.mp4").read_bytes() == CLIP
                for name, expected in (("denied.mp4", "403"), ("moved.mp4", "302")):
                    with pytest.raises(MediaFetchError, match=expected):
                        await fetch_media(session, [MediaInput(f"assets/cine/{name}", f"{base}/automatos/{name}")], project, allowed)
                with pytest.raises(MediaFetchError, match="allowlist"):
                    await fetch_media(session, [MediaInput("assets/cine/x.mp4", f"{base}/other/hook.mp4")], project, allowed)
            assert requests == ["/automatos/hook.mp4", "/automatos/denied.mp4", "/automatos/moved.mp4"]

    asyncio.run(go())


def test_the_fetcher_stops_at_the_size_limit(settings, tmp_path):
    from dataclasses import replace

    async def go():
        requests: List[str] = []
        async with TestServer(storage_app(requests)) as storage:
            base = f"http://127.0.0.1:{storage.port}"
            small = replace(replace_prefixes(settings, f"{base}/automatos/"), max_media_bytes=1024)
            async with aiohttp.ClientSession() as session:
                with pytest.raises(MediaFetchError, match="byte limit"):
                    await fetch_media(session, [MediaInput("assets/cine/hook.mp4", f"{base}/automatos/hook.mp4")], tmp_path, small)

    asyncio.run(go())


def test_staging_writes_the_composition_its_files_and_gsap(settings, tmp_path):
    parsed = parse_bundle(fixture_bundle(), settings, {})
    project = tmp_path / "project"
    stage_project(parsed, project, settings.gsap_path)
    index = (project / "index.html").read_text()
    assert "Rendered on brand" in index and "--brand-accent:#e96235;" in index
    assert (project / "assets" / "vendor" / "gsap.min.js").stat().st_size > 10_000
    assert (project / "assets" / "audio").is_dir()


def _line(line_id, at, seconds, path=Path("x.wav")):
    return PlacedLine(id=line_id, source="kokoro", at=at, path=path, seconds=seconds)


def test_lines_that_overlap_or_overrun_are_refused_before_the_check():
    assert voice_findings([_line("l01", 0.3, 2.0), _line("l02", 2.4, 1.0)], 5.0) == ()
    overlap = voice_findings([_line("l01", 0.3, 2.5), _line("l02", 2.4, 1.0)], 5.0)
    assert [finding["code"] for finding in overlap] == ["voice_lines_overlap"]
    overrun = voice_findings([_line("l01", 4.0, 2.0)], 5.0)
    assert [finding["code"] for finding in overrun] == ["voice_line_overruns"]
    unreadable = voice_findings([_line("l01", 1.0, None)], 5.0)
    assert [finding["code"] for finding in unreadable] == ["voice_line_unreadable"]


def test_tts_speaks_with_kokoro_through_the_api(settings):
    async def go():
        async with TestClient(TestServer(create_app(settings))) as client:
            request = {"lines": [{"id": "l01", "text": "Orders. Stock. Customers. The books."}]}
            response = await client.post("/tts", data=json.dumps(request), headers=AUTH)
            body = await response.json()
            assert response.status == 200, body
            line = body["lines"][0]
            assert line["id"] == "l01" and line["seconds"] > 1.0 and line["sample_rate"] == 24000
            assert len(line["segments"]) >= 3 and "audio_base64" not in line
            unknown = await client.post("/tts", data=json.dumps({**request, "voice": "zz_nobody"}), headers=AUTH)
            assert unknown.status == 400 and (await unknown.json())["error"] == "voice_refused"

    asyncio.run(go())


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_size(data: bytes):
    """(width, height) from a PNG's IHDR."""
    assert data[:8] == PNG_SIGNATURE, data[:8]
    return struct.unpack(">II", data[16:24])


def _pixel(settings, png: Path, x: int, y: int):
    """One pixel of a PNG as (r, g, b), read with the image's own ffmpeg."""
    raw = subprocess.run(
        [settings.ffmpeg_bin, "-v", "error", "-i", str(png), "-vf", f"crop=1:1:{x}:{y}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return tuple(raw[:3])


def test_a_preview_snapshots_the_checked_composition_at_each_moment(settings, tmp_path):
    """US-106: the real check, then the real `hyperframes snapshot`: small frames and a short reel."""
    body = {**fixture_bundle(), "preview": {"at": [2.0, 0.5]}}

    async def go():
        async with TestClient(TestServer(create_app(settings))) as client:
            response = await client.post("/render", data=json.dumps(body), headers=AUTH)
            accepted = await response.json()
            assert response.status == 202, accepted
            job = accepted
            for _ in range(240):
                job = await (await client.get(f"/render/{accepted['id']}", headers=AUTH)).json()
                if job["status"] in ("done", "failed"):
                    break
                await asyncio.sleep(0.5)
            print(json.dumps(job, indent=2)[:4000])
            assert job["status"] == "done", job
            assert job["report"]["check"]["errors"] == 0
            assert [output["name"] for output in job["outputs"]] == ["preview-01.png", "preview-02.png", "preview.mp4"]
            frames, reel = job["outputs"][:2], job["outputs"][2]
            assert [frame["at"] for frame in frames] == [0.5, 2.0]
            for frame in frames:
                data = await (await client.get(frame["path"], headers=AUTH)).read()
                assert _png_size(data) == (540, 960) == (frame["width"], frame["height"])
                (tmp_path / frame["name"]).write_bytes(data)
            # The frames are the composition itself: its brand background, not a blank page.
            assert all(abs(a - b) <= 3 for a, b in zip(_pixel(settings, tmp_path / "preview-02.png", 8, 8), (0x14, 0x17, 0x1C)))
            assert reel["probe"]["video_codec"] == "h264" and (reel["width"], reel["height"]) == (540, 960)
            assert abs(reel["duration"] - 1.0) <= 0.2
            assert "preview_seconds" in job["report"]["timings"]

    asyncio.run(go())


def _png_colour_type(data: bytes) -> int:
    """The IHDR colour type: 2 is RGB, 6 is RGBA."""
    return data[25]


def test_a_still_renders_the_checked_composition_as_full_size_pngs(settings, tmp_path):
    """US-107: the real check, then the real `hyperframes snapshot`, at the composition's own size, flattened to RGB."""
    silent = {key: value for key, value in fixture_bundle().items() if key != "audio"}

    async def render(client, body):
        response = await client.post("/render", data=json.dumps(body), headers=AUTH)
        accepted = await response.json()
        assert response.status == 202, accepted
        job = accepted
        for _ in range(240):
            job = await (await client.get(f"/render/{accepted['id']}", headers=AUTH)).json()
            if job["status"] in ("done", "failed"):
                break
            await asyncio.sleep(0.5)
        print(json.dumps(job, indent=2)[:4000])
        assert job["status"] == "done", job
        assert job["report"]["check"]["errors"] == 0
        return job

    async def go():
        async with TestClient(TestServer(create_app(settings))) as client:
            job = await render(client, {**silent, "still": {"at": [0.5, 2.0]}})
            assert [output["name"] for output in job["outputs"]] == ["render-01.png", "render-02.png"]
            assert [(o["kind"], o["index"], o["at"], o["aspect"]) for o in job["outputs"]] == [
                ("still", 1, 0.5, "9:16"),
                ("still", 2, 2.0, "9:16"),
            ]
            for output in job["outputs"]:
                assert "duration" not in output
                data = await (await client.get(output["path"], headers=AUTH)).read()
                assert _png_size(data) == (1080, 1920) == (output["width"], output["height"])
                assert _png_colour_type(data) == 2 and len(data) == output["bytes"]
                (tmp_path / output["name"]).write_bytes(data)
            # A still is the composition itself: its brand background, at full size.
            assert all(abs(a - b) <= 3 for a, b in zip(_pixel(settings, tmp_path / "render-02.png", 8, 8), (0x14, 0x17, 0x1C)))
            assert "still_seconds" in job["report"]["timings"]

            single = await render(client, {**silent, "still": {"at": [1.0]}})
            assert [output["name"] for output in single["outputs"]] == ["render.png"]

    asyncio.run(go())
