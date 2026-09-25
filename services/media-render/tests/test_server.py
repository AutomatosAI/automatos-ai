"""The HTTP API (US-102) with a stub pipeline: the token gate, the job lifecycle,
the 422 refusal, and the render queue's admission rule. The real pipeline's
checks are in test_pipeline.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile
from aiohttp.test_utils import TestClient, TestServer

from helpers import TOKEN, bundle
from media_render.kokoro_tts import SpokenLine
from media_render.pipeline import CheckOutcome, PipelineError, RenderResult
from media_render.server import TOKEN_HEADER, create_app
from media_render.tts import Speaker

AUTH = {TOKEN_HEADER: TOKEN}
FINDING = {"section": "lint", "severity": "error", "code": "non_deterministic_code", "message": "Math.random()"}


class StubPipeline:
    """Checks pass (or refuse) at once; each render holds its slot until released."""

    def __init__(self, refuse: bool = False) -> None:
        self.refuse = refuse
        self.checked: List[str] = []
        self.started: List[str] = []
        self.running: Dict[str, str] = {}
        self.max_running = 0
        self.max_per_workspace = 0
        self._gates: Dict[str, asyncio.Event] = {}

    def gate(self, reference: str) -> asyncio.Event:
        return self._gates.setdefault(reference, asyncio.Event())

    async def prepare_and_check(self, job):
        self.checked.append(job.bundle.reference)
        if self.refuse:
            return CheckOutcome(ok=False, findings=(FINDING,), report={"findings": [FINDING]})
        return CheckOutcome(ok=True, report={"check": {"ok": True}, "timings": {"check_seconds": 0.1}})

    async def render(self, job):
        reference = job.bundle.reference
        self.started.append(reference)
        self.running[reference] = job.workspace_id
        self.max_running = max(self.max_running, len(self.running))
        per_workspace = max(list(self.running.values()).count(ws) for ws in self.running.values())
        self.max_per_workspace = max(self.max_per_workspace, per_workspace)
        try:
            await self.gate(reference).wait()
        finally:
            del self.running[reference]
        job.output_dir.mkdir(parents=True, exist_ok=True)
        (job.output_dir / "render.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 fake render")
        output = {"name": "render.mp4", "aspect": "9:16", "width": 1080, "height": 1920, "bytes": 30, "duration": 3.0}
        return RenderResult(outputs=(output,), timings={"render_seconds": 0.2})


def serve(settings, pipeline=None, speaker=None):
    return TestClient(TestServer(create_app(settings, pipeline=pipeline or StubPipeline(), speaker=speaker, library={})))


async def post(client, body, headers=AUTH):
    response = await client.post("/render", data=json.dumps(body), headers=headers)
    return response.status, await response.json()


async def status_of(client, job_id):
    response = await client.get(f"/render/{job_id}", headers=AUTH)
    return (await response.json())["status"]


async def settle(predicate, attempts=200):
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("the condition never held")


def test_health_is_open_and_every_other_path_needs_the_token(settings):
    async def go():
        async with serve(settings) as client:
            health = await client.get("/health")
            assert health.status == 200
            body = await health.json()
            assert body["status"] == "healthy" and body["service"] == "media-render"
            assert body["versions"]["hyperframes"] == "0.8.62"
            for method, path in (
                ("POST", "/render"),
                ("GET", "/render/0123"),
                ("GET", "/render/0123/output/render.mp4"),
                ("POST", "/tts"),
            ):
                for headers in ({}, {TOKEN_HEADER: "wrong"}):
                    response = await client.request(method, path, headers=headers)
                    assert response.status == 401, (method, path, headers)
            # The right token passes the gate (an unknown job is then a 404).
            assert (await client.get("/render/0123", headers=AUTH)).status == 404

    asyncio.run(go())


def test_a_checked_bundle_is_accepted_rendered_and_served(settings):
    async def go():
        pipeline = StubPipeline()
        async with serve(settings, pipeline) as client:
            status, job = await post(client, bundle(reference="post-1"))
            assert status == 202 and job["status"] == "rendering"
            assert job["composition"] == {"duration": 3.0, "width": 1080, "height": 1920, "aspect": "9:16"}
            assert job["report"]["check"] == {"ok": True}
            pipeline.gate("post-1").set()
            await settle(lambda: pipeline.running == {} and "post-1" in pipeline.started)
            for _ in range(100):
                if await status_of(client, job["id"]) == "done":
                    break
                await asyncio.sleep(0.01)
            done = await (await client.get(f"/render/{job['id']}", headers=AUTH)).json()
            assert done["status"] == "done"
            assert done["outputs"][0]["path"] == f"/render/{job['id']}/output/render.mp4"
            assert {"check_seconds", "render_seconds", "queue_seconds"} <= set(done["report"]["timings"])
            served = await client.get(done["outputs"][0]["path"], headers=AUTH)
            assert served.status == 200 and (await served.read()).startswith(b"\x00\x00\x00\x18ftyp")
            assert (await client.get(f"/render/{job['id']}/output/other.mp4", headers=AUTH)).status == 404

    asyncio.run(go())


def test_a_bundle_that_fails_its_check_is_refused_with_422_and_never_renders(settings):
    async def go():
        pipeline = StubPipeline(refuse=True)
        async with serve(settings, pipeline) as client:
            status, body = await post(client, bundle(reference="bad"))
            assert status == 422
            assert body["error"] == "check_failed" and body["findings"] == [FINDING]
            assert pipeline.checked == ["bad"] and pipeline.started == []
            assert await status_of(client, body["id"]) == "rejected"
            assert not (Path(settings.work_dir) / body["id"]).exists(), "a refused job keeps no files"

    asyncio.run(go())


def test_three_submissions_from_two_workspaces_two_run_and_the_third_queues(settings):
    async def go():
        pipeline = StubPipeline()
        async with serve(settings, pipeline) as client:
            ids = {}
            for reference, workspace in (("a1", "ws-a"), ("a2", "ws-a"), ("b1", "ws-b")):
                status, job = await post(client, bundle(workspace, reference=reference))
                assert status == 202
                ids[reference] = job["id"]
            await settle(lambda: len(pipeline.started) == 2)
            assert sorted(pipeline.started) == ["a1", "b1"]
            assert [await status_of(client, ids[r]) for r in ("a1", "a2", "b1")] == ["rendering", "queued", "rendering"]
            queued = await (await client.get(f"/render/{ids['a2']}", headers=AUTH)).json()
            assert queued["queue_position"] == 1

            # b1 finishing frees a slot, but a2 still waits for its own workspace's a1.
            pipeline.gate("b1").set()
            await settle(lambda: "b1" not in pipeline.running)
            await asyncio.sleep(0.05)
            assert "a2" not in pipeline.started
            pipeline.gate("a1").set()
            await settle(lambda: "a2" in pipeline.started)
            pipeline.gate("a2").set()
            await settle(lambda: not pipeline.running)
            assert pipeline.max_running == 2 and pipeline.max_per_workspace == 1

    asyncio.run(go())


def test_the_render_queue_is_first_come_first_served_across_workspaces(settings):
    async def go():
        pipeline = StubPipeline()
        async with serve(settings, pipeline) as client:
            for reference, workspace in (("a", "ws-a"), ("b", "ws-b"), ("c", "ws-c"), ("d", "ws-d")):
                assert (await post(client, bundle(workspace, reference=reference)))[0] == 202
            await settle(lambda: len(pipeline.started) == 2)
            for finished, expected in (("a", "c"), ("b", "d")):
                pipeline.gate(finished).set()
                await settle(lambda expected=expected: expected in pipeline.started)
            assert pipeline.started == ["a", "b", "c", "d"]
            pipeline.gate("c").set()
            pipeline.gate("d").set()
            await settle(lambda: not pipeline.running)
            assert pipeline.max_running == 2

    asyncio.run(go())


def test_bad_requests_are_refused_before_any_work(settings):
    async def go():
        pipeline = StubPipeline()
        async with serve(settings, pipeline) as client:
            response = await client.post("/render", data="{not json", headers=AUTH)
            assert response.status == 400 and (await response.json())["error"] == "invalid_json"
            status, body = await post(client, bundle(variables={}))
            assert status == 400 and body["error"] == "invalid_bundle" and "headline" in body["message"]
            assert pipeline.checked == []

    asyncio.run(go())


def test_a_full_queue_answers_503(settings):
    async def go():
        pipeline = StubPipeline()
        async with serve(replace(settings, max_active_jobs=2), pipeline) as client:
            for reference in ("a", "b"):
                assert (await post(client, bundle(f"ws-{reference}", reference=reference)))[0] == 202
            response = await client.post("/render", data=json.dumps(bundle("ws-c", reference="c")), headers=AUTH)
            assert response.status == 503 and response.headers["Retry-After"]
            for reference in ("a", "b"):
                pipeline.gate(reference).set()
            await settle(lambda: not pipeline.running)

    asyncio.run(go())


def test_a_pipeline_failure_fails_the_job_with_its_status(settings):
    class Unreachable(StubPipeline):
        async def prepare_and_check(self, job):
            raise PipelineError("media_fetch_failed", "assets/cine/hook.mp4: storage answered 403", status=502)

    async def go():
        async with serve(settings, Unreachable()) as client:
            status, body = await post(client, bundle(reference="x"))
            assert status == 502 and body["error"] == "media_fetch_failed"
            job = await (await client.get(f"/render/{body['id']}", headers=AUTH)).json()
            assert job["status"] == "failed" and job["error"]["code"] == "media_fetch_failed"

    asyncio.run(go())


def fake_synthesize(text, output, settings, *, voice, speed, lang):
    rate = 24000
    t = np.arange(int(0.8 * rate)) / rate
    samples = (0.4 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(output), samples, rate)
    return SpokenLine(path=output, seconds=0.8, sample_rate=rate, segments=((0.0, 0.8),))


def test_tts_returns_durations_segments_and_the_wav_on_request(settings):
    async def go():
        speaker = Speaker(settings, synthesize=fake_synthesize)
        async with serve(settings, speaker=speaker) as client:
            request = {"lines": [{"id": "l01", "text": "Hello."}, {"id": "l02", "text": "Again."}], "include_audio": True}
            response = await client.post("/tts", data=json.dumps(request), headers=AUTH)
            assert response.status == 200
            body = await response.json()
            assert (body["voice"], body["speed"], body["lang"]) == ("af_heart", 0.95, "en-us")
            assert [line["id"] for line in body["lines"]] == ["l01", "l02"]
            assert body["lines"][0]["seconds"] == 0.8
            assert body["lines"][0]["segments"] == [{"start": 0.0, "end": 0.8}]
            assert body["lines"][0]["audio_base64"]
            bad = await client.post("/tts", data=json.dumps({"lines": [], "voice": "af_heart"}), headers=AUTH)
            assert bad.status == 400
            wrong = await client.post("/tts", data=json.dumps({"lines": [{"id": "l01", "text": "x"}], "speed": 9}), headers=AUTH)
            assert wrong.status == 400

    asyncio.run(go())
