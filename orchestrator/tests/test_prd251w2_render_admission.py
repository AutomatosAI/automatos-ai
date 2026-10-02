"""PRD-251 Wave 2, P251W1-RVW-4 — the orchestrator's side of media-render's per-workspace bound.

media-render admits at most MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE unfinished
jobs for one workspace and refuses the next with 429 ``workspace_busy`` and a
Retry-After, while it still admits the other workspaces
(services/media-render/tests/test_server.py pins the service). The real client
talks to media-render over ``httpx.MockTransport``:

* a 429 ``workspace_busy`` is read with its code, its status and its Retry-After;
* ``submit_when_free`` waits Retry-After and submits again, exactly as it does
  for a full renderer (503 ``busy``);
* once waiting would pass the render's deadline it raises ``workspace_busy`` at
  once; a post's render then fails saying the workspace has too many renders in
  progress, and generate_document answers 429 saying the same;
* compose and the Railway manifest carry the setting.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import httpx  # noqa: E402
import yaml  # noqa: E402

import core.models  # noqa: E402,F401  (registers every mapper)
import core.media_render_client as media_render_client  # noqa: E402
import modules.socials.render as render  # noqa: E402
from core.media_render_client import (  # noqa: E402
    WORKSPACE_BUSY,
    MediaRenderClient,
    MediaRenderError,
    MediaRenderUnavailable,
)

RENDER_URL = "http://media-render:8090"
WS = "11111111-2222-3333-4444-555555555555"
JOB_ID = "c" * 32
RETRY_AFTER = "30"
WORKSPACE_BUSY_BODY = {
    "error": "workspace_busy",
    "message": "this workspace already has 4 renders in progress, the most one workspace may have at once; "
    "try again when one of them ends",
}
BUSY_BODY = {"error": "busy", "message": "20 renders are already in progress; try again shortly"}


def _workspace_busy() -> httpx.Response:
    return httpx.Response(429, json=WORKSPACE_BUSY_BODY, headers={"Retry-After": RETRY_AFTER})


def _busy() -> httpx.Response:
    return httpx.Response(503, json=BUSY_BODY, headers={"Retry-After": RETRY_AFTER})


def _accepted() -> httpx.Response:
    return httpx.Response(202, json={"id": JOB_ID, "status": "queued", "outputs": [], "report": {}})


class Renderer:
    """POST /render answers in turn from ``answers``; the last answer repeats."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.submitted = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/render")
        self.submitted.append(json.loads(request.content)["workspace_id"])
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        return answer()


@pytest.fixture
def renderer_url(monkeypatch):
    for cfg in {id(m.config): m.config for m in (media_render_client, render)}.values():
        monkeypatch.setattr(cfg, "SOCIALS_RENDER_URL", RENDER_URL, raising=False)


@pytest.fixture
def waits(monkeypatch):
    """The client's sleeps, recorded instead of slept: only media_render_client's view of asyncio changes."""
    recorded = []

    async def sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr(media_render_client, "asyncio", SimpleNamespace(sleep=sleep))
    return recorded


def _submit_when_free(renderer: Renderer, deadline_in: float, poll_seconds: float = 5.0):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(renderer.handler)) as http:
            deadline = time.monotonic() + deadline_in
            return await MediaRenderClient(http).submit_when_free(
                {"workspace_id": WS}, deadline=deadline, poll_seconds=poll_seconds
            )

    return asyncio.run(go())


def test_a_workspace_busy_answer_carries_its_code_status_and_retry_after(renderer_url):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _workspace_busy())) as http:
            return await MediaRenderClient(http).submit({"workspace_id": WS})

    with pytest.raises(MediaRenderError) as refused:
        asyncio.run(go())
    assert refused.value.code == WORKSPACE_BUSY == "workspace_busy"
    assert refused.value.status == 429 and refused.value.retry_after == float(RETRY_AFTER)
    assert "4 renders in progress" in str(refused.value)
    assert not isinstance(refused.value, MediaRenderUnavailable)


def test_submit_when_free_waits_out_workspace_busy_like_busy_then_submits(renderer_url, waits):
    renderer = Renderer(_workspace_busy, _busy, _workspace_busy, _accepted)
    record = _submit_when_free(renderer, deadline_in=600)
    assert record["id"] == JOB_ID and record["status"] == "queued"
    assert renderer.submitted == [WS] * 4
    assert waits == [float(RETRY_AFTER)] * 3, "each refusal is waited out for its own Retry-After"


def test_workspace_busy_without_retry_after_waits_the_poll_interval(renderer_url, waits):
    def no_header() -> httpx.Response:
        return httpx.Response(429, json=WORKSPACE_BUSY_BODY)

    renderer = Renderer(no_header, _accepted)
    assert _submit_when_free(renderer, deadline_in=600, poll_seconds=7.0)["id"] == JOB_ID
    assert waits == [7.0]


def test_workspace_busy_past_the_deadline_is_raised_without_waiting(renderer_url, waits):
    renderer = Renderer(_workspace_busy)
    with pytest.raises(MediaRenderError) as refused:
        _submit_when_free(renderer, deadline_in=float(RETRY_AFTER) - 1)
    assert refused.value.code == WORKSPACE_BUSY
    assert renderer.submitted == [WS] and waits == []


def test_a_post_still_refused_at_its_deadline_fails_saying_its_workspace_is_busy(renderer_url, waits):
    renderer = Renderer(_workspace_busy)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(renderer.handler)) as http:
            return await render._submit(MediaRenderClient(http), {"workspace_id": WS}, time.monotonic() + 1)

    with pytest.raises(render.RenderFailure) as failed:
        asyncio.run(go())
    assert failed.value.code == WORKSPACE_BUSY
    assert failed.value.message == render.WORKSPACE_BUSY_MESSAGE
    assert "too many renders in progress" in failed.value.message
    assert failed.value.report == {"code": WORKSPACE_BUSY}


def test_generate_document_answers_429_when_its_workspace_is_busy():
    import api.document_generation as documents_module

    refused = MediaRenderError(WORKSPACE_BUSY, WORKSPACE_BUSY_BODY["message"], status=429, retry_after=30.0)
    answer = documents_module._social_render_error(refused)
    assert answer.status_code == 429
    assert answer.detail == documents_module.RENDER_WORKSPACE_BUSY
    assert "too many renders in progress" in answer.detail


def test_compose_and_railway_carry_the_workspace_bound():
    compose = yaml.safe_load((_ROOT / "docker-compose.yml").read_text())
    environment = compose["services"]["media-render"]["environment"]
    assert environment["MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE"] == "${MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE:-4}"
    manifest = json.loads((_ROOT / "infrastructure" / "railway-manifest.json").read_text())
    keys = manifest["services"]["media-render"]["env_keys"]
    assert "MEDIA_RENDER_MAX_ACTIVE_JOBS_PER_WORKSPACE" in keys and keys == sorted(keys)
