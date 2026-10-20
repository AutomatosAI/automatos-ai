"""PRD-264 EUAI-06a (EU AI Act Art. 50(2)) — a document's social render says which inputs are AI-made.

``generate_document`` with a social format renders a template through media-render
(``modules/documents/generation_service.py``). No slot file joins that render, so its
bundle declares the template's own visuals and, when the template speaks its script, the
synthetic voice (``core.media_render_bundle.with_template_inputs``). media-render marks
each file from it (``services/media-render/media_render/provenance.py``) and logs a job
that names no inputs as an older caller's. Driven through the real document path and the
real client, over the render-quota harness (media-render over httpx.MockTransport).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx
import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import tests.test_prd251w1_render_quota as quota  # noqa: E402
from core.media_render_bundle import AI_INPUTS  # noqa: E402

env = quota.env

SPOKEN_PLAN = {"voice": {"voice": "af_heart", "speed": 0.95, "lines": [{"id": "l01", "at": 0.3, "text": "Three weeks to go."}]}}
TEMPLATE_ONLY = {"footage": False, "stills": False, "voice": False, "uploads": False, "template": True}


def _sent_bundles(env):
    """The bundles POST /render received, with every request still answered by the harness."""
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/render":
            sent.append(json.loads(request.content))
        return env.renderer.handler(request)

    return sent, handler


@pytest.mark.parametrize(
    "audio_plan, expected",
    [(None, TEMPLATE_ONLY), (SPOKEN_PLAN, {**TEMPLATE_ONLY, "voice": True})],
    ids=["silent", "spoken"],
)
def test_a_documents_social_render_declares_its_ai_inputs(env, monkeypatch, tmp_path, audio_plan, expected):
    quota._documents(monkeypatch, tmp_path)
    if audio_plan is not None:
        monkeypatch.setattr(quota, "BLOCKS_60", {**quota.BLOCKS_60, "audio_plan": audio_plan})
    sent, handler = _sent_bundles(env)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await quota._generate(env, http, 1)

    result = asyncio.run(scenario())

    assert result.format == "mp4"
    (bundle,) = sent
    assert bundle[AI_INPUTS] == expected
