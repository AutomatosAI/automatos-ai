"""PRD-251 Wave 3 (final review, P251W3) — a publisher call reaches Composio once.

The Composio SDK re-sends a call on a timeout, a dropped connection, a 408, 409,
429 or 5xx (its default ``max_retries`` is 2) with no idempotency key: a publish the
platform had already taken would post twice, under the publisher's own "never sent
twice blindly" rule (``publish_steps``). Against the real SDK over a mock transport
that answers the execute POST with 504:

* a call made inside ``upload_spec.publisher_call()`` (every ``execute_with_uploads``
  call) sends ONE execute POST, and the publisher reads the 504 as "may be live";
* any other call keeps the SDK's re-send, as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

httpx = pytest.importorskip("httpx")
composio = pytest.importorskip("composio")

from core.composio import client as composio_client  # noqa: E402
from core.composio import upload_spec  # noqa: E402
from modules.socials.capabilities import SEEDED_CHANNELS  # noqa: E402
from modules.socials.publish_steps import MAY_BE_LIVE, _failure  # noqa: E402

ACTION = "TWITTER_CREATION_OF_A_POST"


@pytest.fixture
def gateway(monkeypatch):
    """A ComposioClient whose SDK handles talk to a gateway that times out the execute."""
    posts = []

    def handler(request):
        if request.method == "POST" and "/tools/execute/" in request.url.path:
            posts.append(request.url.path)
            return httpx.Response(504, json={"error": "gateway timeout"})
        return httpx.Response(200, json={})

    def sdk(**kwargs):
        handle = composio.Composio(**kwargs)
        handle._client._client = httpx.Client(transport=httpx.MockTransport(handler), base_url=handle._client.base_url)
        return handle

    monkeypatch.setattr(composio_client, "_get_composio", lambda: sdk)
    monkeypatch.setattr(composio_client, "composio_action_denial", lambda action: None)
    return composio_client.ComposioClient(api_key="test-key"), posts


def test_a_publisher_call_that_times_out_at_the_gateway_is_sent_once(gateway):
    client, posts = gateway
    with upload_spec.publisher_call():
        result = client.execute_action(ACTION, {"text": "Launch"}, entity_id="ws")

    assert result["success"] is False and "504" in result["error"]
    assert len(posts) == 1  # the SDK did not send it again
    publish_step = next(s for s in SEEDED_CHANNELS["twitter"].kinds["text"] if s.action == ACTION)
    failure = _failure(publish_step, result)
    assert failure.transient is False and MAY_BE_LIVE in failure.message


def test_any_other_call_keeps_the_sdk_s_own_retries(gateway):
    client, posts = gateway
    result = client.execute_action(ACTION, {"text": "Launch"}, entity_id="ws")
    assert result["success"] is False and len(posts) == 3
    assert upload_spec.is_publisher_call() is False  # the mark does not outlive its block
