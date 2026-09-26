"""F105 (26 Sep): a turn waits for a Composio lookup for a bounded time.

The lookups left the event loop (core.composio.off_loop), but a turn still
waited on them however long the SDK took, and the SDK's own bound is 60 s a
try, three tries. A turn now gives up after COMPOSIO_LOOKUP_TIMEOUT_SECONDS
and goes on without Composio tools, with a WARNING naming the step and the
app. A playbook step or agent run then skips the hint lookup, which would wait
on the same SDK. The SDK handle that serves lookups gives up as soon,
un-retried, so a hung call frees its thread.

A fake SDK hangs; the tests time the turn around it. No database, no SDK.
"""
import asyncio
import copy
import logging
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from config import config

WS = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
PROMPT = "Email the café its invoice."
HANG_SECONDS = 3.0
BOUND_SECONDS = 0.3
COMPOSIO_EXECUTE = {"type": "function", "function": {
    "name": "composio_execute",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}}},
}}
SEND_EMAIL_TOOL = {"type": "function", "function": {
    "name": "GMAIL_SEND_EMAIL", "description": "Send an email.",
    "parameters": {"type": "object", "properties": {"to": {"type": "string"}}},
}}


@pytest.fixture
def hung_sdk(monkeypatch):
    """The tool search's SDK call hangs (until the test ends); the hint lookup
    is counted, never answered by the SDK."""
    import modules.tools.services.composio_tool_service as tool_service

    released = threading.Event()
    hints = []

    class _Client:
        def search_actions_for_step(self, **kwargs):
            released.wait(HANG_SECONDS)
            return []

    class _HintService:
        def __init__(self, db):
            pass

        def build_hints(self, **kwargs):
            hints.append(kwargs)
            return SimpleNamespace(hint_lines=[], matched_actions=[], strategy_used="none", allowed_apps=[])

    monkeypatch.setattr(config, "COMPOSIO_LOOKUP_TIMEOUT_SECONDS", BOUND_SECONDS, raising=False)
    monkeypatch.setattr(tool_service.ComposioToolService, "_resolve_allowed_apps", lambda self, a, w: ["GMAIL"])
    monkeypatch.setattr(tool_service.ComposioToolService, "_resolve_entity_id", lambda self, w: "entity-c1")
    monkeypatch.setattr(tool_service, "get_composio_client", lambda: _Client())
    monkeypatch.setattr("modules.tools.services.composio_hint_service.ComposioHintService", _HintService)
    monkeypatch.setattr("core.database.database.SessionLocal", lambda: MagicMock(name="the lookup's session"))
    yield SimpleNamespace(hints=hints)
    released.set()  # free the stuck lookup thread


async def _timed(step):
    started = time.monotonic()
    result = step()
    if asyncio.iscoroutine(result):
        result = await result
    return result, time.monotonic() - started


@pytest.mark.asyncio
async def test_a_chat_turn_goes_on_without_composio_when_the_sdk_hangs(hung_sdk, caplog):
    from consumers.chatbot.service import StreamingChatService

    chat = StreamingChatService.__new__(StreamingChatService)
    chat.db, chat.workspace_id = MagicMock(name="the turn's session"), WS
    messages = [{"role": "system", "content": "identity"}, {"role": "user", "content": PROMPT}]

    with caplog.at_level(logging.WARNING):
        (tools, found), took = await _timed(lambda: chat._inject_composio_tools(
            messages, [COMPOSIO_EXECUTE], PROMPT, 1, None, False, None))

    assert took < 1.5, f"the turn waited {took:.1f} s on a hung Composio SDK"
    assert (tools, found) == ([COMPOSIO_EXECUTE], None)
    assert len(messages) == 2
    (warning,) = [r.getMessage() for r in caplog.records if "composio-lookup" in r.getMessage()]
    assert "chat turn (agent 1): tool search" in warning and "gmail" in warning  # the searched toolkit


@pytest.mark.asyncio
async def test_an_agent_run_skips_the_hints_after_the_search_gave_up(hung_sdk):
    import modules.agents.factory.agent_factory as af

    factory = af.AgentFactory.__new__(af.AgentFactory)
    factory.db_session, factory.logger = MagicMock(name="the run's session"), logging.getLogger("f105")
    schemas = [copy.deepcopy(COMPOSIO_EXECUTE)]

    _, took = await _timed(lambda: factory._inject_composio_hints(
        schemas, [{"role": "system", "content": "You are Scribe."}], SimpleNamespace(agent_id=101), PROMPT, WS))

    assert took < 1.5, f"the run waited {took:.1f} s on a hung Composio SDK"
    assert hung_sdk.hints == []


def test_the_lookup_handle_gives_up_as_soon_and_is_not_retried(monkeypatch):
    import core.composio.client as composio_client

    built = []

    class _SDK:
        def __init__(self, **kwargs):
            built.append(kwargs)

    monkeypatch.setattr(composio_client, "_get_composio", lambda: _SDK)
    monkeypatch.setattr(composio_client, "_get_composio_openai_provider", lambda: object)
    monkeypatch.setattr(config, "COMPOSIO_LOOKUP_TIMEOUT_SECONDS", 7.0, raising=False)
    client = composio_client.ComposioClient(api_key="a-key")

    client.toolset, client.composio
    lookups, everything_else = built

    assert (lookups.get("timeout"), lookups.get("max_retries")) == (7.0, 0)
    assert "timeout" not in everything_else and "max_retries" not in everything_else  # execution keeps the SDK's


def test_a_turns_action_list_is_fetched_through_the_lookup_handle(monkeypatch):
    from core.composio.client import ComposioClient
    from modules.tools.services.composio_hint_service import ComposioHintService

    asked = []

    def _handle(name):
        return SimpleNamespace(tools=SimpleNamespace(get=lambda **kwargs: asked.append(name) or [SEND_EMAIL_TOOL]))

    client = ComposioClient.__new__(ComposioClient)  # no key, no SDK
    client._toolset, client._composio = _handle("lookup handle"), _handle("execution handle")
    monkeypatch.setattr("core.composio.client.get_composio_client", lambda: client)
    params = {}

    ComposioHintService(db=None)._enrich_params_from_sdk([("GMAIL", ["GMAIL_SEND_EMAIL"])], params)

    assert asked == ["lookup handle"]
    assert "GMAIL_SEND_EMAIL" in params
