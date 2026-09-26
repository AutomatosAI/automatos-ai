"""F105 (26 Sep): a Composio lookup never freezes the event loop.

Before its model call, a chat turn, a playbook step and an agent run look up
the Composio actions to offer: ComposioToolService.get_tools_for_step (an SDK
search) and, when that finds nothing, ComposioHintService.build_hints (one SDK
tools.get per matched app). Both are sync and ran on the event loop. The
refresh-4 retest's /health probe waited 3.87 s behind one (req 26db0f433654,
01:53:33-37Z), and 16 of the 21 scheduler slips between 01:41 and 01:57Z were
these lookups, 5.6 s at worst.

Fake services hold their thread the way the SDK calls do; the tests watch the
loop around each of the three callers. No database, no SDK.
"""
import asyncio
import contextvars
import copy
import logging
import threading
import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from modules.tools.services.composio_hint_service import ComposioHintResult
from modules.tools.services.composio_tool_service import ComposioToolResult

WS = uuid.UUID("00000000-0000-0000-0000-0000000000c1")
SDK_SECONDS = 0.4
PROMPT = "Email the café its invoice."
HINT = {"role": "system", "content": "Use GMAIL_SEND_EMAIL."}
COMPOSIO_EXECUTE = {"type": "function", "function": {
    "name": "composio_execute",
    "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "params": {"type": "object"}}},
}}
_caller = contextvars.ContextVar("f105_composio_caller", default=None)


class _Lookups:
    """What the fake services saw, per call: thread, the caller's context, session."""

    def __init__(self):
        self.calls = []
        self._lock = threading.Lock()

    def hold(self, name, db):
        with self._lock:
            self.calls.append(SimpleNamespace(name=name, thread=threading.get_ident(), caller=_caller.get(), db=db))
        time.sleep(SDK_SECONDS)


@pytest.fixture
def lookups(monkeypatch):
    seen = _Lookups()

    class _ToolService:
        def __init__(self, db):
            self.db = db

        def get_tools_for_step(self, **kwargs):
            seen.hold("tools", self.db)
            return ComposioToolResult()  # nothing found: the hints run next

    class _HintService:
        def __init__(self, db):
            self.db = db

        def build_hints(self, **kwargs):
            seen.hold("hints", self.db)
            return ComposioHintResult(hint_lines=[HINT["content"]], allowed_apps=["GMAIL"],
                                      matched_actions=["GMAIL_SEND_EMAIL"], strategy_used="token_filtered")

    monkeypatch.setattr("modules.tools.services.composio_tool_service.ComposioToolService", _ToolService)
    monkeypatch.setattr("modules.tools.services.composio_hint_service.ComposioHintService", _HintService)
    return seen


@pytest.fixture
def sessions(monkeypatch):
    """The sessions the lookups open for themselves (the fake services use none)."""
    opened = []

    class _Session:
        closed = False

        def close(self):
            self.closed = True

    def _open():
        opened.append(_Session())
        return opened[-1]

    monkeypatch.setattr("core.database.database.SessionLocal", _open)
    return opened


async def _watched(step):
    """``step()``'s result (awaited when it is a coroutine) and the longest the
    loop stood still while it ran."""
    gaps, done = [], asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    result = step()
    if asyncio.iscoroutine(result):
        result = await result
    done.set()
    await beat
    return result, max(gaps)


def _chat(session):
    from consumers.chatbot.service import StreamingChatService

    chat = StreamingChatService.__new__(StreamingChatService)
    chat.db, chat.workspace_id = session, WS
    return chat


@pytest.mark.asyncio
async def test_a_chat_turn_keeps_the_loop_running_while_it_looks_up_composio(lookups, sessions):
    chat = _chat(MagicMock(name="the turn's session"))
    messages = [{"role": "system", "content": "identity"}, {"role": "system", "content": "context"},
                {"role": "user", "content": PROMPT}]

    _, stood_still = await _watched(lambda: chat._inject_composio_tools(
        messages, [COMPOSIO_EXECUTE], PROMPT, 1, None, False, None))

    assert stood_still < 0.3, f"the loop stood still for {stood_still:.2f} s during a turn's Composio lookups"
    assert [call.name for call in lookups.calls] == ["tools", "hints"]
    assert messages[2] == HINT


@pytest.mark.asyncio
async def test_an_agent_run_keeps_the_loop_running_while_it_looks_up_composio(lookups, sessions):
    import modules.agents.factory.agent_factory as af

    factory = af.AgentFactory.__new__(af.AgentFactory)
    factory.db_session, factory.logger = MagicMock(name="the run's session"), logging.getLogger("f105")
    schemas = [copy.deepcopy(COMPOSIO_EXECUTE)]
    messages = [{"role": "system", "content": "You are Scribe."}]

    _, stood_still = await _watched(lambda: factory._inject_composio_hints(
        schemas, messages, SimpleNamespace(agent_id=101), PROMPT, WS))

    assert stood_still < 0.3, f"the loop stood still for {stood_still:.2f} s during a run's Composio lookups"
    assert [call.name for call in lookups.calls] == ["tools", "hints"]
    assert messages[1] == HINT
    assert schemas[0]["function"]["parameters"]["properties"]["action"]["enum"] == ["GMAIL_SEND_EMAIL"]


@pytest.mark.asyncio
async def test_a_playbook_step_keeps_the_loop_running_while_it_looks_up_composio(monkeypatch, lookups, sessions):
    from api import recipe_executor as rex
    import modules.agents.factory.agent_factory as agent_factory
    import modules.context as context_mod
    import modules.tools.tool_router as tool_router
    import services.cli_ticket_lane as cli_lane

    offered = []

    class _LLM:
        async def generate_response(self, messages, tools):
            offered.append(list(messages))
            return SimpleNamespace(tool_calls=None, content="Sent the invoice.", usage=None)

    class _Factory:
        def __init__(self, db_session):
            pass

        async def activate_agent(self, agent_id):
            return SimpleNamespace(llm_manager=_LLM())

    class _Context:
        def __init__(self, db):
            pass

        async def build_context(self, **kwargs):
            return SimpleNamespace(system_prompt="system", tools=[])

    monkeypatch.setattr(cli_lane, "is_cli_agent", lambda db, agent_id: False)
    monkeypatch.setattr(agent_factory, "AgentFactory", _Factory)
    monkeypatch.setattr(context_mod, "ContextService", _Context)
    monkeypatch.setattr(tool_router, "get_tool_router", lambda: MagicMock(name="tool_router"))

    _, stood_still = await _watched(lambda: rex._execute_step(
        db=MagicMock(name="the step's session"), agent=SimpleNamespace(id=7, name="SCRIBE"),
        clean_prompt=PROMPT, workspace_id=WS, max_iterations=2, recipe_execution_id="exec-f105"))

    assert stood_still < 0.3, f"the loop stood still for {stood_still:.2f} s during a step's Composio lookups"
    assert [call.name for call in lookups.calls] == ["tools", "hints"]
    assert HINT in offered[0]


@pytest.mark.asyncio
async def test_the_lookups_run_on_threads_of_their_own_with_the_callers_context_and_their_own_session(
        lookups, sessions):
    turn_session = MagicMock(name="the turn's session")
    chat = _chat(turn_session)
    token = _caller.set("req-26db0f433654")
    try:
        await _watched(lambda: chat._inject_composio_tools(
            [{"role": "system", "content": "identity"}], None, PROMPT, 1, None, False, None))
    finally:
        _caller.reset(token)

    assert [call.thread != threading.get_ident() for call in lookups.calls] == [True, True]
    assert [call.caller for call in lookups.calls] == ["req-26db0f433654"] * 2
    assert [call.db is not turn_session for call in lookups.calls] == [True, True]
    assert [session.closed for session in sessions] == [True, True]


# ── the client, now used from several lookup threads at once ─────────────────

def test_one_client_is_built_when_several_lookups_ask_at_once(monkeypatch):
    import core.composio.client as composio_client

    built = []

    def _slow_init(self, *args, **kwargs):
        built.append(self)
        time.sleep(0.1)

    monkeypatch.setattr(composio_client.ComposioClient, "__init__", _slow_init)
    monkeypatch.setattr(composio_client, "_client_instance", None)
    got = []
    askers = [threading.Thread(target=lambda: got.append(composio_client.get_composio_client())) for _ in range(4)]
    for asker in askers:
        asker.start()
    for asker in askers:
        asker.join()

    assert len(built) == 1
    assert [client is built[0] for client in got] == [True] * 4


def test_a_name_lookup_survives_another_lookup_caching_an_app_meanwhile():
    from core.composio.client import ComposioClient

    client = ComposioClient.__new__(ComposioClient)  # no key, no SDK
    client._toolset = SimpleNamespace()  # what the lazy `toolset` property hands out
    client._schema_cache_ttl = 3600

    class _GmailSchemas(dict):
        """Looking in GMAIL's schemas is the moment another lookup thread caches SLACK's."""

        def __contains__(self, name):
            client._schema_cache.setdefault("SLACK", {"SLACK_SEND_MESSAGE": {"type": "function"}})
            return super().__contains__(name)

    client._schema_cache = {"GMAIL": _GmailSchemas(GMAIL_SEND_EMAIL={"type": "function"}),
                            "NOTION": {"NOTION_CREATE_PAGE": {"type": "function"}}}
    client._schema_cache_ts = {"GMAIL": time.monotonic(), "NOTION": time.monotonic()}

    found = client.get_action_schemas_by_name(
        action_names=["NOTION_CREATE_PAGE"], entity_id="ws-c1", app_names=["gmail", "notion"])

    assert [(r["app_name"], r["action_name"]) for r in found] == [("NOTION", "NOTION_CREATE_PAGE")]
