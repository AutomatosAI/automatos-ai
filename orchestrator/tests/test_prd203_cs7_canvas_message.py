"""PRD-203 C·S7 — the canvas prompt wire: send_message → client.query.

Extends the fake-SDK-factory session tests with the turn ingress that today
exists nowhere. Proves:
  * send_message calls the SDK client's query() with the (trimmed) user prompt —
    the call that turns the connected-but-idle session into a working one;
  * a mutating tool call still flows a permission.request event through the gate
    (the prompt → streamed turns → approve loop is reachable end to end);
  * no live session → not_found; empty prompt → rejected (no query fired).

Container-free: SDK client mocked (injected factory); the gate's lazy
PermissionResult* import is satisfied by a fake claude_agent_sdk module.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

_WORKER_DIR = Path(__file__).resolve().parents[2] / "services" / "workspace-worker"
sys.path.insert(0, str(_WORKER_DIR))
try:
    import canvas_events as ce
    import canvas_session_service as css
finally:
    sys.path.remove(str(_WORKER_DIR))


class _Allow:
    def __init__(self, updated_input: Optional[Dict[str, Any]] = None) -> None:
        self.updated_input = updated_input


class _Deny:
    def __init__(self, message: str = "") -> None:
        self.message = message


@pytest.fixture(autouse=True)
def _fake_sdk():
    mod = ModuleType("claude_agent_sdk")
    mod.PermissionResultAllow = _Allow  # type: ignore[attr-defined]
    mod.PermissionResultDeny = _Deny  # type: ignore[attr-defined]
    sys.modules["claude_agent_sdk"] = mod
    try:
        yield
    finally:
        sys.modules.pop("claude_agent_sdk", None)


class _FakeSDKClient:
    """SDK client whose query() records the prompts sent to it (C·S7)."""

    def __init__(self, option_kwargs: Dict[str, Any]) -> None:
        self.option_kwargs = option_kwargs
        self.queries: List[str] = []
        self._closed = asyncio.Event()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        self._closed.set()

    async def query(self, prompt: str) -> None:
        self.queries.append(prompt)

    async def receive_messages(self):
        yield SimpleNamespace(subtype="init", data={"session_id": "sdk-msg-1"})
        await self._closed.wait()


class _Factory:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.clients: List[_FakeSDKClient] = []

    def __call__(self, option_kwargs: Dict[str, Any]) -> _FakeSDKClient:
        self.calls.append(option_kwargs)
        client = _FakeSDKClient(option_kwargs)
        self.clients.append(client)
        return client


def _manager(tmp_path, events=None):
    factory = _Factory()

    async def sink(ev):
        if events is not None:
            events.append(ev)

    mgr = css.CanvasSessionManager(
        str(tmp_path),
        sdk_client_factory=factory,
        init_timeout=2.0,
        event_sink=sink if events is not None else None,
    )
    return mgr, factory


def test_send_message_invokes_client_query_with_prompt(tmp_path):
    async def scenario():
        mgr, factory = _manager(tmp_path)
        await mgr.start_session("ws-msg")

        result = await mgr.send_message("ws-msg", "  add validation to X and push  ")
        assert result["success"] is True
        # The call that today exists nowhere — with the trimmed prompt.
        assert factory.clients[0].queries == ["add validation to X and push"]

        await mgr.stop_session("ws-msg")

    asyncio.run(scenario())


def test_send_message_no_live_session_is_not_found(tmp_path):
    async def scenario():
        mgr, _ = _manager(tmp_path)
        result = await mgr.send_message("ws-none", "hello")
        assert result["success"] is False
        assert result["not_found"] is True

    asyncio.run(scenario())


def test_send_empty_prompt_rejected_no_query(tmp_path):
    async def scenario():
        mgr, factory = _manager(tmp_path)
        await mgr.start_session("ws-empty")
        result = await mgr.send_message("ws-empty", "   ")
        assert result["success"] is False
        assert factory.clients[0].queries == []  # nothing sent to the SDK
        await mgr.stop_session("ws-empty")

    asyncio.run(scenario())


def test_prompt_loop_reaches_permission_request_on_mutating_tool(tmp_path):
    """End to end: after a prompt is sent, a mutating tool call the agent makes
    flows a permission.request event (prompt → work → approve is reachable)."""

    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, factory = _manager(tmp_path, events)
        await mgr.start_session("ws-loop")

        # The prompt goes to the SDK...
        await mgr.send_message("ws-loop", "add validation and push")
        assert factory.clients[0].queries == ["add validation and push"]

        # ...and a mutating tool the agent then issues pauses for approval.
        gate = factory.calls[0]["can_use_tool"]
        task = asyncio.create_task(
            gate("Edit", {"file_path": "a.py", "old_string": "1", "new_string": "2"}, None)
        )
        await asyncio.sleep(0.05)
        assert not task.done(), "a mutating tool must block awaiting approval"

        perms = [e for e in events if e["event_type"] == ce.EVENT_PERMISSION_REQUEST]
        assert perms, "expected a permission.request event on the mutating tool call"
        req_id = perms[-1]["data"]["request_id"]

        decided = await mgr.decide("ws-loop", req_id, True)
        assert decided["success"] is True
        result = await asyncio.wait_for(task, timeout=2.0)
        assert isinstance(result, _Allow)

        await mgr.stop_session("ws-loop")

    asyncio.run(scenario())
