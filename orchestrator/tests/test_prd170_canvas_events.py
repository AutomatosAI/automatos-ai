"""PRD-170 S3 — canvas event serializer contract (container-free, no SDK).

Targets the worker-side ``canvas_events`` serializer that maps headless Claude
Agent SDK messages onto the platform's Redis event envelope. Proven here with
DUCK-TYPED fake SDK messages (SimpleNamespace) — the serializer never imports
``claude_agent_sdk``, so no container/SDK is needed.

Guarantees:
  * assistant text -> canvas.assistant.text; a tool_use -> canvas.tool.call;
    a file-mutating tool ALSO emits canvas.file.edit (so the tree refreshes);
    a result -> canvas.turn.complete.
  * every emitted event_type is inside the CLOSED CANVAS_EVENT_TYPES vocabulary
    and carries the current schema_version (drift guard — the frontend vitest
    mirrors this).
  * no secret material leaks: a tool_use whose input carries a token surfaces
    only tool_name + path, never the raw input dict / token.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

_WORKER_DIR = (
    Path(__file__).resolve().parents[2] / "services" / "workspace-worker"
)
sys.path.insert(0, str(_WORKER_DIR))
try:
    import canvas_events as ce
    import canvas_session_service as css
finally:
    sys.path.remove(str(_WORKER_DIR))

WS = "ws-1234-5678"


# ---------------------------------------------------------------------------
# Fake SDK message/block surface (duck-typed)
# ---------------------------------------------------------------------------
def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _tool_use_block(name: str, tool_input: dict, tool_id: str = "tu_1"):
    return SimpleNamespace(type="tool_use", name=name, id=tool_id, input=tool_input)


def _assistant(*blocks):
    return SimpleNamespace(role="assistant", content=list(blocks))


def _result(is_error: bool = False, num_turns: int = 1):
    return SimpleNamespace(role="result", is_error=is_error, num_turns=num_turns)


def _system_init():
    return SimpleNamespace(subtype="init", data={"session_id": "abc"})


# ---------------------------------------------------------------------------
# Vocabulary / envelope invariants
# ---------------------------------------------------------------------------
def _assert_envelope(ev: dict):
    assert ev["schema_version"] == ce.CANVAS_EVENT_SCHEMA_VERSION
    assert ev["event_type"] in ce.CANVAS_EVENT_TYPES
    assert ev["workspace_id"] == WS
    assert isinstance(ev["data"], dict)
    assert isinstance(ev["timestamp"], str) and ev["timestamp"]


def test_assistant_text_maps_to_text_event():
    events = ce.serialize_sdk_message(WS, _assistant(_text_block("Hello world")))
    assert len(events) == 1
    ev = events[0]
    _assert_envelope(ev)
    assert ev["event_type"] == ce.EVENT_ASSISTANT_TEXT
    assert ev["data"]["text"] == "Hello world"


def test_empty_text_block_is_skipped():
    events = ce.serialize_sdk_message(WS, _assistant(_text_block("")))
    assert events == []


def test_tool_use_maps_to_tool_call_event_with_path():
    events = ce.serialize_sdk_message(
        WS, _assistant(_tool_use_block("Read", {"file_path": "src/app.py"}))
    )
    assert len(events) == 1
    _assert_envelope(events[0])
    assert events[0]["event_type"] == ce.EVENT_TOOL_CALL
    assert events[0]["data"]["tool_name"] == "Read"
    assert events[0]["data"]["path"] == "src/app.py"


def test_file_mutating_tool_also_emits_file_edit_event():
    events = ce.serialize_sdk_message(
        WS, _assistant(_tool_use_block("Write", {"file_path": "README.md"}))
    )
    types = [e["event_type"] for e in events]
    assert ce.EVENT_TOOL_CALL in types
    assert ce.EVENT_FILE_EDIT in types
    edit = next(e for e in events if e["event_type"] == ce.EVENT_FILE_EDIT)
    assert edit["data"]["path"] == "README.md"
    for e in events:
        _assert_envelope(e)


def test_edit_tool_without_path_emits_only_tool_call():
    # A malformed edit tool_use (no path) must not synthesize a file-edit event.
    events = ce.serialize_sdk_message(WS, _assistant(_tool_use_block("Edit", {})))
    types = [e["event_type"] for e in events]
    assert types == [ce.EVENT_TOOL_CALL]
    assert events[0]["data"]["path"] is None


def test_result_message_maps_to_turn_complete():
    events = ce.serialize_sdk_message(WS, _result(is_error=False, num_turns=3))
    assert len(events) == 1
    _assert_envelope(events[0])
    assert events[0]["event_type"] == ce.EVENT_TURN_COMPLETE
    assert events[0]["data"]["is_error"] is False
    assert events[0]["data"]["num_turns"] == 3


def test_init_system_message_is_skipped():
    assert ce.serialize_sdk_message(WS, _system_init()) == []


def test_multiple_blocks_preserve_order():
    msg = _assistant(
        _text_block("Let me read the file."),
        _tool_use_block("Read", {"file_path": "a.py"}),
        _text_block("Now I will edit it."),
        _tool_use_block("Edit", {"file_path": "a.py"}),
    )
    events = ce.serialize_sdk_message(WS, msg)
    types = [e["event_type"] for e in events]
    # text, tool.call, text, tool.call + file.edit  (order preserved)
    assert types == [
        ce.EVENT_ASSISTANT_TEXT,
        ce.EVENT_TOOL_CALL,
        ce.EVENT_ASSISTANT_TEXT,
        ce.EVENT_TOOL_CALL,
        ce.EVENT_FILE_EDIT,
    ]


# ---------------------------------------------------------------------------
# Security: no secret material leaks into events (S5-adjacent, blocks)
# ---------------------------------------------------------------------------
def test_tool_input_secrets_never_appear_in_events():
    secret = "ghp_SUPERSECRETTOKEN1234567890"
    msg = _assistant(
        _tool_use_block(
            "Bash",
            {"command": f"git push https://{secret}@github.com/x/y.git", "cwd": "."},
        )
    )
    events = ce.serialize_sdk_message(WS, msg)
    blob = repr(events)
    assert secret not in blob
    # The raw input dict is NOT echoed; only structural fields surface.
    assert "command" not in blob


def test_session_status_and_permission_helpers_are_in_vocabulary():
    s = ce.session_status_event(WS, "running")
    _assert_envelope(s)
    assert s["event_type"] == ce.EVENT_SESSION_STATUS
    assert s["data"]["status"] == "running"

    p = ce.permission_request_event(WS, "Bash", path="./run.sh", request_id="r1")
    _assert_envelope(p)
    assert p["event_type"] == ce.EVENT_PERMISSION_REQUEST
    assert p["data"]["tool_name"] == "Bash"


def test_envelope_rejects_unknown_event_type():
    with pytest.raises(ValueError):
        ce._envelope(WS, "canvas.bogus.event", {})


# ---------------------------------------------------------------------------
# S3 bridge — the manager pump forwards serialized events to an injected sink
# ---------------------------------------------------------------------------
class _BridgeSDKClient:
    """SDK client that streams an init message then a scripted turn, then ends
    (so the pump completes and emits a stopped status)."""

    def __init__(self, option_kwargs, sdk_session_id: str) -> None:
        self.option_kwargs = option_kwargs
        self.sdk_session_id = sdk_session_id
        self.disconnected = False

    async def connect(self) -> None:  # noqa: D401
        return None

    async def disconnect(self) -> None:
        self.disconnected = True

    async def receive_messages(self):
        yield SimpleNamespace(subtype="init", data={"session_id": self.sdk_session_id})
        yield _assistant(
            _text_block("Creating the README."),
            _tool_use_block("Write", {"file_path": "README.md"}),
        )
        yield _result(is_error=False, num_turns=1)
        # generator ends -> pump sees clean exit -> stopped status event


def test_pump_bridges_events_to_sink(tmp_path):
    """End-to-end (container-free): start a session with a scripted client and an
    injected sink; assert the sink receives status.running, the turn's
    text/tool.call/file.edit, turn.complete, and finally status.stopped — all in
    the closed vocabulary."""

    async def scenario():
        received: List[Dict[str, Any]] = []

        async def sink(event: Dict[str, Any]) -> None:
            received.append(event)

        def factory(option_kwargs):
            return _BridgeSDKClient(option_kwargs, "sdk-bridge-1")

        mgr = css.CanvasSessionManager(
            str(tmp_path),
            sdk_client_factory=factory,
            init_timeout=2.0,
            event_sink=sink,
        )
        await mgr.start_session("ws-bridge")
        # Let the pump drain the scripted messages.
        for _ in range(50):
            if any(e["event_type"] == ce.EVENT_SESSION_STATUS
                   and e["data"].get("status") == css.STATUS_STOPPED
                   for e in received):
                break
            await asyncio.sleep(0.02)
        await mgr.stop_session("ws-bridge")

        types = [e["event_type"] for e in received]
        # Every event is in the closed vocabulary + carries the schema version.
        for e in received:
            assert e["schema_version"] == ce.CANVAS_EVENT_SCHEMA_VERSION
            assert e["event_type"] in ce.CANVAS_EVENT_TYPES
            assert e["workspace_id"] == "ws-bridge"

        assert ce.EVENT_SESSION_STATUS in types  # running (and stopped)
        assert ce.EVENT_ASSISTANT_TEXT in types
        assert ce.EVENT_TOOL_CALL in types
        assert ce.EVENT_FILE_EDIT in types
        assert ce.EVENT_TURN_COMPLETE in types

        statuses = [
            e["data"]["status"] for e in received
            if e["event_type"] == ce.EVENT_SESSION_STATUS
        ]
        assert css.STATUS_RUNNING in statuses
        assert css.STATUS_STOPPED in statuses

    asyncio.run(scenario())


def test_sink_failure_does_not_break_pump(tmp_path):
    """A raising sink is swallowed — a dropped UI event never kills the session."""

    async def scenario():
        async def bad_sink(event: Dict[str, Any]) -> None:
            raise RuntimeError("sink down")

        def factory(option_kwargs):
            return _BridgeSDKClient(option_kwargs, "sdk-bridge-2")

        mgr = css.CanvasSessionManager(
            str(tmp_path),
            sdk_client_factory=factory,
            init_timeout=2.0,
            event_sink=bad_sink,
        )
        started = await mgr.start_session("ws-badsink")
        assert started["success"] is True  # start survived the failing sink
        await asyncio.sleep(0.1)
        stopped = await mgr.stop_session("ws-badsink")
        assert stopped["success"] is True

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# S2 — provision-on-demand: a wizard workspace (no dir yet) is cold-provisioned
# on first canvas open; the start result reports it.
# ---------------------------------------------------------------------------
def test_first_open_cold_provisions_wizard_workspace(tmp_path):
    async def scenario():
        def factory(option_kwargs):
            return _BridgeSDKClient(option_kwargs, "sdk-prov-1")

        mgr = css.CanvasSessionManager(
            str(tmp_path), sdk_client_factory=factory, init_timeout=2.0
        )
        # No directory exists for this workspace yet (wizard workspace).
        assert not (tmp_path / "ws-wizard").exists()

        started = await mgr.start_session("ws-wizard")
        assert started["success"] is True
        # First open created the workspace dir → provisioned True.
        assert started["provisioned"] is True
        assert (tmp_path / "ws-wizard").is_dir()

        await mgr.stop_session("ws-wizard")

        # A subsequent open of an existing workspace is NOT a cold provision.
        mgr2 = css.CanvasSessionManager(
            str(tmp_path), sdk_client_factory=factory, init_timeout=2.0
        )
        reopened = await mgr2.start_session("ws-wizard")
        assert reopened["success"] is True
        assert reopened["provisioned"] is False
        await mgr2.stop_session("ws-wizard")

    asyncio.run(scenario())
