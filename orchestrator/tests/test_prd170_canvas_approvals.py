"""PRD-170 S4 — canvas approval loop (container-free).

Proves the headline guarantee — *nothing mutating applies without approval* —
against the worker-side gate with the Claude Agent SDK MOCKED:

  * ``canvas_approvals`` pure policy: bash ALWAYS gated (even with auto-accept),
    file edits gated unless auto-accept; the permission payload carries the
    old/new diff (Write reads the current file; Edit surfaces the hunk) and never
    a secret; the async registry register→resolve→fail_all round-trip.
  * the session manager's can_use_tool gate end to end (injected fake SDK): a
    mutating tool BLOCKS in the callback, emits a ``permission.request`` event,
    and only unblocks on ``decide`` — approve → Allow, deny → Deny(message);
    auto-accept short-circuits an edit to Allow WITHOUT a prompt but still gates
    bash; confinement still hard-denies an escape before any approval.

Pure stdlib + pytest: no DB, no docker, no ``claude_agent_sdk`` import — the
gate's Allow/Deny are asserted through duck-typed stand-ins for
PermissionResultAllow / PermissionResultDeny injected into ``sys.modules``.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

_WORKER_DIR = (
    Path(__file__).resolve().parents[2] / "services" / "workspace-worker"
)
sys.path.insert(0, str(_WORKER_DIR))
try:
    import canvas_approvals as ca
    import canvas_events as ce
    import canvas_session_service as css
finally:
    sys.path.remove(str(_WORKER_DIR))


# ---------------------------------------------------------------------------
# A duck-typed claude_agent_sdk so the gate's PermissionResult* are inspectable.
# The manager imports these lazily INSIDE the callback, so we install a fake
# module keyed to the exact names it imports.
# ---------------------------------------------------------------------------
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


# ===========================================================================
# Pure policy — canvas_approvals
# ===========================================================================
def test_requires_approval_bash_always_even_with_auto_accept():
    assert ca.requires_approval("Bash", auto_accept_edits=False) is True
    # Auto-accept NEVER covers a shell command — a destructive command can't be
    # silently auto-run (matches the frontend diffApproval model).
    assert ca.requires_approval("Bash", auto_accept_edits=True) is True


def test_requires_approval_file_edits_honor_auto_accept():
    for tool in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        assert ca.requires_approval(tool, auto_accept_edits=False) is True
        assert ca.requires_approval(tool, auto_accept_edits=True) is False


def test_read_only_tools_never_gated():
    for tool in ("Read", "Glob", "Grep", "WebFetch"):
        assert ca.requires_approval(tool, auto_accept_edits=False) is False


def test_write_payload_carries_before_and_after_from_disk(tmp_path):
    f = tmp_path / "app.py"
    f.write_text("old body\n")
    payload = ca.build_permission_payload(
        "Write",
        {"file_path": str(f), "content": "new body\n"},
        "perm_x",
        root=tmp_path,
    )
    assert payload["tool_name"] == "Write"
    assert payload["old_content"] == "old body\n"
    assert payload["new_content"] == "new body\n"
    assert payload["request_id"] == "perm_x"


def test_write_payload_new_file_has_empty_old(tmp_path):
    payload = ca.build_permission_payload(
        "Write",
        {"file_path": str(tmp_path / "brand_new.py"), "content": "print(1)\n"},
        "perm_y",
        root=tmp_path,
    )
    assert payload["old_content"] == ""  # new file → all-additions diff
    assert payload["new_content"] == "print(1)\n"


def test_edit_payload_surfaces_the_hunk():
    payload = ca.build_permission_payload(
        "Edit",
        {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 2"},
        "perm_z",
    )
    assert payload["old_content"] == "x = 1"
    assert payload["new_content"] == "x = 2"


def test_bash_payload_carries_command_and_no_diff():
    payload = ca.build_permission_payload(
        "Bash", {"command": "pytest -q"}, "perm_b"
    )
    assert payload["command"] == "pytest -q"
    assert "old_content" not in payload and "new_content" not in payload


def test_permission_payload_never_leaks_secret_kwargs():
    # A tool input that happens to carry a token-looking field: the payload only
    # surfaces the known structural keys, never arbitrary input.
    payload = ca.build_permission_payload(
        "Write",
        {"file_path": "a.py", "content": "ok", "authorization": "Bearer sk-secret"},
        "perm_s",
    )
    assert "authorization" not in str(payload)


def test_pending_registry_register_resolve_roundtrip():
    async def scenario():
        reg = ca.PendingApprovals()
        fut = await reg.register("r1", "Write")
        assert reg.pending_ids() == {"r1": "Write"}
        found = await reg.resolve("r1", True)
        assert found is True
        assert await fut is True
        assert reg.pending_ids() == {}
        # Resolving an unknown id is a no-op miss.
        assert await reg.resolve("nope", False) is False

    asyncio.run(scenario())


def test_pending_registry_fail_all_denies_waiters():
    async def scenario():
        reg = ca.PendingApprovals()
        f1 = await reg.register("a", "Bash")
        f2 = await reg.register("b", "Edit")
        await reg.fail_all()
        assert await f1 is False
        assert await f2 is False
        assert reg.pending_ids() == {}

    asyncio.run(scenario())


# ===========================================================================
# Session-manager gate — the full can_use_tool loop (mocked SDK)
# ===========================================================================
class _GateFakeClient:
    """Minimal client so start_session succeeds and the pump idles."""

    def __init__(self, option_kwargs: Dict[str, Any]) -> None:
        self.option_kwargs = option_kwargs
        self._closed = asyncio.Event()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        self._closed.set()

    async def receive_messages(self):
        yield SimpleNamespace(subtype="init", data={"session_id": "sdk-gate-1"})
        await self._closed.wait()


class _GateFactory:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, option_kwargs: Dict[str, Any]) -> _GateFakeClient:
        self.calls.append(option_kwargs)
        return _GateFakeClient(option_kwargs)


async def _start(tmp_path, workspace_id, events):
    factory = _GateFactory()

    async def sink(ev):
        events.append(ev)

    mgr = css.CanvasSessionManager(
        str(tmp_path),
        sdk_client_factory=factory,
        init_timeout=2.0,
        event_sink=sink,
    )
    await mgr.start_session(workspace_id)
    gate = factory.calls[0]["can_use_tool"]
    root = (tmp_path / workspace_id).resolve()
    return mgr, gate, root


def test_gate_blocks_edit_until_approved_then_allows(tmp_path):
    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, gate, root = await _start(tmp_path, "ws-appr", events)

        # Fire a file edit through the gate; it must BLOCK (not resolve yet).
        task = asyncio.create_task(
            gate("Edit", {"file_path": "a.py", "old_string": "1", "new_string": "2"}, None)
        )
        await asyncio.sleep(0.05)
        assert not task.done(), "gate must block awaiting the human decision"

        # A permission.request event was emitted carrying the diff + request_id.
        perms = [e for e in events if e["event_type"] == ce.EVENT_PERMISSION_REQUEST]
        assert perms, "expected a permission.request event"
        req_id = perms[-1]["data"]["request_id"]
        assert perms[-1]["data"]["old_content"] == "1"
        assert perms[-1]["data"]["new_content"] == "2"

        # Approve → the callback resolves to Allow (edit applies).
        decided = await mgr.decide("ws-appr", req_id, True)
        assert decided["success"] is True
        result = await asyncio.wait_for(task, timeout=2.0)
        assert isinstance(result, _Allow)

        await mgr.stop_session("ws-appr")

    asyncio.run(scenario())


def test_gate_deny_returns_deny_with_message(tmp_path):
    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, gate, root = await _start(tmp_path, "ws-deny", events)

        task = asyncio.create_task(
            gate("Write", {"file_path": "b.py", "content": "x"}, None)
        )
        await asyncio.sleep(0.05)
        req_id = [
            e for e in events if e["event_type"] == ce.EVENT_PERMISSION_REQUEST
        ][-1]["data"]["request_id"]

        await mgr.decide("ws-deny", req_id, False)
        result = await asyncio.wait_for(task, timeout=2.0)
        assert isinstance(result, _Deny)
        assert result.message  # non-empty → fed back to the model (reverts+informs)

        await mgr.stop_session("ws-deny")

    asyncio.run(scenario())


def test_gate_auto_accept_allows_edit_without_prompt_but_still_gates_bash(tmp_path):
    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, gate, root = await _start(tmp_path, "ws-auto", events)

        toggled = await mgr.set_auto_accept("ws-auto", True)
        assert toggled["auto_accept_edits"] is True

        # Edit is auto-approved: resolves immediately, NO permission event.
        before = len(events)
        result = await asyncio.wait_for(
            gate("Edit", {"file_path": "c.py", "old_string": "a", "new_string": "b"}, None),
            timeout=2.0,
        )
        assert isinstance(result, _Allow)
        assert not [
            e for e in events[before:] if e["event_type"] == ce.EVENT_PERMISSION_REQUEST
        ]

        # Bash STILL requires an explicit decision even with auto-accept on.
        task = asyncio.create_task(gate("Bash", {"command": "rm -rf build"}, None))
        await asyncio.sleep(0.05)
        assert not task.done(), "bash must never be auto-accepted"
        req_id = [
            e for e in events if e["event_type"] == ce.EVENT_PERMISSION_REQUEST
        ][-1]["data"]["request_id"]
        await mgr.decide("ws-auto", req_id, True)
        assert isinstance(await asyncio.wait_for(task, timeout=2.0), _Allow)

        await mgr.stop_session("ws-auto")

    asyncio.run(scenario())


def test_gate_confinement_denies_escape_before_any_approval(tmp_path):
    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, gate, root = await _start(tmp_path, "ws-esc", events)

        # An out-of-mount write is HARD-denied by confinement — no human in the
        # loop, no permission event, immediate Deny.
        result = await asyncio.wait_for(
            gate("Write", {"file_path": "../../etc/evil", "content": "x"}, None),
            timeout=2.0,
        )
        assert isinstance(result, _Deny)
        assert not [
            e for e in events if e["event_type"] == ce.EVENT_PERMISSION_REQUEST
        ]

        await mgr.stop_session("ws-esc")

    asyncio.run(scenario())


def test_gate_read_only_tool_allowed_without_prompt(tmp_path):
    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, gate, root = await _start(tmp_path, "ws-ro", events)

        result = await asyncio.wait_for(
            gate("Grep", {"pattern": "TODO", "path": "."}, None), timeout=2.0
        )
        assert isinstance(result, _Allow)
        # Grep re-binds "." to the absolute root → Allow(updated_input=...).
        assert result.updated_input is not None
        assert not [
            e for e in events if e["event_type"] == ce.EVENT_PERMISSION_REQUEST
        ]

        await mgr.stop_session("ws-ro")

    asyncio.run(scenario())


def test_stop_fails_pending_approvals(tmp_path):
    async def scenario():
        events: List[Dict[str, Any]] = []
        mgr, gate, root = await _start(tmp_path, "ws-stop", events)

        task = asyncio.create_task(
            gate("Write", {"file_path": "d.py", "content": "x"}, None)
        )
        await asyncio.sleep(0.05)
        assert not task.done()

        # Stopping the session must release the awaiting callback as a deny.
        await mgr.stop_session("ws-stop")
        result = await asyncio.wait_for(task, timeout=2.0)
        assert isinstance(result, _Deny)

    asyncio.run(scenario())
