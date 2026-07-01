"""PRD-170 S1 — canvas SDK session manager contract tests (container-free).

Targets the worker-side service in ``services/workspace-worker``:
``canvas_session_service.CanvasSessionManager`` + ``canvas_confinement``.

Proven here with the Claude Agent SDK client MOCKED (injected factory):
  * lifecycle state machine: start -> running -> status -> stop -> stopped;
  * resume contract: a brand-new manager (process restart) reads
    ``sdk_session_id`` from the volume state file and passes it as the SDK
    ``resume`` option;
  * one active session per workspace (second start -> conflict);
  * path-escape attempts rejected (tenancy — security, NOT deferrable):
    traversal, absolute escape, symlink escape, null byte, cross-workspace
    bash references; in-root paths are re-bound to absolute in-root paths.

The LIVE container lifecycle/resume runs on CI-with-Docker / the morning
human — marked DEFERRED in scripts/ralph/prd-170.json, not faked here.

Pure stdlib + pytest: no DB, no docker, no claude_agent_sdk import (the
service lazily imports the SDK only inside the default factory/callback,
which these tests never invoke).
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

# The worker is a flat-module service (hyphenated dir, not a package) — put
# its directory on sys.path just long enough to import the modules under
# test, then remove the entry so nothing else resolves against it.
_WORKER_DIR = (
    Path(__file__).resolve().parents[2] / "services" / "workspace-worker"
)
sys.path.insert(0, str(_WORKER_DIR))
try:
    import canvas_confinement as cc
    import canvas_session_service as css
    from workspace_manager import SecurityError
finally:
    sys.path.remove(str(_WORKER_DIR))


# ---------------------------------------------------------------------------
# Fakes — stand-in for ClaudeSDKClient (the worker-image-only dependency)
# ---------------------------------------------------------------------------


class FakeSDKClient:
    """Mimics the ClaudeSDKClient surface the manager uses:
    connect / disconnect / receive_messages (init SystemMessage first)."""

    def __init__(self, option_kwargs: Dict[str, Any], sdk_session_id: str) -> None:
        self.option_kwargs = option_kwargs
        self.sdk_session_id = sdk_session_id
        self.connected = False
        self.disconnected = False
        self._closed = asyncio.Event()

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True
        self._closed.set()

    async def receive_messages(self):
        yield SimpleNamespace(
            subtype="init", data={"session_id": self.sdk_session_id}
        )
        await self._closed.wait()


class FactorySpy:
    """Injected sdk_client_factory: records option kwargs per start."""

    def __init__(self, sdk_session_id: str = "sdk-test-1") -> None:
        self.sdk_session_id = sdk_session_id
        self.calls: List[Dict[str, Any]] = []
        self.clients: List[FakeSDKClient] = []

    def __call__(self, option_kwargs: Dict[str, Any]) -> FakeSDKClient:
        self.calls.append(option_kwargs)
        client = FakeSDKClient(option_kwargs, self.sdk_session_id)
        self.clients.append(client)
        return client


def _manager(
    volume: Path, factory: Optional[FactorySpy] = None
) -> "css.CanvasSessionManager":
    return css.CanvasSessionManager(
        str(volume),
        sdk_client_factory=factory or FactorySpy(),
        init_timeout=2.0,
    )


def _state_file(volume: Path, workspace_id: str) -> Path:
    return volume / workspace_id / ".canvas" / "session.json"


# ---------------------------------------------------------------------------
# AC1 proxy — lifecycle state machine (mocked SDK client)
# ---------------------------------------------------------------------------


def test_lifecycle_start_status_stop(tmp_path):
    async def scenario():
        factory = FactorySpy(sdk_session_id="sdk-life-1")
        mgr = _manager(tmp_path, factory)

        started = await mgr.start_session("ws-life")
        assert started["success"] is True
        assert started["resumed"] is False
        assert started["session"]["status"] == css.STATUS_RUNNING
        assert factory.clients[0].connected is True

        # State (incl. the captured SDK session id) persisted on the volume.
        on_disk = json.loads(_state_file(tmp_path, "ws-life").read_text())
        assert on_disk["status"] == css.STATUS_RUNNING
        assert on_disk["sdk_session_id"] == "sdk-life-1"
        assert on_disk["workspace_id"] == "ws-life"

        status = await mgr.get_status("ws-life")
        assert status["success"] is True
        assert status["live"] is True
        assert status["session"]["status"] == css.STATUS_RUNNING

        stopped = await mgr.stop_session("ws-life")
        assert stopped["success"] is True
        assert stopped["session"]["status"] == css.STATUS_STOPPED
        assert factory.clients[0].disconnected is True

        on_disk = json.loads(_state_file(tmp_path, "ws-life").read_text())
        assert on_disk["status"] == css.STATUS_STOPPED

        after = await mgr.get_status("ws-life")
        assert after["success"] is True
        assert after["live"] is False
        assert after["session"]["status"] == css.STATUS_STOPPED

    asyncio.run(scenario())


def test_status_and_stop_unknown_workspace_not_found(tmp_path):
    async def scenario():
        mgr = _manager(tmp_path)
        status = await mgr.get_status("ws-none")
        assert status["success"] is False and status["not_found"] is True
        stop = await mgr.stop_session("ws-none")
        assert stop["success"] is False and stop["not_found"] is True

    asyncio.run(scenario())


def test_sdk_options_confine_session_to_workspace_mount(tmp_path):
    """The session is wired to its mount: cwd = workspace root, transcript
    dir (CLAUDE_CONFIG_DIR) on the volume, default permission mode, and the
    can_use_tool confinement callback installed."""

    async def scenario():
        factory = FactorySpy()
        mgr = _manager(tmp_path, factory)
        await mgr.start_session("ws-opts")

        opts = factory.calls[0]
        root = (tmp_path / "ws-opts").resolve()
        assert Path(opts["cwd"]) == root
        config_dir = Path(opts["env"]["CLAUDE_CONFIG_DIR"])
        config_dir.relative_to(root)  # raises ValueError if off-volume
        assert opts["permission_mode"] == "default"
        assert callable(opts["can_use_tool"])
        assert "resume" not in opts  # fresh start — nothing to resume

        await mgr.stop_session("ws-opts")

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# AC2 — resume reads state from the volume (mocked)
# ---------------------------------------------------------------------------


def test_resume_reads_state_from_volume_after_restart(tmp_path):
    async def scenario():
        first = FactorySpy(sdk_session_id="sdk-resume-1")
        m1 = _manager(tmp_path, first)
        started = await m1.start_session("ws-resume")
        canvas_id = started["session"]["canvas_session_id"]
        await m1.stop_session("ws-resume")

        # "Restart": a brand-new manager with empty in-memory state —
        # everything it knows must come from the volume.
        second = FactorySpy(sdk_session_id="sdk-resume-1")
        m2 = _manager(tmp_path, second)
        resumed = await m2.start_session("ws-resume")

        assert resumed["success"] is True
        assert resumed["resumed"] is True
        assert second.calls[0]["resume"] == "sdk-resume-1"
        # Canvas session identity is stable across resume.
        assert resumed["session"]["canvas_session_id"] == canvas_id

        await m2.stop_session("ws-resume")

    asyncio.run(scenario())


def test_orphaned_running_state_resumes_instead_of_conflicting(tmp_path):
    """A crash leaves status=running on the volume with no live process.
    A new manager must treat start as resume, not as a conflict."""

    async def scenario():
        first = FactorySpy(sdk_session_id="sdk-orphan-1")
        m1 = _manager(tmp_path, first)
        await m1.start_session("ws-orphan")
        # No stop: simulate worker death — state file still says running.
        on_disk = json.loads(_state_file(tmp_path, "ws-orphan").read_text())
        assert on_disk["status"] == css.STATUS_RUNNING

        second = FactorySpy(sdk_session_id="sdk-orphan-1")
        m2 = _manager(tmp_path, second)
        resumed = await m2.start_session("ws-orphan")
        assert resumed["success"] is True
        assert resumed["resumed"] is True
        assert second.calls[0]["resume"] == "sdk-orphan-1"

        await m2.stop_session("ws-orphan")
        await m1.stop_session("ws-orphan")  # cleanup fake pump of m1

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# AC4 — one active session per workspace
# ---------------------------------------------------------------------------


def test_one_active_session_per_workspace(tmp_path):
    async def scenario():
        factory = FactorySpy()
        mgr = _manager(tmp_path, factory)

        first = await mgr.start_session("ws-one")
        assert first["success"] is True

        dup = await mgr.start_session("ws-one")
        assert dup["success"] is False
        assert dup["conflict"] is True
        assert "already active" in dup["error"]

        # A different workspace is unaffected.
        other = await mgr.start_session("ws-two")
        assert other["success"] is True

        # After stop, the workspace can start again (as a resume).
        await mgr.stop_session("ws-one")
        again = await mgr.start_session("ws-one")
        assert again["success"] is True
        assert again["resumed"] is True

        await mgr.stop_session("ws-one")
        await mgr.stop_session("ws-two")

    asyncio.run(scenario())


def test_failed_connect_reports_failure_and_releases_slot(tmp_path):
    async def scenario():
        def broken_factory(option_kwargs):
            raise RuntimeError("claude CLI not installed")

        mgr = css.CanvasSessionManager(
            str(tmp_path), sdk_client_factory=broken_factory, init_timeout=2.0
        )
        result = await mgr.start_session("ws-broken")
        assert result["success"] is False
        assert "claude CLI not installed" in result["error"]

        on_disk = json.loads(_state_file(tmp_path, "ws-broken").read_text())
        assert on_disk["status"] == css.STATUS_FAILED

        # The failure must not hold the one-session slot.
        ok = await _manager(tmp_path).start_session("ws-broken")
        assert ok["success"] is True

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# AC3 — path-escape attempts rejected (security: unit-proven, NOT deferrable)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "supplied",
    [
        "../../etc/passwd",
        "..",
        "repos/../../other-workspace/secrets",
        "/etc/passwd",
        "/root/.ssh/id_rsa",
        "repos/a\x00b.txt",
    ],
)
def test_confine_path_rejects_escapes(tmp_path, supplied):
    root = tmp_path / "vol" / "ws-sec"
    root.mkdir(parents=True)
    with pytest.raises(SecurityError):
        cc.confine_path(root, supplied)


def test_confine_path_rejects_sibling_workspace_absolute(tmp_path):
    volume = tmp_path / "vol"
    root = volume / "ws-a"
    other = volume / "ws-b"
    root.mkdir(parents=True)
    other.mkdir(parents=True)
    with pytest.raises(SecurityError):
        cc.confine_path(root, str(other / "repos" / "creds.json"))


def test_confine_path_rejects_symlink_escape(tmp_path):
    root = tmp_path / "vol" / "ws-sym"
    (root / "repos").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "repos" / "link").symlink_to(outside)
    with pytest.raises(SecurityError):
        cc.confine_path(root, "repos/link/secrets.txt")


def test_confine_path_rebinds_inside_paths(tmp_path):
    root = tmp_path / "vol" / "ws-ok"
    root.mkdir(parents=True)
    base = root.resolve()
    # Relative paths re-bound under the root.
    assert cc.confine_path(root, "repos/app/main.py") == base / "repos/app/main.py"
    assert cc.confine_path(root, ".") == base
    # Absolute paths INSIDE the root are legitimate (SDK cwd is the root).
    assert cc.confine_path(root, str(base / "reports" / "x.md")) == (
        base / "reports" / "x.md"
    )


def test_tool_confinement_denies_file_escape_and_rebinds_inside(tmp_path):
    root = (tmp_path / "vol" / "ws-tools").resolve()
    root.mkdir(parents=True)

    denied = cc.evaluate_tool_confinement(
        "Write", {"file_path": "../../tmp/evil.sh", "content": "x"}, root
    )
    assert denied.allowed is False
    assert "Write.file_path" in (denied.reason or "")

    denied_abs = cc.evaluate_tool_confinement(
        "Read", {"file_path": "/etc/passwd"}, root
    )
    assert denied_abs.allowed is False

    allowed = cc.evaluate_tool_confinement(
        "Edit",
        {"file_path": "repos/app/a.py", "old_string": "x", "new_string": "y"},
        root,
    )
    assert allowed.allowed is True
    assert allowed.updated_input is not None
    assert allowed.updated_input["file_path"] == str(root / "repos/app/a.py")
    # Non-path fields pass through untouched.
    assert allowed.updated_input["old_string"] == "x"

    rebound_search = cc.evaluate_tool_confinement("Grep", {"path": ".", "pattern": "x"}, root)
    assert rebound_search.allowed is True
    assert rebound_search.updated_input["path"] == str(root)


def test_tool_confinement_bash_cross_tenant_and_traversal_denied(tmp_path):
    volume = tmp_path / "vol"
    root = (volume / "ws-bash").resolve()
    root.mkdir(parents=True)
    other = volume / "ws-victim"
    other.mkdir(parents=True)

    cross = cc.evaluate_tool_confinement(
        "Bash", {"command": f"cat {other}/repos/.env"}, root
    )
    assert cross.allowed is False

    listing = cc.evaluate_tool_confinement(
        "Bash", {"command": f"ls {volume}/"}, root
    )
    assert listing.allowed is False

    traversal = cc.evaluate_tool_confinement(
        "Bash", {"command": "cd .. && ls"}, root
    )
    assert traversal.allowed is False

    bad_cwd = cc.evaluate_tool_confinement(
        "Bash", {"command": "ls", "cwd": "../ws-victim"}, root
    )
    assert bad_cwd.allowed is False


def test_tool_confinement_bash_legitimate_commands_allowed(tmp_path):
    volume = tmp_path / "vol"
    root = (volume / "ws-bash-ok").resolve()
    root.mkdir(parents=True)

    for command in (
        "python3 -m pytest repos/app/tests -q",
        "git -C repos/app status",
        f"cat {root}/repos/app/README.md",  # own root absolute is fine
        "echo 1..10",  # '..' not a path component
    ):
        verdict = cc.evaluate_tool_confinement("Bash", {"command": command}, root)
        assert verdict.allowed is True, (command, verdict.reason)


def test_manager_session_state_excludes_secrets(tmp_path):
    """The persisted/reported session payload carries no token material."""

    async def scenario():
        mgr = _manager(tmp_path)
        started = await mgr.start_session("ws-clean")
        payload = json.dumps(started)
        for marker in ("ANTHROPIC_API_KEY", "sk-ant-", "Bearer "):
            assert marker not in payload
        await mgr.stop_session("ws-clean")

    asyncio.run(scenario())
