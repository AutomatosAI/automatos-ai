"""
Canvas SDK Session Service
==========================
PRD-170 S1: manages ONE headless Claude Agent SDK session per workspace
inside this worker container.

Lifecycle: start (fresh, or resume from volume state) -> running ->
stopped | failed. State and transcript persist on the workspace volume so
sessions survive orchestrator AND worker restarts:

    /workspaces/{workspace_id}/.canvas/session.json   <- session state (ours)
    /workspaces/{workspace_id}/.canvas/claude/        <- CLAUDE_CONFIG_DIR
                                                         (CLI transcript jsonl)

Resume: a later ``start_session`` reads ``sdk_session_id`` from the volume
state file and passes it as the SDK ``resume`` option, so the conversation
continues with its prior transcript.

Tenancy: the session is confined to its workspace mount. The SDK
``can_use_tool`` permission callback routes every tool call through
``canvas_confinement.evaluate_tool_confinement`` — agent-supplied paths
are re-bound/validated against the workspace root server-side; escapes
are denied (see canvas_confinement.py).

The ``claude_agent_sdk`` import is lazy (inside the default factory and
the permission callback) so this module stays importable — and the state
machine unit-testable — without the SDK installed. Tests inject a fake
``sdk_client_factory``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from canvas_confinement import evaluate_tool_confinement
from workspace_manager import WorkspaceManager

logger = logging.getLogger("workspace-worker.canvas")

STATE_DIR_NAME = ".canvas"
STATE_FILE_NAME = "session.json"
CLAUDE_CONFIG_DIR_NAME = "claude"

STATUS_STARTING = "starting"
STATUS_RUNNING = "running"
STATUS_STOPPED = "stopped"
STATUS_FAILED = "failed"

_ACTIVE_STATUSES = (STATUS_STARTING, STATUS_RUNNING)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CanvasSessionState:
    """Persisted session state — JSON on the workspace volume. No secrets."""

    workspace_id: str
    canvas_session_id: str
    sdk_session_id: Optional[str] = None
    status: str = STATUS_STARTING
    created_at: str = ""
    updated_at: str = ""
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class _LiveSession:
    """In-memory handle for a running session (client + message pump)."""

    client: Any
    state: CanvasSessionState
    pump_task: Optional[asyncio.Task] = None
    init_seen: asyncio.Event = field(default_factory=asyncio.Event)
    stopping: bool = False


def _extract_init_session_id(message: Any) -> Optional[str]:
    """Pull the SDK session id from an init SystemMessage (duck-typed)."""
    if getattr(message, "subtype", None) != "init":
        return None
    data = getattr(message, "data", None)
    if isinstance(data, dict):
        sid = data.get("session_id")
        if isinstance(sid, str) and sid:
            return sid
    return None


def _default_sdk_client_factory(option_kwargs: Dict[str, Any]) -> Any:
    """Build a real ClaudeSDKClient (lazy import — the SDK is a worker-image
    dependency; tests and import-time code never need it)."""
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    return ClaudeSDKClient(options=ClaudeAgentOptions(**option_kwargs))


def _make_confinement_callback(root: Path) -> Callable[..., Awaitable[Any]]:
    """SDK ``can_use_tool`` callback enforcing workspace-mount confinement."""

    async def can_use_tool(
        tool_name: str, tool_input: Dict[str, Any], context: Any
    ) -> Any:
        from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

        verdict = evaluate_tool_confinement(tool_name, tool_input, root)
        if not verdict.allowed:
            logger.warning(
                "Canvas confinement denied tool %s: %s", tool_name, verdict.reason
            )
            return PermissionResultDeny(
                message=verdict.reason or "Denied: path outside the workspace mount"
            )
        if verdict.updated_input is not None:
            return PermissionResultAllow(updated_input=verdict.updated_input)
        return PermissionResultAllow()

    return can_use_tool


class CanvasSessionManager:
    """start/resume/status/stop for headless SDK sessions, one per workspace."""

    def __init__(
        self,
        volume_path: str,
        sdk_client_factory: Optional[Callable[[Dict[str, Any]], Any]] = None,
        init_timeout: float = 10.0,
    ) -> None:
        self.volume_path = volume_path
        self._factory = sdk_client_factory or _default_sdk_client_factory
        self._init_timeout = init_timeout
        self._live: Dict[str, _LiveSession] = {}

    # ── Lifecycle ────────────────────────────────────────────────────

    async def start_session(self, workspace_id: str) -> Dict[str, Any]:
        """Start (or resume from volume state) the workspace's session.

        One active session per workspace: a second start while one is live
        returns ``{"success": False, "conflict": True}``.
        """
        existing = self._live.get(workspace_id)
        if existing is not None and existing.state.status in _ACTIVE_STATUSES:
            return {
                "success": False,
                "conflict": True,
                "error": (
                    f"A canvas session is already active for workspace "
                    f"{workspace_id} (one active session per workspace)."
                ),
            }

        ws_manager = WorkspaceManager(workspace_id, self.volume_path)
        ws_manager.ensure_workspace_exists()
        root = ws_manager.root.resolve()

        prior = self._load_state(root)
        resume_id = prior.sdk_session_id if prior else None

        state = CanvasSessionState(
            workspace_id=workspace_id,
            canvas_session_id=(
                prior.canvas_session_id if prior else f"canvas_{uuid.uuid4().hex[:12]}"
            ),
            sdk_session_id=resume_id,
            status=STATUS_STARTING,
            created_at=(prior.created_at if prior and prior.created_at else _utcnow()),
            updated_at=_utcnow(),
        )
        self._persist(root, state)

        option_kwargs: Dict[str, Any] = {
            "cwd": str(root),
            # Transcript + CLI state on the persistent volume -> survives restarts.
            "env": {
                "CLAUDE_CONFIG_DIR": str(root / STATE_DIR_NAME / CLAUDE_CONFIG_DIR_NAME)
            },
            "permission_mode": "default",
            "can_use_tool": _make_confinement_callback(root),
        }
        if resume_id:
            option_kwargs["resume"] = resume_id

        try:
            client = self._factory(option_kwargs)
            await client.connect()
        except Exception as exc:  # noqa: BLE001 — surfaced to the caller
            state.status = STATUS_FAILED
            state.last_error = str(exc)
            state.updated_at = _utcnow()
            self._persist(root, state)
            logger.error(
                "Canvas session start failed for %s: %s", workspace_id[:8], exc
            )
            return {"success": False, "error": f"Failed to start canvas session: {exc}"}

        state.status = STATUS_RUNNING
        state.updated_at = _utcnow()
        self._persist(root, state)

        live = _LiveSession(client=client, state=state)
        self._live[workspace_id] = live
        live.pump_task = asyncio.create_task(
            self._pump(workspace_id, root, live),
            name=f"canvas-pump-{workspace_id[:8]}",
        )

        # Give the SDK init message a bounded chance to land so the
        # sdk_session_id is persisted (and resumable) before we return.
        try:
            await asyncio.wait_for(live.init_seen.wait(), timeout=self._init_timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Canvas init message not seen within %.1fs for %s",
                self._init_timeout,
                workspace_id[:8],
            )

        return {
            "success": True,
            "resumed": bool(resume_id),
            "session": live.state.to_dict(),
        }

    async def get_status(self, workspace_id: str) -> Dict[str, Any]:
        """Report the live session, or the persisted volume state if any."""
        live = self._live.get(workspace_id)
        if live is not None:
            return {"success": True, "live": True, "session": live.state.to_dict()}

        root = (Path(self.volume_path) / workspace_id).resolve()
        state = self._load_state(root)
        if state is None:
            return {
                "success": False,
                "not_found": True,
                "error": f"No canvas session for workspace {workspace_id}",
            }
        return {"success": True, "live": False, "session": state.to_dict()}

    async def stop_session(self, workspace_id: str) -> Dict[str, Any]:
        """Stop the live session; or mark orphaned volume state stopped."""
        live = self._live.pop(workspace_id, None)
        root = (Path(self.volume_path) / workspace_id).resolve()

        if live is None:
            state = self._load_state(root)
            if state is None:
                return {
                    "success": False,
                    "not_found": True,
                    "error": f"No canvas session for workspace {workspace_id}",
                }
            if state.status in _ACTIVE_STATUSES:
                state.status = STATUS_STOPPED
                state.updated_at = _utcnow()
                self._persist(root, state)
            return {"success": True, "live": False, "session": state.to_dict()}

        live.stopping = True
        try:
            await live.client.disconnect()
        except Exception as exc:  # noqa: BLE001 — stop must not fail on teardown
            logger.warning(
                "Canvas disconnect error for %s: %s", workspace_id[:8], exc
            )

        if live.pump_task is not None:
            try:
                await asyncio.wait_for(live.pump_task, timeout=5.0)
            except asyncio.TimeoutError:
                live.pump_task.cancel()
            except Exception:  # noqa: BLE001 — pump errors already logged there
                pass

        live.state.status = STATUS_STOPPED
        live.state.updated_at = _utcnow()
        self._persist(root, live.state)
        return {"success": True, "live": True, "session": live.state.to_dict()}

    # ── Message pump ─────────────────────────────────────────────────

    async def _pump(
        self, workspace_id: str, root: Path, live: _LiveSession
    ) -> None:
        """Consume SDK messages: capture+persist the session id; track exit.

        (S3 extends this to bridge events to the platform SSE channel.)
        """
        try:
            async for message in live.client.receive_messages():
                sdk_id = _extract_init_session_id(message)
                if sdk_id:
                    if sdk_id != live.state.sdk_session_id:
                        live.state.sdk_session_id = sdk_id
                        live.state.updated_at = _utcnow()
                        self._persist(root, live.state)
                    live.init_seen.set()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 — session death is a state, not a crash
            if not live.stopping:
                live.state.status = STATUS_FAILED
                live.state.last_error = str(exc)
                live.state.updated_at = _utcnow()
                self._persist(root, live.state)
                logger.error(
                    "Canvas session pump failed for %s: %s", workspace_id[:8], exc
                )
        else:
            if not live.stopping and live.state.status == STATUS_RUNNING:
                live.state.status = STATUS_STOPPED
                live.state.updated_at = _utcnow()
                self._persist(root, live.state)
                logger.info(
                    "Canvas session process ended for %s", workspace_id[:8]
                )
        finally:
            live.init_seen.set()  # never leave start_session waiting
            if self._live.get(workspace_id) is live:
                self._live.pop(workspace_id, None)

    # ── Volume state ─────────────────────────────────────────────────

    def _state_path(self, root: Path) -> Path:
        return root / STATE_DIR_NAME / STATE_FILE_NAME

    def _persist(self, root: Path, state: CanvasSessionState) -> None:
        """Atomically write session state to the workspace volume."""
        path = self._state_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state.to_dict(), indent=2))
        os.replace(tmp, path)

    def _load_state(self, root: Path) -> Optional[CanvasSessionState]:
        """Read session state from the volume; corrupt/absent -> None."""
        path = self._state_path(root)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text())
            known = {f.name for f in dataclasses.fields(CanvasSessionState)}
            return CanvasSessionState(
                **{k: v for k, v in data.items() if k in known}
            )
        except (OSError, ValueError, TypeError) as exc:
            logger.warning(
                "Unreadable canvas session state at %s: %s", path, exc
            )
            return None


# ── Process-level singleton (worker HTTP handlers) ───────────────────

_manager: Optional[CanvasSessionManager] = None


def get_canvas_manager(volume_path: str) -> CanvasSessionManager:
    """Singleton manager for the worker process (mirrors the orchestrator's
    ``core.workspace_client._get_client`` idiom)."""
    global _manager
    if _manager is None:
        _manager = CanvasSessionManager(volume_path)
    return _manager
