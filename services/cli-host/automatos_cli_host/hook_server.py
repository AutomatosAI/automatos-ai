"""The host's hook socket — one loopback Unix socket, one line in, one line out.

Each connection carries one hook payload (from ``hook_shim``). The server looks
up the session by ``automatos_task_id`` (set in the session's environment) and
hands the payload to that session's handler, which returns the JSON answer
Claude Code should read — a permission decision, ``additionalContext``, or
``{}``. A payload for an unknown session is answered ``{}`` (or a deny on the
gated events) and logged; nothing ever raises into the CLI.

Threaded, blocking sockets: hook calls are rare (tens per turn) and a
``PreToolUse`` hold must be able to block for minutes without starving the
others.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Callable, Dict, Optional

log = logging.getLogger("automatos.cli_host.hooks")

Handler = Callable[[dict], dict]

_GATED_EVENTS = ("PreToolUse", "PermissionRequest")


def deny_answer(event: str, reason: str) -> dict:
    if event == "PermissionRequest":
        return {"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                       "decision": {"behavior": "deny", "message": reason}}}
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                   "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


class HookServer:
    def __init__(self, sock_path: Path):
        self.sock_path = Path(sock_path)
        self._handlers: Dict[str, Handler] = {}
        self._lock = threading.Lock()
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._inode: Optional[int] = None  # the socket file WE created (see stop)

    # ── registry ────────────────────────────────────────────────────────────
    def register(self, task_id: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[str(task_id)] = handler

    def unregister(self, task_id: str) -> None:
        with self._lock:
            self._handlers.pop(str(task_id), None)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        self.sock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.sock_path.unlink()
        except FileNotFoundError:
            pass
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.sock_path))
        os.chmod(self.sock_path, 0o600)
        srv.listen(64)
        srv.settimeout(0.5)
        self._server = srv
        self._inode = os.stat(self.sock_path).st_ino
        self._thread = threading.Thread(target=self._serve, name="automatos-hook-server", daemon=True)
        self._thread.start()

    def owns_socket_file(self) -> bool:
        """Is the file at ``sock_path`` the one THIS server bound?"""
        try:
            return self._inode is not None and os.stat(self.sock_path).st_ino == self._inode
        except FileNotFoundError:
            return False

    def ensure_listening(self) -> bool:
        """Self-heal: if the socket file vanished or was replaced under us (a
        previous host's shutdown unlinking the path after we bound it — seen
        2026-09-03, ticket 69: every hook answered 'host unreachable'), bind a
        fresh socket. Returns True when a re-bind happened."""
        if self._stopping.is_set() or self._server is None or self.owns_socket_file():
            return False
        try:
            self._server.close()
        except OSError:
            pass
        self._stopping.clear()
        self.start()
        return True

    def stop(self) -> None:
        self._stopping.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        # Only remove the file if it is still ours: a newer host may already have
        # bound the same path, and unlinking it would silently take its hooks away.
        if self.owns_socket_file():
            try:
                self.sock_path.unlink()
            except FileNotFoundError:
                pass

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stopping.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    # ── one hook call ───────────────────────────────────────────────────────
    def _handle(self, conn: socket.socket) -> None:
        answer: dict = {}
        try:
            conn.settimeout(10.0)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            payload = json.loads(buf.decode("utf-8", "replace") or "{}")
            if not isinstance(payload, dict):
                payload = {}
            event = payload.get("hook_event_name") or ""
            task_id = str(payload.get("automatos_task_id") or "")
            with self._lock:
                handler = self._handlers.get(task_id)
            if handler is None:
                log.warning("hook %s for unknown session task=%s", event, task_id or "?")
                answer = deny_answer(event, "no Automatos session owns this process") if event in _GATED_EVENTS else {}
            else:
                conn.settimeout(None)  # a PreToolUse hold may take minutes
                answer = handler(payload) or {}
        except Exception:  # noqa: BLE001 — never let a hook call crash the host
            log.exception("hook handling failed")
            answer = {}
        try:
            conn.sendall((json.dumps(answer) + "\n").encode("utf-8"))
        except OSError:
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass
