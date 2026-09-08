"""PRD-239 S7 — a real terminal in the Canvas, served by the host on the loopback.

The Canvas shows the operator's own shell, on their own machine, in a directory
the host may run sessions in. They type ``claude`` (or ``codex``, or anything
installed) themselves — a human using Claude Code interactively is the surface
the subscription covers; the host still never types into a session.

Shape (standard library only):

* one TCP listener on ``127.0.0.1`` (never another interface), an ephemeral port
  unless ``--terminal-port`` is given; the port is announced in the host's
  capabilities so the backend can hand the browser a URL;
* every connection is a WebSocket (RFC 6455, minimal server side) carrying ONE
  grant token in the query string. Grants are minted by the backend for the
  operator, delivered to the host on its next heartbeat, single-use, and expire
  in minutes; the ``Origin`` must be a loopback page;
* the directory comes from the grant (a ticket's real cwd, or an agent's
  working directory) and is resolved against the allow-list exactly like a
  session's; otherwise the ticket's default session folder, else the first root;
* the shell is the user's login shell under a pseudo-terminal with the session
  env hygiene (``build_session_env``); binary frames carry bytes both ways, text
  frames carry ``{"type": "resize", "cols", "rows"}``; closing the socket ends
  the shell's process group.
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import logging
import os
import pty
import re
import signal
import socket
import struct
import subprocess
import termios
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .allowlist import NotAllowed, default_session_cwd, resolve_allowed
from .claude_settings import record_directory_trust
from .env import build_session_env, resolve_binary
from .session import assert_args_honour_invariant
from .transcript import transcript_path

log = logging.getLogger("automatos.cli_host.terminal")

WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
MAX_TERMINALS = 4
GRANT_TTL_SECONDS = 120.0
IDLE_TIMEOUT_SECONDS = 30 * 60
HANDSHAKE_TIMEOUT_SECONDS = 5.0
ALLOWED_ORIGIN_HOSTS = ("localhost", "127.0.0.1", "::1")
DEFAULT_COLS, DEFAULT_ROWS = 120, 32
OPCODE_TEXT, OPCODE_BINARY, OPCODE_CLOSE, OPCODE_PING, OPCODE_PONG = 1, 2, 8, 9, 10
_MAX_REQUEST_BYTES = 8192


# ── RFC 6455 pieces (pure; unit-tested) ──────────────────────────────────────

def accept_key(sec_websocket_key: str) -> str:
    digest = hashlib.sha1((sec_websocket_key.strip() + WS_GUID).encode("ascii")).digest()
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, opcode: int = OPCODE_BINARY) -> bytes:
    """A server→client frame (never masked)."""
    header = bytearray([0x80 | (opcode & 0x0F)])
    n = len(payload)
    if n < 126:
        header.append(n)
    elif n < 65536:
        header.append(126)
        header += struct.pack("!H", n)
    else:
        header.append(127)
        header += struct.pack("!Q", n)
    return bytes(header) + payload


class FrameReader:
    """Incremental parser of client→server frames (masked, per the RFC)."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def feed(self, data: bytes) -> List[Tuple[int, bytes]]:
        self.buf += data
        frames: List[Tuple[int, bytes]] = []
        while True:
            frame = self._pop()
            if frame is None:
                return frames
            frames.append(frame)

    def _pop(self) -> Optional[Tuple[int, bytes]]:
        b = self.buf
        if len(b) < 2:
            return None
        opcode = b[0] & 0x0F
        masked = bool(b[1] & 0x80)
        n = b[1] & 0x7F
        i = 2
        if n == 126:
            if len(b) < 4:
                return None
            n = struct.unpack("!H", bytes(b[2:4]))[0]
            i = 4
        elif n == 127:
            if len(b) < 10:
                return None
            n = struct.unpack("!Q", bytes(b[2:10]))[0]
            i = 10
        mask = b""
        if masked:
            if len(b) < i + 4:
                return None
            mask = bytes(b[i:i + 4])
            i += 4
        if len(b) < i + n:
            return None
        payload = bytes(b[i:i + n])
        if masked:
            payload = bytes(c ^ mask[k % 4] for k, c in enumerate(payload))
        del b[: i + n]
        return opcode, payload


def parse_handshake(raw: bytes) -> Tuple[str, str, Dict[str, str]]:
    """``(method, target, headers)`` of an HTTP request head; header names lower-cased."""
    head = raw.split(b"\r\n\r\n", 1)[0].decode("latin-1")
    lines = head.split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        raise ValueError("malformed request line")
    headers: Dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    return parts[0].upper(), parts[1], headers


def origin_allowed(origin: Optional[str]) -> bool:
    """Only a page served on the loopback may open a terminal. A missing Origin
    (a non-browser client) passes — the grant token is the gate that matters."""
    if not origin:
        return True
    try:
        host = urlparse(origin).hostname
    except ValueError:
        return False
    return host in ALLOWED_ORIGIN_HOSTS


def token_of(target: str) -> Optional[str]:
    parsed = urlparse(target)
    if parsed.path != "/terminal":
        return None
    values = parse_qs(parsed.query).get("token") or []
    return values[0] if values and values[0] else None


# ── grants ───────────────────────────────────────────────────────────────────

@dataclass
class Grant:
    token: str
    cwd: Optional[str]
    task_id: Optional[str]
    expires_at: float
    # PRD-239 S7 (Runtime Canvas): what to run in the PTY — ``None`` = the login
    # shell; ``{"kind": "claude", "session_id": ..., ...}`` = the agent's own
    # Claude Code session, started or resumed, with the human at the keyboard.
    launch: Optional[Dict[str, Any]] = None


class GrantStore:
    """Single-use, expiring grants the backend delivered on the heartbeat."""

    def __init__(self, ttl_seconds: float = GRANT_TTL_SECONDS, clock: Callable[[], float] = time.time) -> None:
        self._grants: Dict[str, Grant] = {}
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._clock = clock

    def admit(self, grants: List[Dict[str, Any]]) -> int:
        added = 0
        now = self._clock()
        with self._lock:
            for raw in grants or []:
                if not isinstance(raw, dict) or not raw.get("token"):
                    continue
                expires = raw.get("expires_at")
                try:
                    expires_at = float(expires) if expires is not None else now + self._ttl
                except (TypeError, ValueError):
                    expires_at = now + self._ttl
                self._grants[str(raw["token"])] = Grant(
                    token=str(raw["token"]),
                    cwd=str(raw["cwd"]) if raw.get("cwd") else None,
                    task_id=str(raw["task_id"]) if raw.get("task_id") is not None else None,
                    expires_at=expires_at,
                    launch=dict(raw["launch"]) if isinstance(raw.get("launch"), dict) else None,
                )
                added += 1
            self._sweep(now)
        return added

    def take(self, token: str) -> Optional[Grant]:
        """The grant for ``token`` — once. Expired or unknown → None."""
        with self._lock:
            self._sweep(self._clock())
            return self._grants.pop(token, None)

    def _sweep(self, now: float) -> None:
        for key in [k for k, g in self._grants.items() if g.expires_at <= now]:
            self._grants.pop(key, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._grants)


# ── launching the agent's own Claude Code session (PRD-239 S7 v2) ───────────

_SESSION_ID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")


class LaunchError(RuntimeError):
    """The grant asked for a session this host cannot start."""


def build_terminal_args(
    claude: str,
    *,
    session_id: str,
    resume: bool,
    system_prompt_path: Optional[Path],
    model: Optional[str],
    task_id: Optional[str],
) -> List[str]:
    """The interactive command for a Runtime Canvas terminal.

    Exactly what the operator gets by typing ``claude`` in that folder — their
    settings at every scope, the folder's CLAUDE.md files, its ``.mcp.json``
    servers — plus the agent's soul appended. Nothing that assumed nobody was
    at the keyboard: no ``--permission-mode acceptEdits`` (the human answers
    Claude's own prompts), no hooks, no ``--worktree``, no positional prompt,
    and none of the unattended lane's ``--setting-sources user`` /
    ``--strict-mcp-config`` narrowing. ``--resume`` continues a session whose
    transcript exists in this folder; otherwise ``--session-id`` starts it
    under the id the backend recorded on the ticket, so the next open resumes it.
    """
    args = [claude, "--resume" if resume else "--session-id", session_id]
    if system_prompt_path is not None:
        args += ["--append-system-prompt-file", str(system_prompt_path)]
    if task_id:
        args += ["--name", f"automatos #{task_id}"]
    if model:
        args += ["--model", str(model)]
    return args


def transcript_exists(cwd: Path, session_id: str, home: Optional[Path] = None) -> bool:
    """Whether ``claude --resume <session_id>`` run in ``cwd`` would find its
    conversation. Claude Code keeps transcripts per project directory, so only
    the exact path counts — a transcript elsewhere would make ``--resume``
    answer "No conversation found" and the terminal die."""
    return transcript_path(str(cwd), session_id, home).exists()


# ── the server ───────────────────────────────────────────────────────────────

class TerminalServer:
    def __init__(
        self,
        allow_roots: List[str],
        default_root: Optional[str],
        *,
        port: int = 0,
        workspace_id: Callable[[], str] = lambda: "",
        shell: Optional[str] = None,
        max_terminals: int = MAX_TERMINALS,
        idle_timeout: float = IDLE_TIMEOUT_SECONDS,
        claude: Optional[str] = None,
        sessions_dir: Optional[Path] = None,
        on_event: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
        claude_home: Optional[Path] = None,
    ) -> None:
        self.allow_roots = list(allow_roots)
        self.default_root = default_root
        # PRD-239 S7 v2: the Runtime Canvas launches the agent's Claude Code
        # session in the PTY; ``on_event`` receives TerminalOpened/TerminalClosed
        # for the ticket so the backend can show the session as attached.
        self._claude = claude
        self._sessions_dir = sessions_dir
        self._on_event = on_event
        self._claude_home = claude_home
        self.requested_port = int(port or 0)
        self.port: Optional[int] = None
        self.grants = GrantStore()
        self._workspace_id = workspace_id
        self._shell = shell or os.environ.get("SHELL") or "/bin/sh"
        self._max = max_terminals
        self._idle_timeout = idle_timeout
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = threading.Event()
        self._active = 0
        self._lock = threading.Lock()

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> int:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", self.requested_port))  # loopback only, by construction
        srv.listen(8)
        srv.settimeout(0.5)
        self._server = srv
        self.port = int(srv.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, name="automatos-terminal-server", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stopping.set()
        srv, self._server = self._server, None
        if srv is not None:
            try:
                srv.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def admit(self, grants: List[Dict[str, Any]]) -> int:
        return self.grants.admit(grants)

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    # ── accept loop ─────────────────────────────────────────────────────────
    def _serve(self) -> None:
        while not self._stopping.is_set():
            srv = self._server
            if srv is None:
                return
            try:
                conn, _addr = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True, name="automatos-terminal").start()

    # ── one connection ──────────────────────────────────────────────────────
    def _handle(self, conn: socket.socket) -> None:
        try:
            self._handle_inner(conn)
        except Exception:  # noqa: BLE001 — one bad client must not take the server down
            log.exception("terminal connection failed")
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _refuse(self, conn: socket.socket, status: str, reason: str) -> None:
        body = reason.encode("utf-8")
        conn.sendall(
            f"HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode("ascii") + body
        )

    def resolve_directory(self, grant: Grant) -> Path:
        """The directory for a grant: its cwd inside the allow-list; else the
        ticket's default session folder; else the first root."""
        if grant.cwd:
            return resolve_allowed(grant.cwd, self.allow_roots, default_root=self.default_root)
        if grant.task_id and self.default_root:
            path = default_session_cwd(self.default_root, self._workspace_id() or "", grant.task_id)
            path.mkdir(parents=True, exist_ok=True)
            return path
        if not self.allow_roots:
            raise NotAllowed("no directories registered")
        return Path(self.allow_roots[0])

    def _handle_inner(self, conn: socket.socket) -> None:
        conn.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = conn.recv(4096)
            if not chunk:
                return
            raw += chunk
            if len(raw) > _MAX_REQUEST_BYTES:
                self._refuse(conn, "431 Request Header Fields Too Large", "request too large")
                return
        method, target, headers = parse_handshake(raw)
        if method != "GET" or headers.get("upgrade", "").lower() != "websocket" or not headers.get("sec-websocket-key"):
            self._refuse(conn, "400 Bad Request", "expected a WebSocket upgrade")
            return
        if not origin_allowed(headers.get("origin")):
            self._refuse(conn, "403 Forbidden", "only a page on this machine may open a terminal")
            return
        token = token_of(target)
        grant = self.grants.take(token) if token else None
        if grant is None:
            self._refuse(conn, "403 Forbidden", "unknown or expired terminal grant")
            return
        try:
            cwd = self.resolve_directory(grant)
        except NotAllowed as exc:
            self._refuse(conn, "403 Forbidden", str(exc))
            return
        if not cwd.is_dir():
            self._refuse(conn, "404 Not Found", f"directory does not exist: {cwd}")
            return
        try:
            command, launched = self.command_for(grant, cwd)
        except LaunchError as exc:
            self._refuse(conn, "503 Service Unavailable", str(exc))
            return
        with self._lock:
            if self._active >= self._max:
                self._refuse(conn, "429 Too Many Requests", f"at most {self._max} terminals at a time")
                return
            self._active += 1
        try:
            conn.sendall(
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                f"Sec-WebSocket-Accept: {accept_key(headers['sec-websocket-key'])}\r\n\r\n".encode("ascii")
            )
            self._bridge(conn, cwd, grant, command, launched)
        finally:
            with self._lock:
                self._active -= 1

    def command_for(self, grant: Grant, cwd: Path) -> Tuple[List[str], Optional[Dict[str, Any]]]:
        """The PTY's command: the login shell, or — for a launch grant — the
        agent's Claude Code session. Returns ``(argv, launched)`` where
        ``launched`` describes the session (``session_id``, ``resumed``) or is
        ``None`` for a plain shell."""
        launch = grant.launch
        if not launch:
            return [self._shell, "-l"], None
        if launch.get("kind") != "claude":
            raise LaunchError(f"unknown launch kind: {launch.get('kind')!r}")
        session_id = str(launch.get("session_id") or "")
        if not _SESSION_ID_RE.match(session_id):
            raise LaunchError("the grant carries no valid session id")
        claude = self._claude or resolve_binary("claude")
        if not claude or not (os.path.isfile(claude) and os.access(claude, os.X_OK)):
            raise LaunchError("Claude Code is not installed on this machine (no runnable `claude` found)")
        system_prompt_path: Optional[Path] = None
        soul = launch.get("system_prompt")
        if isinstance(soul, str) and soul.strip() and self._sessions_dir is not None and grant.task_id:
            session_dir = self._sessions_dir / grant.task_id
            session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            system_prompt_path = session_dir / "system_prompt.md"
            system_prompt_path.write_text(soul, encoding="utf-8")
        resumed = transcript_exists(cwd, session_id, self._claude_home)
        args = build_terminal_args(
            claude,
            session_id=session_id,
            resume=resumed,
            system_prompt_path=system_prompt_path,
            model=str(launch["model"]) if launch.get("model") else None,
            task_id=grant.task_id,
        )
        assert_args_honour_invariant(args)
        try:
            record_directory_trust(cwd, self._claude_home)
        except OSError as exc:
            log.warning("could not record trust for %s: %s", cwd, exc)
        return args, {"session_id": session_id, "resumed": resumed, "agent_name": launch.get("agent_name")}

    def _emit(self, grant: Grant, event: str, payload: Dict[str, Any]) -> None:
        if self._on_event is None or not grant.task_id:
            return
        try:
            self._on_event(grant.task_id, event, payload)
        except Exception:  # noqa: BLE001 — never let reporting break the terminal
            log.debug("terminal event %s for ticket %s not delivered", event, grant.task_id, exc_info=True)

    # ── the PTY bridge ──────────────────────────────────────────────────────
    def _bridge(self, conn: socket.socket, cwd: Path, grant: Grant, command: List[str],
                launched: Optional[Dict[str, Any]]) -> None:
        env = build_session_env(extra={"TERM": "xterm-256color", "AUTOMATOS_TERMINAL": "1",
                                       **({"AUTOMATOS_TASK_ID": grant.task_id} if grant.task_id else {})})
        master, slave = pty.openpty()
        try:
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", DEFAULT_ROWS, DEFAULT_COLS, 0, 0))
        except OSError:
            pass

        def _child_setup() -> None:  # runs in the child after setsid()
            try:
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            except OSError:
                pass

        proc = subprocess.Popen(
            command, stdin=slave, stdout=slave, stderr=slave, cwd=str(cwd), env=env,
            start_new_session=True, preexec_fn=_child_setup, close_fds=True,
        )
        os.close(slave)
        if launched:
            log.info("terminal: Claude Code session %s %s in %s (pid %s, ticket %s)", launched["session_id"],
                     "resumed" if launched["resumed"] else "started", cwd, proc.pid, grant.task_id)
            self._emit(grant, "TerminalOpened", {**launched, "cwd": str(cwd), "pid": proc.pid})
        else:
            log.info("terminal opened in %s (pid %s%s)", cwd, proc.pid, f", ticket {grant.task_id}" if grant.task_id else "")
        closed = threading.Event()

        def _pump_output() -> None:
            try:
                while not closed.is_set():
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        break
                    if not chunk:
                        break
                    try:
                        conn.sendall(encode_frame(chunk, OPCODE_BINARY))
                    except OSError:
                        break
            finally:
                closed.set()

        threading.Thread(target=_pump_output, daemon=True, name="automatos-terminal-out").start()
        reader = FrameReader()
        conn.settimeout(1.0)
        last_input = time.time()
        try:
            while not closed.is_set():
                if proc.poll() is not None:
                    break
                if time.time() - last_input > self._idle_timeout:
                    log.info("terminal in %s idle for %ss — closing", cwd, int(self._idle_timeout))
                    break
                try:
                    data = conn.recv(65536)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not data:
                    break
                for opcode, payload in reader.feed(data):
                    if opcode == OPCODE_CLOSE:
                        closed.set()
                        break
                    if opcode == OPCODE_PING:
                        conn.sendall(encode_frame(payload, OPCODE_PONG))
                        continue
                    if opcode == OPCODE_BINARY:
                        last_input = time.time()
                        os.write(master, payload)
                    elif opcode == OPCODE_TEXT:
                        last_input = time.time()
                        self._control(master, payload)
        finally:
            closed.set()
            try:
                conn.sendall(encode_frame(b"", OPCODE_CLOSE))
            except OSError:
                pass
            self._terminate(proc)
            try:
                os.close(master)
            except OSError:
                pass
            log.info("terminal in %s closed", cwd)
            if launched:
                self._emit(grant, "TerminalClosed", {**launched, "cwd": str(cwd), "exit_code": proc.returncode})

    @staticmethod
    def _control(master: int, payload: bytes) -> None:
        try:
            message = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return
        if isinstance(message, dict) and message.get("type") == "resize":
            try:
                cols = max(2, min(500, int(message.get("cols") or DEFAULT_COLS)))
                rows = max(2, min(200, int(message.get("rows") or DEFAULT_ROWS)))
                fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
            except (OSError, TypeError, ValueError):
                pass

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        for sig, grace in ((signal.SIGHUP, 1.0), (signal.SIGTERM, 2.0), (signal.SIGKILL, 2.0)):
            try:
                os.killpg(proc.pid, sig)
            except OSError:
                return
            deadline = time.time() + grace
            while time.time() < deadline:
                if proc.poll() is not None:
                    return
                time.sleep(0.05)
