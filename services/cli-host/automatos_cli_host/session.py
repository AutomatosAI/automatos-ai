"""One ticket, one supervised interactive Claude Code session (PRD-234 §Design 2).

The turn:

1. resolve the working directory against the host's allowlist; refuse otherwise;
2. write the session's files under the host's state dir (never into the user's
   repo): ``ticket.md`` (the dispatch contract), ``system_prompt.md`` (stable
   per agent — nothing volatile, munder's prompt-cache rule), ``settings.json``
   (hooks → this host); record the trust decision for the directory;
3. spawn the user's own ``claude`` INTERACTIVELY under a pseudo-terminal the
   host only drains: ``--session-id <pre-assigned>`` (or ``--resume`` for a
   continuation), ``--permission-mode acceptEdits``, ``--append-system-prompt-file``,
   ``--settings``, ``--setting-sources user``, ``--strict-mcp-config``,
   ``--add-dir <session dir>``, ``--name``, ``--model`` when the ticket has one,
   ``--worktree`` for git repositories, and a short positional pointer prompt;
4. hooks carry the turn: ``PreToolUse`` is the policy gate, ``PostToolUse`` the
   files touched, ``Notification`` the needs-a-human / limit signals, ``Stop`` the
   end of the turn (with the final text);
5. on ``Stop`` read the transcript for usage, terminate the process, report.

No typing into the TUI, no output parsing. Never ``-p``, never ``--bare``.
"""
from __future__ import annotations

import fcntl
import logging
import os
import pty
import queue
import re
import signal
import struct
import subprocess
import termios
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .allowlist import NotAllowed, resolve_allowed, default_session_cwd
from .claude_settings import has_completed_onboarding, record_directory_trust, write_settings
from .config import HostConfig
from .env import build_session_env, resolve_binary
from .policy import Decision, PolicyContext, bash_allowlist_from_config, decide
from .terminal_log import FILENAME as TERMINAL_LOG_FILENAME, BoundedLog
from .transcript import last_assistant_text, read_usage

log = logging.getLogger("automatos.cli_host.session")

FORBIDDEN_ARGS = ("-p", "--print", "--bare", "--dangerously-skip-permissions", "--permission-mode bypassPermissions")
STOP_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 5.0
PTY_ROWS, PTY_COLS = 50, 200
_OUTPUT_TAIL_BYTES = 16 * 1024
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SessionOutcome:
    status: str                      # success | error | cancelled
    result_text: str = ""
    error: Optional[str] = None
    exit_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    files_touched: List[str] = field(default_factory=list)
    permission_denials: List[Dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None
    transcript_path: Optional[str] = None
    effective_cwd: Optional[str] = None

    def as_result_payload(self, attempt: int) -> Dict[str, Any]:
        return {
            "attempt": attempt,
            "status": self.status,
            "result_text": self.result_text,
            "error": self.error,
            "usage": self.usage,
            "files_touched": self.files_touched,
            "permission_denials": self.permission_denials,
            "session_id": self.session_id,
            "exit_reason": self.exit_reason,
            "transcript_path": self.transcript_path,
        }


def _slug(text: str, limit: int = 40) -> str:
    return _SLUG_RE.sub("-", text).strip("-")[:limit] or "ticket"


def build_system_prompt(ticket: Dict[str, Any]) -> str:
    """Stable per agent: no ids, no dates, no counters (prompt-cache invariant)."""
    name = ticket.get("agent_name") or "an Automatos agent"
    return (
        f"You are {name}, working as a supervised Claude Code session managed by Automatos.\n"
        "The ticket you are working is described in the file named in your first message; "
        "read it fully before acting.\n"
        "Rules of the session: work only inside the directory you were started in; "
        "never push, publish or open pull requests — the manager integrates your work; "
        "keep changes scoped to the ticket's OBJECTIVE and BOUNDARIES; when you are done, "
        "reply with a concise summary of what changed, what you verified, and anything left open.\n"
    )


def build_ticket_file(ticket: Dict[str, Any]) -> str:
    return (
        f"# Ticket #{ticket.get('task_id')} — {ticket.get('title') or ''}\n\n"
        f"{ticket.get('prompt') or ''}\n"
    )


def build_args(
    claude: str,
    *,
    session_id: str,
    resume_session_id: Optional[str],
    system_prompt_path: Path,
    settings_path: Path,
    session_dir: Path,
    ticket_path: Path,
    task_id: Any,
    model: Optional[str],
    worktree_name: Optional[str],
) -> List[str]:
    args = [claude]
    if resume_session_id:
        args += ["--resume", resume_session_id]
    else:
        args += ["--session-id", session_id]
    args += [
        "--permission-mode", "acceptEdits",
        "--append-system-prompt-file", str(system_prompt_path),
        "--settings", str(settings_path),
        "--setting-sources", "user",
        "--strict-mcp-config",
        "--add-dir", str(session_dir),
        "--name", f"automatos #{task_id}",
    ]
    if model:
        args += ["--model", str(model)]
    if worktree_name:
        args += ["--worktree", worktree_name]
    # The positional prompt is a short pointer — nothing sensitive in argv.
    args.append(f"Work the Automatos ticket described in {ticket_path}. Read it first.")
    return args


def assert_args_honour_invariant(args: List[str]) -> None:
    joined = " ".join(args)
    for bad in FORBIDDEN_ARGS:
        if f" {bad} " in f" {joined} " or joined.endswith(f" {bad}"):
            raise RuntimeError(f"forbidden argument in session command: {bad}")


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def _subject_of(payload: Dict[str, Any]) -> Optional[str]:
    """The one thing a tool call is about — a command, a path, a pattern — for the
    ticket's live log. Never the whole tool input."""
    ti = payload.get("tool_input")
    if not isinstance(ti, dict):
        return None
    for key in ("command", "file_path", "notebook_path", "path", "pattern", "url", "query"):
        value = ti.get(key)
        if value:
            return str(value)[:200]
    return None


class Session:
    """Runs one ticket. ``events`` is drained by the host and shipped in batches."""

    def __init__(self, ticket: Dict[str, Any], cfg: HostConfig, allow_roots: List[str],
                 sock_path: Path, default_root: Optional[str], workspace_id: str = ""):
        self.ticket = ticket
        self.workspace_id = workspace_id
        self.cfg = cfg
        self.allow_roots = allow_roots
        self.sock_path = sock_path
        self.default_root = default_root
        self.task_id = str(ticket.get("task_id"))
        self.attempt = int(ticket.get("attempt") or 0)
        self.session_id = str(ticket.get("session_id") or "")
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.cancel_requested = threading.Event()
        self.stopped = threading.Event()
        self.session_started = threading.Event()
        self.ended = threading.Event()
        self.started_at = time.time()
        self.proc: Optional[subprocess.Popen] = None
        self.pgid: Optional[int] = None
        self.effective_cwd: Optional[Path] = None
        self.transcript_path: Optional[str] = None
        self.reported_session_id: Optional[str] = None
        self.last_assistant_message: Optional[str] = None
        self.files_touched: List[str] = []
        self.denials: List[Dict[str, Any]] = []
        self.notifications: List[Dict[str, Any]] = []
        self.output_tail: deque = deque(maxlen=_OUTPUT_TAIL_BYTES)
        self.terminal_log: Optional[BoundedLog] = None
        self._contract_injected = False
        self._policy: Optional[PolicyContext] = None

    # ── hook handling (called on the hook server's threads) ────────────────
    def handle_hook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = payload.get("hook_event_name") or ""
        self._emit(event, payload)
        if event == "SessionStart":
            self.session_started.set()
            self.reported_session_id = payload.get("session_id") or self.reported_session_id
            self.transcript_path = payload.get("transcript_path") or self.transcript_path
            cwd = payload.get("cwd")
            if cwd:
                self.effective_cwd = Path(cwd)
                if self._policy is not None and self.effective_cwd not in self._policy.extra_dirs:
                    self._policy.extra_dirs = (*self._policy.extra_dirs, self.effective_cwd)
            return {}
        if event == "UserPromptSubmit":
            if self._contract_injected:
                return {}
            self._contract_injected = True
            return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                           "additionalContext": build_ticket_file(self.ticket)}}
        if event == "PreToolUse":
            return self._pre_tool_use(payload)
        if event == "PermissionRequest":
            tool = payload.get("tool_name") or "?"
            reason = "a permission prompt reached the TUI — sessions are policy-gated, not prompted"
            self.denials.append({"tool": tool, "reason": reason, "stage": "PermissionRequest"})
            return {"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                           "decision": {"behavior": "deny", "message": reason}}}
        if event == "PostToolUse":
            self._track_file(payload)
            return {}
        if event == "Notification":
            self.notifications.append({"type": payload.get("notification_type"), "message": payload.get("message")})
            return {}
        if event == "Stop":
            self.last_assistant_message = payload.get("last_assistant_message") or self.last_assistant_message
            self.stopped.set()
            return {}
        if event == "SessionEnd":
            self.ended.set()
            return {}
        return {}

    def _pre_tool_use(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        tool = str(payload.get("tool_name") or "")
        tool_input = payload.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        if self._policy is None:
            decision = Decision("deny", "session policy not initialised")
        else:
            decision = decide(tool, tool_input, self._policy)
        if decision.behavior == "ask":
            # Approvals-inbox routing lands with S3; until the backend answers a
            # hold, an 'ask' is an honest deny the ticket surfaces in review.
            decision = Decision("deny", decision.reason + " (approval routing not yet available)")
        if decision.allow:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
        self.denials.append({"tool": tool, "reason": decision.reason, "stage": "PreToolUse",
                             "input": {k: v for k, v in tool_input.items() if k in ("command", "file_path", "path")}})
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": decision.reason}}

    def _track_file(self, payload: Dict[str, Any]) -> None:
        if payload.get("tool_name") in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
            ti = payload.get("tool_input") or {}
            path = ti.get("file_path") or ti.get("notebook_path") if isinstance(ti, dict) else None
            if path and path not in self.files_touched:
                self.files_touched.append(str(path))

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        compact = {
            "event": event,
            "at": time.time(),
            "session_id": payload.get("session_id"),
            "transcript_path": payload.get("transcript_path"),
            "tool_name": payload.get("tool_name"),
            "subject": _subject_of(payload),
            "notification_type": payload.get("notification_type"),
            "message": (payload.get("message") or "")[:500] or None,
        }
        self.events.put({k: v for k, v in compact.items() if v is not None})

    # ── the run ─────────────────────────────────────────────────────────────
    def run(self) -> SessionOutcome:
        try:
            return self._run()
        except Exception as exc:  # noqa: BLE001 — every failure becomes an honest result
            log.exception("session for task %s crashed", self.task_id)
            return self._outcome("error", error=f"host error: {exc}", exit_reason="host_error")

    def _run(self) -> SessionOutcome:
        # 1. where
        try:
            cwd_hint = str(self.ticket.get("cwd") or "").strip()
            if not cwd_hint and self.default_root:
                # No working directory on the agent → the workspace's own sessions
                # folder, which the Deliverables explorer shows live (PRD-234 S2).
                cwd = default_session_cwd(self.default_root, self.workspace_id, self.task_id)
            else:
                cwd = resolve_allowed(cwd_hint or None, self.allow_roots, default_root=self.default_root)
        except NotAllowed as exc:
            return self._outcome("error", error=str(exc), exit_reason="cwd_not_allowed")
        if not cwd.is_dir():
            return self._outcome("error", error=f"working directory does not exist: {cwd}", exit_reason="cwd_missing")

        # 2. preflight — the user's own CLI and login
        claude = self.cfg.claude_binary or resolve_binary("claude")
        if not claude:
            return self._outcome("error", error="Claude Code is not installed on this machine (no `claude` on your PATH). Install it and run `claude login`.", exit_reason="claude_missing")
        if not has_completed_onboarding():
            return self._outcome("error", error="Claude Code has never been run interactively on this machine. Run `claude` once in your terminal and log in, then retry.", exit_reason="claude_not_onboarded")

        # 3. files + trust
        session_dir = self.cfg.sessions_dir / self.task_id
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        ticket_path = session_dir / "ticket.md"
        ticket_path.write_text(build_ticket_file(self.ticket), encoding="utf-8")
        system_prompt_path = session_dir / "system_prompt.md"
        system_prompt_path.write_text(build_system_prompt(self.ticket), encoding="utf-8")
        settings_path = write_settings(session_dir / "settings.json")
        self.terminal_log = BoundedLog(session_dir / TERMINAL_LOG_FILENAME)
        try:
            record_directory_trust(cwd)
        except OSError as exc:
            log.warning("could not record trust for %s: %s", cwd, exc)

        self._policy = PolicyContext(
            cwd=cwd,
            allowed_bash=bash_allowlist_from_config(self.ticket.get("allowed_tools")),
            extra_dirs=(session_dir,),
        )

        # 4. spawn
        worktree = f"automatos-{_slug(str(self.task_id))}" if (self.cfg.use_worktrees and _is_git_repo(cwd)) else None
        args = build_args(
            claude,
            session_id=self.session_id,
            resume_session_id=self.ticket.get("resume_session_id"),
            system_prompt_path=system_prompt_path,
            settings_path=settings_path,
            session_dir=session_dir,
            ticket_path=ticket_path,
            task_id=self.task_id,
            model=self.ticket.get("model"),
            worktree_name=worktree,
        )
        assert_args_honour_invariant(args)
        package_root = str(Path(__file__).resolve().parents[1])
        inherited_pp = os.environ.get("PYTHONPATH", "")
        env = build_session_env(extra={
            "AUTOMATOS_HOST_SOCK": str(self.sock_path),
            "AUTOMATOS_TASK_ID": self.task_id,
            "AUTOMATOS_HOOK_WAIT_SECONDS": "560",
            # Hooks run from the session's directory: the shim (`python -m
            # automatos_cli_host.hook_shim`) must find this package from there.
            "PYTHONPATH": package_root + (os.pathsep + inherited_pp if inherited_pp else ""),
        })
        master, slave = pty.openpty()
        try:
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", PTY_ROWS, PTY_COLS, 0, 0))
        except OSError:
            pass

        def _child_setup() -> None:  # runs in the child after setsid()
            try:
                fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
            except OSError:
                pass

        self.proc = subprocess.Popen(
            args, stdin=slave, stdout=slave, stderr=slave, cwd=str(cwd), env=env,
            start_new_session=True, preexec_fn=_child_setup, close_fds=True,
        )
        os.close(slave)
        self.pgid = self.proc.pid
        threading.Thread(target=self._drain, args=(master,), daemon=True, name=f"pty-drain-{self.task_id}").start()
        log.info("task %s: session %s started (pid %s) in %s%s", self.task_id, self.session_id, self.proc.pid, cwd,
                 f" worktree={worktree}" if worktree else "")

        # 5. wait for Stop / exit / cancel / timeout
        deadline = self.started_at + self.cfg.session_timeout_seconds
        exit_reason = "completed"
        while True:
            if self.stopped.is_set():
                self.ended.wait(STOP_GRACE_SECONDS)
                break
            if self.proc.poll() is not None:
                exit_reason = "exited_before_stop"
                break
            if self.cancel_requested.is_set():
                exit_reason = "cancelled"
                break
            if time.time() > deadline:
                exit_reason = "timeout"
                break
            if not self.session_started.is_set() and time.time() - self.started_at > self.cfg.startup_timeout_seconds:
                exit_reason = "no_session_start"
                break
            time.sleep(0.25)
        self._terminate()
        return self._collect(exit_reason, cwd)

    def _drain(self, master: int) -> None:
        try:
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                if not chunk:
                    break
                self.output_tail.extend(chunk)
                if self.terminal_log is not None:
                    self.terminal_log.write(chunk)
        finally:
            try:
                os.close(master)
            except OSError:
                pass
            if self.terminal_log is not None:
                self.terminal_log.close()

    def _terminate(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.pgid or self.proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            self.proc.wait(KILL_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.pgid or self.proc.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                self.proc.wait(KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                log.error("task %s: process %s survived SIGKILL", self.task_id, self.proc.pid)

    def request_cancel(self) -> None:
        self.cancel_requested.set()

    def _collect(self, exit_reason: str, cwd: Path) -> SessionOutcome:
        transcript = Path(self.transcript_path) if self.transcript_path else None
        usage = read_usage(transcript) if transcript and transcript.exists() else {}
        text = self.last_assistant_message or (last_assistant_text(transcript) if transcript and transcript.exists() else None) or ""
        if exit_reason == "completed":
            status = "success"
            error = None
        elif exit_reason == "cancelled":
            status, error = "cancelled", "cancelled by the operator"
        elif exit_reason == "timeout":
            status, error = "error", f"session exceeded {int(self.cfg.session_timeout_seconds)} s"
        elif exit_reason == "no_session_start":
            tail = bytes(self.output_tail).decode("utf-8", "replace")[-1500:]
            status, error = "error", (
                f"claude did not start a session within {int(self.cfg.startup_timeout_seconds)} s — "
                "it is probably showing a login screen or a dialog. Run `claude` in that directory once "
                f"and log in, then retry. Last output:\n{tail}"
            )
        else:
            tail = bytes(self.output_tail).decode("utf-8", "replace")[-1500:]
            code = self.proc.returncode if self.proc else None
            status, error = "error", f"claude exited (code {code}) before finishing the turn. Last output:\n{tail}"
        return self._outcome(status, result_text=text, error=error, exit_reason=exit_reason, usage=usage, cwd=cwd)

    def _outcome(self, status: str, *, result_text: str = "", error: Optional[str] = None,
                 exit_reason: str = "", usage: Optional[Dict[str, Any]] = None, cwd: Optional[Path] = None) -> SessionOutcome:
        return SessionOutcome(
            status=status, result_text=result_text, error=error, exit_reason=exit_reason,
            usage=usage or {}, files_touched=list(self.files_touched),
            permission_denials=list(self.denials),
            session_id=self.reported_session_id or self.session_id or None,
            transcript_path=self.transcript_path,
            effective_cwd=str(self.effective_cwd or cwd) if (self.effective_cwd or cwd) else None,
        )


def host_capabilities(cfg: HostConfig) -> Dict[str, Any]:
    """What the host announces: the user's CLI, its version, onboarding state, platform."""
    import platform
    import sys

    claude = cfg.claude_binary or resolve_binary("claude")
    version = None
    if claude:
        try:
            out = subprocess.run([claude, "--version"], capture_output=True, text=True, timeout=15, check=False)
            version = (out.stdout or out.stderr or "").strip().split("\n")[0][:80] or None
        except (OSError, subprocess.SubprocessError):
            version = None
    return {
        "host_version": __version__,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "claude": {"path": claude, "version": version, "onboarded": has_completed_onboarding()} if claude else None,
        "providers": ["claude"] if claude else [],
        "worktrees": cfg.use_worktrees,
    }
