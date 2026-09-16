"""One ticket, one supervised interactive CLI session (PRD-234 §Design 2).

The turn is the same for every CLI; the adapter says how the CLI is spelled
(CLI adapter design, ``docs/architecture/CLI-RUNTIME-ADAPTER-DESIGN.md``):

1. resolve the working directory against the host's allowlist; refuse otherwise;
2. ask the adapter to preflight — the operator's own binary, logged in with the
   operator's own plan (never a key);
3. write the session's files under the host's state dir (never into the user's
   repo): ``ticket.md`` (the dispatch contract), ``system_prompt.md`` (stable
   per agent — nothing volatile, munder's prompt-cache rule), and whatever the
   adapter ``prepare()``s (Claude: a hooks-only ``settings.json``; Codex: a
   config home); record the trust decision for the directory;
4. spawn the user's own CLI INTERACTIVELY under a pseudo-terminal the host only
   drains, with the argv the adapter builds from its preset and a short
   positional pointer prompt;
5. hooks carry the turn over the bus: ``PreToolUse`` is the policy gate
   (``ToolIntent`` — what the call does, not what the CLI calls it),
   ``PostToolUse`` the files touched, ``Notification`` the needs-a-human / limit
   signals, ``Stop`` the end of the turn (with the final text). The adapter
   translates each payload in and each reply out;
6. on the turn's end read the transcript (the adapter knows where and how),
   terminate the process, report.

No typing into the TUI, no output parsing. Never a headless mode, never a
bypass of our gate — each preset names what that means for its binary.
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
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
from .adapters import NotServed, UnknownCli, adapter_for, adapters
from .adapters.base import LaunchContext, Reply, ToolClass
from .allowlist import NotAllowed, resolve_allowed, default_session_cwd
from .config import HostConfig
from .env import build_session_env
from .policy import Decision, PolicyContext, bash_allowlist_from_config, decide
from .presets import REGISTRY, TURN_END_PROCESS_EXIT, TURN_END_STOP_HOOK
from .terminal_log import FILENAME as TERMINAL_LOG_FILENAME, BoundedLog
from .transcript import empty_usage, usage_delta

log = logging.getLogger("automatos.cli_host.session")

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
            # PRD-239: where the session REALLY ran (a --worktree for a git repo) —
            # the directory `claude --resume` and the editor links must open.
            "effective_cwd": self.effective_cwd,
        }


def _slug(text: str, limit: int = 40) -> str:
    return _SLUG_RE.sub("-", text).strip("-")[:limit] or "ticket"


SESSION_RULES = (
    "The ticket you are working is described in the file named in your first message; "
    "read it fully before acting.\n"
    "Rules of the session: work only inside the directory you were started in; "
    "never push, publish or open pull requests — the manager integrates your work; "
    "keep changes scoped to the ticket's OBJECTIVE and BOUNDARIES; when you are done, "
    "reply with a concise summary of what changed, what you verified, and anything left open.\n"
)


def build_system_prompt(ticket: Dict[str, Any], cli_label: str = "Claude Code") -> str:
    """Stable per agent: no ids, no dates, no counters (prompt-cache invariant).

    PRD-239 S1: the backend renders the agent's soul — description, persona and
    skills — as ``system_prompt`` on the ticket (stable per agent); it sits
    between the introduction and the session rules. Without it the prompt is
    exactly the name and the rules, as before.
    """
    name = ticket.get("agent_name") or "an Automatos agent"
    intro = f"You are {name}, working as a supervised {cli_label} session managed by Automatos.\n"
    soul = ticket.get("system_prompt")
    soul = soul.strip() if isinstance(soul, str) else ""
    if soul:
        return intro + "\n" + soul + "\n\n" + SESSION_RULES
    return intro + SESSION_RULES


def build_ticket_file(ticket: Dict[str, Any]) -> str:
    return (
        f"# Ticket #{ticket.get('task_id')} — {ticket.get('title') or ''}\n\n"
        f"{ticket.get('prompt') or ''}\n"
    )


def assert_args_honour_invariant(args: Sequence[str], forbidden: Sequence[str]) -> None:
    """The preset's forbidden arguments never reach a command line (design §9.2)."""
    joined = " ".join(args)
    for bad in forbidden:
        if f" {bad} " in f" {joined} " or joined.endswith(f" {bad}"):
            raise RuntimeError(f"forbidden argument in session command: {bad}")


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def compact_event(event: str, payload: Dict[str, Any], subject: Optional[str] = None) -> Dict[str, Any]:
    """The event the backend receives: the few facts the board needs, never the
    whole hook payload. ``cwd`` (PRD-235 W2) is the session's effective working
    directory — SessionStart carries it — so the ticket can deep-link the editor.
    ``subject`` is the one thing a tool call is about (the adapter's ``ToolIntent``)."""
    compact = {
        "event": event,
        "at": time.time(),
        "session_id": payload.get("session_id"),
        "transcript_path": payload.get("transcript_path"),
        "cwd": payload.get("cwd"),
        "tool_name": payload.get("tool_name"),
        "subject": subject,
        "notification_type": payload.get("notification_type"),
        "message": (payload.get("message") or "")[:500] or None,
    }
    return {k: v for k, v in compact.items() if v is not None}


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
        # The adapter for the ticket's CLI (design §4). A CLI this host cannot run is
        # an honest error at preflight, never a silent fallback to another CLI.
        self.adapter = None
        self._adapter_error: Optional[str] = None
        try:
            self.adapter = adapter_for(ticket.get("provider"), getattr(cfg, "cli_binaries", None))
        except (UnknownCli, NotServed) as exc:
            self._adapter_error = str(exc)
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
        self._usage_before: Optional[Dict[str, Any]] = None
        self.reported_session_id: Optional[str] = None
        self.last_assistant_message: Optional[str] = None
        self.files_touched: List[str] = []
        self.denials: List[Dict[str, Any]] = []
        # PRD-235 W2 S3: permission questions the operator answers from the Canvas.
        self._pending_asks: Dict[str, threading.Event] = {}
        self._ask_answers: Dict[str, bool] = {}
        self._ask_lock = threading.Lock()
        self.notifications: List[Dict[str, Any]] = []
        self.output_tail: deque = deque(maxlen=_OUTPUT_TAIL_BYTES)
        self.terminal_log: Optional[BoundedLog] = None
        self._contract_injected = False
        self._policy: Optional[PolicyContext] = None

    @property
    def cli(self) -> str:
        return self.adapter.id if self.adapter is not None else str(self.ticket.get("provider") or "?")

    # ── hook handling (called on the hook server's threads) ────────────────
    def handle_hook(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """One hook in, one answer out — through the adapter both ways (design §5)."""
        if self.adapter is None:
            return {}
        payload = self.adapter.normalize_event(raw)
        if payload is None:
            return {}
        event = payload.get("hook_event_name") or ""
        self._emit(event, payload)
        reply = self._reply_for(event, payload)
        return self.adapter.render_response(event, reply) or {}

    def _reply_for(self, event: str, payload: Dict[str, Any]) -> Reply:
        if event == "SessionStart":
            self.session_started.set()
            self.reported_session_id = payload.get("session_id") or self.reported_session_id
            self.transcript_path = payload.get("transcript_path") or self.transcript_path
            # A resumed session's transcript already holds earlier turns; snapshot
            # them now (before this turn's prompt lands) so the result reports
            # only what THIS run used (2026-09-09 analytics).
            if self.ticket.get("resume_session_id") and self.transcript_path and self._usage_before is None:
                path = Path(self.transcript_path)
                self._usage_before = self.adapter.read_usage(path) if path.exists() else empty_usage()
            cwd = payload.get("cwd")
            if cwd:
                self.effective_cwd = Path(cwd)
                if self._policy is not None and self.effective_cwd not in self._policy.extra_dirs:
                    self._policy.extra_dirs = (*self._policy.extra_dirs, self.effective_cwd)
            return Reply.none()
        if event == "UserPromptSubmit":
            if self._contract_injected:
                return Reply.none()
            self._contract_injected = True
            return Reply.with_context(build_ticket_file(self.ticket))
        if event == "PreToolUse":
            return self._pre_tool_use(payload)
        if event == "PermissionRequest":
            tool = payload.get("tool_name") or "?"
            reason = "a permission prompt reached the TUI — sessions are policy-gated, not prompted"
            self.denials.append({"tool": tool, "reason": reason, "stage": "PermissionRequest"})
            return Reply.deny(reason)
        if event == "PostToolUse":
            self._track_file(payload)
            return Reply.none()
        if event == "Notification":
            self.notifications.append({"type": payload.get("notification_type"), "message": payload.get("message")})
            return Reply.none()
        if event == "Stop":
            self.last_assistant_message = payload.get("last_assistant_message") or self.last_assistant_message
            self.stopped.set()
            return Reply.none()
        if event == "SessionEnd":
            self.ended.set()
            return Reply.none()
        return Reply.none()

    def _pre_tool_use(self, payload: Dict[str, Any]) -> Reply:
        tool = str(payload.get("tool_name") or "")
        tool_input = payload.get("tool_input") or {}
        if not isinstance(tool_input, dict):
            tool_input = {}
        intent = self.adapter.tool_intent(tool, tool_input)
        if self._policy is None:
            decision = Decision("deny", "session policy not initialised")
        else:
            decision = decide(intent, self._policy)
        if decision.behavior == "ask":
            decision = self._ask_operator(tool, intent.subject, decision.reason)
        if decision.allow:
            return Reply.allow()
        self.denials.append({"tool": tool, "reason": decision.reason, "stage": "PreToolUse",
                             "input": {k: v for k, v in tool_input.items() if k in ("command", "file_path", "path")}})
        return Reply.deny(decision.reason)

    def _ask_operator(self, tool: str, subject: Optional[str], reason: str) -> Decision:
        """PRD-235 W2 S3: hold this tool call while the operator answers a card on the
        ticket's Canvas. The question travels with the next event flush; the answer
        comes back on that same channel (``resolve_ask``). No answer within
        ``ask_timeout`` seconds → deny, honestly worded."""
        request_id = uuid.uuid4().hex
        done = threading.Event()
        with self._ask_lock:
            self._pending_asks[request_id] = done
        self.events.put({
            "event": "PermissionRequest", "at": time.time(), "request_id": request_id,
            "tool_name": tool, "subject": subject, "reason": reason,
            "session_id": self.reported_session_id or self.session_id,
        })
        timeout = float(getattr(self.cfg, "ask_timeout", 120.0) or 120.0)
        answered = done.wait(timeout)
        with self._ask_lock:
            self._pending_asks.pop(request_id, None)
            approved = self._ask_answers.pop(request_id, None)
        if answered and approved:
            return Decision("allow")
        if answered:
            return Decision("deny", f"{reason} — denied by the operator")
        return Decision("deny", f"{reason} — no answer from the operator within {int(timeout)} s")

    def resolve_ask(self, request_id: str, approved: bool) -> bool:
        """The backend delivered the operator's answer for a pending question."""
        with self._ask_lock:
            ev = self._pending_asks.get(str(request_id))
            if ev is None:
                return False
            self._ask_answers[str(request_id)] = bool(approved)
        ev.set()
        return True

    def _intent_of(self, payload: Dict[str, Any]):
        tool = payload.get("tool_name")
        if not tool or self.adapter is None:
            return None
        ti = payload.get("tool_input")
        return self.adapter.tool_intent(str(tool), ti if isinstance(ti, dict) else {})

    def _track_file(self, payload: Dict[str, Any]) -> None:
        intent = self._intent_of(payload)
        if intent is not None and intent.cls is ToolClass.FILE_WRITE:
            for path in intent.paths:
                if path not in self.files_touched:
                    self.files_touched.append(str(path))

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        intent = self._intent_of(payload)
        self.events.put(compact_event(event, payload, subject=intent.subject if intent else None))

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

        # 2. preflight — which CLI, the user's own binary and login
        if self.adapter is None:
            return self._outcome("error", error=self._adapter_error or "no CLI adapter", exit_reason="cli_not_served")
        preset = self.adapter.preset
        if preset.turn_end not in (TURN_END_STOP_HOOK, TURN_END_PROCESS_EXIT):
            return self._outcome("error", error=f"{preset.label}: turn end {preset.turn_end!r} is not implemented by this host",
                                 exit_reason="turn_end_unsupported")
        refusal = self.adapter.preflight()
        if refusal is not None:
            return self._outcome("error", error=refusal.message, exit_reason=refusal.code)
        binary = self.adapter.resolve_binary()

        # 3. files + what the adapter prepares (settings/config home + trust)
        session_dir = self.cfg.sessions_dir / self.task_id
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        ticket_path = session_dir / "ticket.md"
        ticket_path.write_text(build_ticket_file(self.ticket), encoding="utf-8")
        system_prompt_path = session_dir / "system_prompt.md"
        system_prompt_path.write_text(build_system_prompt(self.ticket, preset.label), encoding="utf-8")
        self.terminal_log = BoundedLog(session_dir / TERMINAL_LOG_FILENAME)

        self._policy = PolicyContext(
            cwd=cwd,
            allowed_bash=bash_allowlist_from_config(self.ticket.get("allowed_tools")),
            extra_dirs=(session_dir,),
        )

        # PRD-239: a per-agent choice — a single repo gets a worktree per ticket
        # (the checkout stays untouched); a workspace of many repos, whose own
        # git tracks next to nothing, must not (the worktree would be empty).
        wants_worktree = self.ticket.get("worktree", True) is not False
        worktree = f"automatos-{_slug(str(self.task_id))}" if (self.cfg.use_worktrees and wants_worktree and _is_git_repo(cwd)) else None
        ctx = LaunchContext(
            cwd=cwd, session_dir=session_dir, ticket_path=ticket_path, system_prompt_path=system_prompt_path,
            task_id=self.task_id, session_id=self.session_id, resume_session_id=self.ticket.get("resume_session_id"),
            model=self.ticket.get("model"), worktree_name=worktree, agent_id=str(self.ticket.get("agent_id") or "") or None,
            state_dir=getattr(self.cfg, "state_dir", None),
        )
        prepared = self.adapter.prepare(ctx)

        # 4. spawn
        args = self.adapter.launch_args(ctx, prepared)
        assert_args_honour_invariant(args, preset.forbidden_args)
        package_root = str(Path(__file__).resolve().parents[1])
        inherited_pp = os.environ.get("PYTHONPATH", "")
        env = build_session_env(preset, extra={
            "AUTOMATOS_HOST_SOCK": str(self.sock_path),
            "AUTOMATOS_TASK_ID": self.task_id,
            "AUTOMATOS_CLI": preset.id,
            "AUTOMATOS_HOOK_WAIT_SECONDS": "560",
            # Hooks run from the session's directory: the shim (`python -m
            # automatos_cli_host.hook_shim`) must find this package from there.
            "PYTHONPATH": package_root + (os.pathsep + inherited_pp if inherited_pp else ""),
            **prepared.env,
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
        log.info("task %s: %s session %s started (pid %s) in %s%s", self.task_id, preset.id, self.session_id, self.proc.pid, cwd,
                 f" worktree={worktree}" if worktree else "")

        # 5. wait for the turn's end / exit / cancel / timeout — the preset says how a turn ends
        deadline = self.started_at + self.cfg.session_timeout_seconds
        exit_reason = "completed"
        hook_driven = preset.turn_end == TURN_END_STOP_HOOK
        while True:
            if hook_driven and self.stopped.is_set():
                self.ended.wait(STOP_GRACE_SECONDS)
                break
            if self.proc.poll() is not None:
                exit_reason = "exited_before_stop" if hook_driven else "completed"
                break
            if self.cancel_requested.is_set():
                exit_reason = "cancelled"
                break
            if time.time() > deadline:
                exit_reason = "timeout"
                break
            if hook_driven and not self.session_started.is_set() and time.time() - self.started_at > self.cfg.startup_timeout_seconds:
                exit_reason = "no_session_start"
                break
            time.sleep(0.25)
        self._terminate()
        return self._collect(exit_reason, cwd, binary or preset.binary)

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

    def _collect(self, exit_reason: str, cwd: Path, binary: str) -> SessionOutcome:
        preset = self.adapter.preset
        transcript = Path(self.transcript_path) if self.transcript_path else None
        if transcript is None:
            # No hook told us (a print-mode CLI): the adapter knows where it would be.
            guess = self.adapter.transcript_path(str(self.effective_cwd or cwd), self.reported_session_id or self.session_id)
            transcript = guess if guess and guess.exists() else None
            self.transcript_path = str(transcript) if transcript else None
        usage = self.adapter.read_usage(transcript) if transcript and transcript.exists() else {}
        if usage and self._usage_before is not None:
            usage = usage_delta(usage, self._usage_before)
        text = self.last_assistant_message or (self.adapter.last_text(transcript) if transcript and transcript.exists() else None) or ""
        if not text and preset.turn_end == TURN_END_PROCESS_EXIT:
            text = bytes(self.output_tail).decode("utf-8", "replace").strip()   # print mode: stdout IS the answer
        name = os.path.basename(binary)
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
                f"{name} did not start a session within {int(self.cfg.startup_timeout_seconds)} s — "
                f"it is probably showing a login screen or a dialog. Run `{name}` in that directory once "
                f"and log in, then retry. Last output:\n{tail}"
            )
        else:
            tail = bytes(self.output_tail).decode("utf-8", "replace")[-1500:]
            code = self.proc.returncode if self.proc else None
            status, error = "error", f"{name} exited (code {code}) before finishing the turn. Last output:\n{tail}"
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
    """What the host announces: every CLI the registry knows — present, version,
    served (preflight would pass) and, when not, why; ``providers`` is the served
    subset, which the backend's claim filter reads (design §8.2). Never a credential."""
    import platform
    import sys

    binaries = getattr(cfg, "cli_binaries", None) or {}
    served = adapters(binaries)
    clis: Dict[str, Any] = {}
    for cli_id, preset in REGISTRY.items():
        if cli_id in served:
            clis[cli_id] = served[cli_id].detect()
        else:
            clis[cli_id] = {"path": None, "version": None, "served": False, "tier": preset.tier,
                            "reason": f"{preset.label} is not served by this host yet (no adapter)"}
    return {
        "host_version": __version__,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "clis": clis,
        "providers": [cli_id for cli_id, info in clis.items() if info.get("served")],
        "worktrees": cfg.use_worktrees,
        # PRD-239 S6: the directories this host may run sessions in — the backend
        # checks an agent's working_directory against them before it is saved.
        "allow_dirs": [str(p) for p in (cfg.allow_dirs or [])],
    }
