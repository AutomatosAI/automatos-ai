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

import logging
import os
import queue
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__
from . import turn_end, usage_limit
from .adapters import NotServed, UnknownCli, adapter_for, adapters
from .adapters.base import LaunchContext, Reply, ToolClass
from .allowlist import NotAllowed, default_session_cwd, resolve_allowed, session_deliverables_dir
from .brand_files import brand_files_note, write_brand_files
from .config import HostConfig
from .env import build_session_env, hook_pythonpath
from .permission_modes import MODE_EDITS, MODE_PLAN, PLAN_EVENT, PLAN_WITH_OPERATOR, plan_text, save_plan, session_mode
from .policy import PLAN_BASH_ALLOW, Decision, PolicyContext, bash_allowlist_from_config, decide, platform_secret_roots
from .permission_request import PERMISSION_REQUEST_REJUDGE, AllowedCalls, request_of
from .permission_request import answer as permission_answer
from .ptyproc import KILL, TERMINATE, PtyChild
from .ptyproc import spawn as spawn_on_pty
from .presets import HOOK_WAIT_SECONDS, REGISTRY, TURN_END_PROCESS_EXIT, TURN_END_STOP_HOOK, hold_seconds
from .session_prompt import build_system_prompt, build_ticket_file
from .session_files import CREDENTIAL_SESSION_FILES, land_session_deliverables, session_deliverables
from .terminal_log import FILENAME as TERMINAL_LOG_FILENAME, BoundedLog
from .transcript import empty_usage, usage_delta

log = logging.getLogger("automatos.cli_host.session")

STOP_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 5.0
PTY_ROWS, PTY_COLS = 50, 200
_OUTPUT_TAIL_BYTES = 16 * 1024
# F167: a PreToolUse event carries the host's decision and, for a hold, the
# operator's answer. The backend keeps them on the ticket's ``recent_tools``.
DECISION_REASON_CHARS = 300
ANSWER_APPROVED, ANSWER_DENIED, ANSWER_NONE = "approved", "denied", "no answer"
_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class SessionOutcome:
    status: str                      # success | error | cancelled | usage_limit | host_stopped
    result_text: str = ""
    error: Optional[str] = None
    exit_reason: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    files_touched: List[str] = field(default_factory=list)
    permission_denials: List[Dict[str, Any]] = field(default_factory=list)
    session_id: Optional[str] = None
    transcript_path: Optional[str] = None
    effective_cwd: Optional[str] = None
    resets_at: Optional[str] = None   # F083: when a usage_limit pause ends (ISO, host clock)

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
            "resets_at": self.resets_at,
        }


def _slug(text: str, limit: int = 40) -> str:
    return _SLUG_RE.sub("-", text).strip("-")[:limit] or "ticket"


def assert_secret_not_in_args(args: Sequence[str], secret: Optional[str]) -> None:
    """PRD-245 W1: the ticket's own credential rides a 0600 file, never argv —
    argv is world-readable in ``ps`` and lands in the host log."""
    if not secret:
        return
    if any(secret in str(arg) for arg in args):
        raise RuntimeError("the session token reached the command line")


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
            self.adapter = adapter_for(ticket.get("provider"), getattr(cfg, "cli_binaries", None), getattr(cfg, "session_sandbox", None))
        except (UnknownCli, NotServed) as exc:
            self._adapter_error = str(exc)
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.cancel_requested = threading.Event()
        self.stopped_by_host: Optional[str] = None     # F015: why the HOST stopped this session
        self.stopped = threading.Event()
        self.session_started = threading.Event()
        self.ended = threading.Event()
        self.started_at = time.time()
        self.proc: Optional[PtyChild] = None
        self.pgid: Optional[int] = None
        self.effective_cwd: Optional[Path] = None
        self.session_dir: Optional[Path] = None
        self.transcript_path: Optional[str] = None
        self._usage_before: Optional[Dict[str, Any]] = None
        self.reported_session_id: Optional[str] = None
        self.last_assistant_message: Optional[str] = None
        self.files_touched: List[str] = []
        self.denials: List[Dict[str, Any]] = []
        self._allowed_calls = AllowedCalls()            # what the gate allowed: a CLI's own re-ask is answered alike
        # PRD-235 W2 S3: permission questions the operator answers from the Canvas.
        self._pending_asks: Dict[str, threading.Event] = {}
        self._ask_answers: Dict[str, bool] = {}
        self._ask_lock = threading.Lock()
        self.notifications: List[Dict[str, Any]] = []
        self.output_tail: deque = deque(maxlen=_OUTPUT_TAIL_BYTES)
        self.terminal_log: Optional[BoundedLog] = None
        self._contract_injected = False
        self._policy: Optional[PolicyContext] = None
        # PRD-253 Wave P: the mode is chosen before the ticket file is written.
        self.permission_mode: str = MODE_EDITS
        self.plan_turn = False                        # Plan on a CLI with no plan tool: the final message is the plan
        self.plan: Optional[Dict[str, Any]] = None    # {"text", "approved_in_turn"} once the turn produced one
        self._full_bash: Sequence[str] = ()

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
        if event != "PreToolUse":  # F167: a tool call is reported with its decision (_pre_tool_use)
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
            return Reply.with_context(self._turn_context())
        if event == "PreToolUse":
            return self._pre_tool_use(payload)
        if event == "PermissionRequest":
            return self._permission_request(payload)
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

    def _turn_context(self) -> str:
        """What the session is told with its first prompt: the ticket — and, for a CLI
        with no system-prompt flag (Codex, GitHub Copilot), the agent's system prompt
        ahead of it. Design §6.9: there the soul rides this hook, not argv; before,
        only the ticket did, and such a session never saw its rules or its tools."""
        ticket = build_ticket_file(self.ticket, self.default_root, self.plan_turn)
        preset = self.adapter.preset
        if preset.system_prompt_flag:
            return ticket
        return f"{build_system_prompt(self.ticket, preset.label)}\n{ticket}"

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
        verdict, answer, request_id = decision, None, None
        if decision.behavior == "ask" and intent.cls is ToolClass.PLAN:
            verdict, answer, request_id = self._plan_card(tool, intent, decision, plan_text(tool_input))
        elif decision.behavior == "ask":
            verdict, answer, request_id = self._ask_operator(tool, intent.subject, decision.reason)
        # F167: what the host decided, and why, is on the ticket — a call that ran
        # with nobody asked never reads as one the operator approved.
        # ``event_id``: a batch re-posted after a lost response counts once on the ticket.
        self.events.put({**compact_event("PreToolUse", payload, subject=intent.subject), "event_id": uuid.uuid4().hex,
                         "decision": decision.behavior, "reason": decision.reason[:DECISION_REASON_CHARS],
                         **({"answer": answer, "request_id": request_id} if answer else {})})
        if verdict.allow:
            self._allowed_calls.add(tool, tool_input)
            return Reply.allow()
        if verdict.reason != PLAN_WITH_OPERATOR:    # a plan handed to the operator is not a refusal
            self.denials.append({"tool": tool, "reason": verdict.reason, "stage": "PreToolUse",
                                 "input": {k: v for k, v in tool_input.items() if k in ("command", "file_path", "path")}})
        return Reply.deny(verdict.reason)

    def _permission_request(self, payload: Dict[str, Any]) -> Reply:
        """The CLI asked its own permission (``permission_request.py``): denied, or —
        for a preset that rejudges — the gate's verdict on the same call."""
        tool, tool_input = request_of(payload)
        mode = self.adapter.preset.permission_request
        now = decide(self.adapter.tool_intent(tool, dict(tool_input)), self._policy) if (
            mode == PERMISSION_REQUEST_REJUDGE and self._policy is not None) else None
        allow, reason = permission_answer(mode, allowed_before=self._allowed_calls.holds(tool, dict(tool_input)),
                                          behavior=now.behavior if now else None, reason=now.reason if now else "")
        if allow:
            return Reply.allow()
        self.denials.append({"tool": tool or "?", "reason": reason, "stage": "PermissionRequest"})
        return Reply.deny(reason)

    def _plan_card(self, tool: str, intent: Any, decision: Decision, text: str) -> Tuple[Decision, Optional[str], Optional[str]]:
        """The plan a CLI presents in its turn (Claude Code's ExitPlanMode) is a card
        (#845). Approved in time: the session carries on as Edit automatically.
        Unanswered (PRD-253 Wave P): the plan goes to the operator as the Plan card
        instead — the turn ends, the ticket parks on it, and Approve resumes this
        session. Declined: it keeps planning."""
        if self.plan is not None and not self.plan.get("approved_in_turn"):
            return Decision("deny", PLAN_WITH_OPERATOR), None, None     # already with the operator
        save_plan(self._plan_dir, text)
        verdict, answer, request_id = self._ask_operator(tool, intent.subject, decision.reason)
        if verdict.allow:
            self._policy.permission_mode = MODE_EDITS                 # the plan is approved: work as Edit automatically
            self._policy.allowed_bash = self._full_bash
            self.plan = {"text": text, "approved_in_turn": True}
        elif answer == ANSWER_NONE:
            self.plan = {"text": text, "approved_in_turn": False}
            verdict = Decision("deny", PLAN_WITH_OPERATOR)
        return verdict, answer, request_id

    def _ask_operator(self, tool: str, subject: Optional[str], reason: str) -> Tuple[Decision, str, str]:
        """PRD-235 W2 S3: hold this tool call while the operator answers a card on the
        ticket's Canvas. The question travels with the next event flush; the answer
        comes back on that same channel (``resolve_ask``). No answer within
        ``ask_timeout`` seconds → deny, honestly worded. Returns the verdict, the
        answer (``ANSWER_APPROVED`` / ``ANSWER_DENIED`` / ``ANSWER_NONE``) and the
        question's request id (the backend matches a reported approval to it)."""
        request_id = uuid.uuid4().hex
        done = threading.Event()
        with self._ask_lock:
            self._pending_asks[request_id] = done
        self.events.put({
            "event": "PermissionRequest", "at": time.time(), "request_id": request_id,
            "tool_name": tool, "subject": subject, "reason": reason,
            "session_id": self.reported_session_id or self.session_id,
        })
        timeout = hold_seconds(getattr(self.cfg, "ask_timeout", None))    # the hook's own wait bounds it (D4)
        answered = done.wait(timeout)
        with self._ask_lock:
            self._pending_asks.pop(request_id, None)
            approved = self._ask_answers.pop(request_id, None)
        if answered and approved:
            return Decision("allow"), ANSWER_APPROVED, request_id
        if answered:
            return Decision("deny", f"{reason} — denied by the operator"), ANSWER_DENIED, request_id
        return (Decision("deny", f"{reason} — no answer from the operator within {int(timeout)} s"), ANSWER_NONE,
                request_id)

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

    def _session_tools(self) -> Optional[Dict[str, Any]]:
        """The Automatos tools this ticket may call: the names, the URL built from
        the host's OWN backend address, and the per-ticket token — or ``None``
        when the claim offered none (an older backend; the session runs as it did
        before the bridge existed)."""
        names = self.ticket.get("session_tools")
        token = str(self.ticket.get("session_token") or "").strip()
        path = str(self.ticket.get("session_tools_path") or "").strip()
        if not names or not token or not path:
            return None
        base = str(getattr(self.cfg, "url", "") or "").strip().rstrip("/")
        if not base:
            log.warning("task %s: Automatos tools offered but this host has no backend URL", self.task_id)
            return None
        return {"names": [str(n) for n in names], "url": f"{base}{path}", "token": token}

    def _run(self) -> SessionOutcome:
        cwd = self._working_dir()
        if isinstance(cwd, SessionOutcome):
            return cwd
        refused = self._preflight() or self._refused_here(cwd)
        if refused is not None:
            return refused
        preset = self.adapter.preset
        binary = self.adapter.resolve_binary()
        self._choose_mode(preset)
        ticket_path, system_prompt_path = self._write_session_files(preset.label)
        session_tools = self._session_tools()
        self._set_policy(cwd, preset, session_tools)
        ctx, worktree = self._launch_context(cwd, ticket_path, system_prompt_path, session_tools)
        prepared = self.adapter.prepare(ctx)
        self._spawn(preset, ctx, prepared, session_tools, cwd, worktree)
        exit_reason = self._wait_for_turn(preset)
        self._terminate()
        return self._collect(exit_reason, cwd, binary or preset.binary)

    def _working_dir(self) -> "Path | SessionOutcome":
        """1. where the session runs, or why it cannot."""
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
        return cwd

    def _preflight(self) -> Optional[SessionOutcome]:
        """2. which CLI, the user's own binary and login."""
        if self.adapter is None:
            return self._outcome("error", error=self._adapter_error or "no CLI adapter", exit_reason="cli_not_served")
        preset = self.adapter.preset
        if preset.turn_end not in (TURN_END_STOP_HOOK, TURN_END_PROCESS_EXIT):
            return self._outcome("error", error=f"{preset.label}: turn end {preset.turn_end!r} is not implemented by this host",
                                 exit_reason="turn_end_unsupported")
        refusal = self.adapter.preflight()
        if refusal is not None:
            return self._outcome("error", error=refusal.message, exit_reason=refusal.code)
        return None

    def _refused_here(self, cwd: Path) -> Optional[SessionOutcome]:
        """2b. what the CLI can tell about THIS folder before spawn (PRD-253 S1.6)."""
        refusal = self.adapter.refuse_here(cwd)
        return self._outcome("error", error=refusal.message, exit_reason=refusal.code) if refusal else None

    def _choose_mode(self, preset: Any) -> None:
        """The session's permission mode — before the ticket file, which names a Plan
        turn: Plan on a CLI with no plan tool of its own ends with the plan as the
        turn's final message (PRD-253 Wave P)."""
        self.permission_mode = session_mode(getattr(self.cfg, "permission_mode", None), self.ticket.get("permission_mode"),
                                            plan_approved=bool(self.ticket.get("plan_approved")))
        self.plan_turn = self.permission_mode == MODE_PLAN and not preset.plan_stance

    def _write_session_files(self, cli_label: str) -> Tuple[Path, Path]:
        """3. the ticket, the system prompt and the terminal log beside the session."""
        session_dir = self.cfg.sessions_dir / self.task_id
        session_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.session_dir = session_dir
        ticket_path = session_dir / "ticket.md"
        brand_root = session_deliverables_dir(self.default_root, self.task_id) or session_dir  # F332: where it saves
        brand = brand_files_note(write_brand_files(self.ticket, brand_root))
        ticket_path.write_text(build_ticket_file(self.ticket, self.default_root, self.plan_turn) + brand, encoding="utf-8")
        system_prompt_path = session_dir / "system_prompt.md"
        system_prompt_path.write_text(build_system_prompt(self.ticket, cli_label), encoding="utf-8")
        self.terminal_log = BoundedLog(session_dir / TERMINAL_LOG_FILENAME)
        return ticket_path, system_prompt_path

    def _set_policy(self, cwd: Path, preset: Any, session_tools: Optional[Dict[str, Any]]) -> None:
        """The gate for this session: its folders, its allowlist and its permission mode."""
        session_dir = self.session_dir
        # The ticket file NAMES a deliverables folder, so the session has to be
        # able to write there. For a folder-less ticket that folder IS the cwd,
        # but a ticket with its own working directory runs somewhere else — and
        # the instruction would then point at a path the gate refuses, which is
        # an instruction that cannot be followed. Named and writable, or neither.
        deliverables = session_deliverables_dir(self.default_root, str(self.task_id))
        extra_dirs = (session_dir,)
        if deliverables is not None and deliverables.resolve() != cwd.resolve():
            try:
                deliverables.mkdir(parents=True, exist_ok=True, mode=0o755)
            except OSError as exc:
                log.warning("deliverables folder %s not created: %s", deliverables, exc)
            extra_dirs = (*extra_dirs, deliverables)
        self._plan_dir = deliverables or session_dir
        self._full_bash = bash_allowlist_from_config(self.ticket.get("allowed_tools"))
        self._policy = PolicyContext(
            permission_mode=self.permission_mode,
            cwd=cwd,
            allowed_bash=PLAN_BASH_ALLOW if self.permission_mode == MODE_PLAN else self._full_bash,
            plan_tool=bool(preset.plan_stance),
            extra_dirs=extra_dirs,
            session_tools=tuple(session_tools.get("names") or ()) if session_tools else (),
            # F042: the platform's own .env / credential key and this host's state
            # (its token) are out of reach, whatever folder the ticket runs in.
            secret_roots=platform_secret_roots(),
            off_limits=(Path(self.cfg.state_dir).expanduser(),),
        )

    def _launch_context(self, cwd: Path, ticket_path: Path, system_prompt_path: Path,
                        session_tools: Optional[Dict[str, Any]]) -> Tuple[LaunchContext, Optional[str]]:
        """What the adapter launches from, and the worktree the session gets (if any)."""
        session_dir = self.session_dir
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
            session_tools=session_tools, plan_first=self.permission_mode == MODE_PLAN,
            extra_dirs=tuple(self._policy.extra_dirs), hook_socket=self.sock_path,
        )
        return ctx, worktree

    def _spawn(self, preset: Any, ctx: LaunchContext, prepared: Any, session_tools: Optional[Dict[str, Any]],
               cwd: Path, worktree: Optional[str]) -> None:
        """4. spawn the CLI on a pty, with the environment its hooks need."""
        args = self.adapter.launch_args(ctx, prepared)
        assert_args_honour_invariant(args, preset.forbidden_args)
        assert_secret_not_in_args(args, (session_tools or {}).get("token"))
        env = build_session_env(preset, extra={
            "AUTOMATOS_HOST_SOCK": str(self.sock_path),
            "AUTOMATOS_HOST_PID": str(os.getpid()),      # the shim talks to this process only (F234)
            "AUTOMATOS_TASK_ID": self.task_id,
            "AUTOMATOS_CLI": preset.id,
            "AUTOMATOS_HOOK_WAIT_SECONDS": str(HOOK_WAIT_SECONDS),
            "PYTHONPATH": hook_pythonpath(),
            **prepared.env,
        })
        self.proc = spawn_on_pty(args, cwd=cwd, env=env, rows=PTY_ROWS, cols=PTY_COLS)
        self.pgid = self.proc.pid
        threading.Thread(target=self._drain, args=(self.proc,), daemon=True, name=f"pty-drain-{self.task_id}").start()
        log.info("task %s: %s session %s started (pid %s) in %s%s", self.task_id, preset.id, self.session_id, self.proc.pid, cwd,
                 f" worktree={worktree}" if worktree else "")

    def _wait_for_turn(self, preset: Any) -> str:
        """5. wait for the turn's end / exit / cancel / timeout — the preset says how a
        turn ends (``turn_end.py``). A gated CLI proves its gate loaded (SessionStart)
        whatever ends its turn (PRD-253 S0.2)."""
        deadline = self.started_at + self.cfg.session_timeout_seconds
        hook_driven = preset.turn_end == TURN_END_STOP_HOOK
        proof_window = turn_end.startup_timeout(preset, self.cfg.startup_timeout_seconds)
        ended_at: Optional[float] = None
        while True:
            now = time.time()
            if hook_driven and self.stopped.is_set():
                self.ended.wait(STOP_GRACE_SECONDS)
                return turn_end.COMPLETED
            if self.proc.poll() is not None:
                return self._exit_reason(preset)
            if self.cancel_requested.is_set():
                return "cancelled"
            if now > deadline:
                return "timeout"
            if turn_end.is_gated(preset) and not self.session_started.is_set() and now - self.started_at > proof_window:
                return "no_session_start"
            if not hook_driven and self.ended.is_set():
                ended_at = ended_at or now     # a print-mode CLI said goodbye; give it a moment to exit
                if now - ended_at > turn_end.EXIT_GRACE_AFTER_SESSION_END_SECONDS:
                    return self._exit_reason(preset)
            time.sleep(0.25)

    def _exit_reason(self, preset: Any) -> str:
        return turn_end.exit_reason(preset, returncode=self.proc.poll() if self.proc else None,
                                    session_started=self.session_started.is_set(), stopped=self.stopped.is_set())

    def _drain(self, child: PtyChild) -> None:
        try:
            while True:
                try:
                    chunk = child.read(65536)
                except OSError:
                    break
                if not chunk:
                    break
                self.output_tail.extend(chunk)
                if self.terminal_log is not None:
                    self.terminal_log.write(chunk)
        finally:
            child.close()
            if self.terminal_log is not None:
                self.terminal_log.close()

    def _terminate(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if not self.proc.stop(((TERMINATE, KILL_GRACE_SECONDS), (KILL, KILL_GRACE_SECONDS))):
            log.error("task %s: process %s survived being killed", self.task_id, self.proc.pid)

    def request_cancel(self, host_reason: Optional[str] = None) -> None:
        """Stop the session. ``host_reason`` when the host itself is stopping
        (F015): the operator did not cancel the ticket, the machine stopped
        serving it — it goes back to the queue, not to ``cancelled``."""
        if host_reason and not self.cancel_requested.is_set():
            self.stopped_by_host = host_reason
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
        ungated = exit_reason == turn_end.UNGATED_EXIT
        if ungated:
            text = ""                       # PRD-253 S0.2: nothing a run without the gate produced is reported
        elif transcript and transcript.exists():
            text += "".join(f"\n\n[{note}]" for note in self.adapter.record_notes(transcript))
        status, error = turn_end.describe(
            exit_reason, cli=os.path.basename(binary), returncode=self.proc.returncode if self.proc else None,
            tail=bytes(self.output_tail).decode("utf-8", "replace")[-1500:],
            startup_window=turn_end.startup_timeout(preset, self.cfg.startup_timeout_seconds),
            session_timeout=self.cfg.session_timeout_seconds, stopped_by_host=self.stopped_by_host,
        )
        resets = None
        if status == "error":
            # F083: the CLI's plan window closed. That is a pause — the ticket goes
            # back to the queue and the host stops claiming for this CLI — not a
            # failed attempt, and never "check your key".
            said = bytes(self.output_tail).decode("utf-8", "replace")[-4000:] + "\n" + (text or "")
            if usage_limit.is_usage_limit(said):
                until, known = usage_limit.pause(said, _local_now())
                status, error, resets = "usage_limit", usage_limit.describe(self.cli, until, known), until.isoformat()
        files = [] if ungated else [*self.files_touched, *self._land_deliverables(cwd)]
        files = self._report_plan(text, files) if status == "success" else files
        self._shred_session_credentials()
        outcome = self._outcome(status, result_text=text, error=error, exit_reason=exit_reason, usage=usage, cwd=cwd,
                                files_touched=files)
        outcome.resets_at = resets
        return outcome

    def _report_plan(self, text: str, files: List[str]) -> List[str]:
        """PRD-253 Wave P: a plan this turn produced reaches the backend before the
        turn's result — an event in the final flush, which files it as the Plan card
        (or records an approval given in the turn) — and lands as plan.md beside the
        deliverables. A Plan turn's plan is the one it presented, else its final
        message — on Claude Code too, when the turn ends without ExitPlanMode."""
        if self.plan is None and self.permission_mode == MODE_PLAN and text.strip():
            self.plan = {"text": text.strip(), "approved_in_turn": False}
        if self.plan is None:
            return files
        saved = save_plan(self._plan_dir, self.plan["text"])
        self.events.put({"event": PLAN_EVENT, "at": time.time(), **self.plan})
        return [*files, str(saved)] if saved is not None and str(saved) not in files else files

    def _shred_session_credentials(self) -> None:
        """The turn is over: delete the files holding this ticket's token.

        The backend kills the credential on the row, but the PLAINTEXT is what
        an attacker copies, and it sat in the session folder indefinitely —
        every finished ticket leaving one more readable token behind. Deleting
        it costs nothing: the config is rewritten from the claim on every spawn,
        so a resume gets a fresh file with a fresh token.
        """
        if self.session_dir is None:
            return
        for name in CREDENTIAL_SESSION_FILES:
            try:
                (self.session_dir / name).unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                log.warning("could not remove %s for task %s: %s", name, self.task_id, exc)

    def _land_deliverables(self, cwd: Path) -> List[str]:
        """PRD-245 S0.7: what the session wrote in its own folder is copied into the
        ticket's deliverables folder, where the backend registers it as today.
        Nothing to do without a default root."""
        dest = session_deliverables_dir(self.default_root, self.task_id)
        if dest is None or self.session_dir is None:
            return []
        found = session_deliverables(self.files_touched, self.session_dir, self.effective_cwd or cwd)
        return land_session_deliverables(found, self.session_dir, dest)

    def _outcome(self, status: str, *, result_text: str = "", error: Optional[str] = None,
                 exit_reason: str = "", usage: Optional[Dict[str, Any]] = None, cwd: Optional[Path] = None,
                 files_touched: Optional[List[str]] = None) -> SessionOutcome:
        return SessionOutcome(
            status=status, result_text=result_text, error=error, exit_reason=exit_reason,
            usage=usage or {}, files_touched=list(self.files_touched if files_touched is None else files_touched),
            permission_denials=list(self.denials),
            session_id=self.reported_session_id or self.session_id or None,
            transcript_path=self.transcript_path,
            effective_cwd=str(self.effective_cwd or cwd) if (self.effective_cwd or cwd) else None,
        )


def _local_now():
    from datetime import datetime

    return datetime.now().astimezone()


def host_capabilities(cfg: HostConfig) -> Dict[str, Any]:
    """What the host announces: every CLI the registry knows — present, version,
    served (preflight would pass) and, when not, why; ``providers`` is the served
    subset, which the backend's claim filter reads (design §8.2). Never a credential."""
    import platform
    import sys

    binaries = getattr(cfg, "cli_binaries", None) or {}
    served = adapters(binaries, getattr(cfg, "session_sandbox", None))
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
