"""The adapter contract (CLI adapter design §4.2) — seven methods, all defaulted.

``PresetAdapter`` implements everything from the preset alone; a CLI subclasses
only for what its preset cannot express (where its config home lives, a
translating ``normalize_event``/``render_response``, a different transcript).

Above this seam nothing knows which CLI runs: ``session.py`` asks the adapter to
preflight, prepare, build argv, translate each hook in and each reply out, say
what a tool call *does* (``ToolIntent``, what the policy gates on), and read the
usage at the end.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..env import resolve_binary
from ..presets import (
    PROMPT_FLAG, PROMPT_POSITIONAL, PROMPT_TYPE_INTO_TUI, CliPreset,
)


class ToolClass(Enum):
    FILE_READ = "file_read"
    FILE_WRITE = "file_write"
    SHELL = "shell"
    WEB = "web"
    BENIGN = "benign"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ToolIntent:
    """What a tool call DOES, not what it is called — the policy's whole input."""
    tool: str
    cls: ToolClass
    paths: Tuple[str, ...] = ()       # everything the call would touch
    command: Optional[str] = None     # the shell command, if any

    @property
    def subject(self) -> Optional[str]:
        """The one thing the call is about, for the ticket's live log — never the
        whole tool input."""
        value = self.command or (self.paths[0] if self.paths else None)
        return str(value)[:200] if value else None


@dataclass(frozen=True)
class Reply:
    """The bus's answer to a hook, before the adapter renders it for the CLI."""
    kind: str                         # allow | deny | block | context | none
    reason: str = ""
    context: str = ""

    @staticmethod
    def allow() -> "Reply":
        return Reply("allow")

    @staticmethod
    def deny(reason: str) -> "Reply":
        return Reply("deny", reason=reason)

    @staticmethod
    def with_context(text: str) -> "Reply":
        return Reply("context", context=text)

    @staticmethod
    def none() -> "Reply":
        return Reply("none")


@dataclass
class LaunchContext:
    """Everything a launch needs, gathered by the session before spawn."""
    cwd: Path
    session_dir: Path
    ticket_path: Path
    system_prompt_path: Path
    task_id: str
    session_id: str
    resume_session_id: Optional[str] = None
    model: Optional[str] = None
    worktree_name: Optional[str] = None
    agent_id: Optional[str] = None
    state_dir: Optional[Path] = None  # the host's state dir — per-agent config homes live under it
    hook_command: str = ""            # the shim invocation the CLI's hook config points at


@dataclass(frozen=True)
class Refusal:
    """Why a session must not start: the ticket's ``exit_reason`` and the sentence."""
    code: str
    message: str


@dataclass
class Prepared:
    """What ``prepare()`` produced: environment additions (a config home, …) and
    argv fragments (a settings file, a trust bypass, …) the launch must carry."""
    env: Dict[str, str] = field(default_factory=dict)
    args: List[str] = field(default_factory=list)


class PresetAdapter:
    """The default adapter: everything the preset can say, said once here."""

    preset: CliPreset

    def __init__(self, preset: CliPreset, binary: Optional[str] = None) -> None:
        self.preset = preset
        self._binary = binary  # an explicit --cli-binary path; None = the operator's PATH

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def id(self) -> str:
        return self.preset.id

    def resolve_binary(self) -> Optional[str]:
        """The runnable binary, or None. An explicit ``--cli-binary`` path is checked
        like a PATH lookup — a path that does not run is "missing", never "present"."""
        if self._binary:
            return resolve_binary(self._binary)
        return resolve_binary(self.preset.binary)

    def version(self) -> Optional[str]:
        binary = self.resolve_binary()
        if not binary:
            return None
        try:
            out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=15, check=False)
            return (out.stdout or out.stderr or "").strip().split("\n")[0][:80] or None
        except (OSError, subprocess.SubprocessError):
            return None

    def logged_in(self) -> Optional[Refusal]:
        """None = logged in with the operator's own plan; else why not. The preset's
        probe, interpreted by the adapter. The base knows no probe: a CLI without
        one is never served (nothing to check = not safe to spend on)."""
        probe = self.preset.auth_probe
        if probe is None:
            return Refusal(f"{self.preset.id}_no_login_check", f"{self.preset.label}: no login check is implemented for this CLI")
        return Refusal(probe.code, probe.refusal)

    def preflight(self) -> Optional[Refusal]:
        """None = ready. Otherwise the code + operator-facing sentence that land on
        the ticket: binary missing, or logged in the wrong way."""
        if not self.resolve_binary():
            return Refusal(f"{self.preset.id}_missing",
                           self.preset.install_hint or f"{self.preset.label} is not installed on this machine")
        return self.logged_in()

    def detect(self) -> Dict[str, Any]:
        """What the host announces for this CLI: present, version, served (= preflight
        would pass) and, when not, why. Never a credential."""
        binary = self.resolve_binary()
        refusal = self.preflight()
        return {
            "path": binary,
            "version": self.version() if binary else None,
            "served": refusal is None,
            "reason": refusal.message if refusal else None,
            "tier": self.preset.tier,
        }

    # ── launch ──────────────────────────────────────────────────────────────
    def prepare(self, ctx: LaunchContext) -> Prepared:
        """Write what the session needs before spawn. The base writes nothing: a
        CLI with hooks must say where its hook config lives."""
        return Prepared()

    def launch_args(self, ctx: LaunchContext, prepared: Prepared) -> List[str]:
        """Full argv from the preset. The positional prompt is a short pointer —
        nothing sensitive in argv."""
        p = self.preset
        binary = self.resolve_binary() or p.binary
        args: List[str] = [binary]
        resuming = bool(ctx.resume_session_id)
        if resuming:
            if p.resume_subcommand:
                args += [p.resume_subcommand, ctx.resume_session_id]
            elif p.resume_flag:
                args += [p.resume_flag, ctx.resume_session_id]
        elif p.session_id_flag:
            args += [p.session_id_flag, ctx.session_id]
        if p.cwd_flag:
            args += [p.cwd_flag, str(ctx.cwd)]
        args += list(p.ungated_stance)
        if p.system_prompt_flag:
            args += [p.system_prompt_flag, str(ctx.system_prompt_path)]
        args += list(prepared.args)
        args += list(p.required_args)
        if p.add_dir_flag:
            args += [p.add_dir_flag, str(ctx.session_dir)]
        if p.name_flag:
            args += [p.name_flag, f"automatos #{ctx.task_id}"]
        if ctx.model and p.model_flag:
            args += [p.model_flag, str(ctx.model)]
        if ctx.worktree_name and p.worktree_args and not (resuming and p.worktree_excludes_resume):
            args += list(p.worktree_args)
            if p.worktree_takes_name:
                args.append(ctx.worktree_name)   # Claude names the worktree; Codex manages its own
        pointer = f"Work the Automatos ticket described in {ctx.ticket_path}. Read it first."
        if p.initial_prompt == PROMPT_POSITIONAL:
            args.append(pointer)
        elif p.initial_prompt == PROMPT_FLAG:
            args += [p.initial_prompt_flag or "", pointer]
        elif p.initial_prompt == PROMPT_TYPE_INTO_TUI:
            pass   # the pointer is typed after boot by whoever drives the TUI (not this host, today)
        return args

    def terminal_args(self, binary: str, *, session_id: str, resume: bool,
                      system_prompt_path: Optional[Path], model: Optional[str], task_id: Optional[str]) -> List[str]:
        """The interactive command for a Runtime Canvas terminal: exactly what the
        operator gets by typing the CLI in that folder, plus the agent's soul when
        the CLI can take one on the command line. Nothing that assumes nobody is
        at the keyboard."""
        p = self.preset
        args: List[str] = [binary]
        if resume:
            if p.resume_subcommand:
                args += [p.resume_subcommand, session_id]
            elif p.resume_flag:
                args += [p.resume_flag, session_id]
        elif p.session_id_flag:
            args += [p.session_id_flag, session_id]
        if system_prompt_path is not None and p.system_prompt_flag:
            args += [p.system_prompt_flag, str(system_prompt_path)]
        if task_id and p.name_flag:
            args += [p.name_flag, f"automatos #{task_id}"]
        if model and p.model_flag:
            args += [p.model_flag, str(model)]
        return args

    # ── the bus ─────────────────────────────────────────────────────────────
    def normalize_event(self, raw: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        """Raw hook payload → the bus shape. Identity when the CLI is Claude-shaped;
        ``None`` = an event this CLI emits that the bus does not model (dropped)."""
        event = raw.get("hook_event_name")
        if isinstance(event, str) and event and event not in self.preset.hook_events:
            return None   # this CLI emits it, the bus does not model it
        return dict(raw)

    def render_response(self, event: str, reply: Reply) -> Optional[Dict[str, Any]]:
        """The bus's reply → what THIS CLI reads on the hook's stdout. The base
        renders the Claude wire shape (Claude and Codex). ``None`` = write nothing."""
        if reply.kind == "none":
            return None
        if reply.kind == "context" and event == "UserPromptSubmit":
            return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": reply.context}}
        if event == "PermissionRequest":
            if reply.kind == "deny":
                return {"hookSpecificOutput": {"hookEventName": "PermissionRequest",
                                               "decision": {"behavior": "deny", "message": reply.reason}}}
            return None
        if event == "PreToolUse":
            if reply.kind == "allow":
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}}
            if reply.kind == "deny":
                return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                               "permissionDecisionReason": reply.reason}}
        if reply.kind == "block":
            return {"decision": "block", "reason": reply.reason}
        return None

    def offline_deny(self, event: str, reason: str) -> Optional[Dict[str, Any]]:
        """What the shim writes on its own when the host is unreachable: a deny in
        this CLI's shape for the held events — never a prompt nobody watches."""
        if event not in self.preset.hold_events:
            return None
        return self.render_response(event, Reply.deny(reason))

    # ── tools ───────────────────────────────────────────────────────────────
    def tool_intent(self, tool_name: str, tool_input: Mapping[str, Any]) -> ToolIntent:
        """The base knows no tools: everything is UNKNOWN (denied by the policy)."""
        return ToolIntent(tool=tool_name, cls=ToolClass.UNKNOWN)

    # ── the record ──────────────────────────────────────────────────────────
    def read_usage(self, transcript: Path) -> Dict[str, Any]:
        """Normalized token counts for ``llm_usage``; never a price."""
        return {}

    def last_text(self, transcript: Path) -> Optional[str]:
        return None

    def transcript_path(self, cwd: str, session_id: str, home: Optional[Path] = None) -> Optional[Path]:
        return None

    def transcript_exists(self, cwd: Path, session_id: str, home: Optional[Path] = None) -> bool:
        p = self.transcript_path(str(cwd), session_id, home)
        return bool(p and p.exists())

    def record_trust(self, cwd: Path, home: Optional[Path] = None) -> bool:
        """Record the operator's registration decision where the CLI reads it, if
        the CLI has such a dialog. Backup-first, minimal. False = nothing to do."""
        return False


def hook_command(python: Optional[str] = None) -> str:
    """The hook command every preset's config points at: this interpreter, this
    package's shim module."""
    import json
    import sys
    return f"{json.dumps(python or sys.executable)} -m automatos_cli_host.hook_shim"


__all__ = [
    "LaunchContext", "Prepared", "PresetAdapter", "Refusal", "Reply", "ToolClass", "ToolIntent", "hook_command",
]
