"""GitHub Copilot CLI — the hooks tier, run as ``copilot -p`` (PRD-253 W1–W2).

Copilot sends Claude's payload for its PascalCase hook events, so translation is
identity and the shim is reused verbatim; the camelCase-only ``permissionRequest``
and ``notification`` name themselves on the hook's command line. This adapter's
weight is where things live and what a session never gets:

* a config home per agent per host (``copilot_home.py``): a whitelisted
  ``config.json``, our hooks file and nothing else — the operator's ``~/.copilot``
  is never written, no token is ever copied;
* the login read from the account POINTER (the OS credential store holds the
  token), else Copilot's own ``gh`` fallback — never a credential through this
  host (D6);
* no allow flag, ever (D3): in ``-p``, a call the gate did not allow is refused by
  Copilot itself; held-event hooks outlast the shim's wait, because a timed-out
  Copilot hook FAILS OPEN (D4);
* the session record (``copilot_record.py``): usage in tokens and the plan's own
  AI credits, never a price;
* the Automatos tools as an HTTP MCP server from a 0600 file (D7), GitHub's own
  MCP server off (``--disable-builtin-mcps``);
* Copilot's command sandbox under the session when the host sandboxes (S2.2, O3).
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..sandbox import SessionSandbox
from .base import LaunchContext, Prepared, PresetAdapter, Refusal, Reply, ToolClass, ToolIntent, hook_command
from .claude import build_mcp_config
from .codex import patch_paths
from .copilot_home import (
    CONFIG_FILENAME, HOME_ENV, SETTINGS_FILENAME, account_of, build_home, gh_login, holds_plaintext_token,
    hooks_disabled_in, hooks_document, managed_hooks_only, operator_home, read_json, seeded_config, seeded_settings,
    write_private,
)
from .copilot_record import events_path, last_message, mcp_server_blocked, read_events_usage
from .copilot_sandbox import sandbox_settings, unavailable_reason

MCP_SERVER_NAME = "automatos"
MCP_CONFIG_FILENAME = "mcp.json"          # the session dir's credential file, shredded at turn end
DEFAULT_HOST = "https://github.com"
LOGIN_ROUTE_COPILOT = "copilot"           # Copilot's own login: the token in the OS credential store
LOGIN_ROUTE_GH = "gh"                     # Copilot's gh fallback: it runs `gh auth token` itself
# Before 1.0.57 a crashing preToolUse hook ALLOWED the call; 1.0.70 made exit 2 a deny.
VERSION_FLOOR: Tuple[int, int, int] = (1, 0, 70)
_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
ENV_TOKENS = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")
OTEL_HEADERS_ENV = "OTEL_EXPORTER_OTLP_HEADERS"
SECRET_ENV_FLAG = "--secret-env-vars"
MCP_BLOCKED_NOTE = ("the Automatos tools did not load in this session — Copilot reported the server as {status}; "
                    "an organisation's MCP policy that allows only registry servers blocks it")
# How a Copilot MCP tool may be named in a Claude-format payload. Copilot keeps the
# server and the tool as two fields natively and builds the joined name in its
# native layer, so every plausible spelling maps to the same tool (verify at build).
_MCP_TOOL_RE = re.compile(
    rf"^(?:mcp__{MCP_SERVER_NAME}__|{MCP_SERVER_NAME}(?:-|/|__|\.))(?P<tool>[A-Za-z0-9_]+)$"
    rf"|^{MCP_SERVER_NAME}\((?P<paren>[A-Za-z0-9_]+)\)$"
)

# What a Copilot tool call does (S1.4). PascalCase payloads carry Claude's tool
# names with Copilot's input keys; native lowercase names map the same way.
_TOOL_CLASSES: Mapping[str, ToolClass] = {
    **{name: ToolClass.SHELL for name in ("Bash", "bash", "powershell")},
    **{name: ToolClass.FILE_READ for name in ("Read", "view", "Grep", "grep", "rg", "Glob", "glob")},
    **{name: ToolClass.FILE_WRITE for name in ("Write", "create")},
    **{name: ToolClass.WEB for name in ("WebFetch", "WebSearch", "web_fetch", "web_search")},
    # a plan list, a question (not offered under --no-ask-user), reading or stopping a
    # shell the gate already judged, and Copilot's own bookkeeping touch nothing.
    **{name: ToolClass.BENIGN for name in (
        "TodoWrite", "update_todo", "AskUserQuestion", "ask_user", "read_bash", "stop_bash", "list_bash",
        "read_powershell", "stop_powershell", "list_powershell", "report_intent", "task_complete",
        "fetch_copilot_cli_documentation")},
}
EDIT_TOOLS = frozenset({"Edit", "edit", "str_replace_editor", "apply_patch", "MultiEdit"})
_GLOB_KEYS: Mapping[str, str] = {"Grep": "glob", "grep": "glob", "rg": "glob", "Glob": "pattern", "glob": "pattern"}
_PATH_KEYS = ("path", "file_path", "notebook_path")


def _paths(ti: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(str(ti[k]) for k in _PATH_KEYS if ti.get(k))


def parse_version(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    match = _VERSION_RE.search(text or "")
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def mcp_config(session_tools: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Copilot's ``--additional-mcp-config`` document: Claude's HTTP server entry
    (the token literal in a header, D7) with every one of its tools enabled."""
    document = build_mcp_config(session_tools)
    if document is None:
        return None
    server = document["mcpServers"][MCP_SERVER_NAME]
    return {"mcpServers": {MCP_SERVER_NAME: {**server, "tools": ["*"]}}}


class CopilotAdapter(PresetAdapter):
    def __init__(self, preset, binary: Optional[str] = None, home: Optional[Path] = None,
                 sandbox: Optional[SessionSandbox] = None, run: Callable[..., Any] = subprocess.run) -> None:
        super().__init__(preset, binary, sandbox)
        self._home = home          # the operator's home (tests give a fake one); None = Path.home()
        self._run = run
        self._agent_home: Optional[Path] = None   # set by prepare(): the record lives there
        self._login: Optional[Tuple[Optional[str], Optional[str], Optional[Refusal]]] = None
        self._version: Optional[str] = None

    # ── identity ────────────────────────────────────────────────────────────
    def operator_home(self) -> Path:
        return operator_home(self._home)

    def version(self) -> Optional[str]:
        if self._version is None:
            self._version = super().version()
        return self._version

    def login(self) -> Tuple[Optional[str], Optional[str], Optional[Refusal]]:
        """``(login, route, refusal)``: who the sessions run as and how — Copilot's own
        login (the operator's config names the account and holds no token) or ``gh``
        (Copilot runs ``gh auth token`` itself). Never reads a credential."""
        if self._login is None:
            self._login = self._probe_login()
        return self._login

    def _probe_login(self) -> Tuple[Optional[str], Optional[str], Optional[Refusal]]:
        config = read_json(self.operator_home() / CONFIG_FILENAME)
        settings = read_json(self.operator_home() / SETTINGS_FILENAME)
        account = account_of(config)
        plaintext = holds_plaintext_token(config, settings)
        if account and not plaintext:
            return account, LOGIN_ROUTE_COPILOT, None
        via_gh = gh_login(account or DEFAULT_HOST, self._run)
        if via_gh:
            return via_gh, LOGIN_ROUTE_GH, None
        return None, None, self._login_refusal(plaintext)

    def _login_refusal(self, plaintext: bool) -> Refusal:
        if plaintext:
            return Refusal("copilot_plaintext_token",
                           "GitHub Copilot's login on this machine is a token stored in plain text, and `gh` is not "
                           "logged in. Turn on the OS credential store (on Linux or WSL2, a Secret Service such as "
                           "gnome-keyring) and run `copilot login` again — Automatos never copies a credential.")
        if any(os.environ.get(name) for name in ENV_TOKENS):
            return Refusal("copilot_not_logged_in",
                           "GitHub Copilot is logged in here only through a token in the environment, and sessions "
                           "never carry tokens. Run `copilot login` (or `gh auth login` with an account that has a "
                           "Copilot seat), then retry.")
        probe = self.preset.auth_probe
        return Refusal(probe.code, probe.refusal)

    def logged_in(self) -> Optional[Refusal]:
        return self.login()[2]

    def _too_old(self) -> Optional[Refusal]:
        found = parse_version(self.version())
        if found is not None and found >= VERSION_FLOOR:
            return None
        floor = ".".join(str(part) for part in VERSION_FLOOR)
        said = self.version() or "an unreadable version"
        return Refusal("copilot_too_old", f"GitHub Copilot CLI {said} is older than {floor}, the first version whose "
                                          "hooks fail closed. Run `copilot update`, then retry.")

    def _managed_hooks_only(self) -> Optional[Refusal]:
        policy = managed_hooks_only()
        if policy is None:
            return None
        return Refusal("copilot_managed_hooks_only",
                       f"Your organisation's Copilot policy ({policy}) runs only administrator-deployed hooks, so "
                       "Automatos' gate cannot load. Copilot sessions need user hooks: ask your Copilot "
                       "administrator to allow them.")

    def preflight(self) -> Optional[Refusal]:
        """Binary, version floor, organisation policy, login, sandbox — in that order."""
        if not self.resolve_binary():
            return Refusal("copilot_missing", self.preset.install_hint or "GitHub Copilot CLI is not installed")
        refusal = self._too_old() or self._managed_hooks_only() or self.logged_in()
        if refusal is not None:
            return refusal
        reason = unavailable_reason(self.sandbox)
        return Refusal("copilot_sandbox_unavailable", reason) if reason else None

    def refuse_here(self, cwd: Path) -> Optional[Refusal]:
        settings = hooks_disabled_in(cwd)
        if settings is None:
            return None
        return Refusal("copilot_hooks_disabled_here",
                       f"{settings} switches every Copilot hook off for this repository, so Automatos' gate could "
                       "not load. Remove `disableAllHooks` there, or give the agent another folder.")

    def detect(self) -> Dict[str, Any]:
        out = super().detect()
        login, route, _ = self.login() if out["path"] else (None, None, None)
        return {**out, "login": login, "login_route": route}

    # ── launch ──────────────────────────────────────────────────────────────
    def agent_home(self, ctx: LaunchContext) -> Path:
        return self.config_home_for(ctx.state_dir or ctx.session_dir.parent.parent, ctx.agent_id)

    def config_home_for(self, state_dir: Path, agent_id: Any) -> Path:
        return state_dir / "agents" / (str(agent_id) if agent_id not in (None, "") else "shared") / ".copilot"

    def use_config_home(self, home: Path) -> None:
        self._agent_home = home

    def prepare(self, ctx: LaunchContext) -> Prepared:
        """The agent's home rebuilt, the Automatos tools file, the sandbox settings,
        and the folders the gate grants — named to Copilot's own path check too."""
        home = self.agent_home(ctx)
        operator = self.operator_home()
        config, settings = read_json(operator / CONFIG_FILENAME), read_json(operator / SETTINGS_FILENAME)
        build_home(home, config=seeded_config(config),
                   settings=seeded_settings(settings, config, self._sandbox_block(ctx)),
                   hooks=hooks_document(self.preset.hook_events, self.preset.hook_timeout, hook_command()))
        self._agent_home = home
        args: List[str] = []
        document = mcp_config(ctx.session_tools)
        if document is not None and self.preset.mcp_config_flag:
            path = write_private(ctx.session_dir / MCP_CONFIG_FILENAME, document)
            args += [self.preset.mcp_config_flag, f"@{path}"]
        for folder in ctx.extra_dirs:
            if Path(folder) != ctx.session_dir and self.preset.add_dir_flag:
                args += [self.preset.add_dir_flag, str(folder)]
        if os.environ.get(OTEL_HEADERS_ENV):
            args += [SECRET_ENV_FLAG, OTEL_HEADERS_ENV]     # session shells never see its value
        return Prepared(env={HOME_ENV: str(home)}, args=args)

    def _sandbox_block(self, ctx: LaunchContext) -> Optional[Dict[str, Any]]:
        """Copilot's command sandbox under the session (S2.2): a saved
        ``sandbox.enabled`` turns it on; None when the host does not sandbox."""
        if self.sandbox is None or not self.sandbox.enabled:
            return None
        from ..policy import platform_secret_roots  # policy imports the adapters' base
        return sandbox_settings(self.sandbox, writable=(ctx.session_dir, *ctx.extra_dirs),
                                secret_roots=platform_secret_roots(),
                                off_limits=(ctx.state_dir,) if ctx.state_dir else (),
                                sockets=(ctx.hook_socket,) if ctx.hook_socket else ())

    # ── the bus ─────────────────────────────────────────────────────────────
    def render_response(self, event: str, reply: Reply) -> Optional[Dict[str, Any]]:
        """Claude's wire shape, which Copilot reads for Claude-format hooks — plus an
        explicit allow for a permission request the gate rejudged (S1.4)."""
        if event == "PermissionRequest" and reply.kind == "allow":
            return {"hookSpecificOutput": {"hookEventName": "PermissionRequest", "decision": {"behavior": "allow"}}}
        return super().render_response(event, reply)

    # ── tools ───────────────────────────────────────────────────────────────
    def tool_intent(self, tool_name: str, tool_input: Mapping[str, Any]) -> ToolIntent:
        ti = tool_input if isinstance(tool_input, Mapping) else {}
        name = str(tool_name or "")
        platform = _MCP_TOOL_RE.match(name)
        if platform:
            return ToolIntent(tool=name, cls=ToolClass.PLATFORM, command=platform.group("tool") or platform.group("paren"))
        if name in EDIT_TOOLS:
            return self._edit_intent(name, ti, tool_input)
        cls = _TOOL_CLASSES.get(name, ToolClass.UNKNOWN)
        if cls is ToolClass.SHELL:
            return ToolIntent(tool=name, cls=cls, command=str(ti.get("command") or ""))
        if cls in (ToolClass.FILE_READ, ToolClass.FILE_WRITE):
            glob_key = _GLOB_KEYS.get(name)
            globs = (str(ti[glob_key]),) if glob_key and ti.get(glob_key) else ()
            return ToolIntent(tool=name, cls=cls, paths=_paths(ti), globs=globs)
        if cls is ToolClass.WEB:
            return ToolIntent(tool=name, cls=cls, paths=tuple(str(ti[k]) for k in ("url", "query") if ti.get(k)))
        return ToolIntent(tool=name, cls=cls)

    @staticmethod
    def _edit_intent(name: str, ti: Mapping[str, Any], raw: Any) -> ToolIntent:
        """An edit names its file, lists them (``apply_patch`` actions) or carries a
        patch whose headers name them; the editor's ``view`` command only reads. A
        write that names none of these is refused by the gate (PRD-253 S0.1)."""
        if str(ti.get("command") or "") == "view":
            return ToolIntent(tool=name, cls=ToolClass.FILE_READ, paths=_paths(ti))
        actions = ti.get("actions") if isinstance(ti.get("actions"), list) else []
        listed = tuple(str(a["path"]) for a in actions if isinstance(a, Mapping) and a.get("path"))
        patch = raw if isinstance(raw, str) else str(ti.get("input") or ti.get("patch") or "")
        return ToolIntent(tool=name, cls=ToolClass.FILE_WRITE, paths=_paths(ti) or listed or patch_paths(patch))

    # ── the record ──────────────────────────────────────────────────────────
    def transcript_path(self, cwd: str, session_id: str, home: Optional[Path] = None) -> Optional[Path]:
        """``<COPILOT_HOME>/session-state/<id>/events.jsonl`` — the agent's home once
        prepared; without one (the Canvas terminal), the operator's own."""
        root = self._agent_home or ((home / ".copilot") if home is not None else self.operator_home())
        return events_path(root, session_id) if session_id else None

    def read_usage(self, transcript: Path) -> Dict[str, Any]:
        return read_events_usage(transcript)

    def last_text(self, transcript: Path) -> Optional[str]:
        return last_message(transcript)

    def record_notes(self, transcript: Path) -> List[str]:
        """S2.1: the sentence for the ticket when the Automatos server did not load."""
        status = mcp_server_blocked(transcript, MCP_SERVER_NAME)
        return [MCP_BLOCKED_NOTE.format(status=status)] if status else []


__all__ = ["CopilotAdapter", "MCP_BLOCKED_NOTE", "VERSION_FLOOR", "mcp_config", "parse_version"]
