"""GitHub Copilot's config home, one per agent per host (PRD-253 S1.2, D2).

``COPILOT_HOME`` = ``<state>/agents/<agent>/.copilot``. The session index lives in
it (``session-state/``), so it is per agent, never per ticket — a home per ticket
would break ``--resume`` — and the operator's own ``~/.copilot`` is never written.
Rebuilt on every spawn, in the formats of Copilot CLI 1.0.91:

* ``config.json`` — Copilot's STATE: which account is logged in (the operator's
  ``loggedInUsers``/``lastLoggedInUser`` pointer — never a token: with
  ``storeTokenPlaintext`` the file can hold one) and no trusted folder;
* ``settings.json`` — Copilot's SETTINGS since they moved out of config.json:
  fixed values (no memory, no auto-update, no silent model swap, no ``ask_user``,
  hooks on) plus the operator's own co-author and proxy choices, and the sandbox
  block when the host sandboxes (``copilot_sandbox.py``);
* ``hooks/automatos.json`` — the only file in Copilot's user hooks directory, in
  Claude's own hook format, which Copilot reads unmodified and answers with
  Claude-shaped snake_case payloads (1.0.6, 1.0.21, 1.0.62) — so the shim and the
  gate read Copilot exactly as they read Claude Code;
* no ``mcp-config.json`` — the operator's MCP servers never ride along (PRD-245 D9).

Also here: how the operator is logged in (never reading a credential), and the
organisation and repository settings that would switch our hooks off.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

CONFIG_FILENAME = "config.json"
SETTINGS_FILENAME = "settings.json"
HOOKS_DIRNAME = "hooks"
HOOKS_FILENAME = "automatos.json"
OPERATOR_MCP_FILENAME = "mcp-config.json"
HOME_ENV = "COPILOT_HOME"

# config.json: WHICH account — host and login, never a token.
ACCOUNT_KEYS = ("loggedInUsers", "lastLoggedInUser")
# settings.json: the operator's own choices a session keeps.
OPERATOR_SETTINGS = ("includeCoAuthoredBy", "proxyUrl", "proxyKerberosServicePrincipal")
FIXED_SETTINGS: Dict[str, Any] = {
    "banner": "never",
    "autoUpdate": False,
    "memory": False,
    "showTipsOnStartup": False,
    "ide": {"autoConnect": False},
    "continueOnAutoMode": False,   # a rate limit never silently swaps the agent's model
    "askUser": False,              # questions go through ask_human (PRD-245 W2), as with --no-ask-user
    "storeTokenPlaintext": False,
    "disableAllHooks": False,      # our hooks ARE the gate
}
PLAINTEXT_FLAG = "storeTokenPlaintext"
# Where a plaintext login token may sit in the operator's config.json. Checked for
# PRESENCE only — the value is never read into anything (verify at build, S1.2).
PLAINTEXT_TOKEN_KEYS = ("copilotTokens", "copilot_tokens", "tokens", "authTokens", "githubTokens")
EXPERIMENTAL_KEY = "experimental"
SANDBOX_KEY = "sandbox"

# Events whose Claude-format entry carries a tool matcher.
TOOL_EVENTS = frozenset({"PreToolUse", "PostToolUse", "PermissionRequest"})
# Events that also name themselves on the command line (PRD-253 S0.4): should
# Copilot send their payload without ``hook_event_name``, the shim still knows them.
ARGV_NAMED_EVENTS = frozenset({"PermissionRequest", "Notification"})

# Organisation policy: a managed hooks-only policy loads no hook of ours (O4).
MANAGED_POLICY_DIRS: Tuple[Path, ...] = (
    Path("/etc/github-copilot/policy.d"),
    Path("/Library/Application Support/GitHubCopilot/policy.d"),
)
MANAGED_HOOKS_ONLY_KEY = "allowManagedHooksOnly"
# A repository that switches every hook off for anyone working in it.
REPO_SETTINGS = (Path(".github") / "copilot" / "settings.json", Path(".github") / "copilot" / "settings.local.json")
DISABLE_HOOKS_KEY = "disableAllHooks"

_GH_LOGIN_RE = re.compile(r"Logged in to (?P<host>\S+) (?:account|as) (?P<login>[A-Za-z0-9-]+)")
GH_STATUS_TIMEOUT_SECONDS = 10


# ── files ────────────────────────────────────────────────────────────────────

def read_json(path: Path) -> Dict[str, Any]:
    """A JSON object from ``path``; ``{}`` when it is missing, unreadable or not an object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_private(path: Path, document: Mapping[str, Any]) -> Path:
    """Write ``document`` as JSON, created mode 0600 — never chmod'd afterwards, so
    no window leaves it readable and no older file's wider mode survives."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")
    return path


# ── the operator ─────────────────────────────────────────────────────────────

def operator_home(home: Optional[Path] = None, environ: Optional[Mapping[str, str]] = None) -> Path:
    """The operator's own Copilot home: their ``COPILOT_HOME``, else ``~/.copilot``."""
    env = os.environ if environ is None else environ
    raw = str(env.get(HOME_ENV) or "").strip()
    return Path(raw).expanduser() if raw else (home or Path.home()) / ".copilot"


def account_of(config: Mapping[str, Any]) -> Optional[str]:
    """``login@host`` of the operator's last login, from the account POINTER — never
    a token. Copilot keeps the pointer's shape native, so either shape is read: an
    object with ``login``/``host``, or a string."""
    last = config.get("lastLoggedInUser")
    if isinstance(last, Mapping):
        login = str(last.get("login") or "").strip()
        host = re.sub(r"^https?://", "", str(last.get("host") or "github.com")).strip("/")
        return f"{login}@{host}" if login else None
    if isinstance(last, str) and last.strip():
        return last.strip()
    return None


def holds_plaintext_token(config: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    """Whether the operator's login is a token IN a file, not the OS credential
    store. Key presence only: no value is read."""
    flagged = config.get(PLAINTEXT_FLAG) is True or settings.get(PLAINTEXT_FLAG) is True
    return flagged or any(config.get(key) for key in PLAINTEXT_TOKEN_KEYS)


def gh_login(host: str, run: Callable[..., Any] = subprocess.run) -> Optional[str]:
    """The account ``gh`` is logged in to on ``host`` — Copilot's own ``gh-cli``
    login, which reads the token itself. ``--show-token`` is never passed."""
    hostname = re.sub(r"^https?://", "", host).split("@")[-1].strip("/") or "github.com"
    try:
        out = run(["gh", "auth", "status", "--hostname", hostname], capture_output=True, text=True,
                  timeout=GH_STATUS_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if getattr(out, "returncode", 1) != 0:
        return None
    match = _GH_LOGIN_RE.search(f"{getattr(out, 'stdout', '')}\n{getattr(out, 'stderr', '')}")
    return f"{match.group('login')}@{match.group('host')}" if match else None


# ── policies that switch our hooks off ───────────────────────────────────────

def managed_hooks_only(dirs: Optional[Sequence[Path]] = None) -> Optional[Path]:
    """The managed policy file that runs only administrator-deployed hooks, if any."""
    for folder in MANAGED_POLICY_DIRS if dirs is None else dirs:
        try:
            files = sorted(folder.glob("*.json"))
        except OSError:
            continue
        for path in files:
            if read_json(path).get(MANAGED_HOOKS_ONLY_KEY) is True:
                return path
    return None


def hooks_disabled_in(cwd: Path) -> Optional[Path]:
    """The repository settings file that switches every hook off, in the ticket's
    folder or any folder above it up to the git root."""
    for folder in (cwd, *cwd.parents):
        for rel in REPO_SETTINGS:
            if read_json(folder / rel).get(DISABLE_HOOKS_KEY) is True:
                return folder / rel
        if (folder / ".git").exists():
            return None
    return None


# ── the agent's home ─────────────────────────────────────────────────────────

def seeded_config(operator_config: Mapping[str, Any]) -> Dict[str, Any]:
    """The agent home's ``config.json``: the account pointer, and no folder trusted
    — in ``-p`` an untrusted folder keeps repo hooks, MCP servers and extensions out."""
    kept = {key: operator_config[key] for key in ACCOUNT_KEYS if key in operator_config}
    return {**kept, "trustedFolders": []}


def seeded_settings(operator_settings: Mapping[str, Any], operator_config: Mapping[str, Any],
                    sandbox: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The agent home's ``settings.json``: the operator's whitelisted choices (from
    their settings.json, else the legacy config.json), the fixed values, and the
    sandbox block — which needs Copilot's experimental features on."""
    kept = {key: operator_settings.get(key, operator_config.get(key)) for key in OPERATOR_SETTINGS
            if key in operator_settings or key in operator_config}
    extra = {EXPERIMENTAL_KEY: True, SANDBOX_KEY: dict(sandbox)} if sandbox else {}
    return {**kept, **FIXED_SETTINGS, **extra}


def hooks_document(events: Sequence[str], timeout: Callable[[str], Any], command: str) -> Dict[str, Any]:
    """Every bus event this CLI delivers, in Claude's hook format, pointing at the shim."""
    hooks: Dict[str, Any] = {}
    for event in sorted(events):
        cmd = f"{command} --event {event}" if event in ARGV_NAMED_EVENTS else command
        entry: Dict[str, Any] = {"hooks": [{"type": "command", "command": cmd, "timeout": timeout(event)}]}
        hooks[event] = [{"matcher": "*", **entry} if event in TOOL_EVENTS else entry]
    return {"version": 1, "hooks": hooks}


def build_home(home: Path, *, config: Mapping[str, Any], settings: Mapping[str, Any],
               hooks: Mapping[str, Any]) -> Path:
    """(Re)build the agent's home: config, settings and our hooks file rewritten;
    any other hooks file and any MCP config removed."""
    home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(home, 0o700)
    write_private(home / CONFIG_FILENAME, config)
    write_private(home / SETTINGS_FILENAME, settings)
    hooks_dir = home / HOOKS_DIRNAME
    hooks_dir.mkdir(exist_ok=True, mode=0o700)
    for stale in hooks_dir.glob("*.json"):
        stale.unlink()
    write_private(hooks_dir / HOOKS_FILENAME, hooks)
    try:
        (home / OPERATOR_MCP_FILENAME).unlink()
    except FileNotFoundError:
        pass
    return home


__all__ = [
    "ACCOUNT_KEYS", "CONFIG_FILENAME", "FIXED_SETTINGS", "HOOKS_DIRNAME", "HOOKS_FILENAME", "OPERATOR_SETTINGS",
    "SETTINGS_FILENAME", "account_of", "build_home", "gh_login", "holds_plaintext_token", "hooks_disabled_in",
    "hooks_document", "managed_hooks_only", "operator_home", "read_json", "seeded_config", "seeded_settings",
    "write_private",
]
