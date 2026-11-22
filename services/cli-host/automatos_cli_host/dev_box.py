"""Dev-box mode on the host: the session's GitHub token, and its value kept out of reports.

``--dev-box`` says this machine is the owner's development machine: a Claude Code
session commits, pushes its own branch and opens a pull request itself, with a
fine-grained token of the owner's own GitHub account. GitHub's repository ruleset (admin
bypass "pull requests only") is what stops a direct push to the default branch; the gate
is the second layer. The pieces:

* the sandbox first: with the session sandbox off (``--no-session-sandbox``) the mode
  never applies, whatever the flag says — the token never reaches a session the
  operating system does not confine. The host says so once at start;
* the token: ``<state dir>/agent-github-token``, one line, mode 0600, owned by the
  host's user. Missing, open to others, a link, or not a token: a warning in the log,
  and the session runs exactly as it would without the mode (no token, no GitHub
  network, the gate and the brief as before). Read fresh for every session, so a new
  token is used by the next ticket with no restart;
* the identity: the host user's own ``user.name`` and ``user.email`` (their global git
  config, read once at start), passed as ``GIT_AUTHOR_*``/``GIT_COMMITTER_*``; the
  sandbox cannot read ``~/.gitconfig``, and git refuses to commit without one;
* the session's environment: ``env.github_env`` (``GH_TOKEN``, git's credential config
  in ``GIT_CONFIG_*`` variables, gh's own config folder, the identity); the sandbox
  reaches GitHub (``sandbox.GITHUB_DOMAINS``); the gate judges git and gh by
  ``dev_box_policy``; the system prompt says how to publish
  (``session_prompt.DEV_BOX_SECTION``);
* redaction: the token's exact value is replaced in what the host keeps or sends —
  the terminal log, the output tail, the events, the result, the plan file
  (``SessionCredential.scrub``, ``StreamScrubber``, ``ScrubbedQueue``).

Only Claude Code sessions (``DEV_BOX_CLIS``): GitHub Copilot would take ``GH_TOKEN``
as its own login, and Codex's sandbox has no per-host network list to open.
"""
from __future__ import annotations

import functools
import logging
import os
import queue
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .dev_box_policy import allowed_bash
from .dev_box_rules import DevBoxRules, github_owner, rules
from .env import github_env, resolve_binary

log = logging.getLogger("automatos.cli_host.dev_box")

DEV_BOX_CLIS = frozenset({"claude"})
# A GitHub token is one word: ``ghp_…``, ``github_pat_…``. Anything else is not loaded.
TOKEN_RE = re.compile(r"^[A-Za-z0-9_]{20,255}$")
MAX_TOKEN_FILE_BYTES = 1024
OTHERS_ACCESS_MASK = 0o077                       # any bit for the group or others
NO_GITHUB = "this session runs without GitHub access"
NO_SANDBOX = ("dev-box mode is off for every session: the session sandbox is off (--no-session-sandbox), "
              "and the host never hands its GitHub token to a session the operating system does not confine")
REDACTED = "[GH_TOKEN redacted]"
GIT_PROBE_TIMEOUT_SECONDS = 5.0
ORIGIN_URL_KEY = "remote.origin.url"
ORIGIN_HEAD_REF = "refs/remotes/origin/HEAD"
IDENTITY_KEYS = ("user.name", "user.email")
_IDENTITY_VALUE_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")
GH_CONFIG_DIR_PREFIX = "automatos-gh-"


def sandboxed(cfg: Any) -> bool:
    """The host confines its Claude Code sessions in the operating system's sandbox."""
    sandbox = getattr(cfg, "session_sandbox", None)
    return bool(sandbox is not None and sandbox.enabled)


def enabled(cfg: Any) -> bool:
    """Dev-box mode applies on this host: the flag, and the sandbox under every session."""
    return bool(getattr(cfg, "dev_box", False)) and sandboxed(cfg)


def serves(cfg: Any, cli_id: str) -> bool:
    """Dev-box mode applies on this host and covers this CLI."""
    return enabled(cfg) and cli_id in DEV_BOX_CLIS


def _file_problem(info: os.stat_result, path: Path) -> Optional[str]:
    """Why the token file is not one the host loads, from its own metadata."""
    if not stat.S_ISREG(info.st_mode):
        return f"{path} is not a regular file"
    if os.name == "nt":
        return f"{path}: the host cannot check a file's permissions on Windows, so it loads no token there"
    if info.st_mode & OTHERS_ACCESS_MASK:
        return f"{path} is open to other users (mode {oct(info.st_mode & 0o777)}): run chmod 600 {path}"
    if info.st_uid != os.getuid():
        return f"{path} belongs to another user"
    return None


def read_token(path: Path) -> Tuple[Optional[str], Optional[str]]:
    """(the token, None) or (None, why not). The file is opened without following a
    link and judged on the descriptor it opened; its content never reaches a message."""
    try:
        fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError:
        return None, f"no GitHub token at {path}"
    except OSError as exc:
        return None, f"the GitHub token file {path} cannot be opened ({exc.strerror}; a link is refused)"
    try:
        with os.fdopen(fd, "rb") as handle:
            problem = _file_problem(os.fstat(handle.fileno()), path)
            raw = b"" if problem else handle.read(MAX_TOKEN_FILE_BYTES)
    except OSError as exc:
        return None, f"the GitHub token file {path} cannot be read ({exc.strerror})"
    if problem:
        return None, problem
    token = raw.decode("utf-8", "replace").strip()
    return (token, None) if TOKEN_RE.match(token) else (None, f"{path} does not hold a GitHub token (one line, the token alone)")


def session_token(cfg: Any) -> Optional[str]:
    """The token for one session, or None with the reason in the log."""
    token, problem = read_token(Path(cfg.dev_box_token_path))
    if problem:
        log.warning("dev-box mode: %s — %s", problem, NO_GITHUB)
    return token


# ── git: the checkout's facts and the host user's identity ──────────────────

def _git_output(*args: str) -> Optional[str]:
    """One line git prints, read without the network; None when it cannot say."""
    git = resolve_binary("git")
    if git is None:
        return None
    try:
        done = subprocess.run([git, *args], capture_output=True, text=True, timeout=GIT_PROBE_TIMEOUT_SECONDS,
                              check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("dev-box mode: git %s failed (%s)", " ".join(args[:3]), exc)
        return None
    value = done.stdout.strip()
    return value if done.returncode == 0 and value else None


def read_git_identity() -> Optional[Tuple[str, str]]:
    """The host user's own (name, email) from their global git config, or None with a warning."""
    values = tuple(_git_output("config", "--global", "--get", key) for key in IDENTITY_KEYS)
    if not all(value and _IDENTITY_VALUE_RE.match(value) for value in values):
        log.warning("dev-box mode: no usable user.name and user.email in your global git config — sessions get no "
                    "git identity, and git refuses to commit (git config --global user.name / user.email)")
        return None
    return values[0], values[1]


@functools.lru_cache(maxsize=1)
def git_identity() -> Optional[Tuple[str, str]]:
    """``read_git_identity``, once per host (the start-up line reads it first)."""
    return read_git_identity()


def log_summary(cfg: Any) -> None:
    """At start: whether Claude Code sessions will publish, with which file and identity."""
    if not sandboxed(cfg):
        log.warning("%s", NO_SANDBOX)
        return
    path = Path(cfg.dev_box_token_path)
    _, problem = read_token(path)
    if problem:
        log.warning("dev-box mode: %s — Claude Code sessions run without GitHub access until it is there", problem)
    else:
        log.info("dev-box mode: Claude Code sessions push their branch and open pull requests with the token in %s",
                 path)
    git_identity()


def repo_rules(cwd: Path) -> DevBoxRules:
    """The rules for the repository a session runs in: origin's owner on github.com (every
    remote and GitHub write must share it) and origin's own default branch (protected
    beside the named list), read from the checkout. Outside a git repository, the named list."""
    if not (Path(cwd) / ".git").exists():
        return rules()
    url = _git_output("-C", str(cwd), "config", "--get", ORIGIN_URL_KEY)
    head = _git_output("-C", str(cwd), "symbolic-ref", "--quiet", "--short", ORIGIN_HEAD_REF)     # "origin/main"
    default = head.split("/", 1)[1] if head and "/" in head else None
    return rules(default_branch=default, origin_owner=github_owner(url) if url else None)


# ── one session's credential, and keeping its value out of what the host sends ──

class SessionCredential:
    """The GitHub token one session carries in dev-box mode (``token`` None: it carries
    none and runs as it would without the mode), its identity and gh config folder,
    and the scrubbing that replaces the token's exact value in text, bytes, and the
    lists and dicts the host reports. With no token, ``scrub`` hands every value back."""

    def __init__(self, token: Optional[str] = None, identity: Optional[Tuple[str, str]] = None) -> None:
        self.token = token or None
        self.identity = identity if self.token else None
        self.gh_config_dir = (Path(tempfile.gettempdir()) / f"{GH_CONFIG_DIR_PREFIX}{uuid.uuid4().hex}"
                              if self.token else None)

    @classmethod
    def for_session(cls, cfg: Any, cli_id: str) -> "SessionCredential":
        """The token, when dev-box mode applies to this CLI and the file loads cleanly."""
        token = session_token(cfg) if serves(cfg, cli_id) else None
        return cls(token, git_identity() if token else None)

    @property
    def carried(self) -> bool:
        return self.token is not None

    def environment(self) -> Optional[Dict[str, str]]:
        """What the session's environment gets (``env.github_env``), or None."""
        if not self.token:
            return None
        return github_env(self.token, str(self.gh_config_dir), self.identity)

    def discard(self) -> None:
        """The session is over: remove gh's config folder if gh or the session made one
        (a link there is removed, never followed)."""
        folder = self.gh_config_dir
        try:
            if folder is not None and folder.is_symlink():
                folder.unlink()
            elif folder is not None and folder.exists():
                shutil.rmtree(folder)
        except OSError as exc:
            log.warning("dev-box mode: %s not removed: %s", folder, exc)

    def scrub(self, value: Any) -> Any:
        return self._clean(value) if self.token else value

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.replace(self.token, REDACTED)
        if isinstance(value, bytes):
            return value.replace(self.token.encode(), REDACTED.encode())
        if isinstance(value, dict):
            return {key: self._clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self._clean(item) for item in value)
        return value

    def stream(self) -> "StreamScrubber":
        return StreamScrubber(self.token.encode() if self.token else None)


def _partial_tail(data: bytes, secret: bytes) -> int:
    """How many bytes at the end of ``data`` could be the start of ``secret``."""
    for size in range(min(len(data), len(secret) - 1), 0, -1):
        if data.endswith(secret[:size]):
            return size
    return 0


class StreamScrubber:
    """The same for a byte stream read in chunks (the session's terminal): a token split
    between two reads is held back until the next one completes or clears it."""

    def __init__(self, secret: Optional[bytes]) -> None:
        self._secret = secret or None
        self._held = b""

    def feed(self, chunk: bytes) -> bytes:
        if not self._secret:
            return chunk
        data = (self._held + chunk).replace(self._secret, REDACTED.encode())
        keep = _partial_tail(data, self._secret)
        self._held = data[len(data) - keep:] if keep else b""
        return data[:len(data) - keep]

    def flush(self) -> bytes:
        held, self._held = self._held, b""
        return held


class ScrubbedQueue(queue.Queue):
    """A session's event queue: every event is scrubbed on its way in."""

    def __init__(self, scrub: Callable[[Any], Any]) -> None:
        super().__init__()
        self._scrub = scrub

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        super().put(self._scrub(item), block, timeout)


__all__ = [
    "DEV_BOX_CLIS", "REDACTED", "ScrubbedQueue", "SessionCredential", "StreamScrubber", "allowed_bash", "enabled",
    "git_identity", "log_summary", "read_git_identity", "read_token", "repo_rules", "sandboxed", "serves",
    "session_token",
]
