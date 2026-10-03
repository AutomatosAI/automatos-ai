"""The backend's view of the CLI registry (CLI adapter design §8.1).

The host owns everything operational about a CLI (``services/cli-host/
automatos_cli_host/presets.py``: binary, flags, hooks, env). The backend needs
only what it validates and books: the id, the label the picker shows, the rule
a saved model must satisfy, and the ``llm_usage.provider`` slug a session's
spend is tagged with. Two copies of one list drift, so a test asserts the ids
here equal the host's rows (``test_cli_presets_parity.py``) — no build step, no
generated code, and drift fails CI.

Pure module: no DB, no config.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

PROVIDER_CLAUDE = "claude"
PROVIDER_CODEX = "codex"
PROVIDER_COPILOT = "copilot"

# What ``claude --model`` accepts: an alias or a full model id. Deliberately
# narrow — a session agent never carries an OpenRouter id (PRD-223: the model
# route used to validate nothing).
_CLAUDE_MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku", "fable", "default"})
_CLAUDE_MODEL_ID_RE = re.compile(r"^claude-[a-z0-9][a-z0-9.\-]*(\[1m\])?$")
# Codex and GitHub Copilot: permissive (design D-2) — the CLI refuses a bad model
# honestly, and a hardcoded list rots. Copilot's models are its plan's and the
# organisation's policy's (auto, claude-sonnet-4.6, gpt-5.4, …): never a route.
_CODEX_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9.\-]*$")


def _claude_model_ok(model: str) -> bool:
    return model in _CLAUDE_MODEL_ALIASES or bool(_CLAUDE_MODEL_ID_RE.match(model))


def _codex_model_ok(model: str) -> bool:
    return bool(_CODEX_MODEL_RE.match(model))


@dataclass(frozen=True)
class CliPresetInfo:
    id: str
    label: str
    usage_slug: str                       # ``llm_usage.provider`` — a slug of its own, never an API provider
    model_ok: Callable[[str], bool]       # the rule a saved model must satisfy (blank = the CLI's default)
    model_hint: str                       # the help line under the picker's model field
    model_placeholder: str
    # PRD-253 S0.5: how the operator continues a session in their own terminal
    # ({session_id}), and the note a CLI whose home is per agent needs with it.
    takeover: str = ""
    takeover_note: str = ""

    def public(self) -> Dict[str, str]:
        """What the picker renders (design §8.3) — never the rule itself."""
        return {"id": self.id, "label": self.label, "model_hint": self.model_hint, "model_placeholder": self.model_placeholder}


# A CLI whose config home is per agent (Codex's CODEX_HOME) resumes its sessions
# only from that home, which the operator's own terminal does not point at.
AGENT_HOME_NOTE = "or open the ticket's Runtime Canvas terminal, which starts it in the agent's own home"

CLI_PRESETS: Dict[str, CliPresetInfo] = {
    PROVIDER_CLAUDE: CliPresetInfo(
        PROVIDER_CLAUDE, "Claude Code", "claude_code", _claude_model_ok,
        "Claude Code's own aliases, lowercase (fable · opus · sonnet · haiku) or a full id such as claude-opus-5. "
        "Blank = the CLI's default. The model must be available to your login; it is not one of the API models below.",
        "fable · opus · sonnet · haiku · or claude-opus-5",
        takeover="claude --resume {session_id}",
    ),
    PROVIDER_CODEX: CliPresetInfo(
        PROVIDER_CODEX, "Codex", "codex", _codex_model_ok,
        "A model your ChatGPT plan offers in Codex (for example gpt-5.5). Blank = the CLI's default from your ~/.codex/config.toml.",
        "gpt-5.5",
        takeover="codex resume {session_id}",
        takeover_note=AGENT_HOME_NOTE,
    ),
    # PRD-253: GitHub Copilot CLI, run as ``copilot -p`` with its config home per agent.
    PROVIDER_COPILOT: CliPresetInfo(
        PROVIDER_COPILOT, "GitHub Copilot", "copilot_cli", _codex_model_ok,
        "A model your Copilot plan and your organisation's policy enable in Copilot CLI (for example auto, "
        "claude-sonnet-4.6 or gpt-5.4). Blank = the CLI's default.",
        "auto · claude-sonnet-4.6 · gpt-5.4",
        takeover="copilot --resume {session_id}",
        takeover_note=AGENT_HOME_NOTE,
    ),
}


def registry_public() -> List[Dict[str, str]]:
    return [info.public() for info in CLI_PRESETS.values()]


def _info(cli_provider: Optional[str]) -> Optional[CliPresetInfo]:
    """The row for a ticket's CLI; no provider at all is Claude Code (what every
    session agent ran on before the field existed, as the claim defaults it)."""
    return CLI_PRESETS.get(str(cli_provider or PROVIDER_CLAUDE).strip().lower())


def session_heading(cli_provider: Optional[str]) -> str:
    """The task report's section title for a session: the CLI's own name."""
    info = _info(cli_provider)
    return f"## {info.label if info else 'CLI'} session"


def takeover_line(cli_provider: Optional[str], session_id: Optional[str], cwd: Optional[str]) -> Optional[str]:
    """How to continue this session in the operator's own terminal, in THIS CLI's
    spelling (PRD-253 S0.5); None for a CLI the registry does not know."""
    info = _info(cli_provider)
    if info is None or not info.takeover or not session_id:
        return None
    cd = f"cd {cwd} && " if cwd else ""
    note = f" — {info.takeover_note}" if info.takeover_note else ""
    return f"- Take over in your terminal: `{cd}{info.takeover.format(session_id=session_id)}`{note}"
CLI_PROVIDERS = tuple(CLI_PRESETS)

# How a session's spend is tagged in ``llm_usage.provider`` (never a registry API
# provider: a Claude Code session is the user's plan, not an Anthropic API key)
# and the human label the analytics page shows.
USAGE_PROVIDER_SLUGS = {p.id: p.usage_slug for p in CLI_PRESETS.values()}
USAGE_PROVIDER_LABELS = {p.usage_slug: p.label for p in CLI_PRESETS.values()}
BILLING_SUBSCRIPTION = "subscription"


def usage_provider_slug(cli_provider: Optional[str]) -> str:
    """``claude`` → ``claude_code``; an unknown CLI keeps its name."""
    key = str(cli_provider or "").strip().lower()
    return USAGE_PROVIDER_SLUGS.get(key, key or "unknown")


def is_valid_cli_model(provider: str, model: Optional[str]) -> bool:
    """``None``/empty = the CLI's own default; otherwise provider-shaped."""
    if model is None or model == "":
        return True
    if not isinstance(model, str):
        return False
    info = CLI_PRESETS.get(provider)
    return bool(info and info.model_ok(model.strip()))


# PRD-245 S0.6 — the Bash commands a ticket session runs without asking, as the
# host's policy allows them (``services/cli-host/automatos_cli_host/policy.py``
# ``DEFAULT_BASH_ALLOW``). The session prompt renders this list so the agent
# knows what runs, what is held for the operator and what never runs. The host
# owns the rule; this is the backend's rendering copy, kept in step by
# ``test_cli_presets_parity.py`` the same way ``CLI_PRESETS`` is.
SESSION_BASH_VERBS: Tuple[str, ...] = (
    # the host's DEFAULT_BASH_ALLOW, same members, same order (parity test)
    "git status", "git diff", "git log", "git show", "git branch", "git add", "git commit",
    "git stash", "git restore", "git checkout -b", "git switch -c", "git ls-files",
    "git rev-parse", "git blame", "git describe", "git shortlog", "git remote -v",
    "git worktree list", "git stash list", "ls", "cat", "head", "tail", "wc", "grep", "rg",
    "find", "pwd", "which", "echo", "sort", "uniq", "cut", "tr", "sed", "awk", "date", "diff",
    "stat", "basename", "dirname", "printf", "jq", "file", "tree", "du", "true", "test", "[",
    "[[", "comm", "join", "paste", "nl", "tac", "rev", "fold", "expand", "unexpand", "column",
    "md5sum", "sha1sum", "sha256sum", "cksum", "realpath", "readlink", "seq", "python -m pytest", "python3 -m pytest", "pytest", "npm test", "npm run",
    "pnpm test", "pnpm run", "yarn test", "make test", "make lint", "cargo test", "go test",
    "ruff", "black --check", "mypy", "tsc", "eslint", "vitest",
)
