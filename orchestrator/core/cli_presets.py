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
from typing import Callable, Dict, Optional

PROVIDER_CLAUDE = "claude"
PROVIDER_CODEX = "codex"

# What ``claude --model`` accepts: an alias or a full model id. Deliberately
# narrow — a session agent never carries an OpenRouter id (PRD-223: the model
# route used to validate nothing).
_CLAUDE_MODEL_ALIASES = frozenset({"opus", "sonnet", "haiku", "fable", "default"})
_CLAUDE_MODEL_ID_RE = re.compile(r"^claude-[a-z0-9][a-z0-9.\-]*(\[1m\])?$")
# Codex: permissive (design D-2) — the CLI refuses a bad model honestly, and a
# hardcoded list rots.
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


CLI_PRESETS: Dict[str, CliPresetInfo] = {
    PROVIDER_CLAUDE: CliPresetInfo(PROVIDER_CLAUDE, "Claude Code", "claude_code", _claude_model_ok),
    PROVIDER_CODEX: CliPresetInfo(PROVIDER_CODEX, "Codex", "codex", _codex_model_ok),
}
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
