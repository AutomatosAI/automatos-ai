"""Settings for the simulation runner — env file, paths, defaults.

Nothing in this module is a secret. The env file at ``~/.automatos-sim/env``
holds whatever the operator sets (an OpenRouter key for the judge, a different
model, a different API URL); it is never in the repo.

Every variable is ``SIM_``-prefixed on purpose: ``tests/.env`` points the
nightly API suite at Railway production, and a runner that files thousands of
tasks must never inherit that target by accident. A non-loopback API URL is
refused unless ``SIM_ALLOW_REMOTE=1`` is set explicitly, and so is a Docker
daemon that is not the local socket (``DOCKER_HOST``).

Credential: none, on the local edition. ``get_request_context_hybrid`` treats
an ``X-Api-Key`` header as the single static ``ORCHESTRATOR_API_KEY`` (empty
locally, so any key header is a 401) and honours the per-workspace ``ak_srv_``
keys only as a Bearer token on the board-task read routes. Without a header
the local edition resolves the anonymous operator — the instance's super admin
— and scopes by ``X-Workspace-ID``. That header is therefore the whole
boundary between the run and the operator's own workspace; the runner sends
it on every call and refuses the default workspace by id. ``SIM_API_KEY`` is
only for a stack that requires the static key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

SIM_HOME = Path(os.environ.get("AUTOMATOS_SIM_HOME", str(Path.home() / ".automatos-sim")))
ENV_FILE = SIM_HOME / "env"
RUNS_DIR = SIM_HOME / "runs"
LOGS_DIR = SIM_HOME / "logs"
JUDGE_CACHE_DIR = SIM_HOME / "judge-cache"
PERSONAS_FILE = SIM_HOME / "personas.toml"
CAMPAIGN_DB = SIM_HOME / "campaign.sqlite"

PACKAGE_DIR = Path(__file__).resolve().parent
PACKS_DIR = PACKAGE_DIR / "packs"
REPO_ROOT = PACKAGE_DIR.parents[1]
SCRIPTS_DIR = REPO_ROOT / "orchestrator" / "scripts"
CREATE_WORKSPACE_SCRIPT = SCRIPTS_DIR / "create_test_workspace.py"
PURGE_WORKSPACE_SCRIPT = SCRIPTS_DIR / "purge_test_workspace.py"

# The local operator's own workspace (compose default). A sim run must never
# address it — provision() refuses an id that resolves to it (PRD-247 D5).
DEFAULT_WORKSPACE_ID = "00000000-0000-0000-0000-0000000000c1"
# F208: the OpenRouter key belongs to c1, the platform key's workspace; a sim
# workspace has none ("no key configured"). A night has cost $3-11, so preflight
# refuses to launch one with less than this much credit known.
PLATFORM_KEY_WORKSPACE_ID = DEFAULT_WORKSPACE_ID
MIN_BALANCE_USD = float(os.environ.get("SIM_MIN_BALANCE_USD", "5"))

# system_settings rows the cheap-model switch may touch. The table is global
# (no workspace column), so the runner snapshots these before and restores
# them after — see workspace.snapshot_llm_settings.
LLM_SETTING_KEYS = ("llm_model", "llm_provider", "model", "provider")
LLM_SETTING_CATEGORY_SUFFIX = "_llm"
LLM_SETTING_EXTRA_CATEGORIES = ("chatbot",)

LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1", "host.docker.internal")


class ConfigError(ValueError):
    """A setting the runner cannot proceed with — the message says which."""


@dataclass(frozen=True)
class Settings:
    api_url: str = "http://localhost:8000"
    api_key: str = ""  # static ORCHESTRATOR_API_KEY, only when the stack requires it; empty = anonymous operator
    backend_container: str = "automatos_backend"
    postgres_container: str = "automatos_postgres"
    docker: str = "docker"
    model_provider: str = "openrouter"
    model_id: str = "openai/gpt-4.1-mini"
    judge_model_id: str = "openai/gpt-4.1-mini"
    openrouter_api_key: str = ""
    budget_usd: float = 5.0
    task_timeout_s: int = 900
    chat_timeout_s: int = 240
    poll_s: float = 10.0
    dispatch_grace_s: int = 60
    keep_workspace: bool = False
    set_global_models: bool = True
    judge: bool = True
    allow_remote: bool = False


# env name -> (field, caster)
_ENV_MAP: dict[str, tuple[str, Any]] = {
    "SIM_API_URL": ("api_url", str),
    "SIM_API_KEY": ("api_key", str),
    "SIM_BACKEND_CONTAINER": ("backend_container", str),
    "SIM_POSTGRES_CONTAINER": ("postgres_container", str),
    "SIM_DOCKER": ("docker", str),
    "SIM_MODEL_PROVIDER": ("model_provider", str),
    "SIM_MODEL_ID": ("model_id", str),
    "SIM_JUDGE_MODEL_ID": ("judge_model_id", str),
    "OPENROUTER_API_KEY": ("openrouter_api_key", str),
    "SIM_BUDGET_USD": ("budget_usd", float),
    "SIM_TASK_TIMEOUT_S": ("task_timeout_s", int),
    "SIM_CHAT_TIMEOUT_S": ("chat_timeout_s", int),
    "SIM_POLL_S": ("poll_s", float),
    "SIM_DISPATCH_GRACE_S": ("dispatch_grace_s", int),
    "SIM_KEEP_WORKSPACE": ("keep_workspace", lambda v: v.strip().lower() in ("1", "true", "yes")),
    "SIM_SET_GLOBAL_MODELS": ("set_global_models", lambda v: v.strip().lower() in ("1", "true", "yes")),
    "SIM_JUDGE": ("judge", lambda v: v.strip().lower() in ("1", "true", "yes")),
    "SIM_ALLOW_REMOTE": ("allow_remote", lambda v: v.strip().lower() in ("1", "true", "yes")),
}


def read_env_file(path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` file; blank lines and ``#`` comments are skipped.

    Values may be single- or double-quoted. Missing file -> empty mapping.
    """
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        parsed[key.strip()] = value
    return parsed


def settings_from_mapping(env: Mapping[str, str], base: Settings | None = None) -> Settings:
    """Overlay ``env`` (only ``_ENV_MAP`` names) on ``base``; bad values name the variable."""
    values: dict[str, Any] = {}
    for name, (field_name, caster) in _ENV_MAP.items():
        if name not in env or env[name] == "":
            continue
        try:
            values[field_name] = caster(env[name])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name}={env[name]!r} is not a valid value: {exc}") from exc
    return replace(base or Settings(), **values)


def load_settings(overrides: Mapping[str, Any] | None = None) -> Settings:
    """Defaults <- ``~/.automatos-sim/env`` <- process environment <- CLI overrides."""
    merged = {**read_env_file(), **os.environ}
    settings = settings_from_mapping(merged)
    if overrides:
        known = {f.name for f in fields(Settings)}
        clean = {k: v for k, v in overrides.items() if v is not None and k in known}
        settings = replace(settings, **clean)
    check_target(settings)
    return settings


def check_target(settings: Settings) -> None:
    """Refuse a non-loopback API unless the operator said so in as many words."""
    host = (urlparse(settings.api_url).hostname or "").lower()
    if host in LOOPBACK_HOSTS or settings.allow_remote:
        return
    raise ConfigError(
        f"SIM_API_URL={settings.api_url} is not the local stack. The simulation files "
        "real tasks and spends real model calls; set SIM_ALLOW_REMOTE=1 only on purpose."
    )


LOCAL_DOCKER_SCHEMES = ("unix://", "npipe://")


def check_docker_target(settings: Settings, docker_host: str | None = None) -> str:
    """``docker exec`` reaches whatever daemon ``DOCKER_HOST`` names; refuse a remote one.

    Returns the value for the run record (empty when unset = the local socket).
    """
    host = os.environ.get("DOCKER_HOST", "") if docker_host is None else docker_host
    if not host or host.startswith(LOCAL_DOCKER_SCHEMES) or settings.allow_remote:
        return host
    raise ConfigError(
        f"DOCKER_HOST={host} is not the local daemon; the sim runs scripts and SQL inside containers "
        "there. Unset it, or set SIM_ALLOW_REMOTE=1 only on purpose."
    )


def ensure_dirs() -> None:
    for path in (SIM_HOME, RUNS_DIR, LOGS_DIR, JUDGE_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def public_settings(settings: Settings) -> dict[str, Any]:
    """The settings as they may be written into a run record — no key material."""
    shown = {f.name: getattr(settings, f.name) for f in fields(Settings)}
    shown["openrouter_api_key"] = "set" if settings.openrouter_api_key else ""
    shown["api_key"] = "set" if settings.api_key else ""
    return shown
