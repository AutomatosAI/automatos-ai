"""PRD-234 S1a — the agent runtime kind (``api`` | ``cli``) as pure data rules.

Session mode adds a second execution runtime. A ``cli`` agent's ticket runs as
the user's OWN Claude Code (later Codex, …) session on their machine, driven by
the paired CLI host — never through the LLM factory. Everything that needs to
know which runtime an agent has reads it from here, so the rule lives in ONE
place:

* :func:`runtime_kind_of` — the agent's runtime, defaulting to ``api`` (every
  agent that exists today is an API agent; nothing changes for them).
* :func:`validate_runtime_configuration` — what the agents API accepts.
* :class:`RuntimeMismatchError` — raised/reported when a ``cli`` agent reaches a
  lane that can only execute API agents (the factory). PRD-223's lesson: a lane
  that cannot run the agent fails loudly, never falls through to the API path.

Pure module: no DB, no config import — callers pass ``cli_enabled`` in.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

RUNTIME_API = "api"
RUNTIME_CLI = "cli"
RUNTIME_KINDS = (RUNTIME_API, RUNTIME_CLI)

# Keys on ``Agent.configuration`` (JSON — no migration, shallow-merged on update).
CONFIG_RUNTIME_KEY = "runtime"
CONFIG_PROVIDER_KEY = "provider"
CONFIG_MODEL_KEY = "model"
CONFIG_WORKING_DIRECTORY_KEY = "working_directory"
CONFIG_ALLOWED_TOOLS_KEY = "allowed_tools"
# PRD-239: a single repo gets a worktree per ticket (default); a workspace of many
# repos must not — its own git tracks next to nothing, the worktree would be empty.
CONFIG_WORKTREE_KEY = "worktree_per_ticket"

# The CLI registry (id, label, model rule, usage slug) lives in ``core/cli_presets.py``,
# kept in step with the host's preset table by a parity test; re-exported here so
# every importer keeps its name.
from core.cli_presets import (  # noqa: E402,F401
    BILLING_SUBSCRIPTION, CLI_PRESETS, CLI_PROVIDERS, PROVIDER_CLAUDE, PROVIDER_CODEX,
    USAGE_PROVIDER_LABELS, USAGE_PROVIDER_SLUGS, is_valid_cli_model, usage_provider_slug,
)


class RuntimeMismatchError(RuntimeError):
    """A ``cli`` agent reached an execution lane that can only run API agents."""

    error_code = "runtime_mismatch"

    def __init__(self, agent_id: Any, runtime: str, lane: str = "LLM factory") -> None:
        self.agent_id = agent_id
        self.runtime = runtime
        self.lane = lane
        super().__init__(
            f"Agent {agent_id} runs as a {runtime} session (session mode) — the "
            f"{lane} cannot execute it. File a board ticket for it instead; the "
            f"paired CLI host claims and runs those."
        )

    def as_result(self) -> Dict[str, Any]:
        """The factory's error-dict shape, so every lane that already handles
        ``{"status": "error"}`` fails honestly without a code change."""
        return {
            "status": "error",
            "error": str(self),
            "error_code": self.error_code,
            "runtime": self.runtime,
            "agent_id": self.agent_id,
        }


def runtime_kind_of(configuration: Optional[Mapping[str, Any]]) -> str:
    """The runtime an agent configuration declares; ``api`` when absent/unknown."""
    if not isinstance(configuration, Mapping):
        return RUNTIME_API
    raw = configuration.get(CONFIG_RUNTIME_KEY)
    if isinstance(raw, str) and raw.strip().lower() == RUNTIME_CLI:
        return RUNTIME_CLI
    return RUNTIME_API


def is_cli_agent(configuration: Optional[Mapping[str, Any]]) -> bool:
    return runtime_kind_of(configuration) == RUNTIME_CLI


def validate_working_directory(value: Any) -> List[str]:
    """PRD-239 S6: the errors for a cli agent's ``working_directory``.

    Host-agnostic — the backend cannot see the operator's disk. It refuses what
    can never work (a relative path, a ``..`` segment, control characters); the
    host's allow-list and the Canvas mapping are reported separately by
    ``cli_host_service.workspace_check``.
    """
    if value is None or value == "":
        return []
    if not isinstance(value, str):
        return [f"configuration.{CONFIG_WORKING_DIRECTORY_KEY} must be a string path, got {type(value).__name__}"]
    # Control characters are checked on the value as typed — strip() would hide a
    # trailing newline and let it through.
    if any(ch in value for ch in ("\x00", "\n", "\r")):
        return [f"configuration.{CONFIG_WORKING_DIRECTORY_KEY} contains a control character"]
    path = value.strip()
    if not path.startswith("/"):
        return [f"configuration.{CONFIG_WORKING_DIRECTORY_KEY} must be an absolute path, got {value!r}"]
    if any(segment == ".." for segment in path.split("/")):
        return [f"configuration.{CONFIG_WORKING_DIRECTORY_KEY} must not contain '..' segments, got {value!r}"]
    return []


def validate_runtime_configuration(
    configuration: Optional[Mapping[str, Any]], *, cli_enabled: bool
) -> List[str]:
    """Errors (empty list = valid) for the runtime fields of an agent configuration.

    Rules: ``runtime`` absent or ``api`` ⇒ nothing else is checked (today's
    agents). ``runtime: cli`` ⇒ the instance must have ``CLI_RUNTIME_ENABLED``
    (session mode is a local-edition feature), ``provider`` must be a known CLI,
    and ``model`` (if given) must be a model that CLI accepts.
    """
    errors: List[str] = []
    if not isinstance(configuration, Mapping):
        return errors
    raw_runtime = configuration.get(CONFIG_RUNTIME_KEY)
    if raw_runtime is None:
        return errors
    if not isinstance(raw_runtime, str) or raw_runtime.strip().lower() not in RUNTIME_KINDS:
        errors.append(
            f"configuration.runtime must be one of {list(RUNTIME_KINDS)}, got {raw_runtime!r}"
        )
        return errors
    if raw_runtime.strip().lower() == RUNTIME_API:
        return errors
    if not cli_enabled:
        errors.append(
            "configuration.runtime='cli' requires CLI_RUNTIME_ENABLED=true "
            "(session mode is a local-edition feature)"
        )
    provider = configuration.get(CONFIG_PROVIDER_KEY)
    if provider not in CLI_PROVIDERS:
        errors.append(
            f"configuration.provider must be one of {list(CLI_PROVIDERS)} for a cli "
            f"agent, got {provider!r}"
        )
    elif not is_valid_cli_model(provider, configuration.get(CONFIG_MODEL_KEY)):
        errors.append(
            f"configuration.model {configuration.get(CONFIG_MODEL_KEY)!r} is not a "
            f"{provider} model alias or id"
        )
    errors.extend(validate_working_directory(configuration.get(CONFIG_WORKING_DIRECTORY_KEY)))
    worktree = configuration.get(CONFIG_WORKTREE_KEY)
    if worktree is not None and not isinstance(worktree, bool):
        errors.append(f"configuration.{CONFIG_WORKTREE_KEY} must be true or false, got {worktree!r}")
    return errors
