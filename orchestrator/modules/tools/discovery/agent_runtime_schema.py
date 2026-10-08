"""PRD-256 FX-016: platform_create_agent and platform_update_agent take the agent's runtime.

The two tools are registered inside ``register_agents_actions``, a function past the length
rule; this decorator hands it a registry that adds ``runtime``, ``provider`` and ``model``
to those two schemas as they are registered (the definition is rebuilt, never changed in
place). What the handler does with them is ``agent_runtime``. ``provider`` and ``model``
were already read by create_agent (the API model's provider, and ``model`` as an older name
of ``model_id``) and ``provider`` by update_agent: they are declared now, and their
description says what they mean for each runtime.
"""
from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Dict, Tuple

from core.cli_runtime import CLI_PROVIDERS, RUNTIME_KINDS

RUNTIME_DESCRIPTION = (
    "How the agent runs: 'api' (the platform's models, chosen by model_id) or 'cli' (a session of "
    "the owner's own Claude Code or Codex, on their paired machine). 'A Claude session', 'Claude Code', "
    "'a proper Claude agent' or 'runs on my machine' means runtime 'cli' with provider 'claude'. "
    "{absent}"
)
CREATE_ABSENT = "Omit it for the workspace's default runtime for new agents (api unless set)."
UPDATE_ABSENT = "Omit it to keep the agent's runtime."
PROVIDER_DESCRIPTION = (
    "For a cli agent, which CLI runs its session ({clis}): 'claude' is Claude Code, the default. "
    "For an api agent, the model_id's provider (optional)."
)
MODEL_DESCRIPTION = (
    "For a cli agent, the CLI's own model word, e.g. 'sonnet', 'opus' or 'haiku' for Claude Code "
    "(omit it for the CLI's default). An api agent's model goes in model_id."
)
DESCRIPTION_ADDS = (" Set runtime 'cli' (provider 'claude') when the user asks for a Claude session or "
                    "Claude Code agent.")
TAKES_THE_RUNTIME: Dict[str, str] = {
    "platform_create_agent": CREATE_ABSENT,
    "platform_update_agent": UPDATE_ABSENT,
}
# The keys these schemas now declare: no longer undeclared names the handler reads.
DECLARED = ("runtime", "provider", "model")


def runtime_properties(absent: str) -> Dict[str, Any]:
    """The three schema properties, ``absent`` saying what a call without a runtime gets."""
    return {
        "runtime": {"type": "string", "enum": list(RUNTIME_KINDS),
                    "description": RUNTIME_DESCRIPTION.format(absent=absent)},
        "provider": {"type": "string", "description": PROVIDER_DESCRIPTION.format(clis=", ".join(CLI_PROVIDERS))},
        "model": {"type": "string", "description": MODEL_DESCRIPTION},
    }


def _with_the_runtime(action: Any) -> Any:
    """``action`` rebuilt with the runtime properties, when it is one of the two tools."""
    absent = TAKES_THE_RUNTIME.get(getattr(action, "name", None))
    if absent is None:
        return action
    parameters = dict(action.parameters or {})
    parameters["properties"] = {**(parameters.get("properties") or {}), **runtime_properties(absent)}
    accepts: Tuple[str, ...] = tuple(key for key in action.accepts if key not in DECLARED)
    return dataclasses.replace(action, parameters=parameters, accepts=accepts,
                               description=f"{action.description}{DESCRIPTION_ADDS}")


class _TakesTheRuntime:
    """The registry ``register_agents_actions`` is given: it registers through to the real one."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def register(self, action: Any) -> None:
        self._registry.register(_with_the_runtime(action))


def takes_the_runtime(register_actions: Callable[[Any], None]) -> Callable[[Any], None]:
    """Wrap ``actions_agents.register_agents_actions`` (see the module docstring)."""
    @functools.wraps(register_actions)
    def wrapped(registry: Any) -> None:
        register_actions(_TakesTheRuntime(registry))
    return wrapped


__all__ = ["runtime_properties", "takes_the_runtime"]
