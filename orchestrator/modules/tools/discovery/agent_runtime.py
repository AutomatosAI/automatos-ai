"""PRD-256 FX-016 (F396): Auto can make a Claude session agent.

Night 12: asked for "a proper Claude session", Auto made api agents on the default model
(platform_create_agent wrote ``configuration={}`` and neither the create nor the update
tool had a runtime), and CLUB DESK and MARKET-MANAGER were switched by hand. Both tools now
take ``runtime`` ('api' | 'cli') and, for a cli agent, ``provider`` (the CLI, Claude Code
unless named) and ``model`` (the CLI's own model word, e.g. 'sonnet'; none = the CLI's
default). They are written into ``Agent.configuration`` under the keys the CLI host reads
(``core.cli_runtime``) and checked by the rule the agents API uses
(``validate_runtime_configuration``): a cli agent needs session mode on this server.
A create that names no runtime takes ``DEFAULT_AGENT_RUNTIME`` ('api' unless set).
An update is owner-only already (Decision D1): its card says 'runtime: api → cli'.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple

from core.cli_runtime import (
    CLI_PRESETS, CONFIG_MODEL_KEY, CONFIG_PROVIDER_KEY, CONFIG_RUNTIME_KEY, PROVIDER_CLAUDE, RUNTIME_API,
    RUNTIME_CLI, RUNTIME_KINDS, runtime_kind_of, validate_runtime_configuration,
)
from modules.tools.discovery.card_question_text import change_line, shown

logger = logging.getLogger(__name__)

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

RUNTIME, PROVIDER, MODEL = CONFIG_RUNTIME_KEY, CONFIG_PROVIDER_KEY, CONFIG_MODEL_KEY
SESSION_KEYS = (RUNTIME, PROVIDER, MODEL)
# The keys that name the agent an update is for, not a change to it.
NAMES_THE_AGENT = frozenset({"agent_id", "agent_name"})
# What the owner calls each key on the update card.
CARD_LABELS = ((RUNTIME, "runtime"), (PROVIDER, "session CLI"), (MODEL, "session model"))
SAID_BY_THE_DEFAULT = "the default runtime for new agents (DEFAULT_AGENT_RUNTIME)"
UNKNOWN_RUNTIME = ("Unknown runtime {said!r} ({where}): an agent runs as 'api' (the platform's models) or "
                   "'cli' (a Claude Code or Codex session on the owner's paired machine). Nothing was {done}.")
REFUSED = "The agent can't run as a {runtime} agent here: {errors}. Nothing was {done}."
RUNS_AS = " It runs as a {label} session on the paired machine ({model})."
CLI_DEFAULT_MODEL = "the CLI's default model"
CREATED, CHANGED = "created", "changed"
UPDATE_AGENT = "platform_update_agent"
CHANGE = "{field}: {old} → {new}"


def _said_runtime(params: Mapping[str, Any]) -> Optional[str]:
    """The runtime the call names, lower-cased; None when it names none."""
    raw = params.get(RUNTIME)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip().lower()


def _default_runtime() -> str:
    from config import config

    return str(getattr(config, "DEFAULT_AGENT_RUNTIME", RUNTIME_API) or RUNTIME_API).strip().lower()


def planned_configuration(current: Mapping[str, Any], params: Mapping[str, Any], runtime: str) -> Dict[str, Any]:
    """The agent's configuration once ``runtime`` and the call's provider and model apply.

    An api agent carries no session keys. A cli agent runs Claude Code unless the call (or,
    already a cli agent, its configuration) names another CLI; it keeps its model while its
    CLI stays the same, and a model said as empty means the CLI's default."""
    base = {key: value for key, value in current.items() if key not in SESSION_KEYS}
    if runtime != RUNTIME_CLI:
        return {**base, RUNTIME: RUNTIME_API}
    was_cli = runtime_kind_of(current) == RUNTIME_CLI
    provider = str(params.get(PROVIDER) or (current.get(PROVIDER) if was_cli else None) or PROVIDER_CLAUDE)
    provider = provider.strip().lower()
    if MODEL in params:
        model = params.get(MODEL) or None
    else:
        model = current.get(MODEL) if was_cli and current.get(PROVIDER) == provider else None
    planned = {**base, RUNTIME: RUNTIME_CLI, PROVIDER: provider}
    return {**planned, MODEL: str(model).strip()} if model else planned


def _errors(planned: Mapping[str, Any]) -> List[str]:
    from config import config

    return validate_runtime_configuration(planned, cli_enabled=bool(getattr(config, "CLI_RUNTIME_ENABLED", False)))


def _refusal(text: str) -> Dict[str, Any]:
    return {"success": False, "error": text}


def _unknown(said: str, where: str, done: str) -> Dict[str, Any]:
    return _refusal(UNKNOWN_RUNTIME.format(said=said, where=where, done=done))


def _without_session_keys(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in params.items() if key not in SESSION_KEYS}


def _for_the_handler(params: Mapping[str, Any], planned: Mapping[str, Any]) -> Dict[str, Any]:
    """The call as the agent handler reads it: a cli agent's provider and model are the
    session's, kept off the API model's resolution; an api agent's are the API model's."""
    if planned.get(RUNTIME) == RUNTIME_CLI:
        return _without_session_keys(params)
    return {key: value for key, value in params.items() if key != RUNTIME}


def _runs_as(planned: Mapping[str, Any]) -> str:
    """What the create's message adds: the CLI's name and the model."""
    provider = planned.get(PROVIDER)
    label = CLI_PRESETS[provider].label if provider in CLI_PRESETS else str(provider)
    return RUNS_AS.format(label=label, model=planned.get(MODEL) or CLI_DEFAULT_MODEL)


def _session_fields(planned: Mapping[str, Any]) -> Dict[str, Any]:
    return {"runtime": RUNTIME_CLI, "cli_provider": planned.get(PROVIDER), "cli_model": planned.get(MODEL)}


def _write(db: Any, agent: Any, planned: Mapping[str, Any]) -> None:
    """The agent's configuration becomes ``planned`` (which holds every key it had besides
    the session's), as a new object so the JSON column sees the change."""
    agent.configuration = dict(planned)
    db.flush()


# ── platform_create_agent ──────────────────────────────────────────────────────────


def _create_runtime(params: Mapping[str, Any]) -> Tuple[str, Optional[Dict[str, Any]]]:
    """The new agent's runtime (the call's, else the default) and a refusal for an unknown one."""
    said = _said_runtime(params)
    runtime = said or _default_runtime()
    if runtime not in RUNTIME_KINDS:
        return runtime, _unknown(runtime, "said in the call" if said else SAID_BY_THE_DEFAULT, CREATED)
    return runtime, None


def sets_the_runtime_on_create(create: Handler) -> Handler:
    """Wrap handlers_agents.create_agent: a cli agent is checked before anything is made, made
    with the session's provider and model kept off the API model's resolution, then given its
    session configuration. An api agent is made as before."""
    @functools.wraps(create)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        runtime, refusal = _create_runtime(params)
        if refusal:
            return refusal
        if runtime == RUNTIME_API:
            return await create(db, workspace_id, params)  # its provider and model are the API model's, as before
        planned = planned_configuration({}, params, RUNTIME_CLI)
        errors = _errors(planned)
        if errors:
            return _refusal(REFUSED.format(runtime=RUNTIME_CLI, errors="; ".join(errors), done=CREATED))
        result = await create(db, workspace_id, _without_session_keys(params))
        if not (isinstance(result, dict) and result.get("success")):
            return result
        return _made_a_session_agent(db, workspace_id, result, planned)
    return wrapped


def _made_a_session_agent(db: Any, workspace_id: Any, result: Dict[str, Any],
                          planned: Mapping[str, Any]) -> Dict[str, Any]:
    """The created agent (read back in this workspace) given its session configuration."""
    from core.models import Agent

    made = result.get("agent") or {}
    agent = db.query(Agent).filter(Agent.workspace_id == workspace_id, Agent.id == made.get("id")).first()
    if agent is None:  # create_agent just flushed it: not finding it is a bug, never a quiet api agent
        logger.error("[create_agent] agent %s vanished before its runtime was written (workspace %s)",
                     made.get("id"), workspace_id)
        return _refusal("The agent was made, but as an api agent: its session runtime could not be written.")
    _write(db, agent, planned)
    return {**result, "agent": {**made, **_session_fields(planned)},
            "message": f"{result.get('message', '')}{_runs_as(planned)}"}


# ── platform_update_agent ──────────────────────────────────────────────────────────


def runtime_changes(current: Mapping[str, Any], planned: Mapping[str, Any]) -> List[Tuple[str, Any, Any]]:
    """Each session key that differs, as (what the owner calls it, from, to)."""
    now = {RUNTIME: runtime_kind_of(current), PROVIDER: None, MODEL: None}
    if now[RUNTIME] == RUNTIME_CLI:
        now = {**now, PROVIDER: current.get(PROVIDER), MODEL: current.get(MODEL)}
    return [(label, now[key], planned.get(key)) for key, label in CARD_LABELS if now[key] != planned.get(key)]


def update_plan(agent: Any, params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The configuration the update writes, or None when it does not touch the runtime: it
    names a runtime, or the agent runs as a session and the call names its CLI or model."""
    current = agent.configuration if isinstance(agent.configuration, dict) else {}
    said = _said_runtime(params)
    runtime = said or runtime_kind_of(current)
    if said is None and not (runtime == RUNTIME_CLI and (PROVIDER in params or MODEL in params)):
        return None
    return planned_configuration(current, params, runtime)


def runtime_card_lines(agent: Any, params: Mapping[str, Any]) -> List[str]:
    """The update card's lines for the runtime: '- runtime: api → cli', the CLI and model."""
    planned = update_plan(agent, params)
    if planned is None:
        return []
    return [change_line(label, old, new) for label, old, new in runtime_changes(agent.configuration or {}, planned)]


def _update_checked(db: Any, workspace_id: Any,
                    params: Mapping[str, Any]) -> Tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(the agent, the configuration the update writes or None, the refusal of a runtime that
    is unknown or can't run here)."""
    said = _said_runtime(params)
    if said is not None and said not in RUNTIME_KINDS:
        return None, None, _unknown(said, "said in the call", CHANGED)
    agent = _the_agent(db, workspace_id, params)
    planned = update_plan(agent, params) if agent is not None else None
    errors = _errors(planned) if planned is not None and planned.get(RUNTIME) == RUNTIME_CLI else []
    if errors:
        return agent, planned, _refusal(REFUSED.format(runtime=RUNTIME_CLI, errors="; ".join(errors), done=CHANGED))
    return agent, planned, None


def refused_before_the_card(db: Any, workspace_id: Any, action: str,
                            params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """An agent update whose runtime is unknown or can't run on this server is refused before
    the owner is asked: nothing is asked that the click could not do (F091)."""
    if action != UPDATE_AGENT:
        return None
    return _update_checked(db, workspace_id, params)[2]


def sets_the_runtime_on_update(update: Handler) -> Handler:
    """Wrap handlers_agents.update_agent: the runtime change is checked before anything changes,
    the other fields are changed as before (a refusal there writes no runtime), then the
    runtime is written and named in the change log."""
    @functools.wraps(update)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        agent, planned, refusal = _update_checked(db, workspace_id, params)
        if refusal:
            return refusal
        if planned is None:
            return await update(db, workspace_id, params)
        return await _update_with_runtime(db, workspace_id, agent, params, planned, update)
    return wrapped


def _the_agent(db: Any, workspace_id: Any, params: Mapping[str, Any]) -> Any:
    """The agent update_agent changes (its own lookup, in this workspace), or None."""
    from modules.tools.discovery.handlers_agents import _resolve_agent

    return _resolve_agent(db, workspace_id, params)[0]


async def _update_with_runtime(db: Any, workspace_id: Any, agent: Any, params: Dict[str, Any],
                               planned: Dict[str, Any], update: Handler) -> Dict[str, Any]:
    """The call's other fields through update_agent (when it has any), then the runtime: a
    refusal there writes nothing; a call that changes only the runtime never reaches it."""
    changes = [CHANGE.format(field=label, old=shown(old), new=shown(new))
               for label, old, new in runtime_changes(agent.configuration or {}, planned)]
    rest = _for_the_handler(params, planned)
    others = [key for key in rest if key not in NAMES_THE_AGENT and key not in SESSION_KEYS and not key.startswith("_")]
    if others or not changes:
        result = await update(db, workspace_id, rest)
        if not (isinstance(result, dict) and result.get("success")) or not changes:
            return result
    else:
        result = {"success": True, "agent_id": agent.id, "changes": []}
    _write(db, agent, planned)
    logger.info("[PlatformExecutor] Agent %s runtime: %s", agent.id, ", ".join(changes))
    logged = [*result.get("changes", []), *changes]
    return {**result, "changes": logged, "message": f"Agent '{agent.name}' updated: {', '.join(logged)}"}


__all__ = ["planned_configuration", "refused_before_the_card", "runtime_card_lines", "runtime_changes",
           "sets_the_runtime_on_create", "sets_the_runtime_on_update", "update_plan"]
