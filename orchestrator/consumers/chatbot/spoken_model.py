"""Spoken turns answer on a fast model (``SPOKEN_MODE_MODEL``); typed turns keep the agent's own.

28 Oct 2026: a spoken quick ask (``POST /api/chat`` with ``spokenMode``) took 26-32 s a turn on
Auto's own model, claude-opus-5-5. The 11 Oct benchmark scored claude-haiku-5-5 the same (51/65)
at about 3% of the cost, and a person waiting on a spoken answer hears every second of the wait.

``StreamingChatService._stream_response_with_agent_scoped`` is wrapped. On a spoken turn (Auto
Live's too) the agent the turn activates answers on the spoken model: its first call, every tool
round and any retry. Only the model changes. The agent's persona, tools, memory, prompt and locale
line are the turn's as before. Every call is booked, and its credits charged, on the model that
answered it, because the manager that makes the call names that model. The agent the factory
holds is never changed: the turn gets a copy.

The spoken model is reached by the factory's own rules for the agent's model: the provider/model
correction, the orchestrator seat's model policy (both spoken surfaces are Auto's), and the key
that pays (the workspace's own key first; its own keys only when its credits can't pay for the
platform's). The turn stays on the agent's own model, and says why once at info, when the setting
is empty, when the agent runs on it already, when the policy blocks it, when no key for its
provider is open to the workspace (a BYO-key-only provider in this edition, or no key at all), or
when the agent runs on the workspace's own key and the spoken model would run on the platform's.

Every turn logs the model it answers on: the id, never the person's words.
"""
from __future__ import annotations

import dataclasses
import functools
import logging
from typing import Any, AsyncGenerator, Callable, Optional

from config import config

logger = logging.getLogger(__name__)

TURN_MODEL_LOG = "[chat] turn model=%s spoken=%s"
STAYS_LOG = "[spoken] %s not used (%s): the turn answers on the agent's model %s"
# The spoken model is held to Auto's seat: the quick ask and Auto Live both talk to Auto.
ORCHESTRATOR_SEAT = True

Stream = Callable[..., AsyncGenerator[Any, None]]


@dataclasses.dataclass(frozen=True)
class SpokenModel:
    """Where a spoken turn's calls go: a provider and the provider's own id for the model."""

    provider: str
    model: str


def spoken_model() -> Optional[SpokenModel]:
    """The configured spoken model, or None when the setting is off (an empty model or provider)."""
    model = (config.SPOKEN_MODE_MODEL or "").strip()
    provider = (config.SPOKEN_MODE_PROVIDER or "").strip().lower()
    return SpokenModel(provider=provider, model=model) if model and provider else None


def model_of(runtime: Any) -> Optional[str]:
    """The model an agent runtime's calls go to."""
    return getattr(getattr(getattr(runtime, "llm_manager", None), "config", None), "model", None)


async def spoken_turn_runtime(factory: Any, runtime: Any, workspace_id: Any) -> Any:
    """``runtime`` on the spoken model, for one spoken turn: a copy whose manager calls and books
    that model on the key that pays for it. ``runtime`` itself when the turn can't use it (off,
    already on it, or the reason logged once at info)."""
    wanted = spoken_model()
    if wanted is None:
        return runtime
    provider, model = factory._resolve_provider_for_model(wanted.provider, wanted.model)
    agent_model = model_of(runtime)
    if model == agent_model:
        return runtime
    workspace = factory._effective_workspace(workspace_id, runtime)
    reason = _policy_refusal(factory, workspace, provider, model)
    resolved = None if reason else await _spoken_key(factory, runtime, provider, workspace)
    reason = reason or _key_refusal(runtime, provider, resolved)
    if reason:
        logger.info(STAYS_LOG, model, reason, agent_model)
        return runtime
    manager = _spoken_manager(factory, runtime, provider, model, resolved, workspace)
    return dataclasses.replace(runtime, llm_manager=manager, is_byok=resolved.is_byok,
                               resolved_provider=resolved.provider or provider)


def _policy_refusal(factory: Any, workspace: Any, provider: str, model: str) -> Optional[str]:
    """Why the spoken model may not answer in this workspace, or None when it may."""
    from core.llm.model_policy import check_model_for_agent
    from core.llm.providers import enum_for

    if enum_for(provider) is None:
        return f"{provider} has no chat client"
    allowed, reason = check_model_for_agent(
        factory.db_session, workspace, model, orchestrator_seat=ORCHESTRATOR_SEAT, provider=provider,
    )
    return None if allowed else reason


async def _spoken_key(factory: Any, runtime: Any, provider: str, workspace: Any) -> Any:
    """The key that pays for the spoken model's calls, by the factory's own rule (PRD-265: the
    workspace's own keys only when its credits can't pay for the platform's)."""
    own_keys_only = factory._platform_unfunded(workspace) is not None
    return await factory._resolve_api_key(
        provider, runtime.metadata.name, workspace_id=workspace, own_keys_only=own_keys_only,
    )


def _key_refusal(runtime: Any, provider: str, resolved: Any) -> Optional[str]:
    """Why the key found can't carry the turn, or None when it can."""
    if resolved is None or not resolved.api_key:
        return f"no {provider} key is open to this workspace"
    if runtime.is_byok and not resolved.is_byok:
        return "the agent runs on the workspace's own key"
    return None


def _spoken_manager(factory: Any, runtime: Any, provider: str, model: str, resolved: Any, workspace: Any) -> Any:
    """The spoken model's manager: the agent's settings and output budget, capped by the spoken
    model's own ceiling, on the resolved key, booked to the agent and the workspace."""
    from core.llm import LLMConfig, LLMManager
    from core.llm.providers import enum_for

    agent = runtime.llm_manager.config
    ceiling = factory._model_ceiling(model)
    llm_config = LLMConfig(
        provider=enum_for(provider), model=model, temperature=agent.temperature,
        max_tokens=factory._output_budget(agent.max_tokens, ceiling), output_ceiling=ceiling,
        api_key=resolved.api_key, top_p=agent.top_p, frequency_penalty=agent.frequency_penalty,
        presence_penalty=agent.presence_penalty, stop=agent.stop, timeout=agent.timeout,
    )
    return LLMManager(config=llm_config, workspace_id=workspace, agent_id=runtime.agent_id,
                      is_byok=resolved.is_byok)


class TurnAgents:
    """The turn's agent factory: everything is the service's own factory's, except that the agent
    it activates answers on the turn's model (the spoken model on a spoken turn), and says which."""

    def __init__(self, factory: Any, *, spoken: bool) -> None:
        self._factory = factory
        self._spoken = spoken

    def __getattr__(self, name: str) -> Any:
        return getattr(self._factory, name)

    async def activate_agent(self, agent_id: int, **kwargs: Any) -> Any:
        runtime = await self._factory.activate_agent(agent_id, **kwargs)
        if runtime is None:
            return None
        if self._spoken:
            runtime = await self._on_the_spoken_model(runtime, kwargs.get("workspace_id"))
        logger.info(TURN_MODEL_LOG, model_of(runtime), str(self._spoken).lower())
        return runtime

    async def _on_the_spoken_model(self, runtime: Any, workspace_id: Any) -> Any:
        try:
            return await spoken_turn_runtime(self._factory, runtime, workspace_id)
        except Exception:  # a lookup that failed keeps the turn on the agent's own model
            logger.exception("[spoken] the spoken model could not be set up; the turn answers on %s",
                             model_of(runtime))
            return runtime


def answers_on_the_turns_model(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_response_with_agent_scoped``: the agent the turn
    activates answers on the turn's model, the spoken model on a spoken turn. The service is
    built for one turn; its own factory is put back when the turn ends. A service built
    without a factory runs its turn as it is."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        factory = getattr(chat, "agent_factory", None)
        if factory is not None:
            chat.agent_factory = TurnAgents(factory, spoken=bool(kwargs.get("spoken_mode")))
        try:
            async for chunk in turn(chat, *args, **kwargs):
                yield chunk
        finally:
            if factory is not None:
                chat.agent_factory = factory
    return wrapped


__all__ = [
    "SpokenModel", "TurnAgents", "answers_on_the_turns_model", "model_of", "spoken_model", "spoken_turn_runtime",
]
