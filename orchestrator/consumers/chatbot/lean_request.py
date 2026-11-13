"""What a lean turn's request may add: a model from the agent's allow-list, and app context.

PRD-274, "Automatos for products", Wave 2 (§5.3, §5.4). On ``POST /api/chat`` a request
naming a lean agent (``consumers.chatbot.lean_turn``) may also send:

- ``model``: the model this turn answers on, one of the agent's
  ``configuration.model_allow_list`` entries (``{"provider", "model_id"}``). The turn moves
  to it the way a spoken turn moves to the spoken model (``consumers.chatbot.spoken_model``):
  the same provider correction, model policy and key rules, and when one of them refuses,
  the turn answers on the agent's own model and the log says why. Until headless access
  (Wave 1) puts an allow-list on the product's key, the agent's list is the only one.
- ``appContext``: what the host app knows (a cart, a ticket, the document being edited),
  as JSON or text, at most ``APP_CONTEXT_MAX_CHARS`` characters once rendered. It goes in
  the system prompt for this turn only. It is never saved with the message, never in the
  history, never distilled into memory and never in the turn's trace (its size is).

Anything else is refused before the message is saved: either field for an agent that is
not lean, a model not on the list, and app context over the limit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

MODEL_ALLOW_LIST_KEY = "model_allow_list"
APP_CONTEXT_HEADING = "## App context"
APP_CONTEXT_NOTE = "What the app you are answering inside knows for this turn only. Use it; never repeat it back whole."
APP_CONTEXT_CHARS_KEY = "app_context_chars"


class LeanRequestRefused(ValueError):
    """The request asks for what its agent does not allow; the message says what."""


@dataclass(frozen=True)
class TurnModel:
    """The model a turn answers on: a provider and that provider's id for the model."""

    provider: str
    model: str


@dataclass(frozen=True)
class LeanRequest:
    """What a lean turn's request added: a model, app context, the end user it is for, or none."""

    model: Optional[TurnModel] = None
    app_context: Optional[str] = None
    end_user: Optional[Any] = None  # a server key's turn's end user (core.security.end_user_turn.EndUser)


NOTHING_ASKED = LeanRequest()


def allow_list_errors(configuration: Any) -> List[str]:
    """Errors (empty when valid) for ``configuration.model_allow_list``: absent, or a list of
    ``{"provider": str, "model_id": str}``. The catalogue and the model policy judge each
    model when a turn asks for it."""
    raw = configuration.get(MODEL_ALLOW_LIST_KEY) if isinstance(configuration, Mapping) else None
    if raw is None:
        return []
    shaped = isinstance(raw, list) and all(
        isinstance(e, Mapping) and _text(e.get("provider")) and _text(e.get("model_id")) for e in raw
    )
    if shaped:
        return []
    return [f'configuration.{MODEL_ALLOW_LIST_KEY} must be a list of {{"provider": ..., "model_id": ...}}']


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def checked_lean_request(agent: Any, model: Optional[str], app_context: Any) -> LeanRequest:
    """The request's model and app context for ``agent`` (None when the request named no
    agent), or :class:`LeanRequestRefused` saying why they cannot be used."""
    from consumers.chatbot.lean_turn import is_lean

    if not model and app_context in (None, "", {}, []):
        return NOTHING_ASKED
    if agent is None or not is_lean(agent):
        raise LeanRequestRefused("model and appContext need a lean agent: name one in agentId.")
    return LeanRequest(model=_allowed_model(agent, model), app_context=_rendered(app_context))


def _allowed_model(agent: Any, model: Optional[str]) -> Optional[TurnModel]:
    if not model:
        return None
    configuration = getattr(agent, "configuration", None) or {}
    for entry in configuration.get(MODEL_ALLOW_LIST_KEY) or []:
        if isinstance(entry, Mapping) and entry.get("model_id") == model and _text(entry.get("provider")):
            return TurnModel(provider=entry["provider"].strip().lower(), model=model)
    raise LeanRequestRefused(f"{model} is not on this agent's model allow-list.")


def _rendered(app_context: Any) -> Optional[str]:
    from config import config

    if app_context in (None, "", {}, []):
        return None
    text = app_context if isinstance(app_context, str) else json.dumps(app_context, ensure_ascii=False, sort_keys=True)
    limit = int(getattr(config, "APP_CONTEXT_MAX_CHARS", 8000))
    if len(text) > limit:
        raise LeanRequestRefused(f"appContext is {len(text)} characters; the limit is {limit}.")
    return text


def with_app_context(llm_messages: List[Dict[str, Any]], app_context: Optional[str]) -> List[Dict[str, Any]]:
    """``llm_messages`` with the app context at the end of the system prompt, for this call
    only. A new list; unchanged when there is none."""
    if not app_context:
        return llm_messages
    block = f"{APP_CONTEXT_HEADING}\n{APP_CONTEXT_NOTE}\n\n{app_context}"
    first = llm_messages[0] if llm_messages else {}
    if first.get("role") == "system" and isinstance(first.get("content"), str):
        return [{**first, "content": f"{first['content']}\n\n{block}"}, *llm_messages[1:]]
    return [{"role": "system", "content": block}, *llm_messages]


def with_app_context_size(trace: Optional[Dict[str, Any]], app_context: Optional[str]) -> Optional[Dict[str, Any]]:
    """``trace`` with how long the turn's app context was, never what it said. A new dict."""
    if not app_context:
        return trace
    return {**(trace or {}), APP_CONTEXT_CHARS_KEY: len(app_context)}
