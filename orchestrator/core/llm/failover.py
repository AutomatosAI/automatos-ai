"""The one failover model, and when a call may use it (PRD-256 US-008, Decision D5).

There is none by default. ``LLM_FAILOVER_MODEL`` in config.py is empty, so a provider
that rate-limits or refuses a call fails the turn honestly: the typed error reaches
``consumers/chatbot/turn_errors.py`` and nothing answers on another model. The key is
the only place a failover could be set; when an operator sets it, a refused call is
asked once more on that model, on the same provider, and the reply's ``model`` (which
the receipts frame carries) names the model that answered.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional

from config import config

logger = logging.getLogger(__name__)

# The typed refusals of core/llm/clients/openai_compatible_client.py, by name (as
# turn_errors reads them), and the HTTP status an SDK error carries for a rate limit.
REFUSAL_ERRORS = frozenset({"ProviderRateLimitError", "ProviderModelUnavailableError"})
RATE_LIMITED_STATUS = 429


def failover_model() -> Optional[str]:
    """The operator's failover model, or None (the default: no failover)."""
    value = (getattr(config, "LLM_FAILOVER_MODEL", "") or "").strip()
    return value or None


def _is_refusal(exc: BaseException) -> bool:
    return type(exc).__name__ in REFUSAL_ERRORS or getattr(exc, "status_code", None) == RATE_LIMITED_STATUS


def failover_for(exc: BaseException, model: Optional[str]) -> Optional[str]:
    """The model to ask again after ``exc`` refused ``model``, or None to fail the call."""
    target = failover_model()
    if target is None or not _is_refusal(exc) or target == (model or ""):
        return None
    return target


async def answer_on_failover(manager: Any, target: str, messages: List[Dict[str, Any]],
                             tools: Optional[List[Dict]]) -> Any:
    """Ask ``target`` once, through a manager like ``manager`` (same provider, key and
    usage attribution). Its own refusal is raised: the failover never fails over."""
    from core.llm.manager import LLMManager

    logger.warning("[failover] %s refused; asking the operator's failover model %s",
                   manager.config.model, target)
    sibling = LLMManager(config=dataclasses.replace(manager.config, model=target),
                         service_name=manager.service_name, **manager._tracking_ctx)
    return await sibling.generate_response(messages, tools)


__all__ = ["answer_on_failover", "failover_for", "failover_model"]
