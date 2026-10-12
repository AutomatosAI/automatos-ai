"""The model a new agent takes when none is given (P256-FIX-N13-8, F415).

Night 13: INVOICE DESK, created by Auto at the owner's ask, got
``google/gemini-2.5-flash`` on OpenRouter while the workspace ran Auto on
``anthropic/claude-haiku-5-5``: the create path seeded ``model_config`` from
``core/llm/defaults`` instead of the workspace's own default.

The workspace's default is the ``orchestrator_llm`` provider and model in system
settings (Settings > Orchestrator). Both must be set for the pair to be used; with
either missing the defaults module's pair is used whole, never half of each. The
pair goes through ``vendor_fit.fitted`` (P256-FIX-T1), so a model the settings'
provider does not serve moves to a keyed provider that does.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, get_default_model_config
from core.llm.providers import normalize_slug

ORCHESTRATOR_CATEGORY = "orchestrator_llm"
ORCHESTRATOR_SERVICE = "orchestrator"


def _setting(key: str, legacy_key: str):
    from core.llm import manager

    return (manager.get_system_setting(ORCHESTRATOR_CATEGORY, key)
            or manager.get_system_setting(ORCHESTRATOR_CATEGORY, legacy_key))


def default_route() -> Tuple[str, str]:
    """(provider, model) for a new agent given no model: the workspace's own
    default when the settings name both, fitted to each other, else the defaults."""
    from core.llm.vendor_fit import fitted

    provider, model = _setting("provider", "llm_provider"), _setting("model", "llm_model")
    if not provider or not model:
        return DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
    fitted_provider, fitted_model = fitted(ORCHESTRATOR_SERVICE, model)
    chosen = fitted_provider or provider
    return normalize_slug(chosen) or chosen, fitted_model or model


def new_agent_model_config() -> Dict[str, Any]:
    """A fresh ``model_config`` for a new agent: the defaults module's shape with
    the workspace's default provider and model."""
    provider, model = default_route()
    return {**get_default_model_config(), "provider": provider, "model_id": model}


__all__ = ["default_route", "new_agent_model_config"]
