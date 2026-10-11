"""A model id that fits the provider it is sent to (P256-FIX-T1, F397/F400).

A service manager takes its provider from a settings category (``system_llm`` for the
distiller and the verifier) but a caller may name its own model. Once the System LLM
was Anthropic, the distiller sent ``google/gemini-2.5-flash`` and the verifier
``openai/gpt-4o-mini`` to the Anthropic client: a 404 on every turn.

``fitted`` checks the pair before the client is built. A vendor-prefixed id on a
provider that doesn't host vendor ids, or a bare id that names another vendor
(``claude-…`` on OpenRouter), goes to a provider that serves it and has a key;
with none, the call runs on the category's own model and one warning names the
service, the model and the provider. The defaults the verifier reads when no
setting names its models are picked here for the System LLM's vendor.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List, Optional, Tuple

from core.llm.defaults import (
    ANTHROPIC_VERIFIER_FALLBACK_MODEL,
    ANTHROPIC_VERIFIER_MODEL_MAPPING,
    VERIFIER_FALLBACK_MODEL,
    VERIFIER_MODEL_MAPPING,
)
from core.llm.providers import (
    ADAPTER_AZURE,
    ADAPTER_BEDROCK,
    ADAPTER_HUGGINGFACE,
    REGISTRY,
    ProviderSpec,
    env_api_key,
    get_spec,
    mismatched_vendor,
    normalize_slug,
    openrouter_prefix_for,
    routable_provider_names,
)

logger = logging.getLogger(__name__)

OPENROUTER = "openrouter"
ANTHROPIC = "anthropic"
SYSTEM_CATEGORY = "system_llm"
# Providers whose model field is their own id form (a deployment name, a Bedrock id
# or ARN, a HuggingFace repo): never read as another vendor's.
_OWN_ID_ADAPTERS = frozenset({ADAPTER_AZURE, ADAPTER_BEDROCK, ADAPTER_HUGGINGFACE})
# Distinct (service, model, provider) mismatches warned about once per process.
WARNED_MISMATCHES = 256

Route = Tuple[str, str]  # (provider, model)


def _vendor_for_prefix(head: str) -> Optional[str]:
    """The provider whose OpenRouter prefix is ``head/`` (``google/`` → google)."""
    wanted = f"{head.lower()}/"
    return next((slug for slug, spec in REGISTRY.items() if spec.openrouter_prefix == wanted), None)


def _routes_for_prefixed(spec: ProviderSpec, model: str) -> List[Route]:
    """Where a ``vendor/model`` id can run: its own vendor's id on this provider when
    the prefix names it, OpenRouter as written, then the vendor's direct API."""
    head, _, rest = model.partition("/")
    vendor = _vendor_for_prefix(head)
    if vendor == spec.slug:
        return [(spec.slug, rest), (OPENROUTER, model)]
    return [(OPENROUTER, model)] + ([(vendor, rest)] if vendor else [])


def _routes(spec: ProviderSpec, model: str) -> Optional[List[Route]]:
    """The providers to try for ``model``, best first; None when ``spec`` serves it."""
    if spec.adapter in _OWN_ID_ADAPTERS:
        return None
    if "/" in model:
        return None if spec.hosts_vendor_models else _routes_for_prefixed(spec, model)
    vendor = mismatched_vendor(spec.slug, model.lower(), routable_provider_names())
    if vendor is None:
        return None
    prefix = openrouter_prefix_for(vendor)
    return [(vendor, model)] + ([(OPENROUTER, f"{prefix}{model}")] if prefix else [])


def _has_key(slug: str, service_name: str) -> bool:
    """Does the manager's key chain (credential store, operator workspace, env) hold
    a key for ``slug``? The same tiers ``LLMManager._load_config_from_settings`` reads."""
    from core.llm import manager
    from core.llm.workspace_keys import get_platform_workspace_key

    spec = get_spec(slug)
    value = (spec.enum_value if spec else None) or slug
    cred = manager.get_credential_data(value, service_name=service_name) or {}
    return bool(cred.get("api_key") or cred.get("api_token")
                or get_platform_workspace_key(value) or env_api_key(slug))


@lru_cache(maxsize=WARNED_MISMATCHES)
def _warn_once(service_name: str, model: str, provider: str) -> None:
    logger.warning(
        "[vendor_fit] service '%s' asked for model '%s', which provider '%s' does not "
        "serve, and no provider that serves it has a key: using the category's own model",
        service_name, model, provider,
    )


def fitted(service_name: str, model: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """(provider, model) for a manager given ``model`` on its category's provider.

    Unchanged when no model is given or the settings can't be read (the manager
    raises its own error). A provider that serves the id keeps it; a mismatch moves
    to the first keyed provider that serves it, else the category's own model.
    """
    if not model:
        return None, model
    from core.llm import manager

    try:
        provider, own_model = manager.get_provider_and_model_from_settings(service_name)
    except ValueError:
        return None, model
    spec = get_spec(provider)
    routes = _routes(spec, model) if spec is not None else None
    if routes is None:
        return provider, model
    for slug, routed in routes:
        if _has_key(slug, service_name):
            logger.debug("[vendor_fit] %s: '%s' on '%s' runs as '%s' on '%s'",
                         service_name, model, provider, routed, slug)
            return (get_spec(slug).enum_value or slug), routed
    _warn_once(service_name, model, spec.slug)
    return provider, own_model


def _system_vendor() -> Optional[str]:
    from core.llm.manager import get_system_setting

    raw = get_system_setting(SYSTEM_CATEGORY, "provider") or get_system_setting(SYSTEM_CATEGORY, "llm_provider")
    return normalize_slug(raw)


def default_verifier_mapping() -> str:
    """The verifier's executor → verifier mapping when no setting or env names one."""
    return ANTHROPIC_VERIFIER_MODEL_MAPPING if _system_vendor() == ANTHROPIC else VERIFIER_MODEL_MAPPING


def default_verifier_fallback() -> str:
    """The verifier's model for an executor no mapping covers, when none is set."""
    return ANTHROPIC_VERIFIER_FALLBACK_MODEL if _system_vendor() == ANTHROPIC else VERIFIER_FALLBACK_MODEL


__all__ = ["default_verifier_fallback", "default_verifier_mapping", "fitted"]
