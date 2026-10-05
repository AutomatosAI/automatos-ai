"""A workspace key's own endpoint, for providers that live at the customer's URL (#873).

Azure OpenAI (Microsoft Foundry) has no fixed address: every resource has its own
(``https://<resource>.openai.azure.com``). A workspace key saved without one could
only reach the API service's ``AZURE_OPENAI_ENDPOINT``, which on the hosted edition
is not the customer's resource. The endpoint is now saved with the key
(``user_api_keys.base_url``), and a BYOK ``LLMManager`` hands it to the client.

The server calls that URL, so on the hosted (saas) edition:
- it must be https, and must not resolve into a private, loopback, link-local or
  metadata range, or a ``WEB_ACCESS_DENY`` host, when the key is saved;
- every request to it goes through ``core.security.pinned_http``, which checks the
  address again and connects to exactly that one, with redirects off.

Self-hosted installs skip both: the operator is the user, and a bank's Azure private
endpoint resolves to a 10.x address on purpose.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Optional
from urllib.parse import urlsplit

from config import config
from core.llm import providers
from core.llm.clients.azure_v1 import v1_base_url

EDITION_SAAS = "saas"


class EndpointRefused(ValueError):
    """The endpoint can't be used. The message is written for the user."""


def _saas(edition: Optional[str] = None) -> bool:
    return (edition or config.AUTH_EDITION) == EDITION_SAAS


def takes_endpoint(provider: Optional[str]) -> bool:
    """Does a key for this provider carry its own endpoint?"""
    spec = providers.get_spec(provider)
    return bool(spec and spec.endpoint_placeholder)


def endpoint_required(provider: Optional[str], edition: Optional[str] = None) -> bool:
    """Must a key for this provider be saved with an endpoint?

    On saas the env endpoint belongs to the platform, never to a workspace, so the
    key needs its own. Self-hosted, the operator's ``AZURE_OPENAI_ENDPOINT`` can
    stand in.
    """
    if not takes_endpoint(provider):
        return False
    return _saas(edition) or not config.AZURE_OPENAI_ENDPOINT


def clean_endpoint(provider: Optional[str], raw: Optional[str], edition: Optional[str] = None) -> Optional[str]:
    """The endpoint to store with a key, normalised to its v1 base URL; None when blank.

    Raises ``EndpointRefused`` for a provider that takes no endpoint, a URL with no
    host, or (saas) one the server may not call.
    """
    text = (raw or "").strip()
    if not text:
        return None
    if not takes_endpoint(provider):
        raise EndpointRefused(f"A {provider} key takes no endpoint.")
    if "@" in urlsplit(text if "://" in text else f"https://{text}").netloc:
        # A credential in the URL would be stored and logged in clear; the key field is encrypted.
        raise EndpointRefused("Put the key in the key field, not in the endpoint URL.")
    try:
        url = v1_base_url(text)
    except ValueError as exc:
        raise EndpointRefused(str(exc)) from exc
    refusal = endpoint_refusal(url, edition)
    if refusal:
        raise EndpointRefused(refusal)
    return url


def endpoint_refusal(url: str, edition: Optional[str] = None) -> Optional[str]:
    """Why the server may not call ``url`` in this edition, or None when it may."""
    if not _saas(edition):
        return None
    if not url.lower().startswith("https://"):
        return "The endpoint must start with https://."
    from core.security.web_access import resolve_outbound

    target = resolve_outbound(url, enforce_switch=False)
    return None if target.ok else f"The endpoint can't be used: {target.reason}."


def endpoint_http_client(timeout: float, edition: Optional[str] = None) -> Any:
    """The httpx client for calls to a key's own endpoint: pinned on saas, else None (the SDK's own)."""
    if not _saas(edition):
        return None
    from core.security.pinned_http import pinned_client

    return pinned_client(timeout)


def with_key_endpoint(llm_config: Any, workspace_id: Any) -> Any:
    """``llm_config`` with the endpoint saved with the workspace's key, when its provider takes one.

    Called for a BYOK ``LLMManager`` whose config has no base URL: the mission and the
    chat paths both build their config from the key alone.
    """
    if not workspace_id or not takes_endpoint(llm_config.provider.value):
        return llm_config
    from core.llm.key_resolver import byok_endpoint

    endpoint = byok_endpoint(llm_config.provider.value, workspace_id)
    return replace(llm_config, base_url=endpoint, endpoint_from_key=True) if endpoint else llm_config


__all__ = [
    "EndpointRefused", "clean_endpoint", "endpoint_http_client", "endpoint_refusal",
    "endpoint_required", "takes_endpoint", "with_key_endpoint",
]
