"""The one sentence a model call with no key fails with (#1678).

A self-hoster with only an OpenAI key was told "OpenRouter API key not configured" when Auto,
on Anthropic's claude-haiku-5-5, had no key: the call had fallen back to OpenRouter, which had
none either. The sentence names the provider and model that need the key, and both ways out,
as Settings → Orchestrator's readiness line does (``services.readiness``). It starts with
``NO_KEY_LEAD`` so a chat turn can tell it from a provider's own wording
(``consumers.chatbot.turn_errors``), and it fits the 200 characters a turn error keeps.
"""
from __future__ import annotations

from typing import Optional

NO_KEY_LEAD = "No API key for "
OPENROUTER = "openrouter"
_VOWELS = frozenset("AEIOU")


def provider_label(provider: Optional[str]) -> str:
    """The provider's name as Settings shows it, or its slug when the registry has none."""
    from core.llm.providers import get_spec  # at call time: the registry imports the clients, which import this

    spec = get_spec(provider)
    return spec.label if spec else str(provider or "its provider")


def no_key_sentence(provider: Optional[str], model: Optional[str]) -> str:
    """``No API key for Anthropic claude-haiku-5-5: add an Anthropic or OpenRouter key in
    Settings → Model keys, or choose a model whose provider has a key.`` OpenRouter is offered
    only for a model it can run that isn't on OpenRouter already."""
    from core.llm.providers import normalize_slug, openrouter_prefix_for

    label = provider_label(provider)
    slug = normalize_slug(provider) or provider
    via_openrouter = slug != OPENROUTER and bool(openrouter_prefix_for(slug))
    article = "an" if label[:1].upper() in _VOWELS else "a"
    keys = f"{article} {label} or OpenRouter key" if via_openrouter else f"{article} {label} key"
    named = f"{label} {model}" if model else label
    return (f"{NO_KEY_LEAD}{named}: add {keys} in Settings → Model keys, "
            "or choose a model whose provider has a key.")


__all__ = ["NO_KEY_LEAD", "no_key_sentence", "provider_label"]
