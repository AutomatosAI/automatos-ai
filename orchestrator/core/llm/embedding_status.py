"""Whether this install can embed text, and what each surface says when it cannot.

Memory (the durable store), knowledge search (RAG) and tool ranking all embed through an
``EmbeddingManager``. With no embedding provider (the provider set to ``disabled``, or no key
found for it: OpenRouter by default) the manager falls back to ``DeterministicEmbeddingProvider``,
which refuses to embed (``is_degraded``). Nothing can then be stored, recalled or searched by
meaning. On a self-hosted install, 7 Nov 2026, that showed only in the logs (346 errors in 72 h)
while Auto told its owner "my memory is empty".

The reason code and the sentences live here, in the shared kernel, so every surface says the
same thing: the readiness item (Settings → Orchestrator, "What Auto depends on", and Automatos
Go), the memory health tile (Knowledge → Memory), Auto's memory section and its memory tools.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The reason code: no embedding provider, so memory and knowledge search are off.
NO_EMBEDDING_PROVIDER = "no_embedding_provider"

# What the owner reads, with the fix.
OWNER_SENTENCE = (
    "Memory and knowledge search are off: no embedding provider. Add an embeddings key for your "
    "embedding provider (OpenRouter, unless Settings → System Settings → Embeddings names another) in "
    "Settings → API Keys, then restart the backend so memory picks it up."
)

# What Auto is told, in its memory section and in its memory tools' answers.
MEMORY_OFF = (
    "Memory is off on this install (no embedding provider): nothing can be stored or recalled. When asked "
    "what you remember, or to remember something, say memory is off on this install, never that it is "
    "empty or that nothing is stored. The owner turns it on by adding an embedding key (OpenRouter by "
    "default) in Settings → API Keys and restarting the backend."
)


def provider_ready(manager: Any) -> bool:
    """True when ``manager`` (an ``EmbeddingManager``) has a real provider; False when it fell back
    to the degraded stand-in. Loads the provider as the first embed would: a settings and key-store
    read, so call it off the event loop. The manager keeps what it loaded, so a key added later
    needs a restart."""
    manager._ensure_provider()
    return not getattr(manager.provider, "is_degraded", False)


async def store_cannot_embed(store: Optional[Any]) -> bool:
    """True when ``store`` (the durable memory store: ``embeddings_ready``) is known to have no
    embedding provider, so nothing can be stored in it or recalled from it. No store says nothing,
    and neither does a check that fails (logged): only a provider known to be missing is "off".
    The check reads settings and the key store the first time, so it runs in a thread."""
    if store is None:
        return False
    try:
        return not await asyncio.to_thread(store.embeddings_ready)
    except Exception:  # noqa: BLE001 — logged; the caller's own answer stands
        logger.exception("Could not tell whether the memory store has an embedding provider")
        return False


__all__ = ["MEMORY_OFF", "NO_EMBEDDING_PROVIDER", "OWNER_SENTENCE", "provider_ready", "store_cannot_embed"]
