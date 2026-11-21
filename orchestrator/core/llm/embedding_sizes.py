"""Sizes that agree at start-up (PRD-278 US-502, as the Wave 5 review settled it).

An index takes vectors of one size. Start-up compares the provider's vectors with each index
that is really in use:

- ``S3_VECTORS_DIMENSION``, the document index's, when ``S3_VECTORS_ENABLED`` is on (the index
  is made at that size and refuses any other);
- ``FIELD_EMBEDDING_DIM``, the memory store's collections', when the memory store answers at
  start (``memory_store_answers``). Its address has a default, so a set address says nothing.

**Bedrock refuses to start** when an index in use expects another size than its vectors have:
``VECTOR_STORE_DIMENSIONS``, cut to the model's largest output when that is smaller. Bedrock
is new, so no running deployment is stopped by this, and every new install meets it (Titan V2
gives 1,024; the sizes default to 2,048). With Bedrock every one of these settings is read from
the environment, never from a stored setting, so the message's fix needs no settings page.

**Any other provider never refuses.** When its model is known (OpenRouter's and OpenAI's
tables) and smaller than an index in use, one warning names the settings and both numbers,
and the install starts, as it did before Wave 5.

The size the vectors are cut to is not compared on its own: nothing a write reaches is made at
that size for Bedrock (its vectors report their real length, which ingestion and the code index
follow), and for another provider the comparison would refuse installs that work today.

It runs once per worker, before that worker listens. For Bedrock it reads only the
environment, so every worker and replica sees the same. For another provider it reads the
``embeddings`` system settings: a replica that starts before the boot leader has seeded them
reads the environment instead, and can only warn. A stored size that is not a number never
stops a start (``embedding_manager``: logged once, ``VECTOR_STORE_DIMENSIONS`` used).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

PAGE = "docs/deployment-infrastructure/bedrock-embeddings.md"
BEDROCK = "bedrock"
# How long start-up waits for the memory store to answer before treating it as not in use.
MEMORY_STORE_PROBE_TIMEOUT_S = 3.0


class EmbeddingSettingsError(RuntimeError):
    """The embedding settings cannot work: Bedrock vectors an index in use cannot take, or a
    Bedrock model that is not supported."""


def model_dimension(provider: str, model: Optional[str]) -> Optional[int]:
    """The most dimensions ``provider``'s ``model`` produces, from a table; None when unknown.
    Raises ``EmbeddingSettingsError`` for a Bedrock model that is not supported."""
    from core.llm.clients.bedrock_embedding import UnsupportedBedrockEmbeddingModel, supported_model
    from core.llm.clients.openai_embedding import openai_model_dimension
    from core.llm.clients.openrouter_embedding import OPENROUTER_EMBEDDING_MODELS

    if provider == BEDROCK:
        try:
            return supported_model(model or "").dimension
        except UnsupportedBedrockEmbeddingModel as exc:
            raise EmbeddingSettingsError(str(exc)) from None
    if provider == "openrouter":
        known = OPENROUTER_EMBEDDING_MODELS.get(model or "")
        return known[0] if known else None
    if provider == "openai":
        return openai_model_dimension(model)
    return None


async def memory_store_answers(transport: Any = None) -> bool:
    """True when the memory store (``QDRANT_URL``) answers a listing of its collections within
    ``MEMORY_STORE_PROBE_TIMEOUT_S``: the sign it is in use. No address, or no answer: False."""
    if not config.QDRANT_URL:
        return False
    from core.qdrant.client import QdrantRestClient

    client = QdrantRestClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None,
                              timeout=MEMORY_STORE_PROBE_TIMEOUT_S, transport=transport)
    try:
        await client.get_collections()
        return True
    except Exception as exc:  # noqa: BLE001 — logged; a store that does not answer is not compared
        logger.info("Embedding sizes: the memory store did not answer at start (%s); FIELD_EMBEDDING_DIM "
                    "is not compared", type(exc).__name__)
        return False
    finally:
        await client.aclose()


async def indexes_in_use(field_matters: bool) -> List[Tuple[str, int]]:
    """Each index size setting whose index is in use, with its value. The memory store is asked
    only when ``field_matters`` (its size would not agree), so an agreeing install sends nothing."""
    sizes = []
    if config.S3_VECTORS_ENABLED:
        sizes.append(("S3_VECTORS_DIMENSION", int(config.S3_VECTORS_DIMENSION)))
    if field_matters and await memory_store_answers():
        sizes.append(("FIELD_EMBEDDING_DIM", int(config.FIELD_EMBEDDING_DIM)))
    return sizes


def _named(sizes: List[Tuple[str, int]]) -> str:
    return "; ".join(f"{name} is {size}" for name, size in sizes)


def bedrock_refusal(model: str, produced: int, length: int, gaps: List[Tuple[str, int]]) -> EmbeddingSettingsError:
    """The refusal: the model, its vectors' length, each index in use that expects another size,
    and the fix, made in the environment (the API is down while an operator reads it)."""
    cut = f"VECTOR_STORE_DIMENSIONS is {config.VECTOR_STORE_DIMENSIONS}"
    if int(config.VECTOR_STORE_DIMENSIONS) > produced:
        cut += f", more than the model's {produced}"
    return EmbeddingSettingsError(
        f"EMBEDDING_PROVIDER is bedrock: the model '{model}' produces at most {produced} dimensions, so its "
        f"vectors here are {length} long ({cut}). These indexes in use expect another size: {_named(gaps)}. "
        f"Set each of them to {length} in the environment and start again. With Bedrock every size is read "
        f"from the environment: VECTOR_STORE_DIMENSIONS wins over the stored 'dimensions' system setting, and "
        f"EMBEDDING_PROVIDER and EMBEDDING_MODEL over the stored 'provider' and 'model', so no settings page is "
        f"needed. An index already built at another size has to be built again. See {PAGE}.")


async def _check_bedrock(model: str, produced: int, cut_to: int) -> None:
    length = min(cut_to, produced)
    sizes = await indexes_in_use(field_matters=int(config.FIELD_EMBEDDING_DIM) != length)
    gaps = [(name, size) for name, size in sizes if size != length]
    if gaps:
        raise bedrock_refusal(model, produced, length, gaps)
    logger.info("Embedding sizes agree: Bedrock '%s' vectors are %d long", model, length)


async def _warn_if_small(provider: str, model: Optional[str], produced: int) -> None:
    sizes = await indexes_in_use(field_matters=int(config.FIELD_EMBEDDING_DIM) > produced)
    small_for = [(name, size) for name, size in sizes if size > produced]
    if not small_for:
        return
    logger.warning(
        "The embedding model '%s' (provider %s) produces at most %d dimensions, fewer than these indexes in use "
        "expect: %s. Writes to them will fail until each is set to %d in the environment and the index is built "
        "again. The stored 'provider', 'model' and 'dimensions' system settings win over EMBEDDING_PROVIDER, "
        "EMBEDDING_MODEL and VECTOR_STORE_DIMENSIONS. Starting anyway. See %s.",
        model, provider, produced, _named(small_for), produced, PAGE)


async def verify_embedding_sizes() -> None:
    """Start-up: compare the provider's vectors with the indexes in use. Raises, aborting boot,
    only for Bedrock (a size an index cannot take, or a model that is not supported); for any
    other provider at most one warning. The settings read runs in a worker thread."""
    from core.llm.embedding_manager import read_embedding_choice

    choice = await asyncio.to_thread(read_embedding_choice)
    if not choice.is_set:
        return
    produced = model_dimension(choice.provider, choice.model)
    if produced is None:
        logger.info("Embedding sizes not checked: the dimensions of '%s' (%s) are not known here",
                    choice.model, choice.provider)
        return
    if choice.provider == BEDROCK:
        await _check_bedrock(choice.model, produced, choice.dimension)
    else:
        await _warn_if_small(choice.provider, choice.model, produced)


__all__ = ["EmbeddingSettingsError", "MEMORY_STORE_PROBE_TIMEOUT_S", "bedrock_refusal", "indexes_in_use",
           "memory_store_answers", "model_dimension", "verify_embedding_sizes"]
