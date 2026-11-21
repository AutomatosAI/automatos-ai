"""Sizes that agree, or no start (PRD-278 US-502).

An embedding model that produces fewer dimensions than an index expects cannot fill it: the
first write fails. With a real provider set, start-up compares the most dimensions the model
produces with the three size settings:

- the size the provider's vectors are cut to: ``VECTOR_STORE_DIMENSIONS``, or the embeddings
  ``dimensions`` system setting when it names another size;
- ``S3_VECTORS_DIMENSION``, the document index's, when S3 Vectors is on;
- ``FIELD_EMBEDDING_DIM``, the memory store's, when a Qdrant URL is set.

Any of them larger than the model refuses the start, and the message names each setting, both
numbers and the value to set. A model that produces more is cut to fit, as before. The model's
size is known from a table, never from a probe text: Bedrock's (``bedrock_embedding``, and a
Bedrock model ID it does not hold is refused here too, with the list), OpenRouter's and
OpenAI's. A model none of them knows (a local Hugging Face model, an OpenRouter model not in the
table) is not checked, and one line says so. With no provider set, nothing is checked.

It runs before the server listens, after the boot leader has seeded the settings it reads, in a
worker thread: the settings are a database read. It makes no network call.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)

PAGE = "docs/deployment-infrastructure/bedrock-embeddings.md"


class EmbeddingSettingsError(RuntimeError):
    """The embedding settings cannot work: a model too small for an index, or one not supported."""


def model_dimension(provider: str, model: Optional[str]) -> Optional[int]:
    """The most dimensions ``provider``'s ``model`` produces, from a table; None when unknown.
    Raises ``EmbeddingSettingsError`` for a Bedrock model that is not supported."""
    from core.llm.clients.bedrock_embedding import UnsupportedBedrockEmbeddingModel, supported_model
    from core.llm.clients.openai_embedding import openai_model_dimension
    from core.llm.clients.openrouter_embedding import OPENROUTER_EMBEDDING_MODELS

    if provider == "bedrock":
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


def expected_sizes(cut_to: int, cut_setting: str) -> List[Tuple[str, int]]:
    """Each size setting in use, with its value: the cut size always, an index's when it is on."""
    sizes = [(cut_setting, int(cut_to))]
    if config.S3_VECTORS_ENABLED:
        sizes.append(("S3_VECTORS_DIMENSION", int(config.S3_VECTORS_DIMENSION)))
    if config.QDRANT_URL:
        sizes.append(("FIELD_EMBEDDING_DIM", int(config.FIELD_EMBEDDING_DIM)))
    return sizes


def size_error(provider: str, model: Optional[str], produced: int,
               too_large: List[Tuple[str, int]]) -> EmbeddingSettingsError:
    """The refusal: the model, what it produces, each setting with its value, and what to set."""
    settings = "; ".join(f"{name} is {size}" for name, size in too_large)
    return EmbeddingSettingsError(
        f"The embedding model '{model}' (provider {provider}) produces at most {produced} dimensions, "
        f"fewer than these settings expect: {settings}. Set each to {produced} (all the size settings "
        f"must agree) and start again; an index already built at another size has to be indexed "
        f"again. See {PAGE}.")


def check_embedding_sizes() -> None:
    """Raise ``EmbeddingSettingsError`` when a real provider's model is smaller than a size
    setting, or is a Bedrock model that is not supported. A settings read: run it in a thread."""
    from core.llm.embedding_manager import read_embedding_choice

    choice = read_embedding_choice()
    if not choice.is_set:
        return
    produced = model_dimension(choice.provider, choice.model)
    if produced is None:
        logger.info("Embedding sizes not checked: the dimensions of '%s' (%s) are not known here",
                    choice.model, choice.provider)
        return
    too_large = [(name, size) for name, size in expected_sizes(choice.dimension, choice.dimension_setting)
                 if size > produced]
    if too_large:
        raise size_error(choice.provider, choice.model, produced, too_large)
    logger.info("Embedding sizes agree: '%s' (%s) produces %d dimensions", choice.model, choice.provider, produced)


async def verify_embedding_sizes() -> None:
    """Start-up: ``check_embedding_sizes`` off the event loop. Raises, aborting boot, on a mismatch."""
    await asyncio.to_thread(check_embedding_sizes)


__all__ = ["EmbeddingSettingsError", "check_embedding_sizes", "expected_sizes", "model_dimension",
           "verify_embedding_sizes"]
