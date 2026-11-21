"""
AWS Bedrock Embedding Provider (PRD-278 US-501)
===============================================

``EMBEDDING_PROVIDER=bedrock`` embeds text on ``bedrock-runtime`` (``InvokeModel``), in the
cloud account and region the credentials belong to.

Credentials and region are the Bedrock chat client's, found the way its settings tier finds
them: the stored credential, the operator workspace's key and the env key
(``core.llm.manager.settings_credential_fields``), then the install's own role and region
when ``AWS_USE_WORKLOAD_IDENTITY`` is on (``core.llm.key_resolver.with_workload_identity``),
and the client is built by ``bedrock_client.runtime_client``. Nothing here looks for a
credential or a region on its own.

Supported models are the ones whose request and answer were read in AWS's documentation;
any other model ID is refused, with the list (``supported_model``). Amazon Titan Text
Embeddings V2 takes one text per call, so a batch is the manager's bounded set of single
calls (``EmbeddingManager.generate_embeddings_batch``: at most ``max_concurrent`` in
flight). Each call runs in the default executor (``run_blocking``, as the chat client does),
never on the event loop, with the embedding time limits: connect
``EMBEDDING_CONNECT_TIMEOUT_S``, read ``EMBEDDING_QUERY_TIMEOUT_S``, ``EMBEDDING_MAX_RETRIES``
retries (adaptive: botocore backs off on ``ThrottlingException``), and the whole call,
retries included, awaited for at most ``EMBEDDING_BATCH_TIMEOUT_S``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from config import config

from . import bedrock_client
from .base import BaseEmbeddingProvider, LLMConfig, LLMProvider, run_blocking
from .embedding_usage import CHARS_PER_TOKEN_ESTIMATE, record_embedding_usage

logger = logging.getLogger(__name__)

# The settings tier the provider's credentials are looked up for (core.llm.manager).
EMBEDDINGS_SERVICE = "embeddings"
USAGE_PROVIDER = "aws_bedrock"  # as the chat client books its calls
JSON = "application/json"


@dataclass(frozen=True)
class BedrockEmbeddingModel:
    """A Bedrock embedding model whose request and answer were read in AWS's documentation.

    ``dimensions``: the output sizes the model accepts, largest first; ``max_input_chars``
    and ``max_input_tokens``: its input limits.
    """

    model_id: str
    dimensions: Tuple[int, ...]
    max_input_chars: int
    max_input_tokens: int

    @property
    def dimension(self) -> int:
        """The most dimensions the model produces: what the three size settings may not exceed."""
        return self.dimensions[0]

    @property
    def max_chars(self) -> int:
        """Text past this is cut before the call: the character limit, or the token limit at
        the platform's ~4 characters a token, whichever is smaller."""
        return min(self.max_input_chars, self.max_input_tokens * CHARS_PER_TOKEN_ESTIMATE)


# Amazon Titan Text Embeddings V2 (docs.aws.amazon.com/bedrock/latest/userguide/
# titan-embedding-models.html and model-parameters-titan-embed-text.html): up to 8,192 tokens
# or 50,000 characters in; 1,024 (default), 512 or 256 dimensions out; one text per call.
TITAN_TEXT_EMBEDDINGS_V2 = "amazon.titan-embed-text-v2:0"
BEDROCK_EMBEDDING_MODELS: Dict[str, BedrockEmbeddingModel] = {
    TITAN_TEXT_EMBEDDINGS_V2: BedrockEmbeddingModel(TITAN_TEXT_EMBEDDINGS_V2, (1024, 512, 256), 50_000, 8_192),
}
# PRD-278 D-4: the first Bedrock embedding model, used when EMBEDDING_MODEL names none.
DEFAULT_BEDROCK_EMBEDDING_MODEL = TITAN_TEXT_EMBEDDINGS_V2
DOCS_PAGE = "docs/deployment-infrastructure/bedrock-embeddings.md"


class UnsupportedBedrockEmbeddingModel(ValueError):
    """A Bedrock embedding model ID this provider does not support."""


def supported_model(model_id: str) -> BedrockEmbeddingModel:
    """The supported model ``model_id`` names; raises, listing what is supported, for any other."""
    model = BEDROCK_EMBEDDING_MODELS.get((model_id or "").strip())
    if model is None:
        supported = ", ".join(f"{m.model_id} ({m.dimension} dimensions)" for m in BEDROCK_EMBEDDING_MODELS.values())
        raise UnsupportedBedrockEmbeddingModel(
            f"EMBEDDING_PROVIDER is bedrock, but the embedding model '{model_id}' is not supported. "
            f"Set EMBEDDING_MODEL to one of: {supported}. See {DOCS_PAGE}.")
    return model


def request_dimension(model: BedrockEmbeddingModel, wanted: int) -> int:
    """The output size to ask ``model`` for: the smallest it accepts that holds ``wanted``,
    else its largest (a longer answer is cut to ``wanted``)."""
    fitting = [size for size in model.dimensions if size >= wanted]
    return min(fitting) if fitting else model.dimension


def bedrock_settings_config(model_id: str) -> LLMConfig:
    """The credentials and region the Bedrock chat client's settings tier would use, for the
    embeddings tier: the stored credential, the platform keys, then the install's own role and
    region with ``AWS_USE_WORKLOAD_IDENTITY`` on."""
    from core.llm.key_resolver import with_workload_identity
    from core.llm.manager import settings_credential_fields

    fields = settings_credential_fields(LLMProvider.AWS_BEDROCK, EMBEDDINGS_SERVICE)
    return with_workload_identity(LLMConfig(provider=LLMProvider.AWS_BEDROCK, model=model_id,
                                            api_key=fields.api_key, base_url=fields.base_url,
                                            secret_key=fields.secret_key))


def embedding_client_bounds() -> Any:
    """The embedding time limits and retries, as a botocore ``Config`` for ``runtime_client``."""
    from botocore.config import Config

    return Config(connect_timeout=config.EMBEDDING_CONNECT_TIMEOUT_S, read_timeout=config.EMBEDDING_QUERY_TIMEOUT_S,
                  retries={"total_max_attempts": 1 + int(config.EMBEDDING_MAX_RETRIES), "mode": "adaptive"})


def parse_embedding(payload: Any, size: int) -> Tuple[List[float], int]:
    """The vector and the input's token count in an ``InvokeModel`` answer; raises when the
    answer holds no list of ``size`` numbers."""
    vector = payload.get("embedding") if isinstance(payload, dict) else None
    if (not isinstance(vector, list) or len(vector) != size
            or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in vector)):
        raise ValueError(f"Bedrock answered without an embedding of {size} numbers")
    tokens = payload.get("inputTextTokenCount")
    return [float(x) for x in vector], tokens if isinstance(tokens, int) else 0


def _error_code(exc: Exception) -> str:
    """AWS's error code for a refusal (never its message), else the exception's class."""
    response = getattr(exc, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return str(code) if code else type(exc).__name__


class BedrockEmbeddingProvider(BaseEmbeddingProvider):
    """Embeddings on ``bedrock-runtime``, with the chat client's credentials and region."""

    def _initialize_client(self):
        if bedrock_client.boto3 is None:
            raise ImportError("boto3 package not installed. Run: pip install boto3")
        self.model = supported_model(self.config.model)
        self.request_size = request_dimension(self.model, self.config.dimension)
        settings = bedrock_settings_config(self.model.model_id)
        self.region = settings.base_url or bedrock_client.DEFAULT_REGION
        self.client = bedrock_client.runtime_client(settings.api_key, settings.secret_key, settings.base_url,
                                                    use_role=settings.workload_identity, extra=embedding_client_bounds())
        logger.info("Initialized Bedrock embedding client: model %s, %d dimensions, region %s%s",
                    self.model.model_id, self.get_dimension(), self.region,
                    " (the install's own role)" if settings.workload_identity else "")

    def _invoke(self, text: str) -> Tuple[List[float], int]:
        """One ``InvokeModel`` call for one text. Runs in a worker thread (``run_blocking``)."""
        body = {"inputText": text[:self.model.max_chars], "dimensions": self.request_size, "normalize": True}
        response = self.client.invoke_model(modelId=self.model.model_id, body=json.dumps(body),
                                            contentType=JSON, accept=JSON)
        vector, tokens = parse_embedding(json.loads(response["body"].read()), self.request_size)
        return vector[:self.config.dimension], tokens

    async def generate_embedding(self, text: str) -> List[float]:
        """Embed one text off the event loop, inside the embedding time limits."""
        if not text:
            raise ValueError("Bedrock cannot embed an empty text")
        started = time.monotonic()
        try:
            vector, tokens = await asyncio.wait_for(run_blocking(self._invoke, text),
                                                    timeout=config.EMBEDDING_BATCH_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.error("Bedrock embedding (%s) gave no answer within %gs", self.model.model_id,
                         config.EMBEDDING_BATCH_TIMEOUT_S)
            raise
        except Exception as exc:
            logger.error("Bedrock embedding (%s, region %s) failed: %s", self.model.model_id, self.region,
                         _error_code(exc))
            raise
        usage = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=tokens))
        record_embedding_usage(USAGE_PROVIDER, self.model.model_id, usage, [text], started)
        return vector

    def get_dimension(self) -> int:
        """The length of every vector this provider returns: the configured size, or the
        model's largest output when that is smaller (start-up refuses that case)."""
        return min(self.config.dimension, self.model.dimension)


__all__ = [
    "BEDROCK_EMBEDDING_MODELS", "BedrockEmbeddingModel", "BedrockEmbeddingProvider",
    "DEFAULT_BEDROCK_EMBEDDING_MODEL", "TITAN_TEXT_EMBEDDINGS_V2", "UnsupportedBedrockEmbeddingModel",
    "bedrock_settings_config", "embedding_client_bounds", "parse_embedding", "request_dimension", "supported_model",
]
