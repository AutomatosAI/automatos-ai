"""
AWS Bedrock Provider Implementation
====================================

Every Bedrock model that takes messages, through the Converse API (#1170): Claude (the
5.x models included), Amazon Nova, Meta Llama, Mistral, DeepSeek, OpenAI's gpt-oss.
The request and the answer are translated in :mod:`.bedrock_converse`.

The region is ``config.base_url`` (the manager stores it there). Credentials, per
provider and never process-wide:

- an access key ID with its secret (``config.secret_key``): IAM, signed with SigV4;
- a Bedrock API key (``ABSK…`` long-term, ``bedrock-api-key-…`` short-term): sent as
  a bearer token, the way AWS documents it, through this client's own token provider;
- otherwise the default AWS credential chain (environment, profile, instance role).
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .base import BaseLLMProvider, LLMResponse, accepts_sampling_params, request_max_tokens, run_blocking
from .bedrock_converse import answer, converse_messages, converse_tools, history_tools

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)

DEFAULT_REGION = "us-east-1"
# An AWS access key ID (AKIA… long-term, ASIA… temporary): IAM, never a Bedrock API key.
_ACCESS_KEY_ID = re.compile(r"^(AKIA|ASIA)[A-Z0-9]{16}$")
_NOT_CONFIGURED = (
    "AWS Bedrock credentials not configured. Cannot generate response. "
    "Please configure 'development_aws' credential or set AWS env vars."
)


class _StaticToken:
    """A botocore token provider that answers with one Bedrock API key: this client's own."""

    def __init__(self, token: str) -> None:
        self._token = token

    def load_token(self, **_kwargs: Any) -> Any:
        from botocore.tokens import FrozenAuthToken

        return FrozenAuthToken(self._token)


def runtime_client(api_key: Optional[str], secret_key: Optional[str], region: Optional[str]) -> Any:
    """A ``bedrock-runtime`` client for these credentials (see the module docstring)."""
    import botocore.session

    config = Config(region_name=region or DEFAULT_REGION, retries={"max_attempts": 3, "mode": "adaptive"})
    if api_key and secret_key:
        return boto3.client("bedrock-runtime", aws_access_key_id=api_key, aws_secret_access_key=secret_key,
                            config=config)
    if api_key and not _ACCESS_KEY_ID.match(api_key):
        session = botocore.session.Session()
        session.register_component("token_provider", _StaticToken(api_key))
        return boto3.Session(botocore_session=session).client(
            "bedrock-runtime", config=config.merge(Config(signature_version="bearer")))
    return boto3.client("bedrock-runtime", config=config)


def _plain_error(error: Exception, model_id: str) -> Optional[str]:
    """What to do about a model ID Bedrock won't serve, in plain words; None for any other error."""
    text = str(error)
    if "on-demand throughput isn" in text or "inference profile" in text:
        return (f"AWS Bedrock serves {model_id} only through an inference profile. Set the model to the "
                f"profile for where your data may go: us.{model_id}, eu.{model_id}, apac.{model_id} "
                f"or global.{model_id}.")
    if "model identifier is invalid" in text or "end of its life" in text or "marked by provider as Legacy" in text:
        return (f"AWS Bedrock no longer serves {model_id} (or doesn't know it). Pick a current model ID from "
                "the Bedrock console's model catalog.")
    return None


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider: one Converse request shape for every model family."""

    def _initialize_client(self):
        if boto3 is None:
            raise ImportError("boto3 package not installed. Run: pip install boto3")
        self.model_id = self.config.model
        self.client = runtime_client(self.config.api_key, self.config.secret_key, self.config.base_url)
        logger.info(f"Initialized AWS Bedrock (Converse) client with model: {self.model_id}")

    def _request(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]]) -> Dict[str, Any]:
        """The Converse request: messages, the system prompt, limits and tools."""
        system, turns = converse_messages(messages)
        inference: Dict[str, Any] = {"maxTokens": request_max_tokens(self.config)}
        if accepts_sampling_params(self.model_id):
            inference["temperature"] = self.config.temperature
        request: Dict[str, Any] = {"modelId": self.model_id, "messages": turns, "inferenceConfig": inference}
        if system:
            request["system"] = system
        tool_config = converse_tools(self._sanitize_tools(tools)) if tools else history_tools(turns)
        if tool_config:
            request["toolConfig"] = tool_config
        return request

    def _converse(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None) -> LLMResponse:
        """One Converse call. Runs in the caller's thread, or the executor's (run_blocking)."""
        try:
            response = self.client.converse(**self._request(messages, tools))
        except ClientError as error:
            plain = _plain_error(error, self.model_id)
            if plain:
                raise ValueError(plain) from error
            raise
        return LLMResponse(model=self.model_id, provider="aws_bedrock", **answer(response))

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using AWS Bedrock without blocking the event loop"""
        if self.client is None:
            raise ValueError(_NOT_CONFIGURED)
        try:
            return await run_blocking(self._converse, messages, tools)
        except Exception as e:
            logger.error(f"AWS Bedrock API error: {e}")
            raise

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using AWS Bedrock (synchronous)"""
        if self.client is None:
            raise ValueError(_NOT_CONFIGURED)
        try:
            return self._converse(messages)
        except Exception as e:
            logger.error(f"AWS Bedrock API error: {e}")
            raise
