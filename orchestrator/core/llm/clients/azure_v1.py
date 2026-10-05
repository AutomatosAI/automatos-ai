"""
Azure OpenAI v1 route: endpoint, and what each deployment refuses (#873)
========================================================================

#873 (daarthur): the Azure provider used the legacy ``AzureOpenAI`` client on
``api-version=2024-02-15-preview``. Microsoft Foundry's current route is
``https://<resource>.openai.azure.com/openai/v1/``: the plain OpenAI client, the
deployment name as ``model``, no api-version, and it also reaches Foundry's
non-OpenAI models through Chat Completions.

``v1_base_url`` turns whatever endpoint was saved into that base URL. A request's
``model`` is the customer's deployment name (``prod-chat``, ``gpt5-uk``), so the
name alone can't say whether a deployment is a reasoning model. Instead:

* a 400 that names a parameter as unsupported (temperature and the other
  sampling parameters, or ``max_completion_tokens``) is retried once without it,
  and the refusal is remembered for that endpoint and deployment;
* after a successful call, a response whose ``model`` is a model that refuses
  sampling parameters (``accepts_sampling_params``) is remembered the same way.

The memory is per process and starts empty on restart.
"""
import re
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Iterable, Optional, Tuple
from urllib.parse import urlsplit

from .base import accepts_sampling_params
from .openai_chat_request import COMPLETION_TOKENS, SAMPLING_PARAMS

V1_PATH = "/openai/v1"
# Where a pasted Azure URL stops being the resource and becomes an API path.
# Anything before these is kept: an API Management gateway serves Azure under
# its own prefix (https://apim.bank.example/azure/openai/v1/).
_AZURE_API_PATHS = ("/openai/", "/api/projects/")
_HTTP_BAD_REQUEST = 400
_REFUSAL_WORDS = ("unsupported", "not supported")
# "Unsupported parameter: 'temperature' …", "Unsupported value: 'temperature' does
# not support 0.7 …", "'top_p' is not supported with this model."
_NAMED_PARAM = re.compile(
    r"unsupported (?:parameter|value):\s*'(\w+)'|'(\w+)' is not supported|'(\w+)' does not support"
)
_RETRYABLE = frozenset(SAMPLING_PARAMS) | {COMPLETION_TOKENS}

DeploymentKey = Tuple[str, str]


def v1_base_url(endpoint: str) -> str:
    """The v1 base URL for a saved Azure endpoint.

    Accepts the bare resource URL (with or without a trailing slash), one that
    already ends in ``/openai/v1``, and a pasted deployment, Foundry project or
    query-string URL, which is cut back to the resource. A gateway prefix in
    front of the Azure path is kept. Raises ``ValueError`` when no host can be
    read from it.
    """
    raw = (endpoint or "").strip()
    if raw and "://" not in raw:
        raw = f"https://{raw}"
    parts = urlsplit(raw)
    if not parts.netloc:
        raise ValueError("The Azure endpoint must be a URL such as https://<resource>.openai.azure.com")
    return f"{parts.scheme}://{parts.netloc}{_gateway_prefix(parts.path)}{V1_PATH}/"


def _gateway_prefix(path: str) -> str:
    """The part of ``path`` in front of the Azure API path; all of it when it has none."""
    slashed = f"{path.rstrip('/')}/"
    cuts = [slashed.find(marker) for marker in _AZURE_API_PATHS if marker in slashed]
    return slashed[: min(cuts)] if cuts else slashed.rstrip("/")


@dataclass(frozen=True)
class DeploymentQuirks:
    """What a deployment takes: sampling parameters, and ``max_completion_tokens``."""

    sampling: bool = True
    completion_tokens: bool = True


def _error_fields(exc: Exception) -> Tuple[str, Optional[str]]:
    """The error's text (lower case) and its ``param`` field, when it has one."""
    body: Any = getattr(exc, "body", None)
    if isinstance(body, dict) and isinstance(body.get("error"), dict):
        body = body["error"]
    detail = body if isinstance(body, dict) else {}
    text = " ".join(str(v) for v in (detail.get("message"), detail.get("code"), str(exc)) if v)
    param = detail.get("param")
    return text.lower(), (str(param) if param else None)


def refused_params(exc: Exception, sent: Iterable[str]) -> FrozenSet[str]:
    """The parameters a 400 says the deployment refuses, among those ``sent``
    that a retry may leave out. Empty for any other error."""
    if getattr(exc, "status_code", None) != _HTTP_BAD_REQUEST:
        return frozenset()
    text, param = _error_fields(exc)
    if not any(word in text for word in _REFUSAL_WORDS):
        return frozenset()
    named = {name for match in _NAMED_PARAM.findall(text) for name in match if name}
    if param:
        named.add(param.lower())
    return frozenset(named & set(sent) & _RETRYABLE)


def quirks_after_refusal(quirks: DeploymentQuirks, refused: FrozenSet[str]) -> DeploymentQuirks:
    """The quirks with the refused parameters turned off."""
    return replace(
        quirks,
        sampling=quirks.sampling and not (refused & set(SAMPLING_PARAMS)),
        completion_tokens=quirks.completion_tokens and COMPLETION_TOKENS not in refused,
    )


def quirks_after_response(quirks: DeploymentQuirks, served_model: Optional[str]) -> DeploymentQuirks:
    """The quirks once a response says which model served the deployment."""
    if quirks.sampling and served_model and not accepts_sampling_params(served_model):
        return replace(quirks, sampling=False)
    return quirks


class QuirkMemory:
    """Per-process memory of what each (endpoint, deployment) refuses."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._known: Dict[DeploymentKey, DeploymentQuirks] = {}

    def recall(self, key: DeploymentKey) -> DeploymentQuirks:
        """What is known about the deployment; a deployment's name that names a
        reasoning model (``o3-mini``) counts as known."""
        with self._lock:
            known = self._known.get(key)
        if known is not None:
            return known
        return DeploymentQuirks(sampling=accepts_sampling_params(key[1]))

    def remember(self, key: DeploymentKey, quirks: DeploymentQuirks) -> None:
        """Record the deployment's quirks (a new value; nothing is edited in place)."""
        with self._lock:
            self._known = {**self._known, key: quirks}

    def clear(self) -> None:
        """Forget every deployment (tests)."""
        with self._lock:
            self._known = {}


QUIRKS = QuirkMemory()
