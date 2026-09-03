"""
Provider registry (PRD-236 S0.1)
================================

The ONE list of LLM serving providers. Every surface that used to carry its
own copy — the BYOK list, the workspace allowlist, the marketplace probe
list, the factory's DIRECT_PROVIDERS / provider_map / config_map, the
manager's env map, the key validator, the frontend selects — reads from
here. A provider is code (it needs an adapter and a validator); a
workspace's keys and installs stay data.

Vocabulary: ``slug`` is the user-facing provider name (what ``user_api_keys.provider``,
``byok_overrides`` and ``model_config.provider`` store). ``enum_value`` is the
``LLMProvider`` member the adapter switch keys on (``bedrock`` → ``aws_bedrock``).
``aliases`` are the legacy strings still found in stored configs.

Terms (PRD-236 §Terms): NVIDIA's hosted endpoint is a trial — the user's key
is the user's own agreement with NVIDIA, so it is ``byok_only``: the saas
edition never resolves a platform-level NVIDIA key (the factory skips the
platform tiers, the platform-key endpoint refuses the slot).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from config import config
from core.llm.clients.base import LLMProvider

KIND_DIRECT = "direct"          # the vendor's own API
KIND_AGGREGATOR = "aggregator"  # many vendors behind one paid API (OpenRouter)
KIND_HOSTED_OPEN = "hosted_open"  # open models hosted by a third party (NVIDIA)

ADAPTER_OPENAI = "openai"
ADAPTER_ANTHROPIC = "anthropic"
ADAPTER_GOOGLE = "google"
ADAPTER_AZURE = "azure"
ADAPTER_BEDROCK = "bedrock"
ADAPTER_GROK = "grok"
ADAPTER_HUGGINGFACE = "huggingface"
ADAPTER_OPENAI_COMPATIBLE = "openai_compatible"
ADAPTER_NONE = "none"  # key-only providers (no chat client yet)

VALIDATION_MODELS_LIST = "models_list"
VALIDATION_NONE = "none"

ATTRIBUTION_TITLE = "Automatos AI"


@dataclass(frozen=True)
class ProviderSpec:
    slug: str
    label: str
    kind: str
    adapter: str
    enum_value: Optional[str] = None
    env_key: Optional[str] = None          # config.py attribute holding the platform/env key
    base_url_key: Optional[str] = None     # config.py attribute holding the base URL
    hosts_vendor_models: bool = False      # accepts "vendor/model" ids (openrouter, nvidia)
    byok_only: bool = False                # never a platform key in saas
    free: bool = False                     # the provider does not bill for calls
    price_multiplier: float = 1.0          # W0 interim pricing (W1 prices per catalogue row)
    validation: str = VALIDATION_NONE
    attribution_headers: bool = False      # send HTTP-Referer / X-Title (OpenRouter)
    openrouter_prefix: Optional[str] = None  # vendor → its OpenRouter id prefix
    aliases: Tuple[str, ...] = ()
    embeddings: bool = False
    key_placeholder: str = "Paste your API key"
    docs_url: Optional[str] = None
    terms_note: Optional[str] = None
    rate_limit_note: Optional[str] = None

    @property
    def chat(self) -> bool:
        return self.adapter != ADAPTER_NONE


_NVIDIA_TERMS = (
    "NVIDIA's hosted API is a trial: internal testing and evaluation only, not "
    "production, and no personal, financial or health data (NVIDIA API Trial Terms "
    "§1.2, §1.4, §4.3). Your key is your own agreement with NVIDIA."
)
_NVIDIA_RATE_LIMIT = (
    "NVIDIA's free tier allows about 40 requests per minute per key, and a popular "
    "model can be at capacity (429 within a second even on your first call). The "
    "call fails; it is never rerouted to a paid provider. Wait a minute or pick "
    "another NVIDIA route — the smaller Nemotron models usually answer at once."
)

_SPECS: Tuple[ProviderSpec, ...] = (
    ProviderSpec(
        slug="openai", label="OpenAI", kind=KIND_DIRECT, adapter=ADAPTER_OPENAI,
        enum_value="openai", env_key="OPENAI_API_KEY", validation=VALIDATION_MODELS_LIST,
        openrouter_prefix="openai/", embeddings=True, key_placeholder="sk-…",
        docs_url="https://platform.openai.com/api-keys",
    ),
    ProviderSpec(
        slug="anthropic", label="Anthropic", kind=KIND_DIRECT, adapter=ADAPTER_ANTHROPIC,
        enum_value="anthropic", env_key="ANTHROPIC_API_KEY", validation=VALIDATION_MODELS_LIST,
        openrouter_prefix="anthropic/", key_placeholder="sk-ant-…",
        docs_url="https://console.anthropic.com/settings/keys",
    ),
    ProviderSpec(
        slug="google", label="Google", kind=KIND_DIRECT, adapter=ADAPTER_GOOGLE,
        enum_value="google", env_key="GOOGLE_API_KEY", validation=VALIDATION_MODELS_LIST,
        openrouter_prefix="google/", embeddings=True, key_placeholder="AIza…",
        docs_url="https://aistudio.google.com/app/apikey",
    ),
    ProviderSpec(
        slug="openrouter", label="OpenRouter", kind=KIND_AGGREGATOR, adapter=ADAPTER_OPENAI_COMPATIBLE,
        enum_value="openrouter", env_key="OPENROUTER_API_KEY", base_url_key="OPENROUTER_BASE_URL",
        hosts_vendor_models=True, validation=VALIDATION_MODELS_LIST, attribution_headers=True,
        embeddings=True, key_placeholder="sk-or-…", docs_url="https://openrouter.ai/keys",
    ),
    ProviderSpec(
        slug="nvidia", label="NVIDIA", kind=KIND_HOSTED_OPEN, adapter=ADAPTER_OPENAI_COMPATIBLE,
        enum_value="nvidia", env_key="NVIDIA_API_KEY", base_url_key="NVIDIA_BASE_URL",
        hosts_vendor_models=True, byok_only=True, free=True, price_multiplier=0.0,
        validation=VALIDATION_MODELS_LIST, key_placeholder="nvapi-…",
        docs_url="https://build.nvidia.com/", terms_note=_NVIDIA_TERMS,
        rate_limit_note=_NVIDIA_RATE_LIMIT,
    ),
    ProviderSpec(
        slug="deepseek", label="DeepSeek", kind=KIND_DIRECT, adapter=ADAPTER_OPENAI_COMPATIBLE,
        enum_value="deepseek", env_key="DEEPSEEK_API_KEY", base_url_key="DEEPSEEK_BASE_URL",
        validation=VALIDATION_MODELS_LIST, openrouter_prefix="deepseek/", key_placeholder="sk-…",
        docs_url="https://platform.deepseek.com/api_keys",
    ),
    ProviderSpec(
        slug="azure", label="Azure OpenAI", kind=KIND_DIRECT, adapter=ADAPTER_AZURE,
        enum_value="azure", env_key="AZURE_OPENAI_API_KEY", aliases=("azure_openai",),
        docs_url="https://portal.azure.com",
    ),
    ProviderSpec(
        slug="bedrock", label="AWS Bedrock", kind=KIND_DIRECT, adapter=ADAPTER_BEDROCK,
        enum_value="aws_bedrock", env_key="AWS_ACCESS_KEY_ID", aliases=("aws_bedrock",),
        docs_url="https://console.aws.amazon.com/bedrock",
    ),
    ProviderSpec(
        slug="grok", label="Grok / xAI", kind=KIND_DIRECT, adapter=ADAPTER_GROK,
        enum_value="grok", env_key="XAI_API_KEY", aliases=("x-ai", "xai"),
        openrouter_prefix="x-ai/", docs_url="https://console.x.ai",
    ),
    ProviderSpec(
        slug="cohere", label="Cohere", kind=KIND_DIRECT, adapter=ADAPTER_NONE,
        env_key="COHERE_API_KEY", embeddings=True,
        docs_url="https://dashboard.cohere.com/api-keys",
    ),
    ProviderSpec(
        slug="huggingface", label="HuggingFace", kind=KIND_DIRECT, adapter=ADAPTER_HUGGINGFACE,
        enum_value="huggingface", env_key=None, embeddings=True,  # credential store only
        docs_url="https://huggingface.co/settings/tokens",
    ),
)

REGISTRY: Dict[str, ProviderSpec] = {s.slug: s for s in _SPECS}

_ALIAS_INDEX: Dict[str, str] = {}
for _s in _SPECS:
    for _a in _s.aliases:
        _ALIAS_INDEX[_a] = _s.slug
    if _s.enum_value and _s.enum_value != _s.slug:
        _ALIAS_INDEX[_s.enum_value] = _s.slug


def normalize_slug(provider: Optional[str]) -> Optional[str]:
    """Canonical slug for any provider string the platform has ever stored."""
    if not provider:
        return None
    key = str(provider).strip().lower()
    if key in REGISTRY:
        return key
    return _ALIAS_INDEX.get(key)


def get_spec(provider: Optional[str]) -> Optional[ProviderSpec]:
    slug = normalize_slug(provider)
    return REGISTRY.get(slug) if slug else None


def all_slugs() -> List[str]:
    return [s.slug for s in _SPECS]


def chat_slugs() -> List[str]:
    return [s.slug for s in _SPECS if s.chat]


def byok_slugs() -> List[str]:
    """Providers a workspace may add its own key for (every registered provider)."""
    return all_slugs()


def known_provider_names() -> frozenset:
    """Every string that names a registered provider: slugs, aliases, enum values."""
    names = set(REGISTRY.keys()) | set(_ALIAS_INDEX.keys())
    return frozenset(names)


def routable_provider_names() -> frozenset:
    """Names of providers that have a chat adapter (the factory can route to them).

    Key-only providers (cohere) are deliberately absent: a model configured
    against one keeps taking the legacy unknown-provider path (OpenRouter).
    """
    names = set()
    for spec in _SPECS:
        if not spec.chat:
            continue
        names.add(spec.slug)
        names.update(spec.aliases)
        if spec.enum_value:
            names.add(spec.enum_value)
    return frozenset(names)


def platform_key_allowed(provider: Optional[str], edition: Optional[str] = None) -> bool:
    """May the operator hold a platform-level key for this provider?

    ``byok_only`` providers are refused in the saas edition (PRD-236 §Terms).
    In the local edition the operator IS the user, so their key is their own.
    """
    spec = get_spec(provider)
    if spec is None:
        return False
    edition = (edition or config.AUTH_EDITION or "saas").lower()
    return not (spec.byok_only and edition == "saas")


def platform_key_slugs(edition: Optional[str] = None) -> List[str]:
    return [s.slug for s in _SPECS if platform_key_allowed(s.slug, edition)]


def enum_for(provider: Optional[str]) -> Optional[LLMProvider]:
    spec = get_spec(provider)
    if spec is None or not spec.enum_value:
        return None
    try:
        return LLMProvider(spec.enum_value)
    except ValueError:
        return None


def env_api_key(provider: Optional[str]) -> Optional[str]:
    spec = get_spec(provider)
    if spec is None or not spec.env_key:
        return None
    return getattr(config, spec.env_key, None) or None


def base_url_for(provider: Optional[str]) -> Optional[str]:
    spec = get_spec(provider)
    if spec is None or not spec.base_url_key:
        return None
    return getattr(config, spec.base_url_key, None) or None


def headers_for(provider: Optional[str]) -> Dict[str, str]:
    spec = get_spec(provider)
    if spec is None or not spec.attribution_headers:
        return {}
    return {"HTTP-Referer": config.OPENROUTER_SITE_URL, "X-Title": ATTRIBUTION_TITLE}


def hosts_vendor_models(provider: Optional[str]) -> bool:
    spec = get_spec(provider)
    return bool(spec and spec.hosts_vendor_models)


def openrouter_prefix_for(provider: Optional[str]) -> Optional[str]:
    spec = get_spec(provider)
    return spec.openrouter_prefix if spec else None


def price_multiplier_for(provider: Optional[str]) -> float:
    """W0 interim pricing: a free route books zero. Unknown providers price as-is."""
    spec = get_spec(provider)
    return float(spec.price_multiplier) if spec else 1.0


def to_public_dict(spec: ProviderSpec, edition: Optional[str] = None) -> Dict[str, object]:
    """The registry as the UI sees it — never a key, never a config attribute name."""
    return {
        "slug": spec.slug,
        "label": spec.label,
        "kind": spec.kind,
        "chat": spec.chat,
        "embeddings": spec.embeddings,
        "byok": True,
        "platform_key": platform_key_allowed(spec.slug, edition),
        "hosts_vendor_models": spec.hosts_vendor_models,
        "free": spec.free,
        "key_placeholder": spec.key_placeholder,
        "docs_url": spec.docs_url,
        "terms_note": spec.terms_note,
        "rate_limit_note": spec.rate_limit_note,
    }


def public_registry(edition: Optional[str] = None) -> Dict[str, object]:
    edition = (edition or config.AUTH_EDITION or "saas").lower()
    return {
        "edition": edition,
        "providers": [to_public_dict(s, edition) for s in _SPECS],
    }
