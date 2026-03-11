"""
LLM Manager
===========

Main LLM manager that handles provider selection and configuration.
Supports per-service configuration via system settings.
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional
from functools import lru_cache

from config import config

from .clients.base import LLMProvider, LLMConfig
from .clients.openai_client import OpenAIProvider
from .clients.anthropic_client import AnthropicProvider
from .clients.google_client import GoogleProvider
from .clients.azure_client import AzureProvider
from .clients.huggingface_client import HuggingFaceProvider
from .clients.bedrock_client import BedrockProvider
from .clients.grok_client import GrokProvider
from .clients.openrouter_client import OpenRouterProvider

logger = logging.getLogger(__name__)

# Canonical mapping from service name to settings category
SERVICE_CATEGORY_MAP = {
    "orchestrator": "orchestrator_llm",
    "codegraph": "codegraph",
    "document_processing": "document_processing",
    "chatbot": "chatbot",
    "rag": "rag",
    "embeddings": "embeddings",
    "memory_integration": "memory_integration",
    "nl2sql": "nl2sql",
    "heartbeat": "orchestrator_llm",
    "complexity_assessor": "orchestrator_llm",  # PRD-68: uses orchestrator LLM settings
}


def get_system_setting(
    category: str,
    key: str,
    default_value: Optional[str] = None
) -> Optional[str]:
    """
    Get a system setting value from database.
    
    Args:
        category: Setting category (e.g., 'orchestrator_llm', 'codegraph')
        key: Setting key (e.g., 'provider', 'model')
        default_value: Default value if setting not found
        
    Returns:
        Setting value or default
    """
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(
                SystemSetting.category == category,
                SystemSetting.key == key
            ).first()
            
            if setting and setting.value:
                return setting.value
            return default_value
        finally:
            db.close()
    except ImportError:
        # Database module not available yet - return default during startup
        logger.debug(f"Database module not available yet for {category}.{key}")
        return default_value
    except Exception as e:
        # Database might not be ready - log but don't fail
        logger.debug(f"Failed to get system setting {category}.{key}: {e}")
        return default_value


def get_provider_and_model_from_settings(service_name: str = "orchestrator") -> tuple:
    """
    Get provider and model from system settings.
    
    Args:
        service_name: Service name ('orchestrator', 'codegraph', etc.)
        
    Returns:
        Tuple of (provider_str, model_str)
    """
    category = SERVICE_CATEGORY_MAP.get(service_name, "orchestrator_llm")
    
    # Get provider and model from settings (NO hardcoded defaults)
    # Note: Settings use 'llm_provider' and 'llm_model' keys, not 'provider' and 'model'
    provider_str = get_system_setting(category, "llm_provider") or get_system_setting(category, "provider")
    model_str = get_system_setting(category, "llm_model") or get_system_setting(category, "model")
    
    # Require provider to be set - no fallback
    if not provider_str:
        raise ValueError(
            f"LLM provider not configured for service '{service_name}'. "
            f"Please set {category}.provider in system settings."
        )
    
    # Require model to be set - no fallback
    if not model_str:
        raise ValueError(
            f"LLM model not configured for service '{service_name}' with provider '{provider_str}'. "
            f"Please set {category}.model in system settings."
        )
    
    return provider_str, model_str


from cachetools import TTLCache, cached as ttl_cached
_credential_cache = TTLCache(maxsize=32, ttl=300)

@ttl_cached(cache=_credential_cache)
def get_credential_data(provider: str, environment: str = None, service_name: str = "orchestrator") -> Dict[str, Any]:
    """
    Get credential data for a provider from credential system.
    
    Uses prioritized credential resolution with multiple fallback strategies:
    1. Check system settings for explicit credential name mapping (MVP: orchestrator_llm.credential_name_{provider})
    2. Try standard pattern: {environment}_{provider}_api
    3. Try without _api suffix: {environment}_{provider}
    4. Try provider name directly (case variations)
    5. Try by credential type (find any credential of matching type)
    6. Fallback to environment variables
    
    Args:
        provider: Provider name ('openai', 'anthropic', etc.)
        environment: Environment name ('development', 'production', etc.)
        service_name: Service name ('orchestrator', 'codegraph', etc.) for settings lookup
        
    Returns:
        Dictionary of credential data
    """
    if environment is None:
        environment = config.ENVIRONMENT or 'development'
    
    try:
        from core.credentials.resolver import get_credential_resolver
        
        resolver = get_credential_resolver()
        
        # MVP: Strategy 0 - Check system settings for explicit credential name mapping
        # e.g., orchestrator_llm.credential_name_openai = "development_openai"
        category = SERVICE_CATEGORY_MAP.get(service_name, "orchestrator_llm")
        credential_name_setting_key = f"credential_name_{provider.lower()}"
        
        # Try to get explicit credential name from system settings
        explicit_credential_name = get_system_setting(category, credential_name_setting_key, None)
        
        # Check if explicit credential name is set (not None and not empty string)
        if explicit_credential_name and explicit_credential_name.strip():
            try:
                cred_data = resolver.get_dict(explicit_credential_name, environment=environment, silent=True)
                if cred_data and len(cred_data) > 0:
                    logger.info(
                        f"Found credential '{explicit_credential_name}' for provider '{provider}' "
                        f"(via system setting {category}.{credential_name_setting_key})"
                    )
                    return cred_data
            except Exception as e:
                logger.debug(
                    f"Explicit credential name '{explicit_credential_name}' from settings not found: {e}"
                )
        else:
            logger.debug(
                f"No explicit credential mapping found in {category}.{credential_name_setting_key}, "
                f"falling back to flexible lookup"
            )
        
        # Map provider names to credential type names (for type-based lookup)
        credential_type_map = {
            "openai": "openai_api",
            "anthropic": "anthropic_api",
            "google": "google_api",
            "azure": "azure_openai",
            "huggingface": "huggingface_api",
            "aws_bedrock": "aws_bedrock_api",
            "bedrock": "aws_bedrock_api",
            "grok": "xai_api",
            "xai": "xai_api",
            "openrouter": "openrouter_api"
        }
        
        credential_type = credential_type_map.get(provider.lower())
        if not credential_type:
            logger.warning(f"Unknown provider for credential lookup: {provider}")
            return {}
        
        # Strategy 1: Try standard naming pattern with _api suffix
        credential_name_variations = [
            f"{environment}_{provider}_api",      # production_openai_api (standard)
            f"{environment}_{provider}",          # production_openai (user's format)
            f"{provider}_api",                     # openai_api (simple)
            provider,                              # openai (provider name only)
            provider.lower(),                      # openai (lowercase)
            provider.capitalize(),                 # Openai (capitalized)
            provider.title(),                      # HuggingFace → Huggingface
        ]
        
        # Also try case-insensitive variations
        if provider.lower() == "huggingface":
            credential_name_variations.extend([
                "HuggingFace",
                "huggingface",
                "Huggingface",
                f"{environment}_HuggingFace",
                f"{environment}_huggingface",
            ])
        
        # If not in development, also try development environment variations early
        if environment != 'development':
            credential_name_variations.extend([
                f"development_{provider}_api",
                f"development_{provider}",
            ])
            if provider.lower() == "huggingface":
                credential_name_variations.extend([
                    "development_HuggingFace",
                    "development_huggingface",
                ])
        
        # AWS Bedrock credential variations
        if provider.lower() in ["aws_bedrock", "bedrock"]:
            credential_name_variations.extend([
                f"{environment}_aws_bedrock",
                f"{environment}_bedrock",
                f"{environment}_aws",
                "aws_bedrock",
                "bedrock",
                "aws",
            ])
        
        # Try each credential name variation in the current environment
        for cred_name in credential_name_variations:
            try:
                # If credential name starts with 'development_', try in development environment
                lookup_env = 'development' if cred_name.startswith('development_') else environment
                cred_data = resolver.get_dict(cred_name, environment=lookup_env, silent=True)
                if cred_data and len(cred_data) > 0:
                    logger.info(f"Found credential '{cred_name}' for provider '{provider}' (env: {lookup_env})")
                    return cred_data
            except Exception as e:
                logger.debug(f"Credential name '{cred_name}' not found: {e}")
                continue
        
        # Strategy 2: Try to find by credential type (if name-based lookup failed)
        # Search for any active credential of the matching type in the environment
        try:
            from core.database.database import SessionLocal
            from core.credentials.service import CredentialStore
            from core.models.credentials import CredentialType
            
            db = SessionLocal()
            try:
                store = CredentialStore(db)
                
                # Find credential type ID
                cred_type = store.get_credential_type_by_name(credential_type)
                if cred_type:
                    # Find any active credential of this type in the environment
                    credentials = store.list_credentials(
                        credential_type_id=cred_type.id,
                        environment=environment,
                        active_only=True
                    )
                    
                    if credentials:
                        # Use the first active credential found
                        cred = credentials[0]
                        logger.info(
                            f"Found credential '{cred.name}' (type: {credential_type}) "
                            f"for provider '{provider}'"
                        )
                        decrypted = store.get_decrypted_credential(
                            cred.id,
                            service_name="llm_provider"
                        )
                        return decrypted
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"Type-based credential lookup failed: {e}")
        
        # Strategy 3: If current environment failed, try 'development' as fallback
        if environment != 'development':
            logger.debug(f"Trying 'development' environment as fallback for provider '{provider}'")
            
            # First try name-based lookup in development
            try:
                for cred_name in credential_name_variations:
                    try:
                        cred_data = resolver.get_dict(cred_name, environment='development', silent=True)
                        if cred_data and len(cred_data) > 0:
                            logger.info(f"Found credential '{cred_name}' in 'development' environment for provider '{provider}'")
                            return cred_data
                    except Exception as e:
                        logger.debug(f"Credential name '{cred_name}' not found in development: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Name-based lookup in development failed: {e}")
            
            # Then try type-based lookup in development
            try:
                from core.database.database import SessionLocal
                from core.credentials.service import CredentialStore
                
                db = SessionLocal()
                try:
                    store = CredentialStore(db)
                    cred_type = store.get_credential_type_by_name(credential_type)
                    if cred_type:
                        credentials = store.list_credentials(
                            credential_type_id=cred_type.id,
                            environment='development',
                            active_only=True
                        )
                        
                        if credentials:
                            cred = credentials[0]
                            logger.info(
                                f"Found credential '{cred.name}' in 'development' environment "
                                f"(type: {credential_type}) for provider '{provider}'"
                            )
                            decrypted = store.get_decrypted_credential(
                                cred.id,
                                service_name="llm_provider"
                            )
                            return decrypted
                finally:
                    db.close()
            except Exception as e:
                logger.debug(f"Type-based lookup in development failed: {e}")
        
        logger.debug(
            f"No stored credential for provider '{provider}' (tried: {credential_name_variations}), "
            f"falling back to env var"
        )
        return {}
        
    except Exception as e:
        logger.debug(f"Credential resolver not available: {e}")
        return {}


class LLMManager:
    """Main LLM manager that handles provider selection and configuration"""
    
    def __init__(
        self,
        config: LLMConfig = None,
        service_name: str = "orchestrator",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        workspace_id=None,
        agent_id: Optional[int] = None,
        execution_id: Optional[str] = None,
        request_type: Optional[str] = None,
        is_byok: bool = False,
    ):
        """
        Initialize LLM Manager.

        Args:
            config: Optional LLMConfig (if None, loads from settings/env)
            service_name: Service name for per-service configuration ('orchestrator', 'codegraph', etc.)
            provider: Optional provider override
            model: Optional model override
            workspace_id: Workspace ID for usage tracking
            agent_id: Agent ID for usage tracking
            execution_id: Execution ID for usage tracking
            request_type: Request type label (chat, recipe, orchestrator, etc.)
            is_byok: Whether this uses a BYOK key
        """
        self.service_name = service_name
        self._tracking_ctx: Dict[str, Any] = {
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "execution_id": execution_id,
            "request_type": request_type or service_name,
            "is_byok": is_byok,
        }
        
        if config is None:
            config = self._load_config_from_settings(service_name, provider, model)
        elif not config.api_key:
            # Config provided but no api_key - look up credential
            logger.debug(f"Config provided without api_key, looking up credential for {config.provider.value}")
            cred_data = get_credential_data(config.provider.value, service_name=service_name)
            
            # Extract API key based on provider
            api_key = None
            if config.provider == LLMProvider.HUGGINGFACE:
                api_key = cred_data.get("api_token") or cred_data.get("api_key")
            elif config.provider == LLMProvider.GROK:
                api_key = cred_data.get("api_key") or cred_data.get("api_token")
            else:
                api_key = cred_data.get("api_key") or cred_data.get("api_token")
            
            if api_key:
                config = LLMConfig(
                    provider=config.provider,
                    model=config.model,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    api_key=api_key,
                    base_url=config.base_url
                )
                logger.info(f"Credential found for {config.provider.value}")
        
        self.config = config
        self.provider = None  # Lazy initialization
        
        # Don't create provider immediately - lazy loading
        logger.debug(f"LLMManager initialized for service '{service_name}' with provider '{config.provider.value}', model '{config.model}'")
    
    def _load_config_from_settings(
        self,
        service_name: str,
        provider_override: Optional[str] = None,
        model_override: Optional[str] = None
    ) -> LLMConfig:
        """
        Load LLM configuration from system settings or environment.
        
        Args:
            service_name: Service name for per-service settings
            provider_override: Optional provider override
            model_override: Optional model override
            
        Returns:
            LLMConfig instance
        """
        # Get provider and model from settings (or overrides)
        if provider_override:
            provider_str = provider_override.lower()
        else:
            provider_str, _ = get_provider_and_model_from_settings(service_name)
        
        # Validate provider - route unknown providers with slash-format models
        # through OpenRouter (e.g. provider="qwen" + model="qwen/qwen3-coder-next")
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            if model_override and "/" in model_override:
                logger.info(f"Unknown provider '{provider_str}', routing '{model_override}' through OpenRouter")
                provider = LLMProvider.OPENROUTER
            else:
                raise ValueError(
                    f"Unknown LLM provider: '{provider_str}'. "
                    f"Supported providers: {[p.value for p in LLMProvider]}"
                )
        
        # Get model from settings or use override
        if model_override:
            model = model_override
        else:
            _, model = get_provider_and_model_from_settings(service_name)
            if not model:
                # Try config.LLM_MODEL first, then provider-specific fallbacks
                from config import config as _cfg
                _cfg_model = _cfg.LLM_MODEL
                if _cfg_model:
                    model = _cfg_model
                else:
                    # Provider-specific fallbacks (last resort)
                    default_models = {
                        LLMProvider.OPENAI: "gpt-4",
                        LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
                        LLMProvider.GOOGLE: "gemini-pro",
                        LLMProvider.AZURE: "gpt-4",
                        LLMProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2",
                        LLMProvider.GROK: "grok-2-latest",
                        LLMProvider.OPENROUTER: "meta-llama/llama-3.1-70b-instruct"
                    }
                    model = default_models.get(provider, "gpt-4")
        
        # Get other settings with defaults
        category = SERVICE_CATEGORY_MAP.get(service_name, "orchestrator_llm")
        
        temperature = float(get_system_setting(category, "temperature", "0.7"))
        max_tokens = int(get_system_setting(category, "max_tokens", "2000"))
        
        # Get credential data (pass service_name for settings lookup)
        cred_data = get_credential_data(provider.value, service_name=service_name)
        
        # Extract API key and other credential fields
        api_key = None
        base_url = None
        organization_id = None
        secret_key = None  # For AWS Bedrock IAM auth
        
        if provider == LLMProvider.OPENAI:
            api_key = cred_data.get("api_key")
            base_url = cred_data.get("base_url")
            organization_id = cred_data.get("organization_id")
        elif provider == LLMProvider.ANTHROPIC:
            api_key = cred_data.get("api_key")
            base_url = cred_data.get("base_url")
        elif provider == LLMProvider.GOOGLE:
            api_key = cred_data.get("api_key")
        elif provider == LLMProvider.AZURE:
            api_key = cred_data.get("api_key")
            base_url = cred_data.get("endpoint_url") or cred_data.get("base_url")
        elif provider == LLMProvider.HUGGINGFACE:
            api_key = cred_data.get("api_token") or cred_data.get("api_key")
        elif provider == LLMProvider.AWS_BEDROCK:
            # AWS Bedrock - simplified to use new API Keys by default
            # Primary method: New Bedrock API Key (single key - recommended)
            api_key = cred_data.get("bedrock_api_key")
            
            # Fallback: Traditional IAM method (if API key not found)
            if not api_key:
                api_key = cred_data.get("aws_access_key_id")
                secret_key = cred_data.get("aws_secret_access_key")
            
            # Region is stored in base_url for compatibility with LLMConfig
            aws_region = cred_data.get("aws_region", "us-east-1")
            base_url = aws_region  # Store region in base_url field
        elif provider == LLMProvider.GROK:
            api_key = cred_data.get("api_key") or cred_data.get("api_token")
        elif provider == LLMProvider.OPENROUTER:
            api_key = cred_data.get("api_key") or cred_data.get("api_token")

        # Fallback to environment variables if credentials not found (except HuggingFace)
        if not api_key and provider != LLMProvider.HUGGINGFACE:
            fallback_env_vars = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.GOOGLE: "GOOGLE_API_KEY",
                LLMProvider.AZURE: "AZURE_OPENAI_API_KEY",
                LLMProvider.AWS_BEDROCK: "AWS_ACCESS_KEY_ID",
                LLMProvider.GROK: "XAI_API_KEY",
                LLMProvider.OPENROUTER: "OPENROUTER_API_KEY"
            }
            env_var = fallback_env_vars.get(provider)
            if env_var:
                api_key = getattr(config, env_var, None)

        if not api_key and provider == LLMProvider.HUGGINGFACE:
            raise ValueError(
                "HuggingFace credential not found. Create an active credential such as "
                "'development_huggingface' (or environment-specific variation) in the credential store."
            )
        
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            organization_id=organization_id,
            secret_key=secret_key if provider == LLMProvider.AWS_BEDROCK else None
        )
    
    # Patterns that indicate the configured model is dead/removed, not a transient error
    _DEAD_MODEL_PATTERNS = [
        re.compile(r"no endpoints found", re.IGNORECASE),
        re.compile(r"model not found", re.IGNORECASE),
        re.compile(r"does not exist", re.IGNORECASE),
        re.compile(r"model .+ is not available", re.IGNORECASE),
        re.compile(r"invalid model", re.IGNORECASE),
    ]

    # Provider-specific fallback models (cheap & reliable)
    _DEFAULT_FALLBACK_MODELS = {
        LLMProvider.OPENROUTER: "meta-llama/llama-3.1-70b-instruct",
        LLMProvider.OPENAI: "gpt-4o-mini",
        LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
        LLMProvider.GOOGLE: "gemini-2.0-flash",
        LLMProvider.AZURE: "gpt-4o-mini",
        LLMProvider.GROK: "grok-2-latest",
        LLMProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2",
    }

    def _ensure_provider_initialized(self):
        """Ensure provider is initialized (lazy loading)"""
        if self.provider is None:
            self.provider = self._create_provider(self.config)

    @staticmethod
    def _create_provider(config: LLMConfig):
        """Create the appropriate provider instance from a config."""
        if config.provider == LLMProvider.OPENAI:
            return OpenAIProvider(config)
        elif config.provider == LLMProvider.ANTHROPIC:
            return AnthropicProvider(config)
        elif config.provider == LLMProvider.GOOGLE:
            return GoogleProvider(config)
        elif config.provider == LLMProvider.AZURE:
            return AzureProvider(config)
        elif config.provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceProvider(config)
        elif config.provider == LLMProvider.AWS_BEDROCK:
            return BedrockProvider(config)
        elif config.provider == LLMProvider.GROK:
            return GrokProvider(config)
        elif config.provider == LLMProvider.OPENROUTER:
            return OpenRouterProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {config.provider}")

    def _is_retriable_model_error(self, exc: Exception) -> bool:
        """Return True if the exception indicates a dead/removed model (not transient)."""
        error_text = str(exc)
        # Also check for 404 status codes embedded in exception messages
        if "404" in error_text:
            return True
        return any(p.search(error_text) for p in self._DEAD_MODEL_PATTERNS)

    def _get_fallback_model(self) -> Optional[str]:
        """Get fallback model: user setting > provider default > None."""
        # Check for user-configured fallback
        category = SERVICE_CATEGORY_MAP.get(self.service_name, "orchestrator_llm")
        user_fallback = get_system_setting(category, "fallback_model")
        if user_fallback:
            return user_fallback
        return self._DEFAULT_FALLBACK_MODELS.get(self.config.provider)

    def _build_fallback_config(self, fallback_model: str) -> LLMConfig:
        """Build an LLMConfig that reuses primary credentials but swaps the model."""
        return LLMConfig(
            provider=self.config.provider,
            model=fallback_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization_id=self.config.organization_id,
            secret_key=self.config.secret_key,
        )

    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Any:
        """Generate response using the configured provider, with automatic usage tracking.

        If the primary model returns a dead-model error (404 / "no endpoints found"),
        retries once with a fallback model on the same provider and tags the
        response with ``_used_fallback = True``.
        """
        self._ensure_provider_initialized()
        start = time.monotonic()
        try:
            response = await self.provider.generate_response(messages, tools)
            self._track_usage(response, start)
            return response
        except Exception as exc:
            if not self._is_retriable_model_error(exc):
                self._track_usage(None, start, status="error")
                raise

            # Dead-model path — attempt fallback
            fallback_model = self._get_fallback_model()
            if not fallback_model or fallback_model == self.config.model:
                self._track_usage(None, start, status="error")
                raise

            logger.warning(
                "LLM_MODEL_FAILED: Primary model '%s' on provider '%s' is unavailable (%s). "
                "Retrying with fallback model '%s'. "
                "ACTION REQUIRED: Update your model in Settings > Orchestrator.",
                self.config.model, self.config.provider.value, exc,
                fallback_model,
            )

            try:
                fb_config = self._build_fallback_config(fallback_model)
                fb_provider = self._create_provider(fb_config)
                response = await fb_provider.generate_response(messages, tools)
                # Tag so callers know this came from fallback
                response._used_fallback = True
                response._failed_model = self.config.model
                response._fallback_model = fallback_model
                self._track_usage(response, start, status="fallback")
                return response
            except Exception as fb_exc:
                logger.error(
                    "LLM_FALLBACK_FAILED: Fallback model '%s' also failed: %s",
                    fallback_model, fb_exc,
                )
                self._track_usage(None, start, status="error")
                raise fb_exc from exc

    def generate_response_sync(self, messages: List[Dict[str, str]]) -> Any:
        """Generate response using the configured provider (synchronous), with automatic usage tracking.

        Same fallback logic as ``generate_response``.
        """
        self._ensure_provider_initialized()
        start = time.monotonic()
        try:
            response = self.provider.generate_response_sync(messages)
            self._track_usage(response, start)
            return response
        except Exception as exc:
            if not self._is_retriable_model_error(exc):
                self._track_usage(None, start, status="error")
                raise

            fallback_model = self._get_fallback_model()
            if not fallback_model or fallback_model == self.config.model:
                self._track_usage(None, start, status="error")
                raise

            logger.warning(
                "LLM_MODEL_FAILED: Primary model '%s' on provider '%s' is unavailable (%s). "
                "Retrying with fallback model '%s'. "
                "ACTION REQUIRED: Update your model in Settings > Orchestrator.",
                self.config.model, self.config.provider.value, exc,
                fallback_model,
            )

            try:
                fb_config = self._build_fallback_config(fallback_model)
                fb_provider = self._create_provider(fb_config)
                response = fb_provider.generate_response_sync(messages)
                response._used_fallback = True
                response._failed_model = self.config.model
                response._fallback_model = fallback_model
                self._track_usage(response, start, status="fallback")
                return response
            except Exception as fb_exc:
                logger.error(
                    "LLM_FALLBACK_FAILED: Fallback model '%s' also failed: %s",
                    fallback_model, fb_exc,
                )
                self._track_usage(None, start, status="error")
                raise fb_exc from exc

    def _track_usage(self, response: Any, start: float, status: str = "success") -> None:
        """Track LLM usage via UsageTracker if workspace_id is set."""
        ws = self._tracking_ctx.get("workspace_id")
        if not ws:
            return
        try:
            from .usage_tracker import UsageTracker

            latency_ms = int((time.monotonic() - start) * 1000)

            usage = getattr(response, "usage", None) or {}
            input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

            UsageTracker.track(
                workspace_id=ws,
                model_id=self.config.model or "unknown",
                provider=self.config.provider.value if self.config.provider else "unknown",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                agent_id=self._tracking_ctx.get("agent_id"),
                execution_id=self._tracking_ctx.get("execution_id"),
                request_type=self._tracking_ctx.get("request_type", self.service_name),
                latency_ms=latency_ms,
                status=status,
                is_byok=self._tracking_ctx.get("is_byok", False),
            )
        except Exception as e:
            logger.debug(f"Usage tracking failed: {e}")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider configuration"""
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "service_name": self.service_name
        }


# Convenience function for quick usage
def create_llm_manager(
    service_name: str = "orchestrator",
    provider: str = None,
    model: str = None
) -> LLMManager:
    """
    Create an LLM manager with optional overrides.
    
    Args:
        service_name: Service name for per-service settings
        provider: Optional provider override
        model: Optional model override
        
    Returns:
        LLMManager instance
    """
    return LLMManager(
        service_name=service_name,
        provider=provider,
        model=model
    )

