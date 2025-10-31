"""
LLM Manager
===========

Main LLM manager that handles provider selection and configuration.
Supports per-service configuration via system settings.
"""

import os
import logging
from typing import Dict, Any, List, Optional

from .clients.base import LLMProvider, LLMConfig
from .clients.openai_client import OpenAIProvider
from .clients.anthropic_client import AnthropicProvider
from .clients.google_client import GoogleProvider
from .clients.azure_client import AzureProvider
from .clients.huggingface_client import HuggingFaceProvider

logger = logging.getLogger(__name__)


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
        # Lazy import to avoid circular dependency during startup
        import sys
        if 'database.database' not in sys.modules or not hasattr(sys.modules.get('database.database'), 'SessionLocal'):
            # Database not initialized yet - return default
            return default_value
        
        from database.database import SessionLocal
        from models.system_settings import SystemSetting
        
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
    except Exception as e:
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
    # Map service names to settings categories
    category_map = {
        "orchestrator": "orchestrator_llm",
        "codegraph": "codegraph",
        "document_processing": "document_processing",
        "chatbot": "chatbot",
        "rag": "rag",
        "embeddings": "embeddings",
        "memory_integration": "memory_integration",
        "nl_to_sql": "nl_to_sql"
    }
    
    category = category_map.get(service_name, "orchestrator_llm")
    
    # Get provider and model from settings
    provider_str = get_system_setting(category, "provider", "openai")
    model_str = get_system_setting(category, "model")
    
    # Default models if not set in settings
    if not model_str:
        default_models = {
            "openai": "gpt-4",
            "anthropic": "claude-3-5-sonnet-20241022",
            "google": "gemini-pro",
            "azure": "gpt-4",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
        }
        model_str = default_models.get(provider_str, "gpt-4")
    
    return provider_str, model_str


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
        environment = os.getenv('ENVIRONMENT', 'development')
    
    try:
        from services.credential_resolver import get_credential_resolver
        
        resolver = get_credential_resolver()
        
        # MVP: Strategy 0 - Check system settings for explicit credential name mapping
        # e.g., orchestrator_llm.credential_name_openai = "development_openai"
        settings_category_map = {
            "orchestrator": "orchestrator_llm",
            "codegraph": "codegraph",
            "chatbot": "chatbot",
            "document_processing": "document_processing",
            "rag": "rag",
            "embeddings": "embeddings",
            "memory_integration": "memory_integration",
            "nl_to_sql": "nl_to_sql"
        }
        
        category = settings_category_map.get(service_name, "orchestrator_llm")
        credential_name_setting_key = f"credential_name_{provider.lower()}"
        
        # Try to get explicit credential name from system settings
        explicit_credential_name = get_system_setting(category, credential_name_setting_key, None)
        
        # Check if explicit credential name is set (not None and not empty string)
        if explicit_credential_name and explicit_credential_name.strip():
            try:
                cred_data = resolver.get_dict(explicit_credential_name, environment=environment)
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
            "huggingface": "huggingface_api"
        }
        
        credential_type = credential_type_map.get(provider.lower())
        if not credential_type:
            logger.warning(f"Unknown provider for credential lookup: {provider}")
            return {}
        
        # Strategy 1: Try standard naming pattern with _api suffix
        credential_name_variations = [
            f"{environment}_{provider}_api",      # development_openai_api (standard)
            f"{environment}_{provider}",          # development_openai (user's format)
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
        
        # Try each credential name variation
        for cred_name in credential_name_variations:
            try:
                cred_data = resolver.get_dict(cred_name, environment=environment)
                if cred_data and len(cred_data) > 0:
                    logger.info(f"Found credential '{cred_name}' for provider '{provider}'")
                    return cred_data
            except Exception as e:
                logger.debug(f"Credential name '{cred_name}' not found: {e}")
                continue
        
        # Strategy 2: Try to find by credential type (if name-based lookup failed)
        # Search for any active credential of the matching type in the environment
        try:
            from database.database import SessionLocal
            from services.credential_service import CredentialStore
            from models.credentials import CredentialType
            
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
        
        logger.warning(
            f"Could not find credential for provider '{provider}' using any variation. "
            f"Tried: {credential_name_variations}"
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
        model: Optional[str] = None
    ):
        """
        Initialize LLM Manager.
        
        Args:
            config: Optional LLMConfig (if None, loads from settings/env)
            service_name: Service name for per-service configuration ('orchestrator', 'codegraph', etc.)
            provider: Optional provider override
            model: Optional model override
        """
        self.service_name = service_name
        
        if config is None:
            config = self._load_config_from_settings(service_name, provider, model)
        
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
        
        # Validate provider
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            logger.warning(f"Unknown LLM provider: {provider_str}, defaulting to OpenAI")
            provider = LLMProvider.OPENAI
        
        # Get model from settings or use override
        if model_override:
            model = model_override
        else:
            _, model = get_provider_and_model_from_settings(service_name)
            if not model:
                # Default models for each provider
                default_models = {
                    LLMProvider.OPENAI: "gpt-4",
                    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
                    LLMProvider.GOOGLE: "gemini-pro",
                    LLMProvider.AZURE: "gpt-4",
                    LLMProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2"
                }
                model = default_models.get(provider, "gpt-4")
        
        # Get other settings with defaults
        category_map = {
            "orchestrator": "orchestrator_llm",
            "codegraph": "codegraph",
            "document_processing": "document_processing",
            "chatbot": "chatbot",
            "rag": "rag",
            "embeddings": "embeddings",
            "memory_integration": "memory_integration",
            "nl_to_sql": "nl_to_sql"
        }
        category = category_map.get(service_name, "orchestrator_llm")
        
        temperature = float(get_system_setting(category, "temperature", "0.7"))
        max_tokens = int(get_system_setting(category, "max_tokens", "2000"))
        
        # Get credential data (pass service_name for settings lookup)
        cred_data = get_credential_data(provider.value, service_name=service_name)
        
        # Extract API key and other credential fields
        api_key = None
        base_url = None
        organization_id = None
        
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
        
        # Fallback to environment variables if credentials not found
        if not api_key:
            fallback_env_vars = {
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
                LLMProvider.GOOGLE: "GOOGLE_API_KEY",
                LLMProvider.AZURE: "AZURE_OPENAI_API_KEY",
                LLMProvider.HUGGINGFACE: "HUGGINGFACE_API_TOKEN"
            }
            env_var = fallback_env_vars.get(provider)
            if env_var:
                api_key = os.getenv(env_var)
        
        return LLMConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            organization_id=organization_id
        )
    
    def _ensure_provider_initialized(self):
        """Ensure provider is initialized (lazy loading)"""
        if self.provider is None:
            self.provider = self._create_provider()
    
    def _create_provider(self):
        """Create the appropriate provider instance"""
        if self.config.provider == LLMProvider.OPENAI:
            return OpenAIProvider(self.config)
        elif self.config.provider == LLMProvider.ANTHROPIC:
            return AnthropicProvider(self.config)
        elif self.config.provider == LLMProvider.GOOGLE:
            return GoogleProvider(self.config)
        elif self.config.provider == LLMProvider.AZURE:
            return AzureProvider(self.config)
        elif self.config.provider == LLMProvider.HUGGINGFACE:
            return HuggingFaceProvider(self.config)
        else:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Any:
        """Generate response using the configured provider"""
        self._ensure_provider_initialized()
        return await self.provider.generate_response(messages, tools)
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> Any:
        """Generate response using the configured provider (synchronous)"""
        self._ensure_provider_initialized()
        return self.provider.generate_response_sync(messages)
    
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

