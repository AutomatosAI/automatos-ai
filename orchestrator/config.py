"""
Centralized Configuration Management
=====================================

ONLY PLACE where os.getenv() is called for configuration.
All other files import from here.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Config:
    """
    Central configuration - ONLY place where os.getenv() is called
    Simple property access, no complex logic
    """
    
    # =============================================================================
    # DATABASE - PostgreSQL (required)
    # =============================================================================
    POSTGRES_DB: str = os.getenv("POSTGRES_DB")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT")
    DATABASE_URL: str = os.getenv("DATABASE_URL")  # If set, overrides individual params
    
    # =============================================================================
    # REDIS - Caching/PubSub (optional)
    # =============================================================================
    REDIS_HOST: str = os.getenv("REDIS_HOST")
    REDIS_PORT: str = os.getenv("REDIS_PORT")
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD")
    
    # =============================================================================
    # API SECURITY
    # =============================================================================
    API_KEY: str = os.getenv("API_KEY")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    
    # =============================================================================
    # CORS (Frontend origins)
    # =============================================================================
    CORS_ALLOW_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,https://ui.automatos.app")
    
    # =============================================================================
    # LLM KEYS (Optional - LLM Manager handles these)
    # =============================================================================
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    
    # LLM settings - loaded from database system_settings (fallback to env vars)
    @property
    def LLM_PROVIDER(self) -> str:
        """Get LLM provider from system settings (database) or environment"""
        try:
            from services.llm_provider.manager import get_system_setting
            return get_system_setting("orchestrator_llm", "provider", os.getenv("LLM_PROVIDER", "openai"))
        except:
            return os.getenv("LLM_PROVIDER", "openai")
    
    @property
    def LLM_MODEL(self) -> str:
        """Get LLM model from system settings (database) or environment"""
        try:
            from services.llm_provider.manager import get_system_setting
            # Get from database settings, fallback to env var, then hardcoded default
            return get_system_setting("orchestrator_llm", "model", os.getenv("LLM_MODEL", "gpt-4o"))
        except:
            return os.getenv("LLM_MODEL", "gpt-4o")  # Default to gpt-4o (128K context) instead of gpt-4 (8K)
    
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def IS_PRODUCTION(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"
    
    # =============================================================================
    # DEPLOYMENT (for scripts)
    # =============================================================================
    DEPLOY_HOST: str = os.getenv("DEPLOY_HOST")
    DEPLOY_PORT: int = int(os.getenv("DEPLOY_PORT", "22"))
    DEPLOY_USER: str = os.getenv("DEPLOY_USER", "root")
    
    # =============================================================================
    # FRONTEND ENV (NextAuth)
    # =============================================================================
    NEXTAUTH_SECRET: str = os.getenv("NEXTAUTH_SECRET")
    NEXTAUTH_URL: str = os.getenv("NEXTAUTH_URL")
    NEXT_PUBLIC_API_URL: str = os.getenv("NEXT_PUBLIC_API_URL")
    
    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    ENABLE_BATCH_API: bool = os.getenv("ENABLE_BATCH_API", "false").lower() == "true"
    
    def validate(self) -> bool:
        """
        Validate required configuration
        Returns True if all required config is present
        """
        errors = []
        
        # Check database
        if not all([self.POSTGRES_DB, self.POSTGRES_USER, self.POSTGRES_HOST, self.POSTGRES_PORT]):
            errors.append("Database not configured: POSTGRES_DB, POSTGRES_USER, POSTGRES_HOST, POSTGRES_PORT required")
        
        # Check API key
        if self.REQUIRE_API_KEY and not self.API_KEY:
            errors.append("API_KEY required when REQUIRE_API_KEY=true")
        
        # Print errors if any
        if errors:
            logger.error("❌ Configuration errors:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        return True
    
    def print_config(self, show_secrets: bool = False) -> None:
        """
        Print current configuration (masks secrets by default)
        """
        print("=" * 60)
        print("AUTOMATOS AI CONFIGURATION")
        print("=" * 60)
        
        print(f"Environment: {self.ENVIRONMENT}")
        print(f"Database: {self.POSTGRES_DB}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}")
        print(f"Redis: {self.REDIS_HOST}:{self.REDIS_PORT if self.REDIS_HOST else 'Not configured'}")
        print(f"LLM Provider: {self.LLM_PROVIDER} ({self.LLM_MODEL})")
        
        if show_secrets:
            print(f"OpenAI Key: {self.OPENAI_API_KEY[:10]}..." if self.OPENAI_API_KEY else "Not set")
            print(f"Anthropic Key: {self.ANTHROPIC_API_KEY[:10]}..." if self.ANTHROPIC_API_KEY else "Not set")
            print(f"API Key: {self.API_KEY[:10]}..." if self.API_KEY else "Not set")
        else:
            print(f"OpenAI Key: {'✅ Set' if self.OPENAI_API_KEY else '❌ Not set'}")
            print(f"Anthropic Key: {'✅ Set' if self.ANTHROPIC_API_KEY else '❌ Not set'}")
            print(f"API Key: {'✅ Set' if self.API_KEY else '❌ Not set'}")
        
        print(f"API Key Required: {self.REQUIRE_API_KEY}")
        print("=" * 60)

# Singleton instance
config = Config()

# Backward compatibility alias
orchestrator_config = config

# Validate on import (non-blocking)
# if not config.validate():
#     logger.warning("⚠️  WARNING: Configuration validation failed")

# Export for easy import
__all__ = ['config', 'Config', 'orchestrator_config']
