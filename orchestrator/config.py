"""
Centralized Configuration Management
=====================================

Single source of truth for all configuration.
NO HARDCODED VALUES - Everything from environment.
"""

import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

class Config:
    """
    Central configuration class - NO HARDCODED VALUES
    All configuration from environment variables
    """
    
    # API Configuration
    API_HOST: str = os.getenv("API_URL", "localhost")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_KEY: str = os.getenv("API_KEY", "")
    
    # Construct full API URL based on environment
    @property
    def API_BASE_URL(self) -> str:
        """Get the full API base URL based on environment"""
        if self.API_HOST in ["localhost", "127.0.0.1"]:
            return f"http://{self.API_HOST}:{self.API_PORT}"
        else:
            # Production/staging uses HTTPS
            return f"https://{self.API_HOST}"
    
    # PRD-18: Database Configuration - Try credential system first
    @property
    def DATABASE_URL(self) -> str:
        """Get database URL from credential system or environment variables"""
        try:
            from services.credential_resolver import resolve_postgres_params
            params = resolve_postgres_params()
            return (
                f"postgresql://{params['user']}:{params['password']}@"
                f"{params['host']}:{params['port']}/{params['database']}"
            )
        except:
            # Fallback to environment variables
            return os.getenv(
                "DATABASE_URL", 
                f"postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', '')}@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'orchestrator_db')}"
            )
    
    # Keep individual properties for backward compatibility
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "orchestrator_db")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", "5432"))
    
    # PRD-18: Redis Configuration - Try credential system first
    @property
    def REDIS_URL(self) -> str:
        """Get Redis URL from credential system or environment variables"""
        try:
            from services.credential_resolver import resolve_redis_params
            params = resolve_redis_params()
            
            if params.get('password'):
                return f"redis://:{params['password']}@{params['host']}:{params['port']}"
            return f"redis://{params['host']}:{params['port']}"
        except:
            # Fallback to environment variables
            redis_password = os.getenv("REDIS_PASSWORD", "")
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = os.getenv("REDIS_PORT", "6379")
            
            if redis_password:
                return f"redis://:{redis_password}@{redis_host}:{redis_port}"
            return f"redis://{redis_host}:{redis_port}"
    
    # Keep individual properties for backward compatibility
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    
    # PRD-18: LLM Configuration - Use credential system only (no .env fallback)
    @property
    def OPENAI_API_KEY(self) -> str:
        """Get OpenAI API key from credential system"""
        from services.credential_resolver import resolve_openai_key
        key = resolve_openai_key()
        if not key:
            raise ValueError("OpenAI API key not found. Configure 'development_openai' credential.")
        return key
    
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        """Get Anthropic API key from credential system"""
        from services.credential_resolver import resolve_anthropic_key
        key = resolve_anthropic_key()
        if not key:
            raise ValueError("Anthropic API key not found. Configure 'development_anthropic' credential.")
        return key
    
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # Environment Settings
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def IS_PRODUCTION(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT.lower() == "production"
    
    @property
    def IS_DEVELOPMENT(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT.lower() == "development"
    
    # Deployment Configuration
    DEPLOY_HOST: str = os.getenv("DEPLOY_HOST", "")
    DEPLOY_PORT: int = int(os.getenv("DEPLOY_PORT", "22"))
    DEPLOY_USER: str = os.getenv("DEPLOY_USER", "root")
    
    # Frontend Configuration
    NEXTAUTH_SECRET: str = os.getenv("NEXTAUTH_SECRET", "")
    NEXTAUTH_URL: str = os.getenv("NEXTAUTH_URL", "")
    NEXT_PUBLIC_API_URL: str = os.getenv("NEXT_PUBLIC_API_URL", "")
    
    # Feature Flags
    ENABLE_BATCH_API: bool = os.getenv("ENABLE_BATCH_API", "false").lower() == "true"
    ENABLE_MOCK_DATA: bool = False  # NEVER allow mock data
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "true").lower() == "true"
    
    def validate(self) -> bool:
        """
        Validate required configuration
        Returns True if all required config is present
        """
        errors = []
        
        # Check required API keys
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required for OpenAI provider")
        
        if self.LLM_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY is required for Anthropic provider")
        
        # Check database
        if not self.POSTGRES_PASSWORD:
            errors.append("POSTGRES_PASSWORD is required")
        
        # Print errors if any
        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"  - {error}")
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
        print(f"API URL: {self.API_BASE_URL}")
        print(f"Database: {self.POSTGRES_DB}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}")
        print(f"Redis: {self.REDIS_HOST}:{self.REDIS_PORT}")
        print(f"LLM Provider: {self.LLM_PROVIDER} ({self.LLM_MODEL})")
        
        if show_secrets:
            print(f"OpenAI Key: {self.OPENAI_API_KEY[:10]}..." if self.OPENAI_API_KEY else "Not set")
            print(f"Anthropic Key: {self.ANTHROPIC_API_KEY[:10]}..." if self.ANTHROPIC_API_KEY else "Not set")
        else:
            print(f"OpenAI Key: {'Set' if self.OPENAI_API_KEY else 'Not set'}")
            print(f"Anthropic Key: {'Set' if self.ANTHROPIC_API_KEY else 'Not set'}")
        
        print(f"Mock Data: {'ENABLED ⚠️' if self.ENABLE_MOCK_DATA else 'DISABLED ✅'}")
        print(f"API Key Required: {self.REQUIRE_API_KEY}")
        print("=" * 60)

# Singleton instance
config = Config()

# Validate on import
if not config.validate():
    print("⚠️  WARNING: Configuration validation failed")

# Export for easy import
__all__ = ['config', 'Config']
