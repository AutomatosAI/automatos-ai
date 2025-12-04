"""
Credential Resolver Service
===========================

Replaces os.getenv() calls with secure credential resolution from database.
Provides caching, fallback to environment variables, and audit logging.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from functools import lru_cache

from core.credentials.encryption import get_encryption_service, EncryptionKeyError

logger = logging.getLogger(__name__)


class CredentialNotFoundError(Exception):
    """Raised when credential cannot be resolved"""
    pass


class CredentialResolver:
    """
    Resolves credentials at runtime, replacing os.getenv() calls.
    
    Features:
    - Database credential resolution
    - In-memory caching (5 minute TTL)
    - Fallback to environment variables (transition period)
    - Audit logging of credential access
    - Support for synchronous and asynchronous operations
    
    Usage:
        # OLD
        api_key = os.getenv("OPENAI_API_KEY")
        
        # NEW
        api_key = credential_resolver.get("development_openai", fallback_env="OPENAI_API_KEY")
    """
    
    def __init__(self):
        """Initialize credential resolver"""
        self.encryption_service = get_encryption_service()
        self._cache: Dict[str, tuple] = {}  # {cache_key: (value, expiry_time)}
        self._cache_ttl = timedelta(minutes=5)  # Cache for 5 minutes
        logger.info("Credential resolver initialized")
    
    def get(
        self,
        credential_name: str,
        environment: str = None,
        fallback_env: Optional[str] = None,
        fallback_value: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> str:
        """
        Resolve a credential value (synchronous version).
        
        Args:
            credential_name: Name of credential in database
            environment: Environment (defaults to ENVIRONMENT env var or 'development')
            fallback_env: Environment variable name to fall back to
            fallback_value: Default value if credential not found
            service_name: Name of service requesting credential (for audit)
            
        Returns:
            Credential value (decrypted)
            
        Raises:
            CredentialNotFoundError: If credential not found and no fallback
        """
        # Determine environment
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'development')
        
        # Check cache first
        cache_key = f"{credential_name}:{environment}"
        cached_value = self._get_from_cache(cache_key)
        if cached_value is not None:
            logger.debug(f"Credential '{credential_name}' resolved from cache")
            return cached_value
        
        # Try to resolve from database
        # CRITICAL: Only try database AFTER import phase completes
        try:
            # Actually try importing instead of checking sys.modules
            # This handles both bootstrap phase AND module availability correctly
            from core.database.database import SessionLocal
            from core.credentials.service import CredentialStore
            
            db = SessionLocal()
            try:
                store = CredentialStore(db)
                decrypted_data = store.get_decrypted_credential_by_name(
                    name=credential_name,
                    environment=environment,
                    service_name=service_name or "credential_resolver"
                )
                
                # For credentials with single field, return that field
                # For multi-field credentials, return JSON
                if len(decrypted_data) == 1:
                    value = list(decrypted_data.values())[0]
                else:
                    import json
                    value = json.dumps(decrypted_data)
                
                # Cache the value
                self._add_to_cache(cache_key, value)
                
                logger.info(f"Credential '{credential_name}' resolved from database")
                return value
            
            finally:
                db.close()
        
        except ImportError as e:
            logger.debug(f"Database import failed during credential lookup: {e}")
        except Exception as e:
            logger.debug(f"Database credential lookup skipped: {e}")
        
        # Fallback to environment variable
        if fallback_env:
            env_value = os.getenv(fallback_env)
            if env_value:
                logger.warning(
                    f"Credential '{credential_name}' not found in database, "
                    f"using environment variable '{fallback_env}' as fallback"
                )
                return env_value
        
        # Use fallback value
        if fallback_value is not None:
            logger.warning(f"Using fallback value for credential '{credential_name}'")
            return fallback_value
        
        # No credential found and no fallback
        raise CredentialNotFoundError(
            f"Credential '{credential_name}' not found in {environment} environment "
            f"and no fallback provided"
        )
    
    def get_dict(
        self,
        credential_name: str,
        environment: str = None,
        fallback_env_prefix: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resolve a credential and return as dictionary (for multi-field credentials).
        
        Args:
            credential_name: Name of credential
            environment: Environment
            fallback_env_prefix: Prefix for environment variables (e.g., 'POSTGRES_')
            service_name: Service name for audit
            
        Returns:
            Dictionary of credential values
        """
        if environment is None:
            environment = os.getenv('ENVIRONMENT', 'production')
        
        cache_key = f"dict:{credential_name}:{environment}"
        cached_value = self._get_from_cache(cache_key)
        if cached_value is not None:
            import json
            return json.loads(cached_value)
        
        try:
            # Actually try importing instead of checking sys.modules
            from core.database.database import SessionLocal
            from core.credentials.service import CredentialStore
            
            db = SessionLocal()
            try:
                store = CredentialStore(db)
                decrypted_data = store.get_decrypted_credential_by_name(
                    name=credential_name,
                    environment=environment,
                    service_name=service_name or "credential_resolver"
                )
                
                # Cache as JSON string
                import json
                self._add_to_cache(cache_key, json.dumps(decrypted_data))
                
                logger.info(f"Credential dict '{credential_name}' resolved from database")
                return decrypted_data
            
            finally:
                db.close()
        
        except ImportError as e:
            logger.debug(f"Database import failed during credential lookup: {e}")
        except CredentialNotFoundError as e:
                logger.warning(f"Credential '{credential_name}' not found in database: {e}")
        except Exception as e:
            logger.warning(f"Could not load credential '{credential_name}' from database: {e}")
        
        # Fallback to environment variables if prefix provided
        if fallback_env_prefix:
            logger.debug(f"Using environment variables with prefix '{fallback_env_prefix}'")
            # Return empty dict - caller should handle fallback
            return {}
        
        raise CredentialNotFoundError(f"Credential '{credential_name}' not found - check logs for database error")
    
    def get_credential_field(
        self,
        credential_name: str,
        field_name: str,
        environment: str = None,
        fallback_env: Optional[str] = None,
        service_name: Optional[str] = None
    ) -> str:
        """
        Get a specific field from a credential.
        
        Args:
            credential_name: Name of credential
            field_name: Specific field to retrieve
            environment: Environment
            fallback_env: Fallback environment variable
            service_name: Service name for audit
            
        Returns:
            Field value
        """
        try:
            cred_data = self.get_dict(credential_name, environment, service_name=service_name)
            if field_name in cred_data:
                return cred_data[field_name]
            else:
                raise CredentialNotFoundError(
                    f"Field '{field_name}' not found in credential '{credential_name}'"
                )
        except CredentialNotFoundError:
            if fallback_env:
                env_value = os.getenv(fallback_env)
                if env_value:
                    logger.warning(f"Using environment variable '{fallback_env}' as fallback")
                    return env_value
            raise
    
    # ========================================================================
    # Cache Management
    # ========================================================================
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get value from cache if not expired"""
        if key in self._cache:
            value, expiry = self._cache[key]
            if datetime.utcnow() < expiry:
                return value
            else:
                # Expired - remove from cache
                del self._cache[key]
        return None
    
    def _add_to_cache(self, key: str, value: str) -> None:
        """Add value to cache with TTL"""
        expiry = datetime.utcnow() + self._cache_ttl
        self._cache[key] = (value, expiry)
    
    def clear_cache(self, credential_name: Optional[str] = None) -> None:
        """
        Clear credential cache.
        
        Args:
            credential_name: If provided, clear only this credential. Otherwise clear all.
        """
        if credential_name:
            # Clear all entries for this credential
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(credential_name)]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared cache for credential '{credential_name}'")
        else:
            self._cache.clear()
            logger.info("Cleared all credential cache")
    
    # ========================================================================
    # Convenience Methods for Common Credentials
    # ========================================================================
    
    def get_openai_key(
        self,
        credential_name: str = "development_openai",
        environment: str = None,
        required: bool = False
    ) -> Optional[str]:
        """
        Get OpenAI API key.
        
        Args:
            required: If True, raises error if key not found. If False, returns None.
        
        BOOTSTRAP STRATEGY:
        - LLM keys are OPTIONAL at startup
        - Only required when actually making LLM calls
        - Allows platform to start and user to configure credentials via UI
        """
        try:
            return self.get_credential_field(
                credential_name=credential_name,
                field_name="api_key",
                environment=environment,
                fallback_env="OPENAI_API_KEY",
                service_name="llm_provider"
            )
        except (CredentialNotFoundError, Exception) as e:
            if required:
                raise ValueError(f"OpenAI API key required but not found. Please configure '{credential_name}' credential.") from e
            logger.info("OpenAI API key not configured (will fail if LLM features are used)")
            return None
    
    def get_anthropic_key(
        self,
        credential_name: str = "development_anthropic",
        environment: str = None,
        required: bool = False
    ) -> Optional[str]:
        """
        Get Anthropic API key.
        
        Args:
            required: If True, raises error if key not found. If False, returns None.
        
        BOOTSTRAP STRATEGY: Same as OpenAI - optional at startup
        """
        try:
            return self.get_credential_field(
                credential_name=credential_name,
                field_name="api_key",
                environment=environment,
                fallback_env="ANTHROPIC_API_KEY",
                service_name="llm_provider"
            )
        except (CredentialNotFoundError, Exception) as e:
            if required:
                raise ValueError(f"Anthropic API key required but not found. Please configure '{credential_name}' credential.") from e
            logger.info("Anthropic API key not configured (will fail if LLM features are used)")
            return None
    
    def get_postgres_connection_params(
        self,
        credential_name: str = "development_db",
        environment: str = None
    ) -> Dict[str, Any]:
        """
        Get PostgreSQL connection parameters.
        
        BOOTSTRAP STRATEGY:
        - ALWAYS tries .env file first (for initial setup)
        - Then checks database credentials (after user sets them up)
        - This solves chicken-and-egg: DB needs credentials to store credentials!
        
        Workflow:
        1. Start platform with .env file
        2. User configures credentials in UI
        3. User can optionally delete .env file (credentials in DB take precedence)
        """
        try:
            # Try database credentials first (after bootstrap)
            return self.get_dict(
                credential_name=credential_name,
                environment=environment,
                fallback_env_prefix="POSTGRES_",
                service_name="database"
            )
        except CredentialNotFoundError:
            # ALWAYS fallback to environment variables for infrastructure
            # This is NOT a warning - it's the expected bootstrap flow
            logger.info(f"Using .env file for PostgreSQL (credential '{credential_name}' not in database yet)")
            
            # Get from .env - NO HARDCODED DEFAULTS
            # If .env missing → FAIL (don't guess!)
            host = os.getenv("POSTGRES_HOST")
            port = os.getenv("POSTGRES_PORT")
            database = os.getenv("POSTGRES_DB")
            user = os.getenv("POSTGRES_USER")
            password = os.getenv("POSTGRES_PASSWORD")
            
            # Validate ALL required fields present
            missing = []
            if not host: missing.append("POSTGRES_HOST")
            if not port: missing.append("POSTGRES_PORT")
            if not database: missing.append("POSTGRES_DB")
            if not user: missing.append("POSTGRES_USER")
            if not password: missing.append("POSTGRES_PASSWORD")
            
            if missing:
                raise CredentialNotFoundError(
                    f"PostgreSQL credentials not found in database AND missing from .env file: {', '.join(missing)}. "
                    f"Add to .env file or configure '{credential_name}' credential in UI."
                )
            
            return {
                "host": host,
                "port": int(port),
                "database": database,
                "user": user,
                "password": password
            }
    
    def get_redis_connection_params(
        self,
        credential_name: str = "development_redis",
        environment: str = None
    ) -> Dict[str, Any]:
        """
        Get Redis connection parameters.
        
        BOOTSTRAP STRATEGY: Same as Postgres - .env first, then database credentials
        """
        try:
            # Try database credentials first (after bootstrap)
            return self.get_dict(
                credential_name=credential_name,
                environment=environment,
                fallback_env_prefix="REDIS_",
                service_name="redis"
            )
        except CredentialNotFoundError:
            # ALWAYS fallback to environment variables for infrastructure
            logger.info(f"Using .env file for Redis (credential '{credential_name}' not in database yet)")
            
            # Get from .env - NO HARDCODED DEFAULTS
            # If .env missing → FAIL (don't guess!)
            host = os.getenv("REDIS_HOST")
            port = os.getenv("REDIS_PORT")
            password = os.getenv("REDIS_PASSWORD")
            db = os.getenv("REDIS_DB", "0")  # DB 0 is reasonable default
            
            # Validate required fields present
            missing = []
            if not host: missing.append("REDIS_HOST")
            if not port: missing.append("REDIS_PORT")
            # Password is optional for local dev
            
            if missing:
                raise CredentialNotFoundError(
                    f"Redis credentials not found in database AND missing from .env file: {', '.join(missing)}. "
                    f"Add to .env file or configure '{credential_name}' credential in UI."
                )
            
            return {
                "host": host,
                "port": int(port),
                "password": password if password else None,
                "database": int(db)
            }
    
    def get_github_token(
        self,
        credential_name: str = "github_main",
        environment: str = None
    ) -> str:
        """Get GitHub access token"""
        return self.get_credential_field(
            credential_name=credential_name,
            field_name="access_token",
            environment=environment,
            fallback_env="GITHUB_TOKEN",
            service_name="github_integration"
        )


# ============================================================================
# Singleton Instance
# ============================================================================

_credential_resolver: Optional[CredentialResolver] = None


def get_credential_resolver() -> CredentialResolver:
    """
    Get or create singleton credential resolver instance.
    
    Returns:
        CredentialResolver: Resolver instance
    """
    global _credential_resolver
    if _credential_resolver is None:
        _credential_resolver = CredentialResolver()
    return _credential_resolver


# ============================================================================
# Convenience Functions (for easy migration)
# ============================================================================

def resolve_credential(
    credential_name: str,
    fallback_env: Optional[str] = None,
    fallback_value: Optional[str] = None
) -> str:
    """
    Convenience function to resolve a credential.
    Makes migration from os.getenv() easier.
    
    Args:
        credential_name: Credential name
        fallback_env: Environment variable to fall back to
        fallback_value: Default value if not found
        
    Returns:
        Resolved credential value
    """
    resolver = get_credential_resolver()
    return resolver.get(
        credential_name=credential_name,
        fallback_env=fallback_env,
        fallback_value=fallback_value
    )


def resolve_openai_key(required: bool = False) -> Optional[str]:
    """
    Quick helper to get OpenAI API key.
    
    Args:
        required: If True, raises error if not found. If False, returns None.
    """
    return get_credential_resolver().get_openai_key(required=required)


def resolve_anthropic_key(required: bool = False) -> Optional[str]:
    """
    Quick helper to get Anthropic API key.
    
    Args:
        required: If True, raises error if not found. If False, returns None.
    """
    return get_credential_resolver().get_anthropic_key(required=required)


def resolve_postgres_params() -> Dict[str, Any]:
    """Quick helper to get PostgreSQL connection parameters"""
    return get_credential_resolver().get_postgres_connection_params()


def resolve_redis_params() -> Dict[str, Any]:
    """Quick helper to get Redis connection parameters"""
    return get_credential_resolver().get_redis_connection_params()


def resolve_github_token() -> str:
    """Quick helper to get GitHub token"""
    return get_credential_resolver().get_github_token()

