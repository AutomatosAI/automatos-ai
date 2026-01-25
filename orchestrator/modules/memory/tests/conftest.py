"""
Pytest configuration for Memory module tests
Uses centralized credential system for Redis/PostgreSQL
"""

import pytest
import os

# Set test environment
os.environ['ENVIRONMENT'] = 'development'

@pytest.fixture(scope="session", autouse=True)
def init_redis_for_tests():
    """Initialize Redis client using centralized credential system"""
    from core.credentials.resolver import get_credential_resolver
    from core.redis.client import init_redis_client
    
    try:
        # Get Redis credentials from database or .env
        resolver = get_credential_resolver()
        redis_params = resolver.get_redis_connection_params(
            credential_name="development_redis"
        )
        
        # Initialize global Redis client
        init_redis_client(
            host=redis_params['host'],
            port=redis_params['port'],
            password=redis_params.get('password'),
            db=redis_params.get('database', 0)
        )
        print(f"✅ Redis initialized for tests: {redis_params['host']}:{redis_params['port']}")
    except Exception as e:
        print(f"⚠️  Redis initialization failed: {e}")
        # Tests will handle Redis not being available


@pytest.fixture
def memory_system():
    """
    Create memory system with real services.
    Uses centralized credential system for Redis/PostgreSQL.
    """
    from modules.memory.storage.knowledge_system import HierarchicalMemorySystem
    return HierarchicalMemorySystem()
