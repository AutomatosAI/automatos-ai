"""
Redis Client - Pub/Sub and Caching
==================================

Redis infrastructure for real-time updates and caching.

Usage:
    from core.redis import get_redis_client, RedisClient
"""

from core.redis.client import RedisClient, get_redis_client

__all__ = [
    'RedisClient',
    'get_redis_client',
]

