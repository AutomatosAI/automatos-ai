"""
Database Knowledge Cache Service
=================================
PRD-21: Redis caching for schema metadata and query results

Uses existing Redis instance for:
- Schema metadata (1 hour TTL)
- Query results (5 minute TTL)
- Semantic layer definitions
- Template results
"""

import json
import hashlib
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class DatabaseCacheService:
    """
    Redis-based caching for database knowledge sources.
    Significantly improves performance by caching:
    - Schema metadata (rarely changes)
    - Query results (expensive operations)
    - Semantic definitions
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        """Initialize Redis connection using existing instance."""
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.default_schema_ttl = 3600  # 1 hour
        self.default_query_ttl = 300    # 5 minutes
        self.default_template_ttl = 600  # 10 minutes
        
        # Key prefixes for organization
        self.SCHEMA_PREFIX = "db:schema:"
        self.QUERY_PREFIX = "db:query:"
        self.SEMANTIC_PREFIX = "db:semantic:"
        self.TEMPLATE_PREFIX = "db:template:"
        self.STATS_PREFIX = "db:stats:"
        
        logger.info("Database cache service initialized with Redis")
    
    # ========================================================================
    # Schema Caching (Aggressive caching as requested!)
    # ========================================================================
    
    def cache_schema(
        self,
        source_id: int,
        tenant_id: int,
        schema_data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache database schema metadata.
        Schemas rarely change, so we cache aggressively.
        """
        try:
            key = f"{self.SCHEMA_PREFIX}{tenant_id}:{source_id}"
            ttl = ttl or self.default_schema_ttl
            
            # Add timestamp for cache validation
            schema_data['cached_at'] = datetime.utcnow().isoformat()
            schema_data['expires_at'] = (
                datetime.utcnow() + timedelta(seconds=ttl)
            ).isoformat()
            
            # Store in Redis
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(schema_data)
            )
            
            # Update stats
            self._increment_stat(f"schema_cache_writes:{tenant_id}")
            
            logger.debug(f"Cached schema for source {source_id} (TTL: {ttl}s)")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to cache schema: {e}")
            return False
    
    def get_cached_schema(
        self,
        source_id: int,
        tenant_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached schema if available and not expired.
        """
        try:
            key = f"{self.SCHEMA_PREFIX}{tenant_id}:{source_id}"
            cached = self.redis_client.get(key)
            
            if cached:
                schema_data = json.loads(cached)
                self._increment_stat(f"schema_cache_hits:{tenant_id}")
                logger.debug(f"Schema cache hit for source {source_id}")
                return schema_data
            else:
                self._increment_stat(f"schema_cache_misses:{tenant_id}")
                logger.debug(f"Schema cache miss for source {source_id}")
                return None
                
        except RedisError as e:
            logger.error(f"Failed to get cached schema: {e}")
            return None
    
    def invalidate_schema(self, source_id: int, tenant_id: int) -> bool:
        """
        Invalidate schema cache when schema changes.
        """
        try:
            key = f"{self.SCHEMA_PREFIX}{tenant_id}:{source_id}"
            result = self.redis_client.delete(key)
            
            # Also invalidate semantic layer
            semantic_key = f"{self.SEMANTIC_PREFIX}{tenant_id}:{source_id}"
            self.redis_client.delete(semantic_key)
            
            logger.info(f"Invalidated schema cache for source {source_id}")
            return bool(result)
            
        except RedisError as e:
            logger.error(f"Failed to invalidate schema: {e}")
            return False
    
    # ========================================================================
    # Query Result Caching
    # ========================================================================
    
    def cache_query_result(
        self,
        source_id: int,
        tenant_id: int,
        sql: str,
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache query results with SQL as key.
        Returns cache key for future reference.
        """
        try:
            # Generate cache key from SQL and source
            cache_key = self._generate_query_cache_key(source_id, sql)
            key = f"{self.QUERY_PREFIX}{tenant_id}:{cache_key}"
            ttl = ttl or self.default_query_ttl
            
            # Add metadata
            result['cached_at'] = datetime.utcnow().isoformat()
            result['sql'] = sql
            result['source_id'] = source_id
            
            # Store in Redis
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(result, default=str)  # Handle datetime serialization
            )
            
            self._increment_stat(f"query_cache_writes:{tenant_id}")
            logger.debug(f"Cached query result (key: {cache_key[:8]}..., TTL: {ttl}s)")
            
            return cache_key
            
        except RedisError as e:
            logger.error(f"Failed to cache query result: {e}")
            return ""
    
    def get_cached_query(
        self,
        source_id: int,
        tenant_id: int,
        sql: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query result if available.
        """
        try:
            cache_key = self._generate_query_cache_key(source_id, sql)
            key = f"{self.QUERY_PREFIX}{tenant_id}:{cache_key}"
            
            cached = self.redis_client.get(key)
            
            if cached:
                result = json.loads(cached)
                self._increment_stat(f"query_cache_hits:{tenant_id}")
                logger.debug(f"Query cache hit (key: {cache_key[:8]}...)")
                return result
            else:
                self._increment_stat(f"query_cache_misses:{tenant_id}")
                return None
                
        except RedisError as e:
            logger.error(f"Failed to get cached query: {e}")
            return None
    
    # ========================================================================
    # Semantic Layer Caching
    # ========================================================================
    
    def cache_semantic_layer(
        self,
        source_id: int,
        tenant_id: int,
        metrics: List[Dict],
        dimensions: List[Dict],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache semantic layer definitions (metrics and dimensions).
        """
        try:
            key = f"{self.SEMANTIC_PREFIX}{tenant_id}:{source_id}"
            ttl = ttl or self.default_schema_ttl  # Same as schema TTL
            
            semantic_data = {
                'metrics': metrics,
                'dimensions': dimensions,
                'cached_at': datetime.utcnow().isoformat(),
                'version': 1
            }
            
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(semantic_data)
            )
            
            logger.debug(f"Cached semantic layer for source {source_id}")
            return True
            
        except RedisError as e:
            logger.error(f"Failed to cache semantic layer: {e}")
            return False
    
    def get_cached_semantic_layer(
        self,
        source_id: int,
        tenant_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached semantic layer definitions.
        """
        try:
            key = f"{self.SEMANTIC_PREFIX}{tenant_id}:{source_id}"
            cached = self.redis_client.get(key)
            
            if cached:
                return json.loads(cached)
            return None
            
        except RedisError as e:
            logger.error(f"Failed to get cached semantic layer: {e}")
            return None
    
    # ========================================================================
    # Template Result Caching
    # ========================================================================
    
    def cache_template_result(
        self,
        template_id: int,
        parameters: Dict[str, Any],
        result: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache template execution results.
        """
        try:
            # Create cache key from template + parameters
            param_str = json.dumps(parameters, sort_keys=True)
            cache_key = hashlib.md5(
                f"{template_id}:{param_str}".encode()
            ).hexdigest()
            
            key = f"{self.TEMPLATE_PREFIX}{cache_key}"
            ttl = ttl or self.default_template_ttl
            
            self.redis_client.setex(
                key,
                ttl,
                json.dumps(result, default=str)
            )
            
            return cache_key
            
        except RedisError as e:
            logger.error(f"Failed to cache template result: {e}")
            return ""
    
    # ========================================================================
    # Statistics and Monitoring
    # ========================================================================
    
    def get_cache_stats(self, tenant_id: int) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        """
        try:
            stats = {
                'schema_cache_hits': self._get_stat(f"schema_cache_hits:{tenant_id}"),
                'schema_cache_misses': self._get_stat(f"schema_cache_misses:{tenant_id}"),
                'query_cache_hits': self._get_stat(f"query_cache_hits:{tenant_id}"),
                'query_cache_misses': self._get_stat(f"query_cache_misses:{tenant_id}"),
                'query_cache_writes': self._get_stat(f"query_cache_writes:{tenant_id}")
            }
            
            # Calculate hit rates
            schema_total = stats['schema_cache_hits'] + stats['schema_cache_misses']
            query_total = stats['query_cache_hits'] + stats['query_cache_misses']
            
            stats['schema_hit_rate'] = (
                (stats['schema_cache_hits'] / schema_total * 100) 
                if schema_total > 0 else 0
            )
            stats['query_hit_rate'] = (
                (stats['query_cache_hits'] / query_total * 100)
                if query_total > 0 else 0
            )
            
            return stats
            
        except RedisError as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def clear_tenant_cache(self, tenant_id: int) -> int:
        """
        Clear all cached data for a tenant.
        """
        try:
            pattern = f"db:*:{tenant_id}:*"
            keys = list(self.redis_client.scan_iter(pattern))
            
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries for tenant {tenant_id}")
                return deleted
            
            return 0
            
        except RedisError as e:
            logger.error(f"Failed to clear tenant cache: {e}")
            return 0
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _generate_query_cache_key(self, source_id: int, sql: str) -> str:
        """Generate consistent cache key for queries."""
        # Normalize SQL for better cache hits
        normalized_sql = ' '.join(sql.lower().split())
        return hashlib.md5(
            f"{source_id}:{normalized_sql}".encode()
        ).hexdigest()
    
    def _increment_stat(self, stat_key: str) -> None:
        """Increment a statistics counter."""
        try:
            key = f"{self.STATS_PREFIX}{stat_key}"
            self.redis_client.incr(key)
        except RedisError:
            pass  # Stats are non-critical
    
    def _get_stat(self, stat_key: str) -> int:
        """Get a statistics value."""
        try:
            key = f"{self.STATS_PREFIX}{stat_key}"
            value = self.redis_client.get(key)
            return int(value) if value else 0
        except (RedisError, ValueError):
            return 0


# Singleton instance
_cache_service = None

def get_database_cache_service(redis_url: Optional[str] = None) -> DatabaseCacheService:
    """Get or create the database cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = DatabaseCacheService(redis_url or "redis://localhost:6379/0")
    return _cache_service
