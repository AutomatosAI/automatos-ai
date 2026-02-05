"""
Caching Service - Redis-based caching for expensive operations
===============================================================

Reduces API costs and rate limits by caching:
- Embeddings (OpenAI API)
- Document chunks (processing time)
- Cloud file listings (Composio API)
- Search results (vector database queries)
"""

import hashlib
import json
import logging
from typing import Any, Optional, List, Dict
from datetime import timedelta

import redis
from redis import Redis

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service with workspace isolation"""

    # Cache TTLs (Time To Live)
    TTL_EMBEDDINGS = None  # Never expire (only invalidate on model change)
    TTL_CHUNKS = None  # Never expire (only invalidate on document change)
    TTL_CLOUD_LISTINGS = 300  # 5 minutes (cloud files don't change often)
    TTL_SEARCH_RESULTS = 600  # 10 minutes
    TTL_DOCUMENT_METADATA = 1800  # 30 minutes

    def __init__(self, redis_client: Redis):
        self.redis = redis_client
        logger.info("✅ CacheService initialized")

    def _key(self, namespace: str, key: str, workspace_id: Optional[str] = None) -> str:
        """Generate cache key with workspace isolation"""
        if workspace_id:
            return f"cache:{namespace}:{workspace_id}:{key}"
        return f"cache:{namespace}:{key}"

    def _hash(self, data: Any) -> str:
        """Generate hash for cache key"""
        if isinstance(data, str):
            content = data
        elif isinstance(data, (list, dict)):
            content = json.dumps(data, sort_keys=True)
        else:
            content = str(data)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # =========================================================================
    # EMBEDDING CACHE (Most important - OpenAI API costs)
    # =========================================================================

    def get_embedding(self, text: str, model: str = "default") -> Optional[List[float]]:
        """Get cached embedding by text hash"""
        text_hash = self._hash(text)
        key = self._key("embedding", f"{model}:{text_hash}")

        try:
            cached = self.redis.get(key)
            if cached:
                logger.debug(f"✅ Cache hit: embedding for text hash {text_hash}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_embedding(
        self,
        text: str,
        embedding: List[float],
        model: str = "default"
    ) -> None:
        """Cache embedding (never expires)"""
        text_hash = self._hash(text)
        key = self._key("embedding", f"{model}:{text_hash}")

        try:
            self.redis.set(key, json.dumps(embedding))
            logger.debug(f"💾 Cached embedding for text hash {text_hash}")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def get_embeddings_batch(
        self,
        texts: List[str],
        model: str = "default"
    ) -> Dict[str, Optional[List[float]]]:
        """Get multiple embeddings at once (more efficient)"""
        result = {}
        keys = []

        for text in texts:
            text_hash = self._hash(text)
            keys.append(self._key("embedding", f"{model}:{text_hash}"))

        try:
            cached_values = self.redis.mget(keys)
            for i, (text, cached) in enumerate(zip(texts, cached_values)):
                if cached:
                    result[text] = json.loads(cached)
                    logger.debug(f"✅ Cache hit: embedding {i+1}/{len(texts)}")
                else:
                    result[text] = None

            hit_rate = sum(1 for v in result.values() if v is not None) / len(texts) * 100
            logger.info(f"📊 Embedding cache hit rate: {hit_rate:.1f}% ({len(texts)} texts)")
            return result
        except Exception as e:
            logger.warning(f"Cache batch get error: {e}")
            return {text: None for text in texts}

    def set_embeddings_batch(
        self,
        embeddings: Dict[str, List[float]],
        model: str = "default"
    ) -> None:
        """Cache multiple embeddings at once"""
        try:
            pipeline = self.redis.pipeline()
            for text, embedding in embeddings.items():
                text_hash = self._hash(text)
                key = self._key("embedding", f"{model}:{text_hash}")
                pipeline.set(key, json.dumps(embedding))
            pipeline.execute()
            logger.info(f"💾 Cached {len(embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"Cache batch set error: {e}")

    # =========================================================================
    # DOCUMENT CHUNKS CACHE
    # =========================================================================

    def get_chunks(
        self,
        document_hash: str,
        strategy: str = "default"
    ) -> Optional[List[Dict]]:
        """Get cached document chunks"""
        key = self._key("chunks", f"{document_hash}:{strategy}")

        try:
            cached = self.redis.get(key)
            if cached:
                logger.info(f"✅ Cache hit: chunks for document {document_hash[:8]}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_chunks(
        self,
        document_hash: str,
        chunks: List[Dict],
        strategy: str = "default"
    ) -> None:
        """Cache document chunks (never expires)"""
        key = self._key("chunks", f"{document_hash}:{strategy}")

        try:
            self.redis.set(key, json.dumps(chunks))
            logger.info(f"💾 Cached {len(chunks)} chunks for document {document_hash[:8]}")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    # =========================================================================
    # CLOUD FILE LISTINGS CACHE (Composio API savings)
    # =========================================================================

    def get_cloud_listing(
        self,
        connection_id: int,
        path: str,
        list_type: str = "files"
    ) -> Optional[List[Dict]]:
        """Get cached cloud file/folder listing"""
        listing_key = f"{connection_id}:{path}:{list_type}"
        key = self._key("cloud_listing", listing_key)

        try:
            cached = self.redis.get(key)
            if cached:
                logger.info(f"✅ Cache hit: {list_type} listing for {path}")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_cloud_listing(
        self,
        connection_id: int,
        path: str,
        listing: List[Dict],
        list_type: str = "files"
    ) -> None:
        """Cache cloud file/folder listing (5 min TTL)"""
        listing_key = f"{connection_id}:{path}:{list_type}"
        key = self._key("cloud_listing", listing_key)

        try:
            self.redis.setex(
                key,
                self.TTL_CLOUD_LISTINGS,
                json.dumps(listing)
            )
            logger.info(f"💾 Cached {list_type} listing for {path} (TTL: {self.TTL_CLOUD_LISTINGS}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    def invalidate_cloud_listing(self, connection_id: int) -> None:
        """Invalidate all cached listings for a connection"""
        try:
            pattern = self._key("cloud_listing", f"{connection_id}:*")
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"🗑️  Invalidated {len(keys)} cloud listings for connection {connection_id}")
        except Exception as e:
            logger.warning(f"Cache invalidation error: {e}")

    # =========================================================================
    # SEARCH RESULTS CACHE
    # =========================================================================

    def get_search_results(
        self,
        query: str,
        filters: Optional[Dict] = None,
        workspace_id: Optional[str] = None
    ) -> Optional[List[Dict]]:
        """Get cached search results"""
        search_key = self._hash({"query": query, "filters": filters})
        key = self._key("search", search_key, workspace_id)

        try:
            cached = self.redis.get(key)
            if cached:
                logger.info(f"✅ Cache hit: search results for query '{query[:30]}...'")
                return json.loads(cached)
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None

    def set_search_results(
        self,
        query: str,
        results: List[Dict],
        filters: Optional[Dict] = None,
        workspace_id: Optional[str] = None
    ) -> None:
        """Cache search results (10 min TTL)"""
        search_key = self._hash({"query": query, "filters": filters})
        key = self._key("search", search_key, workspace_id)

        try:
            self.redis.setex(
                key,
                self.TTL_SEARCH_RESULTS,
                json.dumps(results)
            )
            logger.info(f"💾 Cached search results for query '{query[:30]}...' (TTL: {self.TTL_SEARCH_RESULTS}s)")
        except Exception as e:
            logger.warning(f"Cache set error: {e}")

    # =========================================================================
    # STATISTICS & MONITORING
    # =========================================================================

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            info = self.redis.info('stats')
            return {
                "total_keys": self.redis.dbsize(),
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "hit_rate": info.get('keyspace_hits', 0) / max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0), 1) * 100
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}

    def clear_namespace(self, namespace: str, workspace_id: Optional[str] = None) -> int:
        """Clear all keys in a namespace"""
        try:
            pattern = self._key(namespace, "*", workspace_id)
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
                logger.info(f"🗑️  Cleared {len(keys)} keys from namespace '{namespace}'")
                return len(keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear namespace: {e}")
            return 0


# Global cache instance (initialized on first use)
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get or create global cache service instance"""
    global _cache_service

    if _cache_service is None:
        from core.redis.client import get_redis_client
        redis_client = get_redis_client()
        _cache_service = CacheService(redis_client)

    return _cache_service
