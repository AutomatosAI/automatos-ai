"""
Plugin Content Cache Service
=============================

Redis caching layer for frequently accessed plugin content from S3.
Wraps MarketplaceS3Service with Redis get/set to reduce S3 latency.

Cache key format: plugin_content:{slug}:{version}
TTL: configurable via PLUGIN_CACHE_TTL_SECONDS env var (default 1 hour)
Graceful fallback: if Redis unavailable, fetches directly from S3.
"""

import json
import logging
from typing import Dict, Optional

from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class PluginContentCache:
    """Redis-backed cache that wraps MarketplaceS3Service for plugin content."""

    # Key prefixes
    CONTENT_PREFIX = "plugin_content:"
    MANIFEST_PREFIX = "plugin_manifest:"
    FILES_PREFIX = "plugin_files:"

    def __init__(self, s3_service=None):
        """Initialise with optional S3 service and Redis client.

        Parameters
        ----------
        s3_service : MarketplaceS3Service, optional
            If not provided, one will be created lazily on first use.
        """
        self._s3 = s3_service
        self._redis = None
        self._redis_available = True  # optimistic; flip on first failure

        # Read TTL from config
        try:
            from config import config
            self._ttl = config.PLUGIN_CACHE_TTL_SECONDS
        except Exception:
            self._ttl = 3600  # 1 hour default

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_redis(self):
        """Return a usable Redis connection or None."""
        if not self._redis_available:
            return None

        if self._redis is not None:
            return self._redis

        try:
            from core.redis.client import get_redis_client

            client = get_redis_client()
            if client:
                self._redis = client.get_redis()
                return self._redis
            else:
                self._redis_available = False
                return None
        except Exception as e:
            logger.warning("Redis unavailable for plugin cache: %s", e)
            self._redis_available = False
            return None

    def _get_s3(self):
        """Return the MarketplaceS3Service, creating lazily if needed."""
        if self._s3 is None:
            from core.services.marketplace_s3 import MarketplaceS3Service

            self._s3 = MarketplaceS3Service()
        return self._s3

    def _cache_get(self, key: str) -> Optional[str]:
        """Get a value from Redis (returns None on miss or error)."""
        r = self._get_redis()
        if r is None:
            return None
        try:
            return r.get(key)
        except RedisError as e:
            logger.warning("Redis GET failed for %s: %s", key, e)
            return None

    def _cache_set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set a value in Redis with TTL (silently ignores errors)."""
        r = self._get_redis()
        if r is None:
            return
        try:
            r.setex(key, ttl or self._ttl, value)
        except RedisError as e:
            logger.warning("Redis SET failed for %s: %s", key, e)

    def _cache_delete(self, key: str) -> None:
        """Delete a key from Redis (silently ignores errors)."""
        r = self._get_redis()
        if r is None:
            return
        try:
            r.delete(key)
        except RedisError as e:
            logger.warning("Redis DELETE failed for %s: %s", key, e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_plugin_content(
        self, slug: str, version: str
    ) -> Dict[str, str]:
        """Return all plugin files as ``{relative_path: content}``.

        Checks Redis first; falls back to S3 on miss and populates cache.
        """
        cache_key = f"{self.CONTENT_PREFIX}{slug}:{version}"

        # 1. Try cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            try:
                logger.debug("Plugin content cache HIT for %s@%s", slug, version)
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                # Corrupt cache entry — fall through to S3
                self._cache_delete(cache_key)

        # 2. Fetch from S3
        logger.debug("Plugin content cache MISS for %s@%s — fetching from S3", slug, version)
        s3 = self._get_s3()
        file_keys = await s3.list_plugin_files(slug, version)

        prefix = f"plugins/{slug}/{version}/"
        files: Dict[str, str] = {}
        for key in file_keys:
            try:
                content = await s3.get_file(key)
                relative = key[len(prefix):] if key.startswith(prefix) else key
                files[relative] = content
            except Exception as file_err:
                logger.warning("Could not read S3 file %s: %s", key, file_err)

        # 3. Populate cache
        try:
            self._cache_set(cache_key, json.dumps(files))
        except Exception as e:
            logger.warning("Failed to cache plugin content for %s@%s: %s", slug, version, e)

        return files

    async def get_manifest(self, slug: str, version: str) -> Optional[dict]:
        """Return parsed manifest.json, cached in Redis.

        Returns None on any error (S3 unavailable, no manifest, etc.)
        """
        cache_key = f"{self.MANIFEST_PREFIX}{slug}:{version}"

        # 1. Try cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            try:
                return json.loads(cached)
            except (json.JSONDecodeError, TypeError):
                self._cache_delete(cache_key)

        # 2. Fetch from S3
        try:
            s3 = self._get_s3()
            manifest = await s3.get_manifest(slug, version)
        except Exception as e:
            logger.warning("Could not load manifest for %s@%s: %s", slug, version, e)
            return None

        # 3. Populate cache
        try:
            self._cache_set(cache_key, json.dumps(manifest))
        except Exception as e:
            logger.warning("Failed to cache manifest for %s@%s: %s", slug, version, e)

        return manifest

    async def get_file(self, slug: str, version: str, file_path: str) -> Optional[str]:
        """Return a single plugin file, cached in Redis.

        ``file_path`` is the relative path within the plugin (e.g. ``SKILL.md``).
        """
        cache_key = f"{self.FILES_PREFIX}{slug}:{version}:{file_path}"

        # 1. Try cache
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # 2. Fetch from S3
        try:
            s3 = self._get_s3()
            s3_key = f"plugins/{slug}/{version}/{file_path}"
            content = await s3.get_file(s3_key)
        except Exception as e:
            logger.warning("Could not load file %s for %s@%s: %s", file_path, slug, version, e)
            return None

        # 3. Populate cache (raw string, not JSON)
        self._cache_set(cache_key, content)

        return content

    def invalidate_plugin(self, slug: str, version: str) -> None:
        """Clear all cached content for a plugin version.

        Should be called when a plugin is updated or deleted.
        """
        r = self._get_redis()
        if r is None:
            return

        patterns = [
            f"{self.CONTENT_PREFIX}{slug}:{version}",
            f"{self.MANIFEST_PREFIX}{slug}:{version}",
            f"{self.FILES_PREFIX}{slug}:{version}:*",
        ]

        try:
            # Direct key deletes for content and manifest
            r.delete(
                f"{self.CONTENT_PREFIX}{slug}:{version}",
                f"{self.MANIFEST_PREFIX}{slug}:{version}",
            )

            # Scan for individual file keys (pattern match)
            file_pattern = f"{self.FILES_PREFIX}{slug}:{version}:*"
            keys = list(r.scan_iter(file_pattern))
            if keys:
                r.delete(*keys)

            logger.info("Invalidated plugin cache for %s@%s (%d file keys)", slug, version, len(keys))
        except RedisError as e:
            logger.warning("Failed to invalidate plugin cache for %s@%s: %s", slug, version, e)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_plugin_cache: Optional[PluginContentCache] = None


def get_plugin_content_cache(s3_service=None) -> PluginContentCache:
    """Get or create the plugin content cache singleton."""
    global _plugin_cache
    if _plugin_cache is None:
        _plugin_cache = PluginContentCache(s3_service=s3_service)
    return _plugin_cache
