"""
Cache Management API
====================

Monitor and manage Redis caching for cost optimization.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission
from core.auth.hybrid import get_request_context_hybrid
from core.cache import get_cache_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cache", tags=["cache"])


@router.get("/stats")
async def get_cache_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Get cache statistics and performance metrics.

    Shows:
    - Total keys cached
    - Cache hit/miss rates
    - Cost savings (estimated API calls avoided)
    """
    try:
        cache = get_cache_service()
        stats = cache.get_cache_stats()

        return {
            "cache_enabled": True,
            "total_keys": stats.get("total_keys", 0),
            "hits": stats.get("hits", 0),
            "misses": stats.get("misses", 0),
            "hit_rate_percent": round(stats.get("hit_rate", 0), 2),
            "estimated_api_calls_saved": stats.get("hits", 0),
            "estimated_cost_savings_usd": round(stats.get("hits", 0) * 0.0001, 2),  # $0.0001 per embedding
            "namespaces": {
                "embeddings": "OpenAI API calls (never expires)",
                "chunks": "Document processing (never expires)",
                "cloud_listings": "Composio API calls (5 min TTL)",
                "search": "Vector search results (10 min TTL)"
            }
        }

    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/clear/{namespace}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def clear_cache_namespace(
    namespace: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Clear cache for a specific namespace.

    Namespaces:
    - embeddings: Clear all cached embeddings (use after model change)
    - chunks: Clear all cached document chunks
    - cloud_listings: Clear all cloud file listings
    - search: Clear all search results
    """
    try:
        cache = get_cache_service()

        if namespace not in ["embeddings", "chunks", "cloud_listings", "search"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid namespace. Must be one of: embeddings, chunks, cloud_listings, search"
            )

        # Clear namespace with workspace isolation
        keys_deleted = cache.clear_namespace(namespace, str(ctx.workspace_id))

        return {
            "success": True,
            "namespace": namespace,
            "keys_deleted": keys_deleted,
            "message": f"Cleared {keys_deleted} keys from {namespace} cache"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/invalidate/cloud/{connection_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def invalidate_cloud_cache(
    connection_id: int,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    Invalidate all cached cloud listings for a connection.

    Use this after:
    - Uploading new files to Google Drive/Dropbox
    - Deleting files from cloud storage
    - Moving/renaming files
    """
    try:
        cache = get_cache_service()
        cache.invalidate_cloud_listing(connection_id)

        return {
            "success": True,
            "connection_id": connection_id,
            "message": f"Invalidated all cloud listings for connection {connection_id}"
        }

    except Exception as e:
        logger.error(f"Error invalidating cloud cache: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
