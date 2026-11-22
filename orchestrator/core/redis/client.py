"""
Redis Pub/Sub Client for Real-Time Workflow Updates
"""
import json
import logging
import redis
import redis.asyncio as aioredis
from typing import Optional, Dict, Any
from contextlib import contextmanager, asynccontextmanager

from core.redis.connection import (
    ConnectionKind,
    RedisTarget,
    connection_kwargs,
    redis_target_from_config,
)

logger = logging.getLogger(__name__)

#: Several services call the SYNC client on the event loop (memory L1, the
#: cache, the rate limiter): an unbounded socket read there freezes every
#: request in the process. 2026-09-16: a pool pointed at a remote Redis took
#: 26–153 s per call and the chat lane stalled with it. Bound every socket op.
REDIS_SOCKET_TIMEOUT_S = 2.0
REDIS_CONNECT_TIMEOUT_S = 2.0
#: A pooled connection idle longer than this is pinged before reuse, so a
#: server-side close surfaces as a reconnect instead of "closed by server".
REDIS_HEALTH_CHECK_INTERVAL_S = 30


class RedisClient:
    """Redis client for publishing workflow execution updates"""
    
    def __init__(self, target: RedisTarget):
        self.target = target
        self.pool = redis.ConnectionPool(
            **connection_kwargs(target, ConnectionKind.POOL),
            decode_responses=True,
            max_connections=50,
            socket_timeout=REDIS_SOCKET_TIMEOUT_S,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT_S,
            socket_keepalive=True,
            health_check_interval=REDIS_HEALTH_CHECK_INTERVAL_S,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Redis connection pool created for %s", target.describe())

    def get_redis(self):
        """Get a Redis connection from the pool"""
        return redis.Redis(connection_pool=self.pool)

    @contextmanager
    def pubsub_client(self):
        """Context manager for pub/sub client (synchronous)"""
        redis_client = self.get_redis()
        pubsub = redis_client.pubsub()
        try:
            yield pubsub
        finally:
            pubsub.close()
            redis_client.close()

    async def get_async_pubsub(self, channel: str):
        """
        Get an async Redis pubsub client for real-time streaming
        
        This is used by WebSocket endpoints for non-blocking message delivery
        """
        # Connect bounded; no read timeout — a subscription idles by design.
        redis_async = aioredis.Redis(
            **connection_kwargs(self.target, ConnectionKind.ASYNC),
            decode_responses=True,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT_S,
            socket_keepalive=True,
            health_check_interval=REDIS_HEALTH_CHECK_INTERVAL_S,
        )
        pubsub = redis_async.pubsub()
        await pubsub.subscribe(channel)
        self.logger.info(f"✅ Async pubsub subscribed to channel: {channel}")
        return redis_async, pubsub

    def publish(self, channel: str, message: Dict[str, Any]) -> bool:
        """
        Publish a message to a Redis channel
        
        Args:
            channel: Redis channel name
            message: Message dictionary to publish
            
        Returns:
            True if successful, False otherwise
        """
        redis_client = None
        try:
            redis_client = self.get_redis()
            message_str = json.dumps(message)
            redis_client.publish(channel, message_str)
            self.logger.info(f"✅ Published to Redis channel '{channel}': {message.get('type', 'unknown')}")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to publish to Redis channel '{channel}': {e}", exc_info=True)
            return False
        finally:
            if redis_client:
                redis_client.close()

    def publish_workflow_event(
        self,
        workflow_id: int,
        execution_id: int,
        event_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Publish a workflow execution event
        
        Args:
            workflow_id: Workflow ID
            execution_id: Execution ID
            event_type: Event type (e.g., 'execution_started', 'subtask_execution_update')
            data: Event data
            
        Returns:
            True if successful, False otherwise
        """
        channel = f"workflow:{workflow_id}:execution:{execution_id}"
        message = {
            "type": event_type,
            "data": {
                "execution_id": execution_id,
                "workflow_id": workflow_id,
                **data
            }
        }
        return self.publish(channel, message)

    def test_connection(self) -> bool:
        """Test Redis connection"""
        redis_client = None
        try:
            redis_client = self.get_redis()
            redis_client.ping()
            self.logger.info("✅ Redis connection test successful")
            return True
        except Exception as e:
            self.logger.error(f"❌ Redis connection test failed: {e}", exc_info=True)
            return False
        finally:
            if redis_client:
                redis_client.close()


# Global Redis client instance (lazy-initialized)
_redis_client: Optional[RedisClient] = None


def init_redis_client(target: RedisTarget) -> None:
    """Initialize the global Redis client (explicit init)."""
    global _redis_client
    _redis_client = RedisClient(target)
    _redis_client.test_connection()
    logger.info("Redis client initialized")


def get_redis_client() -> Optional[RedisClient]:
    """The global Redis client, built on first use from the settings.

    The address is ``REDIS_URL``, or one built from ``REDIS_HOST``/``REDIS_PORT``
    (see ``core.redis.connection``). Returns None when Redis is not configured
    (an optional service) or cannot be set up.
    """
    if _redis_client is None:
        try:
            target = redis_target_from_config()
        except ValueError as e:
            # Never the address or the error's text: either can hold part of a password.
            logger.error("REDIS_URL is not a usable Redis address (%s). Redis features disabled.", type(e).__name__)
            return None
        if target is None:
            logger.warning("Redis not configured (REDIS_URL or REDIS_HOST/REDIS_PORT missing). Redis features disabled.")
            return None
        try:
            logger.info("Redis target %s", target.describe())
            init_redis_client(target)
        except Exception:
            logger.exception("Failed to initialize Redis client for %s", target.describe())
            return None
    return _redis_client
