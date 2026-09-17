"""
Redis Pub/Sub Client for Real-Time Workflow Updates
"""
import json
import logging
import redis
import redis.asyncio as aioredis
from typing import Optional, Dict, Any
from contextlib import contextmanager, asynccontextmanager

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
    
    def __init__(self, host: str = '127.0.0.1', port: int = 6379, password: Optional[str] = None, db: int = 0):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            password=password,
            db=db,
            decode_responses=True,
            max_connections=50,
            socket_timeout=REDIS_SOCKET_TIMEOUT_S,
            socket_connect_timeout=REDIS_CONNECT_TIMEOUT_S,
            socket_keepalive=True,
            health_check_interval=REDIS_HEALTH_CHECK_INTERVAL_S,
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Redis connection pool created for {host}:{port}")

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
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
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


def init_redis_client(host: str = '127.0.0.1', port: int = 6379, password: Optional[str] = None, db: int = 0):
    """Initialize the global Redis client (explicit init)"""
    global _redis_client
    _redis_client = RedisClient(host=host, port=port, password=password, db=db)
    _redis_client.test_connection()
    logger.info("Redis client initialized")


def get_redis_client() -> Optional[RedisClient]:
    """
    Get the global Redis client instance with lazy initialization.
    Uses centralized config - supports REDIS_URL (Railway, Heroku) or individual vars.
    Returns None if Redis is not configured (optional service).
    """
    global _redis_client
    if _redis_client is None:
        from config import config
        from urllib.parse import urlparse
        
        # Try REDIS_URL first (Railway, Heroku, etc.)
        redis_url = config.REDIS_URL
        if redis_url:
            try:
                parsed = urlparse(redis_url)
                host = parsed.hostname
                port = parsed.port or 6379
                password = parsed.password
                db = int(parsed.path.lstrip('/')) if parsed.path.lstrip('/') else 0
                
                if host and port:
                    try:
                        logger.info("Redis target %s:%s db=%s (from REDIS_URL)", host, port, db)
                        init_redis_client(host=host, port=port, password=password, db=db)
                    except Exception as e:
                        logger.error(f"Failed to initialize Redis client from URL: {e}")
                        return None
                else:
                    logger.warning("Redis URL provided but could not parse host/port")
                    return None
            except Exception as e:
                logger.error(f"Failed to parse REDIS_URL: {e}")
                return None
        else:
            # Fallback to individual environment variables
            host = config.REDIS_HOST
            port = config.REDIS_PORT
            password = config.REDIS_PASSWORD
            
            if not host or not port:
                logger.warning("Redis not configured (REDIS_URL or REDIS_HOST/REDIS_PORT missing). Redis features disabled.")
                return None
            
            try:
                logger.info("Redis target %s:%s (from REDIS_HOST/REDIS_PORT)", host, port)
                init_redis_client(host=host, port=int(port), password=password)
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
                return None
    return _redis_client

