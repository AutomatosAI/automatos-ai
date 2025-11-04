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
            max_connections=50
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
        redis_async = aioredis.Redis(
            host=self.host,
            port=self.port,
            password=self.password,
            db=self.db,
            decode_responses=True
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
    Uses centralized config - NO direct os.getenv() calls.
    Returns None if Redis is not configured (optional service).
    """
    global _redis_client
    if _redis_client is None:
        from config import config
        
        host = config.REDIS_HOST
        port = config.REDIS_PORT
        password = config.REDIS_PASSWORD
        
        if not host or not port:
            logger.warning("Redis not configured (REDIS_HOST/REDIS_PORT missing). Redis features disabled.")
            return None
        
        try:
            init_redis_client(host=host, port=int(port), password=password)
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            return None
    return _redis_client

