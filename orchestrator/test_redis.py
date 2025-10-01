#!/usr/bin/env python3
"""
Test Redis connection for inter-agent communication
"""
import os
import asyncio
import redis.asyncio as redis
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_redis_connection():
    """Test Redis connection with proper configuration"""
    try:
        # Build Redis URL from environment variables
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = os.getenv("REDIS_PORT", "6379")
        redis_password = os.getenv("REDIS_PASSWORD", "")
        
        if redis_password:
            redis_url = f"redis://:{redis_password}@{redis_host}:{redis_port}"
        else:
            redis_url = f"redis://{redis_host}:{redis_port}"
        
        print(f"Connecting to Redis: {redis_url}")
        
        # Create Redis client
        redis_client = await redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # Test connection
        result = await redis_client.ping()
        print(f"✅ Redis connection successful: {result}")
        
        # Test pub/sub
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("test_channel")
        print("✅ Redis pub/sub setup successful")
        
        # Clean up
        await pubsub.unsubscribe("test_channel")
        await redis_client.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_redis_connection())
    if success:
        print("🎉 Redis is ready for inter-agent communication!")
    else:
        print("💥 Redis connection failed - inter-agent communication disabled")

