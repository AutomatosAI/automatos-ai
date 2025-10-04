"""
Add Vector Indexes for Memory System Performance
===============================================

This script adds pgvector indexes to optimize memory retrieval performance.
Run this once to enhance your existing memory system.
"""

import asyncio
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

logger = logging.getLogger(__name__)

async def add_vector_indexes(database_url: str):
    """Add vector indexes to existing memory tables"""
    
    # Convert to async URL if needed
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    engine = create_async_engine(database_url, echo=False)
    
    async with engine.begin() as conn:
        try:
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("✅ pgvector extension enabled")
            
            # Add vector index for memory_items table
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memory_items_embedding_cosine 
                ON memory_items 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100)
            """))
            logger.info("✅ Added cosine similarity index to memory_items")
            
            # Add vector index for knowledge_nodes table
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_embedding_cosine 
                ON knowledge_nodes 
                USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 50)
            """))
            logger.info("✅ Added cosine similarity index to knowledge_nodes")
            
            # Add composite indexes for common queries
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memory_items_agent_level_type 
                ON memory_items (agent_id, memory_level, memory_type)
            """))
            logger.info("✅ Added composite index for agent/level/type queries")
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memory_items_importance_access 
                ON memory_items (importance DESC, last_access DESC)
            """))
            logger.info("✅ Added importance/access index")
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_agent_type 
                ON knowledge_nodes (agent_id, node_type)
            """))
            logger.info("✅ Added knowledge node agent/type index")
            
            # Add partial indexes for performance
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memory_items_long_term 
                ON memory_items (agent_id, created_at DESC) 
                WHERE memory_level = 'long_term'
            """))
            logger.info("✅ Added long-term memory partial index")
            
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_memory_items_high_importance 
                ON memory_items (agent_id, importance DESC) 
                WHERE importance > 0.7
            """))
            logger.info("✅ Added high importance partial index")
            
            logger.info("🚀 All vector indexes added successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to add indexes: {e}")
            raise
    
    await engine.dispose()

if __name__ == "__main__":
    import os
    
    # Get database URL from environment
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/automatos")
    
    print("Adding vector indexes to memory system...")
    asyncio.run(add_vector_indexes(database_url))
    print("Vector indexes added successfully!")
