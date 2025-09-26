#!/usr/bin/env python3
"""
Database Table Creator
=====================
Creates missing tables required for the dashboard
"""

import os
import sys
from pathlib import Path
import psycopg2
import logging
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
parent_dir = str(Path(__file__).resolve().parents[2])
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Load environment variables
load_dotenv()

# Database connection parameters
DB_HOST = os.environ.get('POSTGRES_HOST', 'localhost')
DB_PORT = os.environ.get('POSTGRES_PORT', '5432')
DB_NAME = os.environ.get('POSTGRES_DB', 'automatos_ai')
DB_USER = os.environ.get('POSTGRES_USER', 'postgres')
DB_PASSWORD = os.environ.get('POSTGRES_PASSWORD', 'postgres')

def create_tables():
    """Create missing dashboard tables in the database"""
    
    print("\n=== Creating Dashboard Database Tables ===")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Set autocommit
        conn.autocommit = True
        
        # Create a cursor
        cur = conn.cursor()
        
        # Create context_optimization_metrics table
        print("Creating context_optimization_metrics table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS context_optimization_metrics (
            id SERIAL PRIMARY KEY,
            tokens_saved INTEGER NOT NULL DEFAULT 0,
            compression_ratio FLOAT NOT NULL DEFAULT 1.0,
            optimization_type VARCHAR(50),
            pattern_used VARCHAR(100),
            execution_time_ms INTEGER,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """)
        
        # Create memory_items table
        print("Creating memory_items table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_items (
            id SERIAL PRIMARY KEY,
            agent_id INTEGER,
            memory_type VARCHAR(50) NOT NULL,
            memory_level VARCHAR(50) NOT NULL DEFAULT 'working',
            content TEXT NOT NULL,
            importance FLOAT DEFAULT 0.5,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            last_accessed TIMESTAMP WITH TIME ZONE
        );
        """)
        
        # Create knowledge_nodes table (corrected name)
        print("Creating knowledge_nodes table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            id SERIAL PRIMARY KEY,
            node_type VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            embedding_vector FLOAT[] DEFAULT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """)
        
        # Create collaboration_sessions table
        print("Creating collaboration_sessions table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS collaboration_sessions (
            id SERIAL PRIMARY KEY,
            session_type VARCHAR(50) NOT NULL,
            initiator_agent_id INTEGER,
            participants JSONB NOT NULL DEFAULT '[]',
            status VARCHAR(20) NOT NULL DEFAULT 'active',
            start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            end_time TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """)
        
        # Create memory_consolidations table
        print("Creating memory_consolidations table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS memory_consolidations (
            id SERIAL PRIMARY KEY,
            agent_id INTEGER,
            source_memory_ids INTEGER[] NOT NULL DEFAULT '{}',
            target_memory_id INTEGER,
            consolidation_type VARCHAR(50) NOT NULL,
            improvement_score FLOAT DEFAULT 0.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """)
        
        # Create indexes
        print("Creating indexes...")
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_memory_items_agent_id ON memory_items(agent_id);
        CREATE INDEX IF NOT EXISTS idx_memory_items_memory_level ON memory_items(memory_level);
        CREATE INDEX IF NOT EXISTS idx_memory_items_created_at ON memory_items(created_at);
        CREATE INDEX IF NOT EXISTS idx_context_metrics_created_at ON context_optimization_metrics(created_at);
        CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_created_at ON knowledge_graph_nodes(created_at);
        CREATE INDEX IF NOT EXISTS idx_collaboration_sessions_status ON collaboration_sessions(status);
        CREATE INDEX IF NOT EXISTS idx_memory_consolidations_agent_id ON memory_consolidations(agent_id);
        """)
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        print("\n✅ All dashboard tables created successfully")
        return True
            
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        return False

if __name__ == "__main__":
    create_tables()