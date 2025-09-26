#!/usr/bin/env python3
"""
Database Table Checker
=====================
Checks if required dashboard tables exist in the database
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

# List of tables required for dashboard
REQUIRED_TABLES = [
    'context_optimization_metrics',
    'memory_items',
    'knowledge_nodes',  # Corrected from knowledge_graph_nodes
    'collaboration_sessions',
    'memory_consolidations'
]

def check_tables():
    """Check if all required tables exist in the database"""
    
    print("\n=== Dashboard Database Table Check ===")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Create a cursor
        cur = conn.cursor()
        
        # Check each table
        missing_tables = []
        for table in REQUIRED_TABLES:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table,))
            
            exists = cur.fetchone()[0]
            
            if exists:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"❌ Table '{table}' does not exist")
                missing_tables.append(table)
        
        # Close cursor and connection
        cur.close()
        conn.close()
        
        # Print summary
        if missing_tables:
            print(f"\n❗ Missing {len(missing_tables)} tables: {', '.join(missing_tables)}")
            print("Run create_tables.py to create these tables")
            return False
        else:
            print("\n✅ All required tables exist")
            return True
            
    except Exception as e:
        print(f"❌ Error connecting to database: {str(e)}")
        return False

if __name__ == "__main__":
    check_tables()