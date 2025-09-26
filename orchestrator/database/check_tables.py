"""
Utility to check if required tables exist and report on missing tables
"""

import logging
import os
import sys
from sqlalchemy import create_engine, text

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration
from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REQUIRED_TABLES = [
    'context_optimization_metrics',
    'memory_items',
    'knowledge_graph_nodes',
    'collaboration_sessions',
    'memory_consolidations'
]

def check_tables():
    """Check if required tables exist and return results"""
    try:
        # Get database connection
        config = Config()
        engine = create_engine(config.DATABASE_URL)
        
        missing_tables = []
        existing_tables = []
        
        with engine.connect() as connection:
            for table_name in REQUIRED_TABLES:
                # Check if table exists
                query = text("""
                    SELECT COUNT(*) 
                    FROM information_schema.tables 
                    WHERE table_name = :table_name
                """)
                
                result = connection.execute(query, {"table_name": table_name}).scalar() or 0
                
                if result == 0:
                    missing_tables.append(table_name)
                else:
                    existing_tables.append(table_name)
        
        return {
            "missing_tables": missing_tables,
            "existing_tables": existing_tables,
            "all_tables_exist": len(missing_tables) == 0
        }
        
    except Exception as e:
        logger.error(f"Error checking tables: {e}")
        return {
            "error": str(e),
            "missing_tables": REQUIRED_TABLES,
            "existing_tables": [],
            "all_tables_exist": False
        }

def print_status():
    """Print table status in a readable format"""
    result = check_tables()
    
    if "error" in result:
        print(f"❌ Error checking tables: {result['error']}")
        return False
    
    print("\n=== Dashboard Database Status ===")
    print(f"Required tables: {len(REQUIRED_TABLES)}")
    print(f"Existing tables: {len(result['existing_tables'])}")
    print(f"Missing tables: {len(result['missing_tables'])}")
    
    if result['all_tables_exist']:
        print("\n✅ All required tables exist!")
    else:
        print("\n❌ Missing tables:")
        for table in result['missing_tables']:
            print(f"  - {table}")
        
        print("\n💡 Run the following command to create missing tables:")
        print("python -m database.create_tables")
    
    print("\n=================================")
    return result['all_tables_exist']

if __name__ == "__main__":
    print_status()
