"""
Create missing database tables for dashboard functionality
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

def create_missing_tables():
    """Create missing tables for dashboard functionality"""
    try:
        # Get database connection
        config = Config()
        engine = create_engine(config.DATABASE_URL)
        
        # Read SQL script
        script_path = os.path.join(os.path.dirname(__file__), 'create_missing_tables.sql')
        with open(script_path, 'r') as f:
            sql_script = f.read()
        
        # Execute SQL statements
        with engine.connect() as connection:
            connection.execute(text(sql_script))
            connection.commit()
        
        logger.info("Successfully created missing tables")
        return True
        
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False

if __name__ == "__main__":
    logger.info("Creating missing tables for dashboard functionality...")
    success = create_missing_tables()
    if success:
        logger.info("Tables created successfully!")
    else:
        logger.error("Failed to create tables")
        sys.exit(1)
