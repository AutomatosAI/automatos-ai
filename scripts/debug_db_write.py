
import sys
import os
import logging
from datetime import datetime
from dotenv import load_dotenv

# Add orchestrator to path
sys.path.append(os.path.join(os.getcwd(), "orchestrator"))

# Load environment variables explicitly
load_dotenv(os.path.join(os.getcwd(), "orchestrator", ".env"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from core.database.database import get_database_url
from core.models.tools import Tool, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_db_write():
    url = get_database_url()
    logger.info(f"Connecting to DB: {url.split('@')[-1]}") # Log host part only
    
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    db = Session()
    
    try:
        # Check current count
        initial_count = db.query(Tool).count()
        logger.info(f"Initial tool count: {initial_count}")
        
        # Create dummy tool
        test_name = f"debug_tool_{int(datetime.now().timestamp())}"
        t = Tool(
            name=test_name,
            description="Debug Tool",
            category="system",
            provider="debug",
            version="0.0.1",
            status="active"
        )
        db.add(t)
        db.commit()
        logger.info(f"Committed tool: {test_name}")
        
        # Verify immediately
        verify_count = db.query(Tool).count()
        logger.info(f"Post-commit count: {verify_count}")
        
        # Verify via raw SQL to ensure it's not just session cache
        with engine.connect() as conn:
            raw_count = conn.execute(text("SELECT count(*) FROM tools")).scalar()
            logger.info(f"Raw SQL count: {raw_count}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    test_db_write()
