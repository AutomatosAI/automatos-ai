
"""
Database Configuration and Session Management
============================================

Database setup, connection management, and session handling for Automotas AI.
"""

import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import logging

from models import Base

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "sqlite:///./automotas_ai.db"
)

# Create engine with connection pooling
if "sqlite" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
    )

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Session:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()

def init_database():
    """Initialize database with default data"""
    try:
        create_tables()
        
        with get_db_session() as db:
            from models import SystemConfiguration, RAGConfiguration
            
            # Create default system configurations
            default_configs = [
                {
                    "config_key": "system.max_agents",
                    "config_value": {"value": 100},
                    "description": "Maximum number of agents allowed in the system"
                },
                {
                    "config_key": "system.default_timeout",
                    "config_value": {"value": 300},
                    "description": "Default timeout for agent operations (seconds)"
                },
                {
                    "config_key": "rag.default_model",
                    "config_value": {"value": "sentence-transformers/all-MiniLM-L6-v2"},
                    "description": "Default embedding model for RAG system"
                },
                {
                    "config_key": "workflow.max_concurrent",
                    "config_value": {"value": 10},
                    "description": "Maximum concurrent workflow executions"
                }
            ]
            
            for config_data in default_configs:
                existing = db.query(SystemConfiguration).filter(
                    SystemConfiguration.config_key == config_data["config_key"]
                ).first()
                
                if not existing:
                    config = SystemConfiguration(**config_data)
                    db.add(config)
            
            # Create default RAG configuration
            existing_rag = db.query(RAGConfiguration).filter(
                RAGConfiguration.name == "default"
            ).first()
            
            if not existing_rag:
                default_rag = RAGConfiguration(
                    name="default",
                    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                    chunk_size=1000,
                    chunk_overlap=200,
                    retrieval_strategy="similarity",
                    top_k=5,
                    similarity_threshold=0.7,
                    configuration={
                        "max_tokens": 4000,
                        "temperature": 0.7,
                        "use_reranking": True
                    },
                    is_active=True,
                    created_by="system"
                )
                db.add(default_rag)
            
            db.commit()
            logger.info("Database initialized with default data")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

# Database event listeners
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance (if using SQLite)"""
    if "sqlite" in DATABASE_URL:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

if __name__ == "__main__":
    init_database()
