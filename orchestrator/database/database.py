from dotenv import load_dotenv
load_dotenv()


"""
Database Configuration and Session Management
============================================

Database setup, connection management, and session handling for Automotas AI.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

from .models import Base

logger = logging.getLogger(__name__)

# PRD-18: Database configuration - Try credential system first, fallback to environment variables
def get_database_url() -> str:
    """
    Get database URL from credential system or environment variables.
    Tries credential system first for secure credential management.
    """
    try:
        from services.credential_resolver import get_credential_resolver
        resolver = get_credential_resolver()
        params = resolver.get_postgres_connection_params()
        
        # Build connection string from credential
        return (
            f"postgresql://{params['user']}:{params['password']}@"
            f"{params['host']}:{params['port']}/{params['database']}"
        )
    except Exception as e:
        # Fallback to environment variables (transition period)
        logger.warning(f"Using environment variables for database (credential system not available): {e}")
        
        # Try DATABASE_URL first (Heroku, Railway, etc.)
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            return database_url
        
        # Build from individual env vars - NO HARDCODED DEFAULTS!
        user = os.getenv('POSTGRES_USER')
        password = os.getenv('POSTGRES_PASSWORD')
        host = os.getenv('POSTGRES_HOST')
        port = os.getenv('POSTGRES_PORT')
        database = os.getenv('POSTGRES_DB')
        
        if not all([user, password, host, port, database]):
            missing = []
            if not user: missing.append("POSTGRES_USER")
            if not password: missing.append("POSTGRES_PASSWORD")
            if not host: missing.append("POSTGRES_HOST")
            if not port: missing.append("POSTGRES_PORT")
            if not database: missing.append("POSTGRES_DB")
            
            raise ValueError(
                f"Database credentials not configured. Missing from .env: {', '.join(missing)}. "
                f"Either set these environment variables or configure 'development_db' credential in database."
            )
        
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

DATABASE_URL = get_database_url()

# Create PostgreSQL engine with connection pooling
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

# Database event listeners for PostgreSQL
# No special connection setup needed for PostgreSQL

if __name__ == "__main__":
    init_database()
