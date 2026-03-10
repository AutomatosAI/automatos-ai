from dotenv import load_dotenv
load_dotenv()


"""
Database Configuration and Session Management
============================================

Database setup, connection management, and session handling for Automotas AI.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

from config import config
from core.database.base import Base

logger = logging.getLogger(__name__)

# PRD-18: Database configuration - Try credential system first, fallback to environment variables
def get_database_url() -> str:
    """
    Get database URL from credential system or environment variables.
    Tries credential system first for secure credential management.
    """
    try:
        from core.credentials.resolver import get_credential_resolver
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
        database_url = config.DATABASE_URL
        if database_url:
            return database_url

        # Build from individual env vars - NO HARDCODED DEFAULTS!
        user = config.POSTGRES_USER
        password = config.POSTGRES_PASSWORD
        host = config.POSTGRES_HOST
        port = config.POSTGRES_PORT
        database = config.POSTGRES_DB
        
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

# Enforce SSL in production
_environment = (config.ENVIRONMENT or "development").lower()
_connect_args = {}
if _environment == "production":
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    _parsed = urlparse(DATABASE_URL)
    _params = parse_qs(_parsed.query)
    if "sslmode" not in _params:
        _params["sslmode"] = ["require"]
        DATABASE_URL = urlunparse(_parsed._replace(query=urlencode(_params, doseq=True)))

# Create PostgreSQL engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every hour
    pool_timeout=30,    # Wait up to 30 seconds for a connection
    echo=config.SQL_DEBUG
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
            # Initialize default data here
            pass
            
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise