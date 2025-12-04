"""
Core - Platform Foundation
==========================

Essential infrastructure that the Automatos platform requires to exist.
NOT sellable as separate modules - this is the foundation everything runs on.

Components:
- llm/       - LLM provider management (OpenAI, Anthropic, HuggingFace, etc.)
- redis/     - Redis client for pub/sub and caching
- database/  - Database connection, schema, migrations
- models/    - SQLAlchemy ORM models
- seeds/     - Seed data for initialization
- utils/     - Logging, utilities

Usage:
    from core.llm import create_llm_manager
    from core.redis import get_redis_client
    from core.database import get_db
    from core.models import Agent, Workflow
"""

__all__ = [
    'llm',
    'redis', 
    'database',
    'models',
    'seeds',
    'utils',
]

