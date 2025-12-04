"""
Shared Infrastructure
=====================

Common utilities used by all modules.

Components:
- llm/       - LLM providers, embedding manager, clients
- math/      - Mathematical foundations (entropy, MMR, knapsack, etc.)
- database/  - Database connections, models
- utils/     - Logging, config, validation

Usage:
    from shared.llm import LLMManager, EmbeddingManager
    from shared.math import calculate_entropy, mmr_selection
    from shared.database import get_db_session
"""

# Will export common utilities
__all__ = []

