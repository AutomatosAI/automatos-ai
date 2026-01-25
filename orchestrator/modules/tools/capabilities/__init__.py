"""
Tool Capabilities Module
========================

Provides capability-based action filtering and classification for Composio tools.

Components:
- taxonomy.py: Capability taxonomy definitions
- classifier.py: Heuristic-first action classifier
- models.py: SQLAlchemy models for action metadata
"""

from .taxonomy import (
    CAPABILITY_TAXONOMY,
    DESTRUCTIVE_CAPABILITIES,
    REQUIRES_CONFIRMATION,
    INTENT_TO_CAPABILITIES,
)
from .classifier import ActionClassifier, ActionClassification
from .models import ComposioActionMetadata

__all__ = [
    'CAPABILITY_TAXONOMY',
    'DESTRUCTIVE_CAPABILITIES',
    'REQUIRES_CONFIRMATION',
    'INTENT_TO_CAPABILITIES',
    'ActionClassifier',
    'ActionClassification',
    'ComposioActionMetadata',
]
