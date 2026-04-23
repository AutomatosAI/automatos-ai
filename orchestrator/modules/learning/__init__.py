"""
Learning Module - Self-Improvement System
=========================================

Pattern recognition, playbook mining, feedback learning.
Cross-cutting: integrates with all modules.

Components:
- playbooks/  - Workflow pattern mining, templates
- patterns/   - Pattern detection, extraction, storage
- feedback/   - Feedback collection, analysis, adaptation

Usage:
    from modules.learning import PlaybookMiner

    miner = PlaybookMiner(db_session)
    patterns = miner.mine(min_support=5)

Sellable as: automatos-learning
"""

from .playbooks import PlaybookMiner

__all__ = [
    "PlaybookMiner",
]
