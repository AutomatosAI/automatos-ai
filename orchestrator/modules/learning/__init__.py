"""
Learning Module - Self-Improvement System
=========================================

Pattern recognition, playbook mining, feedback learning.
Cross-cutting: integrates with all modules.

Components:
- engine/     - Learning engine, system updater
- playbooks/  - Workflow pattern mining, templates
- patterns/   - Pattern detection, extraction, storage
- feedback/   - Feedback collection, analysis, adaptation

Usage:
    from modules.learning import LearningSystemUpdater, PlaybookMiner
    
    learning = LearningSystemUpdater(db_session)
    await learning.update_from_execution(workflow_id, execution_id, results, ...)
    
    miner = PlaybookMiner(db_session)
    patterns = miner.mine(min_support=5)

Sellable as: automatos-learning
"""

from .engine import LearningSystemUpdater
from .playbooks import PlaybookMiner

__all__ = [
    "LearningSystemUpdater",
    "PlaybookMiner",
]
