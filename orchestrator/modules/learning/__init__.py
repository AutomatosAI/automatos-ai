"""
Learning Module - Self-Improvement System
=========================================

Pattern recognition, playbook mining, feedback learning.
Cross-cutting: integrates with all modules.

Components:
- patterns/    - Pattern detection, extraction, storage
- playbooks/   - Workflow pattern mining, templates, executor
- feedback/    - Feedback collection, analysis, adaptation
- engine/      - Learning engine, reinforcement, continuous improvement

Usage:
    from modules.learning import LearningService
    
    learning = LearningService()
    await learning.record_feedback(task_id, success=True, quality=0.9)
    patterns = await learning.detect_patterns(user_id)
    playbook = await learning.suggest_playbook(task_type)

Sellable as: automatos-learning
"""

# Will export: LearningService, PatternDetector, PlaybookMiner
__all__ = []

