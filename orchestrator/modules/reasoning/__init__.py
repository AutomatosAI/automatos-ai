"""
Reasoning Module - Multi-Agent Collaborative Reasoning
=======================================================

Consensus-based decision making, conflict resolution, and reasoning quality assessment.

Components:
- collaborative_reasoning.py - CollaborativeReasoningEngine

Usage:
    from modules.reasoning import CollaborativeReasoningEngine
    
    engine = CollaborativeReasoningEngine()
    result = await engine.collaborative_reasoning(db, task_id, user_id, agents)

Sellable as: automatos-reasoning
"""

from .collaborative_reasoning import CollaborativeReasoningEngine

__all__ = [
    "CollaborativeReasoningEngine",
]

