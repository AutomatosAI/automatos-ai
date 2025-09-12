
"""
Multi-Agent Systems Package
===========================

Advanced multi-agent systems implementation for Automatos AI with:
- Collaborative reasoning
- Coordination strategies  
- Emergent behavior monitoring
- Multi-agent optimization
"""

from .collaborative_reasoning import CollaborativeReasoningEngine
from .coordination_manager import CoordinationManager
from .behavior_monitor import EmergentBehaviorMonitor
from .optimization_engine import MultiAgentOptimizer

__all__ = [
    "CollaborativeReasoningEngine",
    "CoordinationManager", 
    "EmergentBehaviorMonitor",
    "MultiAgentOptimizer"
]
