"""
CodeGraph Analysis
==================

Architecture analysis — module detection, coupling metrics, hotspot identification.
"""

from .architecture_analyzer import ArchitectureAnalyzer, ArchitectureReport

__all__ = ["ArchitectureAnalyzer", "ArchitectureReport"]
