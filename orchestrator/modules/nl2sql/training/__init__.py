"""
NL2SQL Training Module
======================

Manages training examples (question/SQL pairs) for few-shot retrieval.
Inspired by Vanna's RAG-based approach and Dataherald's Golden SQL concept.
"""

from .example_store import SQLExampleStore

__all__ = ["SQLExampleStore"]
