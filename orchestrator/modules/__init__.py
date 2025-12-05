"""
Automatos Modules
=================

Standalone, sellable product modules.

Available Modules (Complete):
- search/      - Core search engine (vector store, embeddings, retrieval, optimization) ✅
- rag/         - Document RAG (chunking, ingestion) ✅
- memory/      - Multi-type memory system ✅
- agents/      - Agent lifecycle management ✅
- tools/       - Tool registry and execution ✅
- nl2sql/     - Natural language to SQL ✅
- codegraph/   - Code analysis and search ✅
- learning/    - Self-improvement (patterns, playbooks, feedback) ✅
- reasoning/   - Collaborative reasoning engine ✅
- evaluation/  - Evaluation and benchmarking (structure ready)

Usage:
    from modules.search import ContextOptimizer
    from modules.rag import RAGService, SemanticChunker
    from modules.memory import MemoryService, MemoryType
    from modules.agents import AgentService, AgentFactory
    from modules.tools import ToolService, ToolRegistry
    from modules.learning import LearningSystemUpdater, PlaybookMiner
    from modules.reasoning import CollaborativeReasoningEngine
"""

# Lazy imports - use these to access modules
__all__ = [
    "search",
    "rag", 
    "memory",
    "agents",
    "tools",
    "nl2sql",
    "codegraph",
    "learning",
    "reasoning",
    "evaluation",
]
