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
- learning/    - Playbook mining (PlaybookMiner) ✅

Usage:
    from modules.search import ContextOptimizer
    from modules.rag import RAGService, SemanticChunker
    from modules.memory.unified_memory_service import get_unified_memory_service
    from modules.agents import AgentService, AgentFactory
    from modules.tools import ToolRegistry, get_tools_for_agent
    from modules.learning import PlaybookMiner
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
]
