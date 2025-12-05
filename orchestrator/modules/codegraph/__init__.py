"""
CodeGraph Module
================

Code indexing, analysis, and intelligent search.

Usage:
    from modules.codegraph import CodeGraphService, CodeSymbol
    
    service = CodeGraphService(db_session)
    await service.index_github_project(name, url)
    results = await service.search_code("function that handles auth")

Sellable as: automatos-codegraph
"""

from .codegraph_service import (
    CodeGraphService,
    CodeSymbol,
    CodeRelationship,
    ParseResult
)
from .project_context import (
    ProjectContextAnalyzer,
    ProjectContext,
    get_project_analyzer
)

__all__ = [
    "CodeGraphService",
    "CodeSymbol",
    "CodeRelationship",
    "ParseResult",
    "ProjectContextAnalyzer",
    "ProjectContext",
    "get_project_analyzer"
]
