"""
NL2SQL Intelligence Module
==========================

Smart query enhancements inspired by PandasAI's Agent capabilities.

Features:
- Query clarification (ask user for more info when ambiguous)
- Query rephrasing (improve vague queries)
- Explanation generation (explain results)
- Multi-turn conversation state
- Auto-visualization suggestions
- Semantic layer integration

Usage:
    from modules.nl2sql.intelligence import SmartNL2SQLAgent
    
    agent = SmartNL2SQLAgent(llm_provider, schema_metadata)
    
    # Check if query needs clarification
    clarifications = await agent.get_clarifications("show me sales")
    
    # Rephrase vague query
    better_query = await agent.rephrase_query("show me stuff")
    
    # Get explanation after query
    explanation = await agent.explain_result(sql, data)
    
    # Smart query with all features
    result = await agent.smart_query("show me sales trends")
"""

from .clarifier import QueryClarifier
from .rephraser import QueryRephraser
from .explainer import ResultExplainer
from .visualizer import VisualizationSuggester
from .agent import SmartNL2SQLAgent

__all__ = [
    "QueryClarifier",
    "QueryRephraser", 
    "ResultExplainer",
    "VisualizationSuggester",
    "SmartNL2SQLAgent",
]

