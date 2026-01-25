"""
Smart NL2SQL Agent
==================

A PandasAI-inspired intelligent agent for natural language database queries.
Combines clarification, rephrasing, explanation, and visualization features.

Usage:
    from modules.nl2sql.intelligence import SmartNL2SQLAgent
    
    agent = SmartNL2SQLAgent(llm_provider, schema_metadata)
    
    # Smart query with all features
    result = await agent.query("show me sales trends")
    
    # Or use individual features
    if agent.needs_clarification("show me sales"):
        questions = agent.get_clarifications("show me sales")
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .clarifier import QueryClarifier
from .rephraser import QueryRephraser
from .explainer import ResultExplainer
from .visualizer import VisualizationSuggester, ChartType

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Represents a single turn in the conversation."""
    query: str
    rephrased_query: Optional[str] = None
    sql: Optional[str] = None
    data: Optional[List[Dict]] = field(default_factory=list)
    explanation: Optional[str] = None
    clarifications_asked: Optional[List[str]] = None
    clarifications_received: Optional[Dict[str, str]] = None
    visualization: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


@dataclass
class AgentState:
    """Maintains conversation state across turns."""
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    pending_clarifications: Optional[List[str]] = None
    last_query: Optional[str] = None
    
    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
        self.last_query = turn.query
    
    def get_history(self, max_turns: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return [
            {
                "query": t.query,
                "sql": t.sql,
                "row_count": len(t.data) if t.data else 0,
                "explanation": t.explanation,
            }
            for t in self.turns[-max_turns:]
        ]


class SmartNL2SQLAgent:
    """
    Intelligent NL2SQL agent with multi-turn conversation support.
    
    Features:
    - Query clarification (asks for more info when ambiguous)
    - Query rephrasing (improves vague queries)
    - Result explanation (explains what the data means)
    - Visualization suggestions (recommends chart types)
    - Conversation memory (maintains state across turns)
    """
    
    def __init__(
        self,
        llm_provider,
        schema_metadata: Dict[str, Any],
        semantic_layer: Optional[Dict[str, Any]] = None,
        auto_clarify: bool = True,
        auto_rephrase: bool = True,
        auto_explain: bool = True,
        auto_visualize: bool = True,
    ):
        """
        Initialize the smart agent.
        
        Args:
            llm_provider: LLM provider for AI features
            schema_metadata: Database schema information
            semantic_layer: Optional business metrics definitions
            auto_clarify: Auto-ask clarification questions
            auto_rephrase: Auto-rephrase vague queries
            auto_explain: Auto-generate explanations
            auto_visualize: Auto-suggest visualizations
        """
        self.llm_provider = llm_provider
        self.schema_metadata = schema_metadata
        self.semantic_layer = semantic_layer
        
        # Feature flags
        self.auto_clarify = auto_clarify
        self.auto_rephrase = auto_rephrase
        self.auto_explain = auto_explain
        self.auto_visualize = auto_visualize
        
        # Initialize components
        self.clarifier = QueryClarifier(llm_provider)
        self.rephraser = QueryRephraser(llm_provider)
        self.explainer = ResultExplainer(llm_provider)
        self.visualizer = VisualizationSuggester()
        
        # Initialize state
        self.state = AgentState()
        
        # Import NL2SQL service (from parent module)
        from ..query.nl2sql_service import NaturalLanguageToSQLService
        self.nl2sql_service = NaturalLanguageToSQLService(llm_provider)
    
    def reset(self):
        """Reset conversation state."""
        self.state = AgentState()
    
    # =========================================================================
    # CORE QUERY METHOD
    # =========================================================================
    
    async def query(
        self,
        natural_language_query: str,
        clarification_answers: Optional[Dict[str, str]] = None,
        skip_clarification: bool = False,
        skip_rephrase: bool = False,
        execute_sql: bool = False,
        db_executor: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Process a natural language query with all smart features.
        
        Args:
            natural_language_query: The user's query
            clarification_answers: Answers to previous clarification questions
            skip_clarification: Force skip clarification step
            skip_rephrase: Force skip rephrasing step
            execute_sql: Whether to execute the SQL (requires db_executor)
            db_executor: Async function to execute SQL and return results
            
        Returns:
            Dict with: sql, explanation, clarifications_needed, visualization, data, etc.
        """
        turn = ConversationTurn(query=natural_language_query)
        
        try:
            # Step 1: Apply any pending clarification answers
            working_query = natural_language_query
            if clarification_answers and self.state.pending_clarifications:
                working_query = self.clarifier.apply_clarification(
                    natural_language_query,
                    clarification_answers
                )
                turn.clarifications_received = clarification_answers
                self.state.pending_clarifications = None
            
            # Step 2: Check if clarification needed
            if self.auto_clarify and not skip_clarification:
                if self.clarifier.needs_clarification(working_query, self.schema_metadata):
                    clarifications = self.clarifier.get_clarifications(
                        working_query,
                        self.schema_metadata
                    )
                    if clarifications:
                        turn.clarifications_asked = clarifications
                        self.state.pending_clarifications = clarifications
                        self.state.add_turn(turn)
                        
                        return {
                            "status": "needs_clarification",
                            "query": working_query,
                            "clarifications": clarifications,
                            "message": "Could you provide more details?",
                        }
            
            # Step 3: Rephrase query if needed
            if self.auto_rephrase and not skip_rephrase:
                rephrased, rephrase_reason = self.rephraser.rephrase(
                    working_query,
                    self.schema_metadata
                )
                if rephrased != working_query:
                    turn.rephrased_query = rephrased
                    working_query = rephrased
            
            # Step 4: Generate SQL
            sql, sql_explanation, sql_metadata = self.nl2sql_service.generate_sql(
                question=working_query,
                schema_metadata=self.schema_metadata,
                semantic_layer=self.semantic_layer,
            )
            turn.sql = sql
            
            # Step 5: Execute SQL if requested
            data = []
            if execute_sql and db_executor:
                try:
                    data = await db_executor(sql)
                    turn.data = data
                except Exception as e:
                    turn.error = str(e)
                    logger.error(f"SQL execution error: {e}")
            
            # Step 6: Generate explanation
            if self.auto_explain and data:
                explanation = self.explainer.explain_result(
                    query=natural_language_query,
                    sql=sql,
                    data=data
                )
                turn.explanation = explanation
            else:
                turn.explanation = sql_explanation
            
            # Step 7: Suggest visualization
            visualization = None
            if self.auto_visualize and data:
                chart_type, chart_config = self.visualizer.suggest(data, natural_language_query)
                visualization = {
                    "type": chart_type.value,
                    "config": chart_config,
                    "prompt": self.visualizer.get_chart_prompt(chart_type, chart_config, data),
                }
                turn.visualization = visualization
            
            # Step 8: Generate follow-up questions
            follow_ups = []
            if data:
                follow_ups = self.explainer.generate_follow_up_questions(
                    natural_language_query,
                    data
                )
            
            # Save turn
            self.state.add_turn(turn)
            
            return {
                "status": "success",
                "original_query": natural_language_query,
                "rephrased_query": turn.rephrased_query,
                "sql": sql,
                "explanation": turn.explanation,
                "data": data,
                "row_count": len(data),
                "visualization": visualization,
                "follow_up_questions": follow_ups,
                "metadata": sql_metadata,
            }
            
        except Exception as e:
            turn.error = str(e)
            self.state.add_turn(turn)
            logger.error(f"Agent query error: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "query": natural_language_query,
            }
    
    # =========================================================================
    # INDIVIDUAL FEATURE METHODS (for direct access)
    # =========================================================================
    
    def needs_clarification(self, query: str) -> bool:
        """Check if query needs clarification."""
        return self.clarifier.needs_clarification(query, self.schema_metadata)
    
    def get_clarifications(self, query: str, max_questions: int = 3) -> List[str]:
        """Get clarification questions for a query."""
        return self.clarifier.get_clarifications(query, self.schema_metadata, max_questions)
    
    def rephrase_query(self, query: str) -> Tuple[str, str]:
        """Rephrase a query to be more precise."""
        return self.rephraser.rephrase(query, self.schema_metadata)
    
    def explain_sql(self, sql: str) -> str:
        """Explain what a SQL query does."""
        return self.explainer.explain_sql(sql, self.schema_metadata)
    
    def explain_result(self, query: str, sql: str, data: List[Dict]) -> str:
        """Explain query results."""
        return self.explainer.explain_result(query, sql, data)
    
    def suggest_visualization(self, data: List[Dict], query: str = "") -> Tuple[ChartType, Dict]:
        """Suggest visualization for data."""
        return self.visualizer.suggest(data, query)
    
    def get_alternatives(self, query: str, num: int = 3) -> List[Tuple[str, str]]:
        """Get alternative query phrasings."""
        return self.rephraser.suggest_alternatives(query, self.schema_metadata, num)
    
    # =========================================================================
    # CONVERSATION METHODS
    # =========================================================================
    
    def get_conversation_history(self, max_turns: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.state.get_history(max_turns)
    
    def get_last_result(self) -> Optional[ConversationTurn]:
        """Get the last conversation turn."""
        return self.state.turns[-1] if self.state.turns else None
    
    def has_pending_clarifications(self) -> bool:
        """Check if there are pending clarification questions."""
        return bool(self.state.pending_clarifications)
    
    def get_pending_clarifications(self) -> Optional[List[str]]:
        """Get pending clarification questions."""
        return self.state.pending_clarifications
    
    # =========================================================================
    # CONTEXT METHODS
    # =========================================================================
    
    def add_context(self, key: str, value: Any):
        """Add context information (e.g., user preferences)."""
        self.state.context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get context information."""
        return self.state.context.get(key, default)
    
    def set_schema(self, schema_metadata: Dict[str, Any]):
        """Update schema metadata."""
        self.schema_metadata = schema_metadata
    
    def set_semantic_layer(self, semantic_layer: Dict[str, Any]):
        """Update semantic layer."""
        self.semantic_layer = semantic_layer


# =========================================================================
# FACTORY FUNCTION
# =========================================================================

def create_smart_agent(
    llm_provider,
    schema_metadata: Dict[str, Any],
    **kwargs
) -> SmartNL2SQLAgent:
    """
    Factory function to create a SmartNL2SQLAgent.
    
    Args:
        llm_provider: LLM provider
        schema_metadata: Database schema
        **kwargs: Additional agent options
        
    Returns:
        Configured SmartNL2SQLAgent instance
    """
    return SmartNL2SQLAgent(
        llm_provider=llm_provider,
        schema_metadata=schema_metadata,
        **kwargs
    )

