"""
Tool Router (Phase 3)
=====================

Intelligently routes user queries to the best tool.

Flow:
1. Classify intent (category + action type)
2. Get agent's assigned tools
3. Filter by intent category
4. Select best tool and action
5. Execute or return for LLM

Goal: Reduce chat response time from 60s to < 10s
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from core.services.intent_classifier import IntentClassifier, IntentClassification
from core.services.agent_tool_assignment import AgentToolAssignmentService
from core.metadata_cache.cache_manager import CacheManager
from core.metadata_cache.models import IntentClassificationCache, ToolExecutionLog
from core.redis.client import RedisClient

logger = logging.getLogger(__name__)


class ToolRouter:
    """
    Intelligent tool routing service.
    
    Reduces LLM context by pre-selecting relevant tools based on intent.
    """
    
    def __init__(self, db: Session, redis_client: Optional[RedisClient] = None):
        self.db = db
        self.redis = redis_client
        self.intent_classifier = IntentClassifier()
        self.assignment_service = AgentToolAssignmentService(db)
        self.cache_manager = CacheManager(db, redis_client) if redis_client else None
    
    def route(
        self,
        query: str,
        agent_id: int,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Route a query to the best tool.
        
        Args:
            query: User's natural language query
            agent_id: Agent ID
            conversation_history: Recent messages for context
            
        Returns:
            {
                "success": bool,
                "intent": {...},
                "selected_tools": [...],
                "reasoning": str,
                "execution_strategy": str  # "direct", "llm_select", "multi_tool"
            }
        """
        start_time = datetime.now()
        logger.info(f"[tool-router] Routing query for agent={agent_id}: {query[:100]}...")
        
        try:
            # Step 1: Classify intent (check cache first)
            intent = self._classify_with_cache(query)
            
            # Step 2: Get agent's assigned tools
            assignments = self.assignment_service.get_agent_assignments(agent_id, active_only=True)
            if not assignments:
                logger.warning(f"[tool-router] Agent {agent_id} has no assigned tools")
                return {
                    "success": False,
                    "error": "No tools assigned to this agent",
                    "intent": intent.__dict__,
                    "selected_tools": [],
                    "reasoning": "Agent has no tools available"
                }
            
            # Step 3: Filter tools by intent category
            relevant_tools = self._filter_tools_by_intent(assignments, intent)
            
            if not relevant_tools:
                logger.info(f"[tool-router] No tools match intent category '{intent.category}'")
                # Fallback: try secondary categories or use all tools
                relevant_tools = assignments[:5]  # Top 5 by priority
            
            # Step 4: Select best tool(s)
            selected_tools = self._select_best_tools(relevant_tools, intent, query)
            
            # Step 5: Determine execution strategy
            strategy = self._determine_strategy(selected_tools, intent)
            
            elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            
            result = {
                "success": True,
                "intent": {
                    "category": intent.category,
                    "action_type": intent.action_type,
                    "confidence": intent.confidence,
                    "reasoning": intent.reasoning
                },
                "selected_tools": [
                    {
                        "app_name": tool["app_name"],
                        "display_name": tool["display_name"],
                        "app_type": tool["app_type"],
                        "priority": tool["priority"],
                        "relevance_score": tool.get("relevance_score", 0.5)
                    }
                    for tool in selected_tools
                ],
                "reasoning": self._build_reasoning(intent, selected_tools, strategy),
                "execution_strategy": strategy,
                "routing_time_ms": elapsed_ms
            }
            
            logger.info(f"[tool-router] Routed to {len(selected_tools)} tool(s) in {elapsed_ms}ms (strategy={strategy})")
            return result
            
        except Exception as e:
            logger.error(f"[tool-router] Routing failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "intent": None,
                "selected_tools": [],
                "reasoning": "Routing error occurred"
            }
    
    def _classify_with_cache(self, query: str) -> IntentClassification:
        """Classify intent with cache lookup."""
        # Check cache (exact match)
        cached = self.db.query(IntentClassificationCache).filter(
            IntentClassificationCache.query_text == query
        ).first()
        
        if cached:
            # Update hit stats
            cached.hit_count += 1
            cached.last_hit_at = datetime.utcnow()
            self.db.commit()
            
            logger.debug(f"[tool-router] Intent cache HIT (hits={cached.hit_count})")
            return IntentClassification(
                category=cached.classified_category,
                action_type=cached.classified_action_type or "UNKNOWN",
                confidence=cached.confidence or 0.8,
                reasoning=cached.reasoning or "From cache",
                method="cached"
            )
        
        # Classify and cache
        intent = self.intent_classifier.classify(query)
        
        # Store in cache
        try:
            cache_entry = IntentClassificationCache(
                query_text=query,
                classified_category=intent.category,
                classified_action_type=intent.action_type,
                confidence=intent.confidence,
                reasoning=intent.reasoning
            )
            self.db.add(cache_entry)
            self.db.commit()
            logger.debug(f"[tool-router] Intent cached for future use")
        except Exception as e:
            logger.warning(f"[tool-router] Failed to cache intent: {e}")
        
        return intent
    
    def _filter_tools_by_intent(
        self,
        assignments: List[Dict[str, Any]],
        intent: IntentClassification
    ) -> List[Dict[str, Any]]:
        """Filter tools by intent category."""
        relevant = []
        intent_category_lower = intent.category.lower()
        
        for tool in assignments:
            categories = tool.get("categories", [])
            categories_lower = [c.lower() for c in categories]
            
            # Check if intent category matches tool categories
            if intent_category_lower in categories_lower:
                relevant.append(tool)
            # Check for related categories
            elif self._are_categories_related(intent_category_lower, categories_lower):
                relevant.append(tool)
        
        return relevant
    
    def _are_categories_related(self, category: str, tool_categories: List[str]) -> bool:
        """Check if categories are semantically related."""
        related_groups = {
            "email": ["communication", "productivity"],
            "calendar": ["productivity", "scheduling"],
            "code": ["developer", "internal"],
            "knowledge": ["internal", "ai", "search"],
            "database": ["internal", "query", "analytics"]
        }
        
        related = related_groups.get(category, [])
        return any(r in tool_categories for r in related)
    
    def _select_best_tools(
        self,
        relevant_tools: List[Dict[str, Any]],
        intent: IntentClassification,
        query: str
    ) -> List[Dict[str, Any]]:
        """Select best tool(s) for this query."""
        if not relevant_tools:
            return []
        
        # Score each tool
        scored_tools = []
        for tool in relevant_tools:
            score = self._score_tool(tool, intent, query)
            tool_with_score = tool.copy()
            tool_with_score["relevance_score"] = score
            scored_tools.append(tool_with_score)
        
        # Sort by score (descending) then priority (descending)
        scored_tools.sort(
            key=lambda t: (t["relevance_score"], t["priority"]),
            reverse=True
        )
        
        # Return top 3 tools (or all if fewer)
        return scored_tools[:3]
    
    def _score_tool(
        self,
        tool: Dict[str, Any],
        intent: IntentClassification,
        query: str
    ) -> float:
        """Calculate relevance score for a tool."""
        score = 0.5  # Base score
        
        # Category match
        intent_cat = intent.category.lower()
        tool_cats = [c.lower() for c in tool.get("categories", [])]
        if intent_cat in tool_cats:
            score += 0.3
        
        # Name/description keyword match
        query_lower = query.lower()
        tool_name = tool.get("display_name", "").lower()
        tool_desc = tool.get("description", "").lower()
        
        if tool_name in query_lower or any(word in query_lower for word in tool_name.split()):
            score += 0.2
        
        if tool_desc and any(word in query_lower for word in tool_desc.split()[:10]):
            score += 0.1
        
        # Priority bonus (0-10 -> 0-0.1)
        priority_bonus = min(0.1, tool.get("priority", 0) * 0.01)
        score += priority_bonus
        
        return min(1.0, score)
    
    def _determine_strategy(
        self,
        selected_tools: List[Dict[str, Any]],
        intent: IntentClassification
    ) -> str:
        """
        Determine execution strategy.
        
        Strategies:
        - "direct": Single clear tool, execute immediately
        - "llm_select": Multiple options, let LLM choose
        - "multi_tool": Need to use multiple tools in sequence
        """
        if not selected_tools:
            return "fallback"
        
        if len(selected_tools) == 1 and selected_tools[0]["relevance_score"] > 0.8:
            return "direct"
        
        if len(selected_tools) > 1:
            # Check if tools are complementary (e.g., RAG + CodeGraph)
            categories = set()
            for tool in selected_tools:
                categories.update(tool.get("categories", []))
            
            if len(categories) > 2:
                return "multi_tool"
        
        return "llm_select"
    
    def _build_reasoning(
        self,
        intent: IntentClassification,
        selected_tools: List[Dict[str, Any]],
        strategy: str
    ) -> str:
        """Build human-readable reasoning for the routing decision."""
        parts = [
            f"Intent: {intent.category}/{intent.action_type} (confidence: {intent.confidence:.0%})"
        ]
        
        if selected_tools:
            tool_names = [t["display_name"] for t in selected_tools]
            parts.append(f"Selected tools: {', '.join(tool_names)}")
        else:
            parts.append("No matching tools found")
        
        strategy_desc = {
            "direct": "High confidence - executing directly",
            "llm_select": "Multiple options - letting LLM decide",
            "multi_tool": "Complex query - may need multiple tools",
            "fallback": "No suitable tools found"
        }
        parts.append(f"Strategy: {strategy_desc.get(strategy, strategy)}")
        
        return " | ".join(parts)
    
    def log_execution(
        self,
        agent_id: int,
        app_name: str,
        action_name: str,
        user_id: int,
        workspace_id: str,
        user_query: str,
        status: str,
        execution_time_ms: int,
        router_decision: Dict[str, Any],
        output_result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Log tool execution for analytics."""
        try:
            log_entry = ToolExecutionLog(
                agent_id=agent_id,
                app_name=app_name,
                action_name=action_name,
                user_id=user_id,
                workspace_id=workspace_id,
                user_query=user_query,
                status=status,
                execution_time_ms=execution_time_ms,
                router_decision=router_decision,
                output_result=output_result,
                error_message=error_message
            )
            self.db.add(log_entry)
            self.db.commit()
            logger.debug(f"[tool-router] Execution logged (status={status})")
        except Exception as e:
            logger.warning(f"[tool-router] Failed to log execution: {e}")
