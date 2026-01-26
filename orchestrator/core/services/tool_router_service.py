"""ToolRouterService - Smart tool routing for intelligent action selection.

This service handles the intelligent routing of user queries to the appropriate
tools and actions. It combines intent classification with agent-specific tool
filtering to achieve fast, accurate tool selection.

Key Features:
- Intent classification (EMAIL, CALENDAR, CODE, etc.)
- Confidence-based action selection
- Agent-specific tool filtering (only use assigned tools)
- Internal tool integration (RAG, CodeGraph, Memory, NL2SQL)
- Execution logging and analytics
- Result caching for repeated queries

Architecture:
    User Query → Intent Classification → Tool Filtering → Action Selection → Execution

Usage:
    service = ToolRouterService(db_session, composio_service)
    result = await service.route_and_execute(agent_id, "Summarize my emails from today")
"""

import logging
import hashlib
import json
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

# Import services
from core.services.composio_api_service import ComposioAPIService
from core.services.agent_tool_service import AgentToolService, INTERNAL_TOOLS

# Import models
from core.models.composio_cache import (
    ComposioAppCache,
    ComposioActionCache,
    ToolExecutionLog,
    IntentClassificationCache,
)

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
    """Categories for user intent classification."""
    EMAIL = "EMAIL"
    CALENDAR = "CALENDAR"
    MESSAGING = "MESSAGING"
    CODE = "CODE"
    FILES = "FILES"
    PROJECT = "PROJECT"
    DATABASE = "DATABASE"
    KNOWLEDGE = "KNOWLEDGE"
    MEMORY = "MEMORY"
    SEARCH = "SEARCH"
    GENERAL = "GENERAL"


class ActionType(str, Enum):
    """Types of actions."""
    FETCH = "FETCH"      # Read/get data
    CREATE = "CREATE"    # Create new items
    UPDATE = "UPDATE"    # Modify existing items
    DELETE = "DELETE"    # Remove items
    SEARCH = "SEARCH"    # Search/query
    ANALYZE = "ANALYZE"  # Analyze/process
    EXECUTE = "EXECUTE"  # Execute/run


@dataclass
class IntentClassification:
    """Result of intent classification."""
    category: IntentCategory
    action_type: ActionType
    confidence: float
    reasoning: str
    keywords: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "action_type": self.action_type.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "keywords": self.keywords,
        }


@dataclass
class ToolMatch:
    """Result of tool matching."""
    app_name: str
    action_name: str
    confidence: float
    is_internal: bool
    reasoning: str
    parameters_hint: Dict[str, Any]


@dataclass
class ExecutionResult:
    """Result of tool execution."""
    success: bool
    data: Any
    error: Optional[str]
    execution_time_ms: int
    tool_used: str
    action_used: str


# Category to app mapping
CATEGORY_APP_MAPPING = {
    IntentCategory.EMAIL: ["GMAIL", "OUTLOOK", "EMAIL"],
    IntentCategory.CALENDAR: ["GOOGLE_CALENDAR", "OUTLOOK_CALENDAR", "CALENDAR"],
    IntentCategory.MESSAGING: ["SLACK", "TEAMS", "DISCORD"],
    IntentCategory.CODE: ["GITHUB", "GITLAB", "BITBUCKET", "CODEGRAPH"],
    IntentCategory.FILES: ["GOOGLE_DRIVE", "DROPBOX", "ONEDRIVE", "BOX"],
    IntentCategory.PROJECT: ["JIRA", "TRELLO", "ASANA", "LINEAR"],
    IntentCategory.DATABASE: ["NL2SQL"],
    IntentCategory.KNOWLEDGE: ["RAG"],
    IntentCategory.MEMORY: ["MEMORY"],
    IntentCategory.SEARCH: ["RAG", "GOOGLE_SEARCH"],
}

# Keywords for intent classification
INTENT_KEYWORDS = {
    IntentCategory.EMAIL: [
        "email", "mail", "inbox", "send", "compose", "reply", "forward",
        "attachment", "subject", "recipient", "unread", "draft"
    ],
    IntentCategory.CALENDAR: [
        "calendar", "event", "meeting", "schedule", "appointment", "invite",
        "availability", "free", "busy", "remind"
    ],
    IntentCategory.MESSAGING: [
        "message", "chat", "slack", "teams", "channel", "dm", "direct message",
        "notify", "ping", "mention"
    ],
    IntentCategory.CODE: [
        "code", "github", "repo", "repository", "commit", "pull request", "pr",
        "merge", "branch", "issue", "function", "class", "file", "analyze code"
    ],
    IntentCategory.FILES: [
        "file", "document", "folder", "drive", "upload", "download", "share",
        "storage", "dropbox", "google drive"
    ],
    IntentCategory.PROJECT: [
        "task", "project", "jira", "ticket", "sprint", "backlog", "kanban",
        "board", "assign", "status", "priority"
    ],
    IntentCategory.DATABASE: [
        "database", "sql", "query", "table", "data", "records", "schema",
        "select", "insert", "update", "delete"
    ],
    IntentCategory.KNOWLEDGE: [
        "knowledge", "document", "search docs", "find information", "lookup",
        "what do we know", "rag", "knowledge base"
    ],
    IntentCategory.MEMORY: [
        "remember", "recall", "what did", "previously", "earlier", "history",
        "conversation", "last time", "mentioned"
    ],
}

# Action type keywords
ACTION_TYPE_KEYWORDS = {
    ActionType.FETCH: ["get", "fetch", "read", "show", "display", "list", "what", "see"],
    ActionType.CREATE: ["create", "new", "add", "compose", "write", "make", "send"],
    ActionType.UPDATE: ["update", "edit", "modify", "change", "set"],
    ActionType.DELETE: ["delete", "remove", "clear", "cancel"],
    ActionType.SEARCH: ["search", "find", "look for", "query", "filter"],
    ActionType.ANALYZE: ["analyze", "summarize", "explain", "review", "check"],
}


class ToolRouterService:
    """Smart tool routing service.
    
    This service:
    1. Classifies user intent from natural language queries
    2. Filters available tools based on agent assignments
    3. Selects the best matching tool and action
    4. Executes the tool and returns results
    5. Logs everything for analytics
    """
    
    def __init__(
        self,
        db_session: Session,
        composio_service: Optional[ComposioAPIService] = None,
        llm_client: Optional[Any] = None,
    ):
        """Initialize the service.
        
        Args:
            db_session: SQLAlchemy database session
            composio_service: Optional ComposioAPIService instance
            llm_client: Optional LLM client for advanced classification
        """
        self.db = db_session
        self.composio = composio_service or ComposioAPIService()
        self.agent_tool_service = AgentToolService(db_session)
        self.llm_client = llm_client
    
    # =========================================================================
    # Main Entry Point
    # =========================================================================
    
    async def route_and_execute(
        self,
        agent_id: int,
        user_query: str,
        entity_id: str,
        user_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        additional_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Route a user query to the appropriate tool and execute it.
        
        This is the main entry point for tool routing. It:
        1. Classifies the intent
        2. Gets the agent's available tools
        3. Matches the best tool
        4. Executes the tool
        5. Logs the execution
        
        Args:
            agent_id: ID of the agent handling the query
            user_query: The user's natural language query
            entity_id: Composio entity ID (for external tools)
            user_id: Optional user ID for logging
            workspace_id: Optional workspace ID for logging
            additional_context: Optional additional context
            
        Returns:
            Execution result with tool details
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Classify intent
            intent = await self.classify_intent(user_query)
            logger.info(f"Intent classified: {intent.category} ({intent.confidence})")
            
            # Step 2: Get agent's available tools
            available_tools = await self.agent_tool_service.get_agent_tools(agent_id)
            tool_names = [t["app_name"] for t in available_tools]
            logger.info(f"Agent {agent_id} has tools: {tool_names}")
            
            # Step 3: Filter and match tools
            tool_match = await self._match_tool(intent, available_tools, user_query)
            
            if not tool_match:
                # No matching tool found
                return self._create_no_tool_response(intent, available_tools)
            
            logger.info(f"Matched tool: {tool_match.app_name}.{tool_match.action_name}")
            
            # Step 4: Execute the tool
            result = await self._execute_tool(
                tool_match=tool_match,
                entity_id=entity_id,
                user_query=user_query,
                additional_context=additional_context,
            )
            
            # Step 5: Log execution
            execution_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            await self._log_execution(
                agent_id=agent_id,
                tool_match=tool_match,
                result=result,
                user_query=user_query,
                user_id=user_id,
                workspace_id=workspace_id,
                execution_time_ms=execution_time_ms,
                router_decision={
                    "intent": intent.to_dict(),
                    "tool_match_confidence": tool_match.confidence,
                    "tool_match_reasoning": tool_match.reasoning,
                },
            )
            
            return {
                "success": result.success,
                "data": result.data,
                "error": result.error,
                "tool_used": result.tool_used,
                "action_used": result.action_used,
                "intent": intent.to_dict(),
                "execution_time_ms": execution_time_ms,
            }
            
        except Exception as e:
            logger.error(f"Tool routing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_used": None,
                "action_used": None,
            }
    
    # =========================================================================
    # Intent Classification
    # =========================================================================
    
    async def classify_intent(
        self,
        user_query: str,
        use_cache: bool = True,
    ) -> IntentClassification:
        """Classify user intent from query.
        
        Uses keyword matching for fast classification.
        Can optionally use LLM for complex queries.
        
        Args:
            user_query: The user's query
            use_cache: Whether to use cached classifications
            
        Returns:
            IntentClassification result
        """
        query_lower = user_query.lower()
        
        # Check cache first
        if use_cache:
            cached = await self._get_cached_intent(user_query)
            if cached:
                return cached
        
        # Keyword-based classification
        category_scores = {}
        matched_keywords = {}
        
        for category, keywords in INTENT_KEYWORDS.items():
            score = 0
            matches = []
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1
                    matches.append(keyword)
            if score > 0:
                category_scores[category] = score
                matched_keywords[category] = matches
        
        # Determine best category
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            max_score = category_scores[best_category]
            total_keywords = len(INTENT_KEYWORDS[best_category])
            confidence = min(max_score / 3, 1.0)  # Normalize confidence
            keywords_found = matched_keywords.get(best_category, [])
        else:
            best_category = IntentCategory.GENERAL
            confidence = 0.3
            keywords_found = []
        
        # Determine action type
        action_type = ActionType.FETCH  # Default
        action_scores = {}
        
        for action, keywords in ACTION_TYPE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                action_scores[action] = score
        
        if action_scores:
            action_type = max(action_scores, key=action_scores.get)
        
        # Create classification
        intent = IntentClassification(
            category=best_category,
            action_type=action_type,
            confidence=confidence,
            reasoning=f"Matched keywords: {keywords_found}",
            keywords=keywords_found,
        )
        
        # Cache the result
        if use_cache:
            await self._cache_intent(user_query, intent)
        
        return intent
    
    async def _get_cached_intent(self, query: str) -> Optional[IntentClassification]:
        """Get cached intent classification.
        
        Args:
            query: User query
            
        Returns:
            Cached classification or None
        """
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        
        cached = self.db.query(IntentClassificationCache).filter_by(
            query_hash=query_hash
        ).first()
        
        if cached:
            # Update hit count
            cached.hit_count += 1
            cached.last_hit_at = datetime.utcnow()
            self.db.commit()
            
            return IntentClassification(
                category=IntentCategory(cached.classified_category),
                action_type=ActionType(cached.classified_action_type) if cached.classified_action_type else ActionType.FETCH,
                confidence=cached.confidence or 0.5,
                reasoning=cached.reasoning or "From cache",
                keywords=[],
            )
        
        return None
    
    async def _cache_intent(self, query: str, intent: IntentClassification) -> None:
        """Cache an intent classification.
        
        Args:
            query: User query
            intent: Classification result
        """
        query_hash = hashlib.md5(query.lower().encode()).hexdigest()
        
        cache_entry = IntentClassificationCache(
            query_hash=query_hash,
            query_text=query,
            classified_category=intent.category.value,
            classified_action_type=intent.action_type.value,
            confidence=intent.confidence,
            reasoning=intent.reasoning,
        )
        
        self.db.merge(cache_entry)
        self.db.commit()
    
    # =========================================================================
    # Tool Matching
    # =========================================================================
    
    async def _match_tool(
        self,
        intent: IntentClassification,
        available_tools: List[Dict],
        user_query: str,
    ) -> Optional[ToolMatch]:
        """Match the best tool for the given intent.
        
        Args:
            intent: Classified intent
            available_tools: List of tools available to the agent
            user_query: Original user query
            
        Returns:
            Best matching tool or None
        """
        available_app_names = {t["app_name"] for t in available_tools}
        
        # Get apps for this category
        category_apps = CATEGORY_APP_MAPPING.get(intent.category, [])
        
        # Filter to available apps
        matching_apps = [app for app in category_apps if app in available_app_names]
        
        if not matching_apps:
            # Try to find any tool that might work
            for tool in available_tools:
                if tool["app_type"] == "INTERNAL":
                    # Internal tools are always good fallbacks
                    if intent.category in [IntentCategory.KNOWLEDGE, IntentCategory.SEARCH]:
                        matching_apps.append("RAG")
                    elif intent.category == IntentCategory.MEMORY:
                        matching_apps.append("MEMORY")
                    elif intent.category == IntentCategory.DATABASE:
                        matching_apps.append("NL2SQL")
                    elif intent.category == IntentCategory.CODE:
                        matching_apps.append("CODEGRAPH")
        
        if not matching_apps:
            return None
        
        # Use the first matching app (could be enhanced with priority logic)
        best_app = matching_apps[0]
        is_internal = best_app in INTERNAL_TOOLS
        
        # Select best action
        action_name = await self._select_action(best_app, intent, is_internal)
        
        return ToolMatch(
            app_name=best_app,
            action_name=action_name,
            confidence=intent.confidence,
            is_internal=is_internal,
            reasoning=f"Matched {best_app} for {intent.category.value} intent",
            parameters_hint=await self._extract_parameters(user_query, action_name),
        )
    
    async def _select_action(
        self,
        app_name: str,
        intent: IntentClassification,
        is_internal: bool,
    ) -> str:
        """Select the best action for an app based on intent.
        
        Args:
            app_name: Name of the app
            intent: Classified intent
            is_internal: Whether this is an internal tool
            
        Returns:
            Best action name
        """
        if is_internal:
            # Map intent action type to internal tool actions
            internal_actions = INTERNAL_TOOLS.get(app_name, {}).get("actions", [])
            
            action_map = {
                ActionType.FETCH: ["SEARCH", "RETRIEVE", "QUERY", "GET"],
                ActionType.CREATE: ["STORE", "INDEX", "CREATE"],
                ActionType.UPDATE: ["UPDATE", "MODIFY"],
                ActionType.DELETE: ["DELETE", "REMOVE"],
                ActionType.SEARCH: ["SEARCH", "QUERY", "RETRIEVE"],
                ActionType.ANALYZE: ["ANALYZE", "METRICS", "IMPACT"],
            }
            
            preferred_keywords = action_map.get(intent.action_type, ["SEARCH"])
            
            for action in internal_actions:
                for keyword in preferred_keywords:
                    if keyword in action:
                        return action
            
            # Default to first action
            return internal_actions[0] if internal_actions else f"{app_name}_SEARCH"
        
        else:
            # External (Composio) actions
            actions = self.db.query(ComposioActionCache).filter_by(
                app_name=app_name
            ).all()
            
            if not actions:
                return f"{app_name}_DEFAULT"
            
            # Score actions by relevance to intent
            action_keywords = {
                ActionType.FETCH: ["fetch", "get", "list", "read"],
                ActionType.CREATE: ["create", "send", "add", "new"],
                ActionType.UPDATE: ["update", "edit", "modify"],
                ActionType.DELETE: ["delete", "remove"],
                ActionType.SEARCH: ["search", "find", "query"],
            }
            
            keywords = action_keywords.get(intent.action_type, ["fetch"])
            
            best_action = actions[0]
            best_score = 0
            
            for action in actions:
                score = sum(1 for kw in keywords if kw in action.action_name.lower())
                if score > best_score:
                    best_score = score
                    best_action = action
            
            return best_action.action_name
    
    async def _extract_parameters(
        self,
        user_query: str,
        action_name: str,
    ) -> Dict[str, Any]:
        """Extract potential parameters from user query.
        
        Basic extraction - can be enhanced with LLM.
        
        Args:
            user_query: User's query
            action_name: Selected action
            
        Returns:
            Extracted parameters
        """
        params = {}
        query_lower = user_query.lower()
        
        # Time-based parameters
        if "today" in query_lower:
            params["date"] = "today"
        elif "yesterday" in query_lower:
            params["date"] = "yesterday"
        elif "this week" in query_lower:
            params["date_range"] = "this_week"
        
        # Limit parameters
        if "last" in query_lower:
            import re
            match = re.search(r"last (\d+)", query_lower)
            if match:
                params["limit"] = int(match.group(1))
        
        # Search terms
        if "about" in query_lower:
            idx = query_lower.find("about")
            params["search_term"] = user_query[idx + 6:].strip()
        
        return params
    
    # =========================================================================
    # Tool Execution
    # =========================================================================
    
    async def _execute_tool(
        self,
        tool_match: ToolMatch,
        entity_id: str,
        user_query: str,
        additional_context: Optional[Dict] = None,
    ) -> ExecutionResult:
        """Execute the matched tool.
        
        Args:
            tool_match: The matched tool and action
            entity_id: Composio entity ID
            user_query: Original query
            additional_context: Optional context
            
        Returns:
            Execution result
        """
        start_time = datetime.utcnow()
        
        try:
            if tool_match.is_internal:
                # Execute internal tool
                result = await self._execute_internal_tool(
                    tool_match=tool_match,
                    user_query=user_query,
                    context=additional_context,
                )
            else:
                # Execute Composio tool
                result = await self.composio.execute_action(
                    entity_id=entity_id,
                    action_name=tool_match.action_name,
                    parameters=tool_match.parameters_hint,
                )
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return ExecutionResult(
                success=result.get("success", False),
                data=result.get("data"),
                error=result.get("error"),
                execution_time_ms=execution_time,
                tool_used=tool_match.app_name,
                action_used=tool_match.action_name,
            )
            
        except Exception as e:
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return ExecutionResult(
                success=False,
                data=None,
                error=str(e),
                execution_time_ms=execution_time,
                tool_used=tool_match.app_name,
                action_used=tool_match.action_name,
            )
    
    async def _execute_internal_tool(
        self,
        tool_match: ToolMatch,
        user_query: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute an internal tool.
        
        This delegates to the appropriate internal service.
        
        Args:
            tool_match: Tool match info
            user_query: User query
            context: Additional context
            
        Returns:
            Execution result
        """
        app_name = tool_match.app_name
        action_name = tool_match.action_name
        
        # This is where you'd integrate with your actual internal tools
        # For now, return a placeholder that can be replaced with real implementations
        
        if app_name == "RAG":
            # Integration point for RAG service
            return {
                "success": True,
                "data": {
                    "message": f"RAG search executed for: {user_query}",
                    "action": action_name,
                    # Actual RAG results would go here
                },
            }
        
        elif app_name == "CODEGRAPH":
            # Integration point for CodeGraph service
            return {
                "success": True,
                "data": {
                    "message": f"CodeGraph {action_name} executed",
                    # Actual CodeGraph results would go here
                },
            }
        
        elif app_name == "MEMORY":
            # Integration point for Memory service
            return {
                "success": True,
                "data": {
                    "message": f"Memory {action_name} executed",
                    # Actual Memory results would go here
                },
            }
        
        elif app_name == "NL2SQL":
            # Integration point for NL2SQL service
            return {
                "success": True,
                "data": {
                    "message": f"NL2SQL query executed: {user_query}",
                    # Actual NL2SQL results would go here
                },
            }
        
        return {
            "success": False,
            "error": f"Unknown internal tool: {app_name}",
        }
    
    # =========================================================================
    # Logging
    # =========================================================================
    
    async def _log_execution(
        self,
        agent_id: int,
        tool_match: ToolMatch,
        result: ExecutionResult,
        user_query: str,
        user_id: Optional[int],
        workspace_id: Optional[str],
        execution_time_ms: int,
        router_decision: Dict,
    ) -> None:
        """Log tool execution for analytics.
        
        Args:
            agent_id: Agent ID
            tool_match: Matched tool
            result: Execution result
            user_query: Original query
            user_id: User ID
            workspace_id: Workspace ID
            execution_time_ms: Execution time
            router_decision: How the tool was selected
        """
        log = ToolExecutionLog(
            agent_id=agent_id,
            app_name=tool_match.app_name,
            action_name=tool_match.action_name,
            user_id=user_id,
            workspace_id=workspace_id,
            input_parameters=tool_match.parameters_hint,
            user_query=user_query,
            output_result=result.data if result.success else None,
            status="success" if result.success else "error",
            error_message=result.error,
            execution_time_ms=execution_time_ms,
            router_decision=router_decision,
            cache_hit=False,
        )
        
        self.db.add(log)
        self.db.commit()
    
    def _create_no_tool_response(
        self,
        intent: IntentClassification,
        available_tools: List[Dict],
    ) -> Dict[str, Any]:
        """Create response when no tool matches.
        
        Args:
            intent: Classified intent
            available_tools: Available tools
            
        Returns:
            Response indicating no tool found
        """
        return {
            "success": False,
            "error": f"No tool available for {intent.category.value} intent",
            "tool_used": None,
            "action_used": None,
            "intent": intent.to_dict(),
            "available_tools": [t["app_name"] for t in available_tools],
            "suggestion": f"Connect an app that supports {intent.category.value} operations",
        }
