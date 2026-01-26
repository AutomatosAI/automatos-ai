"""ToolRouterService - smart routing for tool/action selection.

Migrated (code-only) from the rewrite with adjustments to match this codebase:
- Uses `core.composio.client` for external action execution
- Uses `services.agent_tool_service.AgentToolService` for agent tool filtering
- Uses DB cache models in `core.models.composio_cache`

Not yet wired into the main chat execution flow (separate checkpoint).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID as PyUUID

from sqlalchemy.orm import Session

from core.composio.client import get_composio_client
from core.models.composio_cache import ComposioActionCache, IntentClassificationCache, ToolExecutionLog
from services.agent_tool_service import AgentToolService, INTERNAL_TOOLS

logger = logging.getLogger(__name__)


class IntentCategory(str, Enum):
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
    FETCH = "FETCH"
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    SEARCH = "SEARCH"
    ANALYZE = "ANALYZE"
    EXECUTE = "EXECUTE"


@dataclass
class IntentClassification:
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
    app_name: str
    action_name: str
    confidence: float
    is_internal: bool
    reasoning: str
    parameters_hint: Dict[str, Any]


@dataclass
class ExecutionResult:
    success: bool
    data: Any
    error: Optional[str]
    execution_time_ms: int
    tool_used: str
    action_used: str


CATEGORY_APP_MAPPING: Dict[IntentCategory, List[str]] = {
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

INTENT_KEYWORDS: Dict[IntentCategory, List[str]] = {
    IntentCategory.EMAIL: ["email", "mail", "inbox", "send", "reply", "forward", "attachment", "unread", "draft"],
    IntentCategory.CALENDAR: ["calendar", "event", "meeting", "schedule", "appointment", "invite", "availability"],
    IntentCategory.MESSAGING: ["message", "chat", "slack", "teams", "channel", "dm", "mention"],
    IntentCategory.CODE: ["code", "github", "repo", "repository", "commit", "pull request", "pr", "branch", "issue"],
    IntentCategory.FILES: ["file", "document", "folder", "drive", "upload", "download", "share", "dropbox"],
    IntentCategory.PROJECT: ["task", "project", "jira", "ticket", "sprint", "backlog", "kanban", "board"],
    IntentCategory.DATABASE: ["database", "sql", "query", "table", "schema", "select", "insert", "update", "delete"],
    IntentCategory.KNOWLEDGE: ["knowledge", "search docs", "lookup", "rag", "knowledge base"],
    IntentCategory.MEMORY: ["remember", "recall", "earlier", "previously", "history", "last time"],
}

ACTION_TYPE_KEYWORDS: Dict[ActionType, List[str]] = {
    ActionType.FETCH: ["get", "fetch", "read", "show", "list", "what", "see"],
    ActionType.CREATE: ["create", "new", "add", "compose", "write", "make", "send"],
    ActionType.UPDATE: ["update", "edit", "modify", "change", "set"],
    ActionType.DELETE: ["delete", "remove", "clear", "cancel"],
    ActionType.SEARCH: ["search", "find", "look for", "query", "filter"],
    ActionType.ANALYZE: ["analyze", "summarize", "explain", "review", "check"],
}


class ToolRouterService:
    def __init__(self, db_session: Session, llm_client: Optional[Any] = None):
        self.db = db_session
        self.agent_tool_service = AgentToolService(db_session)
        self.llm_client = llm_client
        self.composio = get_composio_client()

    async def route_and_execute(
        self,
        agent_id: int,
        user_query: str,
        entity_id: str,
        user_id: Optional[int] = None,
        workspace_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        start_time = datetime.utcnow()

        try:
            intent = await self.classify_intent(user_query)
            available_tools = await self.agent_tool_service.get_agent_tools(agent_id)

            tool_match = await self._match_tool(intent, available_tools, user_query)
            if not tool_match:
                return self._create_no_tool_response(intent, available_tools)

            result = await self._execute_tool(
                tool_match=tool_match,
                entity_id=entity_id,
                user_query=user_query,
                additional_context=additional_context,
            )

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
            logger.exception("Tool routing failed")
            return {"success": False, "error": str(e), "tool_used": None, "action_used": None}

    async def classify_intent(self, user_query: str, use_cache: bool = True) -> IntentClassification:
        query_norm = (user_query or "").strip()
        query_lower = query_norm.lower()

        if use_cache:
            cached = await self._get_cached_intent(query_norm)
            if cached:
                return cached

        category_scores: Dict[IntentCategory, int] = {}
        matched_keywords: Dict[IntentCategory, List[str]] = {}
        for category, keywords in INTENT_KEYWORDS.items():
            score = 0
            matches: List[str] = []
            for kw in keywords:
                if kw in query_lower:
                    score += 1
                    matches.append(kw)
            if score:
                category_scores[category] = score
                matched_keywords[category] = matches

        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = min(float(category_scores[best_category]) / 3.0, 1.0)
            keywords_found = matched_keywords.get(best_category, [])
        else:
            best_category = IntentCategory.GENERAL
            confidence = 0.3
            keywords_found = []

        action_type = ActionType.FETCH
        action_scores: Dict[ActionType, int] = {}
        for atype, kws in ACTION_TYPE_KEYWORDS.items():
            action_scores[atype] = sum(1 for kw in kws if kw in query_lower)
        if any(action_scores.values()):
            action_type = max(action_scores, key=action_scores.get)

        intent = IntentClassification(
            category=best_category,
            action_type=action_type,
            confidence=confidence,
            reasoning=f"Matched keywords: {keywords_found}",
            keywords=keywords_found,
        )

        if use_cache:
            await self._cache_intent(query_norm, intent)

        return intent

    async def _get_cached_intent(self, query: str) -> Optional[IntentClassification]:
        query_norm = (query or "").strip()
        if not query_norm:
            return None

        cached = (
            self.db.query(IntentClassificationCache)
            .filter(IntentClassificationCache.query_text == query_norm)
            .order_by(IntentClassificationCache.cached_at.desc())
            .first()
        )
        if not cached:
            return None

        cached.hit_count = int(cached.hit_count or 0) + 1
        cached.last_hit_at = datetime.utcnow()
        self.db.commit()

        return IntentClassification(
            category=IntentCategory(cached.classified_category),
            action_type=ActionType(cached.classified_action_type) if cached.classified_action_type else ActionType.FETCH,
            confidence=float(cached.confidence or 0.5),
            reasoning=cached.reasoning or "From cache",
            keywords=[],
        )

    async def _cache_intent(self, query: str, intent: IntentClassification) -> None:
        query_norm = (query or "").strip()
        if not query_norm:
            return

        existing = (
            self.db.query(IntentClassificationCache)
            .filter(IntentClassificationCache.query_text == query_norm)
            .order_by(IntentClassificationCache.cached_at.desc())
            .first()
        )
        if existing:
            existing.classified_category = intent.category.value
            existing.classified_action_type = intent.action_type.value
            existing.confidence = intent.confidence
            existing.reasoning = intent.reasoning
            existing.cached_at = datetime.utcnow()
            self.db.commit()
            return

        row = IntentClassificationCache(
            query_text=query_norm,
            classified_category=intent.category.value,
            classified_action_type=intent.action_type.value,
            confidence=intent.confidence,
            reasoning=intent.reasoning,
        )
        self.db.add(row)
        self.db.commit()

    async def _match_tool(
        self, intent: IntentClassification, available_tools: List[Dict[str, Any]], user_query: str
    ) -> Optional[ToolMatch]:
        available_app_names = {t.get("app_name") for t in available_tools if t.get("app_name")}

        category_apps = CATEGORY_APP_MAPPING.get(intent.category, [])
        matching_apps = [app for app in category_apps if app in available_app_names]

        if not matching_apps:
            if intent.category in (IntentCategory.KNOWLEDGE, IntentCategory.SEARCH) and "RAG" in available_app_names:
                matching_apps = ["RAG"]
            elif intent.category == IntentCategory.MEMORY and "MEMORY" in available_app_names:
                matching_apps = ["MEMORY"]
            elif intent.category == IntentCategory.DATABASE and "NL2SQL" in available_app_names:
                matching_apps = ["NL2SQL"]
            elif intent.category == IntentCategory.CODE and "CODEGRAPH" in available_app_names:
                matching_apps = ["CODEGRAPH"]

        if not matching_apps:
            return None

        best_app = matching_apps[0]
        is_internal = best_app in INTERNAL_TOOLS
        action_name = await self._select_action(best_app, intent, is_internal)

        return ToolMatch(
            app_name=best_app,
            action_name=action_name,
            confidence=intent.confidence,
            is_internal=is_internal,
            reasoning=f"Matched {best_app} for {intent.category.value} intent",
            parameters_hint=await self._extract_parameters(user_query, action_name),
        )

    async def _select_action(self, app_name: str, intent: IntentClassification, is_internal: bool) -> str:
        if is_internal:
            internal_actions = INTERNAL_TOOLS.get(app_name, {}).get("actions", []) or []

            action_map: Dict[ActionType, List[str]] = {
                ActionType.FETCH: ["SEARCH", "RETRIEVE", "QUERY", "GET"],
                ActionType.CREATE: ["STORE", "INDEX", "CREATE"],
                ActionType.UPDATE: ["UPDATE", "MODIFY"],
                ActionType.DELETE: ["DELETE", "REMOVE"],
                ActionType.SEARCH: ["SEARCH", "QUERY", "RETRIEVE"],
                ActionType.ANALYZE: ["ANALYZE", "METRICS", "IMPACT"],
            }

            preferred = action_map.get(intent.action_type, ["SEARCH"])
            for a in internal_actions:
                for kw in preferred:
                    if kw in a:
                        return a
            return internal_actions[0] if internal_actions else f"{app_name}_SEARCH"

        actions = self.db.query(ComposioActionCache).filter(ComposioActionCache.app_name == app_name).all()
        if not actions:
            return f"{app_name}_DEFAULT"

        action_keywords: Dict[ActionType, List[str]] = {
            ActionType.FETCH: ["fetch", "get", "list", "read"],
            ActionType.CREATE: ["create", "send", "add", "new"],
            ActionType.UPDATE: ["update", "edit", "modify"],
            ActionType.DELETE: ["delete", "remove"],
            ActionType.SEARCH: ["search", "find", "query"],
        }
        kws = action_keywords.get(intent.action_type, ["fetch"])

        best = actions[0]
        best_score = -1
        for a in actions:
            name = (a.action_name or "").lower()
            score = sum(1 for kw in kws if kw in name)
            if score > best_score:
                best_score = score
                best = a
        return best.action_name

    async def _extract_parameters(self, user_query: str, action_name: str) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        q = (user_query or "").lower()
        if "today" in q:
            params["date"] = "today"
        elif "yesterday" in q:
            params["date"] = "yesterday"
        elif "this week" in q:
            params["date_range"] = "this_week"
        return params

    async def _execute_tool(
        self,
        tool_match: ToolMatch,
        entity_id: str,
        user_query: str,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        start_time = datetime.utcnow()
        try:
            if tool_match.is_internal:
                raw = await self._execute_internal_tool(tool_match, user_query, additional_context)
            else:
                raw = self.composio.execute_action(
                    action=tool_match.action_name,
                    params=tool_match.parameters_hint or {},
                    entity_id=entity_id,
                )

            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return ExecutionResult(
                success=bool(raw.get("success", False)),
                data=raw.get("data"),
                error=raw.get("error"),
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
        self, tool_match: ToolMatch, user_query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "success": True,
            "data": {
                "message": "Internal tool execution not yet wired",
                "tool": tool_match.app_name,
                "action": tool_match.action_name,
                "query": user_query,
            },
            "error": None,
        }

    async def _log_execution(
        self,
        agent_id: int,
        tool_match: ToolMatch,
        result: ExecutionResult,
        user_query: str,
        user_id: Optional[int],
        workspace_id: Optional[str],
        execution_time_ms: int,
        router_decision: Dict[str, Any],
    ) -> None:
        ws: Optional[PyUUID] = None
        if workspace_id:
            try:
                ws = PyUUID(str(workspace_id))
            except Exception:
                ws = None

        row = ToolExecutionLog(
            agent_id=agent_id,
            app_name=tool_match.app_name,
            action_name=tool_match.action_name,
            user_id=user_id,
            workspace_id=ws,
            input_parameters=tool_match.parameters_hint,
            user_query=user_query,
            output_result=result.data if result.success else None,
            status="success" if result.success else "error",
            error_message=result.error,
            execution_time_ms=execution_time_ms,
            router_decision=router_decision,
            cache_hit=False,
        )
        self.db.add(row)
        self.db.commit()

    def _create_no_tool_response(self, intent: IntentClassification, available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "success": False,
            "error": f"No tool available for {intent.category.value} intent",
            "tool_used": None,
            "action_used": None,
            "intent": intent.to_dict(),
            "available_tools": [t.get("app_name") for t in available_tools if t.get("app_name")],
        }

