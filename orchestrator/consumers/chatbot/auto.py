"""
Auto Brain — The Intelligent Core of Automatos
================================================

Auto is the central intelligence that receives every message and decides:
1. How complex is this task? (Atom → Organism scale)
2. Can I handle it directly, or should I delegate to a sub-agent?
3. Which agent has the right tools for this job?

Auto runs on a premium reasoning model (the workspace's orchestrator LLM)
and delegates 80-90% of work to cheaper sub-agents.

Usage:
    brain = AutoBrain(db, workspace_id)
    assessment = await brain.assess(user_message)
    # assessment.complexity -> "atom" | "molecule" | "cell" | "organ" | "organism"
    # assessment.action -> "respond" | "delegate" | "workflow"
    # assessment.target_agent_id -> int | None
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complexity levels (Progressive Complexity Model from Platform Guide)
# ---------------------------------------------------------------------------

class Complexity(str, Enum):
    """Task complexity on the Atom → Organism scale."""
    ATOM = "atom"            # Simple: greetings, factual, chitchat (<200 tokens)
    MOLECULE = "molecule"    # Needs a tool or specific agent skill (~1K tokens)
    CELL = "cell"            # Needs memory + tool + reasoning (~3K tokens)
    ORGAN = "organ"          # Multi-agent coordination (~6K tokens)
    ORGANISM = "organism"    # Enterprise pipeline, learning + feedback (~12K tokens)


class Action(str, Enum):
    """What Auto should do with this request."""
    RESPOND = "respond"      # Auto responds directly (no delegation)
    DELEGATE = "delegate"    # Route to a single sub-agent
    WORKFLOW = "workflow"    # Trigger multi-agent workflow


@dataclass
class ComplexityAssessment:
    """Result of Auto's complexity assessment."""
    complexity: Complexity
    action: Action
    reasoning: str
    target_agent_id: Optional[int] = None
    target_agent_name: Optional[str] = None
    matched_tools: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class AgentCapability:
    """Summary of an agent's capabilities for matching."""
    agent_id: int
    name: str
    description: str
    app_names: List[str]        # e.g. ["GMAIL", "SLACK", "GITHUB"]
    internal_tools: List[str]   # e.g. ["search_knowledge", "query_database"]
    tags: List[str]


# ---------------------------------------------------------------------------
# Heuristic patterns for fast complexity classification
# ---------------------------------------------------------------------------

_ATOM_PATTERNS = [
    r"^(hi|hello|hey|howdy|yo|thanks|thank you|bye|goodbye|ok|okay|yes|no|sure|cool|nice|great)\b",
    r"^(good (morning|afternoon|evening))\b",
    r"^(what|who) (are|is) (you|automatos|auto)\b",
    r"^(how are you|what can you do)\b",
]

_TOOL_KEYWORDS = {
    "GMAIL": ["email", "mail", "inbox", "gmail", "send email", "read email", "compose"],
    "SLACK": ["slack", "channel", "dm", "direct message", "post message"],
    "GITHUB": ["github", "repo", "repository", "pull request", "pr", "issue", "commit"],
    "GOOGLE_CALENDAR": ["calendar", "meeting", "schedule", "event", "appointment"],
    "JIRA": ["jira", "ticket", "sprint", "backlog", "story", "epic"],
    "GOOGLE_DRIVE": ["drive", "google doc", "spreadsheet", "google sheet"],
    "NOTION": ["notion", "page", "database"],
    "TELEGRAM": ["telegram"],
    "WHATSAPP": ["whatsapp"],
    "TAVILY": ["search the web", "web search", "look up online", "tavily"],
    "WEBSEARCH": ["flight", "flights", "hotel", "hotels", "travel", "book a", "booking"],
    "IMAGE_GENERATION": ["create image", "generate image", "draw", "picture of",
                         "image of", "create a picture", "design a", "illustration"],
}

# Internal tool keyword mapping
# NOTE: Platform tools are checked FIRST (dict insertion order) so they win
# over generic matches like "how many" in query_database.
_INTERNAL_TOOL_KEYWORDS = {
    # --- PRD-64: Platform self-awareness (check first) ---
    "platform_list_agents": ["my agent", "list agent", "what agent", "show agent",
                             "agents", "how many agents"],
    "platform_list_recipes": ["my recipe", "list recipe", "what recipe", "show recipe",
                              "my workflow", "list workflow", "recipes", "how many recipes"],
    "platform_get_llm_usage": ["token usage", "llm usage", "api calls", "how much spent",
                               "my cost", "my spend", "tokens", "how many tokens",
                               "cost", "usage", "spending"],
    "platform_list_documents": ["my document", "uploaded document", "uploaded file",
                                "list document", "documents", "how many documents"],
    "platform_get_workspace_info": ["workspace info", "my workspace", "workspace"],
    "platform_list_connected_apps": ["connected app", "integration", "what app",
                                     "integrations", "connected apps", "my apps"],
    # --- General internal tools (also handled by Auto — no agent routing needed) ---
    "search_knowledge": ["search my", "search document", "find in my", "knowledge base",
                         "in my docs", "in my files", "in our docs", "documentation",
                         "uploaded", "rag", "look up in"],
    "query_database": ["database", "query db", "run query", "sql", "analytics",
                       "from the database"],
    "smart_query_database": ["show me data", "show metrics", "analyze data",
                             "statistics", "report on"],
    "generate_document": ["create report", "generate document", "write report",
                          "create a doc", "create doc", "generate report",
                          "make a report", "write a doc", "export report"],
    "search_codebase": ["search code", "search the code", "codebase", "codegraph",
                        "code search", "in the code", "find the function",
                        "implementation of"],
}

_MULTI_STEP_SIGNALS = [
    r"\b(and then|after that|next|also|additionally)\b",
    r"\b(first|second|third)\b.*\b(then|next|after)\b",
    r"\bstep\s*\d\b",
]

_atom_re = [re.compile(p, re.IGNORECASE) for p in _ATOM_PATTERNS]
_multi_step_re = [re.compile(p, re.IGNORECASE) for p in _MULTI_STEP_SIGNALS]


# ---------------------------------------------------------------------------
# AutoBrain
# ---------------------------------------------------------------------------

class AutoBrain:
    """
    The intelligent core of Automatos.

    Assesses task complexity, selects the right agent, and decides
    whether to respond directly or delegate.
    """

    def __init__(self, db: Session, workspace_id: str):
        self._db = db
        self._workspace_id = workspace_id
        self._agents_cache: Optional[List[AgentCapability]] = None

    async def assess(
        self,
        message: str,
        conversation_length: int = 0,
    ) -> ComplexityAssessment:
        """
        Assess a user message and decide what Auto should do.

        Fast path: heuristic classification (~0ms).
        Slow path: LLM classification for ambiguous cases (future).

        Args:
            message: The user's latest message text.
            conversation_length: Number of messages in conversation so far.

        Returns:
            ComplexityAssessment with complexity, action, and target agent.
        """
        if not message or not message.strip():
            return ComplexityAssessment(
                complexity=Complexity.ATOM,
                action=Action.RESPOND,
                reasoning="Empty message",
                confidence=1.0,
            )

        msg_lower = message.lower().strip()

        # --- Tier 1: Atom detection (fast, high confidence) ---
        if self._is_atom(msg_lower):
            return ComplexityAssessment(
                complexity=Complexity.ATOM,
                action=Action.RESPOND,
                reasoning="Simple greeting or conversational message",
                confidence=0.95,
            )

        # --- Tier 2: Tool & agent matching ---
        matched_external = self._match_external_tools(msg_lower)
        matched_internal = self._match_internal_tools(msg_lower)
        all_matched = matched_external + matched_internal
        is_multi_step = self._is_multi_step(msg_lower)

        # Multi-step with multiple tool domains → Organ (workflow candidate)
        if is_multi_step and len(set(matched_external)) >= 2:
            best_agent = await self._find_best_agent(matched_external)
            return ComplexityAssessment(
                complexity=Complexity.ORGAN,
                action=Action.WORKFLOW if len(matched_external) >= 3 else Action.DELEGATE,
                reasoning=f"Multi-step task spanning {', '.join(set(matched_external))}",
                target_agent_id=best_agent.agent_id if best_agent else None,
                target_agent_name=best_agent.name if best_agent else None,
                matched_tools=list(set(matched_external)),
                confidence=0.80,
            )

        # Single tool domain → Molecule (delegate to agent with that tool)
        if matched_external:
            best_agent = await self._find_best_agent(matched_external)
            complexity = Complexity.CELL if is_multi_step else Complexity.MOLECULE
            return ComplexityAssessment(
                complexity=complexity,
                action=Action.DELEGATE,
                reasoning=f"Task requires {', '.join(set(matched_external))} tools",
                target_agent_id=best_agent.agent_id if best_agent else None,
                target_agent_name=best_agent.name if best_agent else None,
                matched_tools=list(set(matched_external)),
                confidence=0.85,
            )

        # Internal tools only — Auto handles directly (no agent routing needed).
        # All internal tools (platform_*, search_knowledge, query_database,
        # generate_document, search_codebase) are platform-level capabilities
        # available to any agent via ToolRegistry. Routing to a specialized
        # agent adds 3-5s of LLM overhead for zero benefit.
        if matched_internal:
            complexity = Complexity.CELL if is_multi_step else Complexity.MOLECULE
            return ComplexityAssessment(
                complexity=complexity,
                action=Action.RESPOND,
                reasoning=f"Internal tools (Auto handles): {', '.join(set(matched_internal))}",
                matched_tools=list(set(matched_internal)),
                confidence=0.90,
            )

        # Long/complex query with no clear tool match → Cell (needs reasoning)
        if len(message) > 200 or is_multi_step:
            return ComplexityAssessment(
                complexity=Complexity.CELL,
                action=Action.DELEGATE,
                reasoning="Complex query requiring reasoning and context",
                confidence=0.70,
            )

        # Memory-related → Cell (needs context)
        if re.search(r"\b(remember|my name|last time|previously|earlier|what did)\b", msg_lower):
            return ComplexityAssessment(
                complexity=Complexity.CELL,
                action=Action.RESPOND,
                reasoning="Memory recall — Auto handles with Mem0 context",
                confidence=0.85,
            )

        # Default → Delegate to router (it has LLM + semantic to pick the right agent).
        # Previously this was RESPOND, which meant anything not matching hardcoded
        # keywords was answered conversationally — breaking routing for agents like
        # Scout (flights/travel), Graphics Generator (images), etc.
        return ComplexityAssessment(
            complexity=Complexity.MOLECULE,
            action=Action.DELEGATE,
            reasoning="General conversation — Auto responds directly",
            confidence=0.70,
        )

    # ------------------------------------------------------------------
    # Agent selection — find the best agent for the matched tools
    # ------------------------------------------------------------------

    async def _find_best_agent(
        self, required_tools: List[str]
    ) -> Optional[AgentCapability]:
        """
        Find the workspace agent with the best tool coverage for the task.

        Scoring:
        1. Primary: how many required tools does this agent have?
        2. Tiebreaker: among equal scores, prefer the agent with MORE total
           external apps (a "Communication" agent with GMAIL+SLACK+GCAL
           should beat a "Research" agent that happens to also have GMAIL).
        """
        agents = self._get_workspace_agents()
        if not agents:
            return None

        required_set = {t.upper() for t in required_tools}
        best: Optional[AgentCapability] = None
        best_score = 0
        best_total_apps = 0

        for agent in agents:
            agent_apps = {a.upper() for a in agent.app_names}
            overlap = required_set & agent_apps
            score = len(overlap)
            total_apps = len(agent_apps)

            if score > best_score or (score == best_score and score > 0 and total_apps > best_total_apps):
                best_score = score
                best_total_apps = total_apps
                best = agent

        if best and best_score > 0:
            logger.info(
                "[AutoBrain] Best agent for %s: %s (id=%d, score=%d/%d, total_apps=%d)",
                required_tools, best.name, best.agent_id,
                best_score, len(required_set), best_total_apps,
            )
            return best

        return None

    def select_best_agent_id(self, message: str) -> Optional[int]:
        """
        Synchronous convenience method for quick agent selection.

        Used by the chat API when UniversalRouter returns no match.
        Finds the agent whose tools best match the user's request.

        Tiebreaker: agent with more total external apps wins (more specialized).
        """
        msg_lower = message.lower().strip()
        matched = self._match_external_tools(msg_lower)
        if not matched:
            return None

        agents = self._get_workspace_agents()
        if not agents:
            return None

        required_set = {t.upper() for t in matched}
        best_id: Optional[int] = None
        best_score = 0
        best_total_apps = 0

        for agent in agents:
            agent_apps = {a.upper() for a in agent.app_names}
            score = len(required_set & agent_apps)
            total_apps = len(agent_apps)

            if score > best_score or (score == best_score and score > 0 and total_apps > best_total_apps):
                best_score = score
                best_total_apps = total_apps
                best_id = agent.agent_id

        return best_id if best_score > 0 else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_atom(self, msg_lower: str) -> bool:
        """Fast check: is this a simple Atom-level message?"""
        # Very short messages are almost always atoms
        if len(msg_lower) < 15:
            for pattern in _atom_re:
                if pattern.match(msg_lower):
                    return True
            # Single word/emoji
            if len(msg_lower.split()) <= 2:
                return True
        return False

    def _match_external_tools(self, msg_lower: str) -> List[str]:
        """Match user message against external tool keywords."""
        matched = []
        for tool_name, keywords in _TOOL_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    matched.append(tool_name)
                    break
        return matched

    def _match_internal_tools(self, msg_lower: str) -> List[str]:
        """Match user message against internal tool keywords."""
        matched = []
        for tool_name, keywords in _INTERNAL_TOOL_KEYWORDS.items():
            for kw in keywords:
                if kw in msg_lower:
                    matched.append(tool_name)
                    break
        return matched

    def _is_multi_step(self, msg_lower: str) -> bool:
        """Detect multi-step requests."""
        for pattern in _multi_step_re:
            if pattern.search(msg_lower):
                return True
        # Multiple action verbs
        verbs = ["create", "find", "search", "send", "write", "read",
                 "check", "list", "get", "post", "update", "delete",
                 "generate", "summarize", "analyze", "compare"]
        count = sum(1 for v in verbs if v in msg_lower)
        return count >= 3

    def _get_workspace_agents(self) -> List[AgentCapability]:
        """Load and cache workspace agents with their tool assignments."""
        if self._agents_cache is not None:
            return self._agents_cache

        try:
            from core.models.core import Agent
            from core.models.composio_cache import AgentAppAssignment

            # Get active agents in this workspace
            agents = (
                self._db.query(Agent)
                .filter(
                    Agent.workspace_id == self._workspace_id,
                    Agent.status == "active",
                )
                .all()
            )

            result = []
            for agent in agents:
                # Get app assignments
                assignments = (
                    self._db.query(AgentAppAssignment.app_name, AgentAppAssignment.app_type)
                    .filter(
                        AgentAppAssignment.agent_id == agent.id,
                        AgentAppAssignment.is_active == True,  # noqa: E712
                    )
                    .all()
                )

                external_apps = [
                    (a.app_name or "").upper()
                    for a in assignments
                    if a.app_type == "EXTERNAL" and a.app_name
                ]
                internal_tools = [
                    (a.app_name or "").lower()
                    for a in assignments
                    if a.app_type == "INTERNAL" and a.app_name
                ]

                result.append(AgentCapability(
                    agent_id=agent.id,
                    name=agent.name,
                    description=agent.description or "",
                    app_names=external_apps,
                    internal_tools=internal_tools,
                    tags=agent.tags or [],
                ))

            self._agents_cache = result
            logger.info(
                "[AutoBrain] Loaded %d agents for workspace %s",
                len(result), str(self._workspace_id)[:8],
            )
            return result

        except Exception as exc:
            logger.warning("[AutoBrain] Failed to load agents: %s", exc)
            self._agents_cache = []
            return []
