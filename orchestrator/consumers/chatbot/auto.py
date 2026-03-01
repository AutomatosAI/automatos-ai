"""
Auto Brain — The Progressive Complexity Assessor
====================================================

PRD-68: Progressive Complexity Routing (Atom → Organism).

Auto receives every message and determines its complexity level:
  - ATOM: Direct response (greetings, chitchat) — no tools, no memory
  - MOLECULE: Single tool calls without deep memory
  - CELL: Needs memory + tools + reasoning
  - ORGAN: Multi-agent coordination
  - ORGANISM: Full PRD-59 Neural Swarm pipelines

3-Tier Assessment:
  Tier 1: Redis cache lookup (<5ms, free)
  Tier 2: Regex fast-paths (<5ms, free)
  Tier 3: LLM classification (~200ms, ~$0.001)

The ComplexityAssessment flows through the existing wiring:
  api/chat.py → service.py → integration.py → smart_orchestrator.py
where needs_memory and tool_hints drive downstream behavior.
"""

import logging
import re
import os
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Complexity levels (Progressive Complexity Model PRD-68)
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
    # PRD-68: Fields consumed by smart_orchestrator.py
    needs_memory: bool = False
    tool_hints: List[str] = field(default_factory=list)
    needs_multi_agent: bool = False

    def to_dict(self):
        return {
            "complexity": self.complexity.value,
            "action": self.action.value,
            "reasoning": self.reasoning,
            "tool_hints": self.tool_hints,
            "needs_memory": self.needs_memory,
            "needs_multi_agent": self.needs_multi_agent,
            "confidence": self.confidence,
        }


# ---------------------------------------------------------------------------
# Tier 2: Fast Heuristic Patterns
# ---------------------------------------------------------------------------

# Must be the ENTIRE message (with optional punctuation).
# "hello" → atom.  "hello can you create an image" → NOT atom.
_ATOM_PATTERNS = [
    r"^(hi|hello|hey|howdy|yo|sup)[\s!?.,:]*$",
    r"^(thanks|thank you|thx|ty|cheers)[\s!?.,:]*$",
    r"^(bye|goodbye|see ya|later|cya)[\s!?.,:]*$",
    r"^(ok|okay|yes|no|sure|cool|nice|great|awesome|perfect|got it|alright)[\s!?.,:]*$",
    r"^(good\s+(morning|afternoon|evening|night))[\s!?.,:]*$",
    r"^(what|who)\s+(are|is)\s+(you|automatos|auto)[\s!?.]*$",
    r"^how\s+are\s+you[\s!?.]*$",
    r"^what\s+can\s+you\s+do[\s!?.]*$",
]

_PLATFORM_KEYWORDS = {
    "platform_list_agents": [
        "list my agents", "what agents do i have", "show my agents",
        "how many agents do i have", "show me my agents",
    ],
    "platform_list_recipes": [
        "list my recipes", "what recipes do i have", "show my recipes",
        "list my workflows", "show my workflows", "how many recipes",
        "how many workflows",
    ],
    "platform_get_llm_usage": [
        "token usage", "llm usage", "how much have i spent",
        "my api cost", "my token spend", "how many tokens",
        "show my usage", "show my spending",
    ],
    "platform_list_documents": [
        "list my documents", "what documents do i have",
        "show my documents", "how many documents do i have",
        "what files have i uploaded", "show my uploaded files",
    ],
    "platform_get_workspace_info": [
        "workspace info", "my workspace info",
        "tell me about my workspace", "show workspace details",
    ],
    "platform_list_connected_apps": [
        "what apps are connected", "show my integrations",
        "list my connected apps", "list my integrations",
        "what integrations do i have",
    ],
}

_atom_re = [re.compile(p, re.IGNORECASE) for p in _ATOM_PATTERNS]

_MEMORY_PATTERN = re.compile(
    r"\b(do you remember|recall when|my name is|last time we|"
    r"previously we discussed|earlier (i|we|you) said|what did (i|we|you) (say|tell|ask))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# AutoBrain (The Assessor)
# ---------------------------------------------------------------------------

class AutoBrain:
    """
    PRD-68: 3-Tier Progressive Complexity Assessor.

    Evaluates every incoming request to determine the required execution depth
    (Atom → Organism), bypassing heavy tools and memory for simple requests.

    Tier 1: Redis cache (<5ms)
    Tier 2: Regex heuristics (<5ms)
    Tier 3: LLM classification (~200ms, configurable model via system settings)
    """

    def __init__(self, db: Session, workspace_id: str):
        self._db = db
        self._workspace_id = workspace_id
        self._redis = None
        try:
            from core.redis.client import get_redis_client
            self._redis = get_redis_client()
        except Exception:
            logger.debug("[AutoBrain] Redis not available, cache disabled")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def assess(
        self,
        message: str,
        conversation_length: int = 0,
    ) -> ComplexityAssessment:
        """Run the 3-Tier complexity assessment."""
        if not message or not message.strip():
            return ComplexityAssessment(
                complexity=Complexity.ATOM, action=Action.RESPOND,
                reasoning="Empty message", confidence=1.0,
            )

        msg_lower = message.lower().strip()

        # ── Tier 1: Redis cache lookup (<5ms) ──
        cached = self._cache_lookup(msg_lower)
        if cached:
            return cached

        # ── Tier 2: Regex fast-paths (FREE, <5ms) ──
        heur = self._run_fast_heuristics(msg_lower)
        if heur:
            self._cache_store(msg_lower, heur)
            return heur

        # ── Tier 3: LLM classification (~200ms) ──
        llm_result = await self._llm_classify(message, conversation_length)
        self._cache_store(msg_lower, llm_result)
        return llm_result

    # ------------------------------------------------------------------
    # Tier 2: Fast heuristics
    # ------------------------------------------------------------------

    def _run_fast_heuristics(self, msg_lower: str) -> Optional[ComplexityAssessment]:
        # ATOM: Pure chitchat
        if self._is_atom(msg_lower):
            logger.info("[AutoBrain] Tier 2 Atom: '%s'", msg_lower[:50])
            return ComplexityAssessment(
                complexity=Complexity.ATOM, action=Action.RESPOND,
                reasoning="Greeting or chitchat", confidence=0.95,
                needs_memory=False, tool_hints=[], needs_multi_agent=False,
            )

        # MOLECULE: Platform queries
        platform_tool = self._match_platform_query(msg_lower)
        if platform_tool:
            logger.info("[AutoBrain] Tier 2 Platform query: %s", platform_tool)
            return ComplexityAssessment(
                complexity=Complexity.MOLECULE, action=Action.RESPOND,
                reasoning=f"Platform query ({platform_tool})",
                matched_tools=[platform_tool], tool_hints=["platform"],
                confidence=0.90, needs_memory=False, needs_multi_agent=False,
            )

        # CELL: Memory recall
        if self._is_memory_recall(msg_lower):
            logger.info("[AutoBrain] Tier 2 Memory recall")
            return ComplexityAssessment(
                complexity=Complexity.CELL, action=Action.RESPOND,
                reasoning="Explicit memory recall", confidence=0.85,
                needs_memory=True, tool_hints=[], needs_multi_agent=False,
            )

        return None

    # ------------------------------------------------------------------
    # Tier 3: LLM classification
    # ------------------------------------------------------------------

    async def _llm_classify(
        self, message: str, conversation_length: int
    ) -> ComplexityAssessment:
        """Use a lightweight LLM to classify complexity. Any model, any provider."""
        logger.info("[AutoBrain] Tier 3 LLM classifying: '%s'", message[:80])

        agent_summaries = self._get_agent_summaries()

        prompt = f"""Classify this user message for an AI platform.

Available agents: {agent_summaries}
Conversation turn: {conversation_length}

Message: "{message}"

Return ONLY valid JSON:
{{
  "complexity": "atom|molecule|cell|organ|organism",
  "action": "respond|delegate|workflow",
  "tool_hints": ["domain1", "domain2"],
  "needs_memory": true/false,
  "needs_multi_agent": true/false,
  "reasoning": "one sentence"
}}

Rules:
- atom: Greetings, chitchat, simple factual. No tools.
- molecule: Needs ONE tool/agent. "Send email", "check Jira", "search docs".
- cell: Needs tools + memory/conversation context. "Reply to that email we discussed".
- organ: Needs multiple agents coordinating. "Research bug, plan fix, open PR".
- organism: Enterprise-scale multi-step. "Refactor auth across all services".
- tool_hints: short domain keywords like "email", "github", "jira", "code", "database". Empty for atom.
- needs_memory: true if the message references past conversations or user preferences.
- needs_multi_agent: true only for organ/organism level tasks.
- action: "respond" for atom, "delegate" for molecule/cell, "workflow" for organ/organism."""

        try:
            from core.llm import create_llm_manager

            llm = create_llm_manager(service_name="complexity_assessor")
            response = await llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)

            # Extract JSON block
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return ComplexityAssessment(
                    complexity=Complexity(data.get("complexity", "molecule").lower()),
                    action=Action(data.get("action", "delegate").lower()),
                    reasoning=data.get("reasoning", "LLM classified"),
                    confidence=0.85,
                    needs_memory=data.get("needs_memory", False),
                    tool_hints=data.get("tool_hints", []),
                    needs_multi_agent=data.get("needs_multi_agent", False),
                )
        except Exception:
            logger.exception("[AutoBrain] Tier 3 LLM classification failed, falling back to DELEGATE")

        # Fallback: treat as MOLECULE / DELEGATE (current behavior)
        return ComplexityAssessment(
            complexity=Complexity.MOLECULE, action=Action.DELEGATE,
            reasoning="LLM classification failed — defaulting to delegate",
            confidence=0.50, needs_memory=False, tool_hints=[],
            needs_multi_agent=False,
        )

    # ------------------------------------------------------------------
    # Agent summaries for LLM context
    # ------------------------------------------------------------------

    def _get_agent_summaries(self) -> str:
        """Get lightweight agent descriptions for LLM context."""
        try:
            from core.models.agents import Agent
            agents = self._db.query(Agent.name, Agent.description).filter(
                Agent.workspace_id == self._workspace_id,
                Agent.is_active == True,
            ).all()
            if not agents:
                return "No custom agents configured."
            return ", ".join(
                f"{a.name}: {(a.description or '')[:60]}" for a in agents
            )
        except Exception:
            logger.debug("[AutoBrain] Could not load agent summaries")
            return "Agent list unavailable."

    # ------------------------------------------------------------------
    # Redis cache (Tier 1)
    # ------------------------------------------------------------------

    def _cache_lookup(self, msg_lower: str) -> Optional[ComplexityAssessment]:
        if not self._redis:
            return None
        try:
            cache_key = self._make_cache_key(msg_lower)
            raw = self._redis.get(cache_key)
            if raw:
                data = json.loads(raw)
                logger.info("[AutoBrain] Tier 1 Cache hit: '%s'", msg_lower[:50])
                return ComplexityAssessment(
                    complexity=Complexity(data["complexity"]),
                    action=Action(data["action"]),
                    reasoning=data.get("reasoning", "cached") + " (cached)",
                    confidence=data.get("confidence", 0.90),
                    needs_memory=data.get("needs_memory", False),
                    tool_hints=data.get("tool_hints", []),
                    needs_multi_agent=data.get("needs_multi_agent", False),
                )
        except Exception:
            logger.debug("[AutoBrain] Cache lookup failed")
        return None

    def _cache_store(self, msg_lower: str, assessment: ComplexityAssessment) -> None:
        if not self._redis:
            return
        try:
            cache_key = self._make_cache_key(msg_lower)
            ttl = int(os.environ.get("COMPLEXITY_CACHE_TTL_HOURS", "24")) * 3600
            self._redis.setex(cache_key, ttl, json.dumps(assessment.to_dict()))
        except Exception:
            logger.debug("[AutoBrain] Cache store failed, non-critical")

    def _make_cache_key(self, msg_lower: str) -> str:
        h = hashlib.sha256(msg_lower.encode()).hexdigest()[:16]
        return f"complexity:{self._workspace_id}:{h}"

    # ------------------------------------------------------------------
    # Pattern matchers (Tier 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_atom(msg_lower: str) -> bool:
        for pattern in _atom_re:
            if pattern.match(msg_lower):
                return True
        return False

    @staticmethod
    def _match_platform_query(msg_lower: str) -> Optional[str]:
        for tool_name, phrases in _PLATFORM_KEYWORDS.items():
            for phrase in phrases:
                if phrase in msg_lower:
                    return tool_name
        return None

    @staticmethod
    def _is_memory_recall(msg_lower: str) -> bool:
        return bool(_MEMORY_PATTERN.search(msg_lower))
