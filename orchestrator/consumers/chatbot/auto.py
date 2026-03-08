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
import json
import hashlib
import time
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
    # Greetings (with optional name: "hi auto", "morning auto", "hey there")
    r"^(hi|hello|hey|howdy|yo|sup)(\s+\w+)?[\s!?.,:]*$",
    r"^(good\s+)?(morning|afternoon|evening|night)(\s+\w+)?[\s!?.,:]*$",
    r"^(g'day|hiya|heya|oi|ello|mornin)(\s+\w+)?[\s!?.,:]*$",
    # Informal greetings and check-ins
    r"^what'?s\s+up[\s!?.,:]*$",
    r"^how'?s\s+it\s+going[\s!?.,:]*$",
    r"^how\s+are\s+(you|things|ya)[\s!?.,:]*$",
    r"^how'?s\s+everything[\s!?.,:]*$",
    r"^long\s+time\s+no\s+see[\s!?.,:]*$",
    # Thanks / bye / acknowledgements
    r"^(thanks|thank you|thx|ty|cheers)(\s+\w+)?[\s!?.,:]*$",
    r"^(bye|goodbye|see ya|later|cya|see you)(\s+\w+)?[\s!?.,:]*$",
    r"^(ok|okay|yes|no|sure|cool|nice|great|awesome|perfect|got it|alright|grand|brilliant)[\s!?.,:]*$",
    # Identity questions
    r"^(what|who)\s+(are|is)\s+(you|automatos|auto)[\s!?.]*$",
    r"^what\s+can\s+you\s+do[\s!?.]*$",
    # Simple chitchat (no tools needed)
    r"^tell\s+me\s+a\s+joke[\s!?.,:]*$",
    # NOTE: "what time/day is it" intentionally excluded — ATOM path lacks
    # grounding context; let Tier 3 route these to the full prompt.
    r"^(lol|haha|lmao|rofl|ha+)[\s!?.,:]*$",
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
    "platform_list_tools": [
        "what tools", "list my tools", "available tools",
        "what can i use", "show my tools", "what integrations",
        "composio tools", "connected tools",
    ],
    "platform_list_llms": [
        "what models", "available models", "list llms", "list models",
        "what llms", "cheapest model", "show models",
        "openrouter models",
    ],
    "platform_list_datasources": [
        "what data", "data sources", "what databases", "list datasources",
        "what documents", "rag sources", "nl2sql", "queryable databases",
        "what repos are indexed",
    ],
    "platform_workspace_stats": [
        "workspace stats", "platform stats", "usage stats",
        "how many queries", "agent activity", "what's being used",
        "show stats", "show usage",
    ],
    "platform_execute_recipe": [
        "run the recipe", "execute recipe", "trigger recipe",
        "run automation", "start recipe",
    ],
    "platform_get_recipe_execution": [
        "recipe status", "execution status", "recipe result",
        "did the recipe run", "check recipe",
    ],
    "platform_get_system_health": [
        "system health", "platform health", "system status",
        "check health", "health check", "is everything working",
    ],
    "platform_delete_document": [
        "delete document", "remove document",
        "delete from knowledge base",
    ],
    "platform_reprocess_document": [
        "reprocess document", "re-embed document", "reindex document",
        "regenerate chunks", "rebuild embeddings",
    ],
    "platform_delete_recipe": [
        "delete recipe", "remove recipe", "delete automation",
    ],
    "platform_get_activity_feed": [
        "recent activity", "activity feed", "what's been happening",
        "show activity", "what has been running", "activity log",
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
            assessment = ComplexityAssessment(
                complexity=Complexity.ATOM, action=Action.RESPOND,
                reasoning="Greeting or chitchat", confidence=0.95,
                needs_memory=False, tool_hints=[], needs_multi_agent=False,
            )
            logger.info(
                "[AutoBrain] assessed",
                extra={
                    "tier": 2, "complexity": "atom", "action": "respond",
                    "confidence": 0.95, "latency_ms": 0, "cache_hit": False,
                    "workspace_id": self._workspace_id,
                },
            )
            return assessment

        # MOLECULE: Platform queries
        platform_tool = self._match_platform_query(msg_lower)
        if platform_tool:
            assessment = ComplexityAssessment(
                complexity=Complexity.MOLECULE, action=Action.RESPOND,
                reasoning=f"Platform query ({platform_tool})",
                matched_tools=[platform_tool], tool_hints=["platform"],
                confidence=0.90, needs_memory=False, needs_multi_agent=False,
            )
            logger.info(
                "[AutoBrain] assessed",
                extra={
                    "tier": 2, "complexity": "molecule", "action": "respond",
                    "confidence": 0.90, "latency_ms": 0, "cache_hit": False,
                    "workspace_id": self._workspace_id,
                },
            )
            return assessment

        # CELL: Memory recall
        if self._is_memory_recall(msg_lower):
            assessment = ComplexityAssessment(
                complexity=Complexity.CELL, action=Action.RESPOND,
                reasoning="Explicit memory recall", confidence=0.85,
                needs_memory=True, tool_hints=[], needs_multi_agent=False,
            )
            logger.info(
                "[AutoBrain] assessed",
                extra={
                    "tier": 2, "complexity": "cell", "action": "respond",
                    "confidence": 0.85, "latency_ms": 0, "cache_hit": False,
                    "workspace_id": self._workspace_id,
                },
            )
            return assessment

        return None

    # ------------------------------------------------------------------
    # Tier 3: LLM classification
    # ------------------------------------------------------------------

    async def _llm_classify(
        self, message: str, conversation_length: int
    ) -> ComplexityAssessment:
        """Use a lightweight LLM to classify complexity. Any model, any provider.

        Context Engineering: No agent summaries in the prompt — they bias toward
        delegation and waste tokens. Agent routing happens downstream, not here.
        """
        logger.info("[AutoBrain] Tier 3 LLM classifying: '%s'", message[:80])
        t0 = time.monotonic()

        prompt = f"""You are a message complexity classifier for an AI platform.

Analyze the user's message step by step, then classify it.

Message: "{message}"
Conversation turn: {conversation_length}

## Reasoning Steps (think through each):

1. **Intent**: What is the user asking for? (greeting, question, action, complex task)
2. **Tool need**: Does this require external data or actions? (database, email, search, file ops)
3. **Memory need**: Does this reference past conversations or user preferences?
4. **Coordination**: How many systems need to work together?

## Classification levels:

- **atom**: Greetings, chitchat, opinions, simple factual questions, jokes, acknowledgements. NO tools needed. This is the most common category — when in doubt, choose atom.
- **molecule**: Needs ONE tool or action. "Send email", "search docs", "check Jira", "list my agents".
- **cell**: Needs tools + memory/context. "Reply to that email we discussed", "update the report from last week".
- **organ**: Multiple agents coordinating. "Research this bug, plan a fix, open a PR".
- **organism**: Enterprise multi-step pipeline. Rare.

## Examples:

- "Morning Auto" → atom (greeting)
- "How are you?" → atom (chitchat)
- "What's the weather like?" → atom (conversational)
- "Tell me about yourself" → atom (identity question)
- "Send an email to John" → molecule (email tool)
- "What agents do I have?" → molecule (platform query)
- "Search my docs for the Q4 report" → molecule (search tool)
- "Remember last week's meeting? Update those notes" → cell (memory + action)

**Default bias: atom.** Most messages are simpler than they look.

Return ONLY valid JSON:
{{
  "complexity": "atom|molecule|cell|organ|organism",
  "action": "respond|delegate|workflow",
  "tool_hints": [],
  "needs_memory": false,
  "needs_multi_agent": false,
  "reasoning": "one sentence"
}}

action mapping: "respond" for atom, "delegate" for molecule/cell, "workflow" for organ/organism.
tool_hints: short domain keywords like "email", "github", "jira", "code", "database". Empty for atom."""

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
                elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
                assessment = ComplexityAssessment(
                    complexity=Complexity(data.get("complexity", "atom").lower()),
                    action=Action(data.get("action", "respond").lower()),
                    reasoning=data.get("reasoning", "LLM classified"),
                    confidence=0.85,
                    needs_memory=data.get("needs_memory", False),
                    tool_hints=data.get("tool_hints", []),
                    needs_multi_agent=data.get("needs_multi_agent", False),
                )
                logger.info(
                    "[AutoBrain] assessed",
                    extra={
                        "tier": 3,
                        "complexity": assessment.complexity.value,
                        "action": assessment.action.value,
                        "confidence": assessment.confidence,
                        "latency_ms": elapsed_ms,
                        "cache_hit": False,
                        "workspace_id": self._workspace_id,
                    },
                )
                return assessment
        except Exception:
            logger.exception("[AutoBrain] Tier 3 LLM classification failed")

        # Fallback: cheap keyword heuristic before defaulting to ATOM.
        # Messages reaching Tier 3 already passed Tier 2 without matching
        # greetings/platform queries — so they're more likely action-oriented.
        # A quick keyword scan avoids silently dropping real requests.
        if self._has_action_keywords(message):
            logger.info("[AutoBrain] Tier 3 fallback → MOLECULE (action keywords detected)")
            return ComplexityAssessment(
                complexity=Complexity.MOLECULE, action=Action.DELEGATE,
                reasoning="LLM classification failed — action keywords detected, routing to tools",
                confidence=0.40, needs_memory=False, tool_hints=[],
                needs_multi_agent=False,
            )

        return ComplexityAssessment(
            complexity=Complexity.ATOM, action=Action.RESPOND,
            reasoning="LLM classification failed — defaulting to conversational",
            confidence=0.50, needs_memory=False, tool_hints=[],
            needs_multi_agent=False,
        )

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
                logger.info(
                    "[AutoBrain] assessed",
                    extra={
                        "tier": 1, "complexity": data.get("complexity", "?"),
                        "confidence": data.get("confidence", 0.90),
                        "latency_ms": 0, "cache_hit": True,
                        "workspace_id": self._workspace_id,
                    },
                )
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
            from config import config
            ttl = int(config.COMPLEXITY_CACHE_TTL_HOURS or 24) * 3600
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
                # Word-boundary match to avoid false triggers on substrings
                if re.search(r'\b' + re.escape(phrase) + r'\b', msg_lower):
                    return tool_name
        return None

    @staticmethod
    def _has_action_keywords(message: str) -> bool:
        """Cheap scan for action-oriented keywords. Used only as Tier 3 fallback."""
        msg = message.lower()
        return any(kw in msg for kw in (
            "send", "email", "search", "find", "create", "open",
            "run", "fetch", "query", "calendar", "schedule",
            "deploy", "build", "delete", "update", "upload",
            "download", "generate", "analyze", "report",
        ))

    @staticmethod
    def _is_memory_recall(msg_lower: str) -> bool:
        return bool(_MEMORY_PATTERN.search(msg_lower))
