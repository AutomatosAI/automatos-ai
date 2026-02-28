"""
Auto Brain — Thin Gate Before the Universal Router
====================================================

Auto receives every message and makes ONE decision:
  - Is this trivial (greeting, platform meta-query, memory recall)?
    → Handle directly with the orchestrator LLM.
  - Everything else → DELEGATE to the Universal Router.

The router has semantic similarity (Tier 2.5) and LLM classification (Tier 3)
which understand agent descriptions, tags, personas, and tools far better than
keyword maps ever could.

Previous design had hardcoded _TOOL_KEYWORDS and _INTERNAL_TOOL_KEYWORDS that
tried to match user messages to tools via substring matching.  That approach:
  - Missed natural language variations ("create an image" ≠ "create image")
  - Treated any short 1-2 word message as a greeting (atom), swallowing
    legitimate requests like "send email", "find flights", "draw this"
  - Prevented the LLM-powered router from ever seeing most messages

Now: AutoBrain is intentionally narrow.  When in doubt, DELEGATE.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

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


# ---------------------------------------------------------------------------
# Patterns — intentionally narrow.  When in doubt, DON'T match.
# ---------------------------------------------------------------------------

# Greetings and chitchat — must be the ENTIRE message (with optional punctuation).
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

# Platform self-awareness queries — meta-questions ABOUT the platform itself.
# These are handled by Auto's internal tools, not specialized agents.
# Patterns are intentionally specific to avoid false positives.
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
# AutoBrain
# ---------------------------------------------------------------------------

class AutoBrain:
    """
    Thin gate before the Universal Router.

    Only short-circuits for messages that clearly DON'T need a specialized
    agent: greetings, platform meta-queries, and memory recalls.

    Everything else → DELEGATE → Universal Router (semantic + LLM).
    """

    def __init__(self, db: Session, workspace_id: str):
        self._db = db
        self._workspace_id = workspace_id

    async def assess(
        self,
        message: str,
        conversation_length: int = 0,
    ) -> ComplexityAssessment:
        """
        Decide: handle directly (RESPOND) or send to router (DELEGATE)?

        RESPOND = Auto handles with orchestrator LLM (greetings, platform, memory).
        DELEGATE = Universal Router picks the best specialized agent.
        """
        if not message or not message.strip():
            return ComplexityAssessment(
                complexity=Complexity.ATOM,
                action=Action.RESPOND,
                reasoning="Empty message",
                confidence=1.0,
            )

        msg_lower = message.lower().strip()

        # --- Greetings & chitchat → Auto responds directly ---
        if self._is_atom(msg_lower):
            logger.info("[AutoBrain] Atom detected: '%s'", msg_lower[:50])
            return ComplexityAssessment(
                complexity=Complexity.ATOM,
                action=Action.RESPOND,
                reasoning="Greeting or chitchat",
                confidence=0.95,
            )

        # --- Platform self-awareness → Auto responds directly ---
        platform_tool = self._match_platform_query(msg_lower)
        if platform_tool:
            logger.info("[AutoBrain] Platform query: %s", platform_tool)
            return ComplexityAssessment(
                complexity=Complexity.MOLECULE,
                action=Action.RESPOND,
                reasoning=f"Platform query ({platform_tool})",
                matched_tools=[platform_tool],
                confidence=0.90,
            )

        # --- Memory recall → Auto responds directly ---
        if self._is_memory_recall(msg_lower):
            logger.info("[AutoBrain] Memory recall detected")
            return ComplexityAssessment(
                complexity=Complexity.CELL,
                action=Action.RESPOND,
                reasoning="Memory recall - Auto handles with context",
                confidence=0.85,
            )

        # --- Everything else → DELEGATE to Universal Router ---
        # The router has:
        #   Tier 2.5 — Semantic similarity (understands agent descriptions)
        #   Tier 3   — LLM classification (sees all agents + tools + descriptions)
        # Both are far more capable than keyword matching.
        logger.info("[AutoBrain] Delegating to router: '%s'", msg_lower[:80])
        return ComplexityAssessment(
            complexity=Complexity.MOLECULE,
            action=Action.DELEGATE,
            reasoning="Delegating to router for agent selection",
            confidence=0.70,
        )

    # ------------------------------------------------------------------
    # Pattern matchers — intentionally narrow
    # ------------------------------------------------------------------

    @staticmethod
    def _is_atom(msg_lower: str) -> bool:
        """Is this a standalone greeting or chitchat that needs no agent?

        Only matches complete messages like "hi", "thanks!", "ok".
        Does NOT match "hi can you create an image" or "ok now send the email".
        """
        for pattern in _atom_re:
            if pattern.match(msg_lower):
                return True
        return False

    @staticmethod
    def _match_platform_query(msg_lower: str) -> Optional[str]:
        """Match platform self-awareness queries (list agents, usage, etc.).

        Only matches specific phrases about the platform itself, not general
        requests that happen to mention "agents" or "documents".
        """
        for tool_name, phrases in _PLATFORM_KEYWORDS.items():
            for phrase in phrases:
                if phrase in msg_lower:
                    return tool_name
        return None

    @staticmethod
    def _is_memory_recall(msg_lower: str) -> bool:
        """Is this specifically asking about past conversations/memory?"""
        return bool(_MEMORY_PATTERN.search(msg_lower))
