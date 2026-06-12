"""
Smart Memory Manager
====================

Intelligent memory management for the Automatos assistant.

Features:
- Proper Mem0 integration with correct user scoping
- Smart retrieval based on intent classification
- Extracts user facts (name, preferences) from memories
- Caches recent memories for fast access
- Background storage to not block responses
"""

import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# PRD-159 S1 — operational memory taxonomy (Zep ontology incl. `procedure`).
# The distiller emits a {fact, type, importance} object per durable fact; `type`
# is validated against this set and stored as the memory `category` so recall
# and the Explorer can filter operational knowledge by kind.
MEMORY_FACT_TYPES = frozenset({
    "tool_outcome",     # a tool/Composio call's notable result (failure, quirk, new id)
    "task_learning",    # what was learned from a mission/task succeeding or failing
    "playbook_pattern", # a reusable pattern surfaced while running a playbook
    "user_fact",        # stable fact about the user
    "business_fact",    # stable fact about their business/domain
    "preference",       # a stated preference
    "procedure",        # a how-to / standard operating procedure
})
DEFAULT_FACT_TYPE = "task_learning"


@dataclass
class UserContext:
    """Extracted user context from memories."""
    name: Optional[str] = None
    preferences: List[str] = None
    facts: List[str] = None
    recent_topics: List[str] = None
    last_interaction: Optional[datetime] = None

    def __post_init__(self):
        self.preferences = self.preferences or []
        self.facts = self.facts or []
        self.recent_topics = self.recent_topics or []


@dataclass
class MemoryResult:
    """Result of memory retrieval."""
    memories: List[Dict[str, Any]]
    user_context: UserContext
    formatted_context: str
    retrieval_time_ms: float


class SmartMemoryManager:
    """
    Intelligent memory management for chat.

    Key improvements over previous system:
    1. Consistent user_id scoping (workspace + agent)
    2. Smart retrieval based on intent
    3. User fact extraction (name, preferences)
    4. Memory caching for performance
    5. Background storage to not block responses
    """

    def __init__(self):
        self._unified_service = None
        self._cache: Dict[str, Tuple[float, MemoryResult]] = {}
        self._cache_ttl = 120  # 2 minutes
        self._storage_queue: List[Tuple] = []
        self._storage_task = None
        # PRD-159 S5: honest memory_stored SSE — the tier that actually persisted
        # and how many durable facts were written this turn (0 → no SSE).
        self._last_tier: Optional[str] = None
        self._last_l3_facts_stored: int = 0

    @property
    def unified_service(self):
        """Lazy initialization of UnifiedMemoryService."""
        if self._unified_service is None:
            try:
                from modules.memory.unified_memory_service import get_unified_memory_service
                self._unified_service = get_unified_memory_service()
                logger.info("[SmartMemory] UnifiedMemoryService initialized")
            except Exception as e:
                logger.warning(f"[SmartMemory] Could not initialize UnifiedMemoryService: {e}")
        return self._unified_service

    @staticmethod
    def _get_store_max_chars() -> int:
        """Memory store truncation length — non-LLM feature flag in `general`
        (post PRD-136 collapse: memory_management category retired)."""
        try:
            from core.llm.manager import get_system_setting
            val = get_system_setting("general", "memory_store_max_chars", "6000")
            return int(val)
        except Exception:
            return 6000

    def _classify_memory_tier(self, user_message: str, assistant_response: str) -> str:
        """
        Classify where a memory should be stored. Returns "global" or "agent".

        PRD-159 S5: the "both"-tier double-write default is REMOVED. Every
        exchange used to write to BOTH the workspace and agent namespaces,
        doubling the Mem0 rows and the Explorer count for one memory. The default
        is now a SINGLE workspace namespace ("global"); the agent namespace is
        used only when the user gives an explicit agent-scoped instruction. The
        fragile single-character keywords ('#'/'@') that misrouted ordinary
        messages to the agent tier are gone.

        Only the USER message is classified — the assistant response contains
        tool names in explanations that caused false positives.
        """
        combined = user_message.lower()

        # Explicit agent-scoped instructions → store under the agent namespace.
        # These are durable directions about how THIS agent should act, not
        # passing mentions of a tool name.
        strong_agent_keywords = [
            "always cc", "default channel", "for this agent", "in this context",
            "when i ask you", "post to", "send to",
        ]
        if any(kw in combined for kw in strong_agent_keywords):
            return "agent"

        # Everything else → single workspace namespace. No double-write.
        return "global"

    def _get_cache_key(self, workspace_id: str, agent_id: Optional[int], query: str) -> str:
        """Create cache key for memory lookups (includes agent for agent-specific cache)."""
        return f"{workspace_id}:{agent_id}:{query[:50]}"

    async def retrieve_memories(
        self,
        workspace_id: str,
        agent_id: Optional[int],
        query: str,
        limit: int = 8,
        widget_mode: bool = False
    ) -> MemoryResult:
        """
        Retrieve relevant memories for a query.

        Args:
            workspace_id: Workspace ID for scoping
            agent_id: Agent ID for scoping
            query: The user's query to match against
            limit: Maximum memories to retrieve

        Returns:
            MemoryResult with memories and extracted context
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(workspace_id, agent_id, query)
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._cache_ttl:
            logger.debug("[SmartMemory] Using cached memory result")
            return cached[1]

        memories = []
        user_context = UserContext()

        try:
            if self.unified_service:
                # TWO-TIER MEMORY RETRIEVAL via UnifiedMemoryService
                # 1. Global memories (user facts shared across all agents)
                # 2. Agent-specific memories (tool preferences, workflow patterns)

                if widget_mode:
                    # Widget mode: agent-only retrieval — never leak global workspace memories
                    logger.info("[SmartMemory] Widget mode: agent-only retrieval (ws=%s agent=%s)", workspace_id, agent_id)
                    agent_memories = await self.unified_service.search_long_term(
                        workspace_id, query, agent_id=agent_id, limit=limit,
                    )
                    global_memories = []
                else:
                    logger.info("[SmartMemory] Two-tier search: ws=%s agent=%s", workspace_id, agent_id)

                    # Fetch both tiers in parallel
                    global_task = self.unified_service.search_long_term(
                        workspace_id, query, agent_id=None, limit=limit,
                    )
                    agent_task = self.unified_service.search_long_term(
                        workspace_id, query, agent_id=agent_id, limit=limit,
                    )
                    global_memories, agent_memories = await asyncio.gather(global_task, agent_task)

                # Merge: global first (who the user is), then agent-specific (how they use this agent)
                global_memories = global_memories or []
                agent_memories = agent_memories or []

                logger.info("[SmartMemory] Found %d global + %d agent-specific memories", len(global_memories), len(agent_memories))

                # Combine with global first, agent-specific second
                # Mark agent-specific memories so we can format them differently
                for mem in agent_memories:
                    mem["_tier"] = "agent"

                memories = global_memories + agent_memories

                if memories:
                    for i, mem in enumerate(memories[:3]):
                        tier = mem.get("_tier", "global")
                        logger.info("[SmartMemory]   Memory %d [%s]: %s...", i + 1, tier, mem.get("memory", "")[:80])

                    # Extract user context from memories
                    user_context = self._extract_user_context(memories)
                else:
                    logger.info("[SmartMemory] No memories found for user")

        except Exception as e:
            logger.warning("[SmartMemory] Retrieval failed: %s", e, exc_info=True)

        # Format memories for LLM context
        formatted = self._format_memories_for_llm(memories, user_context)

        retrieval_time = (time.time() - start_time) * 1000

        result = MemoryResult(
            memories=memories,
            user_context=user_context,
            formatted_context=formatted,
            retrieval_time_ms=retrieval_time
        )

        # Cache result
        self._cache[cache_key] = (time.time(), result)

        # Track memory access for analytics (fire-and-forget)
        try:
            self._track_memory_access(workspace_id, len(memories) > 0)
        except Exception:
            pass

        return result

    def _track_memory_access(self, workspace_id: str, had_results: bool) -> None:
        """Record a memory search for hit-rate analytics."""
        try:
            from core.database.database import SessionLocal
            from sqlalchemy import text
            db = SessionLocal()
            try:
                db.execute(
                    text("""
                        INSERT INTO memory_access_log (workspace_id, had_results, created_at)
                        VALUES (:ws, :hit, NOW())
                    """),
                    {"ws": workspace_id, "hit": had_results},
                )
                db.commit()
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[SmartMemory] Access tracking skipped: {e}")

    def _extract_user_context(self, memories: List[Dict]) -> UserContext:
        """
        Extract user facts from memories.

        Looks for:
        - Name mentions ("Name is X", "I'm X", "My name is X")
        - Preferences ("prefer", "like", "favorite")
        - Facts about the user
        """
        context = UserContext()
        name_patterns = ["name is", "i'm", "my name is", "call me"]

        for mem in memories:
            content = mem.get("memory", "").lower() if isinstance(mem.get("memory"), str) else ""

            # Look for name
            if not context.name:
                for pattern in name_patterns:
                    if pattern in content:
                        # Extract name after pattern
                        idx = content.find(pattern) + len(pattern)
                        words = content[idx:].strip().split()
                        if words:
                            # Capitalize the name
                            potential_name = words[0].strip(".,!?")
                            if len(potential_name) > 1:
                                context.name = potential_name.capitalize()
                                break

            # Look for preferences
            if "prefer" in content or "like" in content or "favorite" in content:
                context.preferences.append(mem.get("memory", ""))

            # Add as general fact
            if mem.get("memory"):
                context.facts.append(mem.get("memory"))

        return context

    def _format_memories_for_llm(
        self,
        memories: List[Dict],
        user_context: UserContext
    ) -> str:
        """
        Format memories into a string for LLM context injection.

        Two-tier formatting:
        1. Global memories (who the user is)
        2. Agent-specific memories (how they use this agent's tools)
        """
        if not memories and not user_context.name:
            return ""

        lines = []

        # User identity first (most important)
        if user_context.name:
            lines.append(f"User's name: {user_context.name}")

        # Separate global and agent-specific memories
        global_memories = [m for m in memories if m.get("_tier") != "agent"]
        agent_memories = [m for m in memories if m.get("_tier") == "agent"]

        # Global facts about the user
        if global_memories:
            lines.append("\nAbout this user:")
            for mem in global_memories[:5]:
                memory_text = mem.get("memory", "")
                if memory_text and len(memory_text) > 5:
                    if len(memory_text) > 150:
                        memory_text = memory_text[:150] + "..."
                    lines.append(f"  - {memory_text}")

        # Agent-specific context
        if agent_memories:
            lines.append("\nWith this agent specifically:")
            for mem in agent_memories[:4]:
                memory_text = mem.get("memory", "")
                if memory_text and len(memory_text) > 5:
                    if len(memory_text) > 150:
                        memory_text = memory_text[:150] + "..."
                    lines.append(f"  - {memory_text}")

        return "\n".join(lines)

    async def store_conversation(
        self,
        workspace_id: str,
        agent_id: Optional[int],
        user_message: str,
        assistant_response: str,
        chat_id: Optional[str] = None,
        widget_mode: bool = False
    ) -> bool:
        """
        Store a conversation exchange in memory.

        This runs asynchronously to not block the response.

        Args:
            workspace_id: Workspace ID
            agent_id: Agent ID
            user_message: The user's message
            assistant_response: The assistant's response
            chat_id: Optional chat session ID

        Returns:
            Success status
        """
        if not user_message or not assistant_response:
            return False

        # Skip storing trivial exchanges (pure greetings with no substance)
        # But keep short personal facts like "I'm Gerard" or "Call me G"
        trivial_patterns = {"hi", "hello", "hey", "thanks", "ok", "bye", "yes", "no", "sure"}
        if (
            len(user_message.strip()) < 5
            or user_message.strip().lower().rstrip("!.?") in trivial_patterns
        ):
            logger.debug("[SmartMemory] Skipping storage for trivial exchange")
            return False

        try:
            if not self.unified_service:
                logger.warning("[SmartMemory] No UnifiedMemoryService available for storage")
                return False

            # TWO-TIER MEMORY STORAGE
            # Classify where this memory should be stored
            tier = self._classify_memory_tier(user_message, assistant_response)

            # Widget mode: force agent-only storage — never pollute global with widget customer data
            if widget_mode:
                logger.info("[SmartMemory] Widget mode: forcing agent-only storage")
                tier = "agent"
            # US-015: Store last tier so callers can emit SSE events
            self._last_tier = tier
            logger.info("[SmartMemory] Memory classified as: %s", tier)

            max_chars = self._get_store_max_chars()

            # L3 input curation: distil durable facts from this exchange BEFORE
            # feeding Mem0. Sending the raw transcript made Mem0's server-side
            # extraction emit thin, episodic facts ("User requested…"). Curating
            # here yields durable knowledge instead. The verbatim transcript is
            # still dual-written to L2 below, so nothing is lost.
            facts = await self._distill_durable_facts(
                user_message,
                assistant_response,
                workspace_id=workspace_id,
                agent_id=agent_id,
            )

            # PRD-159 S1: NO raw-exchange fallback. A failed distill
            # (``facts is None``) or an exchange with nothing durable
            # (``facts == []``) writes NOTHING to L3 — the L2 transcript below
            # still preserves the verbatim turn. This is what ends the
            # "user said hello" era: junk never reaches Mem0.
            if facts is None:
                logger.info(
                    "[SmartMemory] Distill failed — L3 skipped; transcript kept in L2"
                )
                facts = []
            distilled = True

            base_metadata = {
                "chat_id": chat_id,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace_id": workspace_id,
                "agent_id": agent_id,
                "distilled": distilled,
            }

            # Each durable fact is stored with its OWN typed metadata (category +
            # importance) so tier/category/importance stay filterable in semantic
            # search and the Explorer (PRD-159 S1/S3/S5). store_two_tier applies
            # one metadata dict per call, so we write per fact — facts/turn is
            # small (0..N), so this stays cheap.
            # PRD-142 W3-S7 (§H): the L3 write is guarded so a Mem0 outage cannot
            # prevent the L2 transcript write below.
            results: List[tuple] = []
            l3_raised = False
            for fact in facts:
                # Defensive: tolerate a bare string (legacy/edge) and skip a
                # malformed fact rather than crash the turn — the L2 transcript
                # write below must NEVER be lost to one bad L3 fact (PRD-142 W3-S7).
                if isinstance(fact, str):
                    fact = {"fact": fact, "type": DEFAULT_FACT_TYPE, "importance": 0.5}
                if not isinstance(fact, dict) or not fact.get("fact"):
                    continue
                fact_meta = {
                    **base_metadata,
                    "category": fact.get("type", DEFAULT_FACT_TYPE),
                    "importance": fact.get("importance", 0.5),
                }
                try:
                    fact_results = await self.unified_service.store_two_tier(
                        workspace_id=workspace_id,
                        messages=[{"role": "user", "content": str(fact["fact"])[:max_chars]}],
                        agent_id=agent_id,
                        tier=tier,
                        metadata=fact_meta,
                    )
                    results.extend(fact_results)
                except Exception:
                    l3_raised = True
                    logger.warning(
                        "[SmartMemory] L3 store_two_tier raised for a fact — L2 "
                        "transcript will still persist", exc_info=True,
                    )

            # PRD-131d Phase 3 / W3-S7 G12: L2 transcript is the SINGLE L2
            # write for a chat turn. Mem0 keeps the distilled facts for
            # retrieval; L2 keeps the verbatim text for audit/review. The
            # older direct ``UnifiedMemoryService.store_exchange`` spawn from
            # smart_orchestrator was a duplicate L2 row this collapse retired.
            try:
                await self.unified_service.store_transcript(
                    workspace_id=workspace_id,
                    turns=[
                        {"role": "user", "content": user_message},
                        {"role": "assistant", "content": assistant_response},
                    ],
                    agent_id=agent_id,
                    conversation_id=chat_id,
                    metadata={"tier": tier, "widget_mode": widget_mode},
                )
            except Exception:
                logger.warning(
                    "[SmartMemory] Transcript storage skipped", exc_info=True,
                )

            # L3 stored OK, or was deliberately skipped (nothing durable) — both
            # count as success; the L2 transcript above preserves the raw
            # exchange regardless. A raised L3 (l3_raised) is reported as a
            # visible failure via False return (§H: never silent), even though
            # L2 still got the verbatim turn.
            l3_ok = any(r[1] and not r[1].get("error") for r in results)
            # PRD-159 S5: how many durable facts actually persisted to L3 this
            # turn — drives the honest memory_stored SSE (0 → no event fired).
            self._last_l3_facts_stored = len(facts) if l3_ok else 0
            success = (l3_ok or not facts) and not l3_raised

            if success:
                if l3_ok:
                    tiers_stored = [r[0] for r in results if r[1] and not r[1].get("error")]
                    logger.info("[SmartMemory] Stored distilled facts in tiers: %s", tiers_stored)
                else:
                    logger.info("[SmartMemory] No durable facts — L3 skipped; transcript kept in L2")
                # Invalidate cache
                self._invalidate_cache(workspace_id, agent_id)
                return True
            else:
                if l3_raised:
                    logger.warning(
                        "[SmartMemory] L3 raised; turn preserved in L2 only",
                    )
                else:
                    errors = [f"{r[0]}: {r[1].get('error') if r[1] else 'None'}" for r in results]
                    logger.warning("[SmartMemory] L3 storage failed: %s", errors)
                return False

        except Exception as e:
            logger.error("[SmartMemory] Storage failed: %s", e, exc_info=True)
            return False

    # ---------------------------------------------------------------
    # L3 input curation — distil durable facts before feeding Mem0
    # ---------------------------------------------------------------

    async def _distill_durable_facts(
        self,
        user_message: str,
        assistant_response: str,
        *,
        workspace_id: str,
        agent_id: Optional[int],
    ) -> Optional[List[Dict[str, Any]]]:
        """Distil 0..N typed durable facts from a chat exchange for the L3 feed.

        Each fact is ``{"fact": str, "type": <taxonomy>, "importance": float}``.
        The distiller runs on the cheap model tier (``config.MEMORY_DISTILL_MODEL``)
        — ~1 LLM call/turn (PRD-159 D11/Q16).

        Returns:
            - ``list[dict]`` of typed facts (possibly empty → nothing durable;
              caller skips L3),
            - ``None`` on LLM/parse failure. PRD-159 S1: the caller stores
              NOTHING on failure (no raw-exchange fallback) — the L2 transcript
              still preserves the verbatim turn.
        """
        prompt = self._build_distill_prompt(user_message, assistant_response)
        try:
            # Imported here (not at module top) so tests can monkeypatch
            # ``core.llm.create_llm_manager`` and have it take effect per call.
            from core.llm import create_llm_manager
            from config import config

            llm = create_llm_manager(
                service_name="memory_integration",
                model=config.MEMORY_DISTILL_MODEL,
                workspace_id=workspace_id,
                agent_id=agent_id,
                request_type="memory_distill",
            )
            response = await llm.generate_response(
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content if hasattr(response, "content") else str(response)
        except Exception:
            logger.warning(
                "[SmartMemory] Fact distillation LLM call failed", exc_info=True
            )
            return None

        return self._parse_distilled_facts(content)

    @staticmethod
    def _build_distill_prompt(user_message: str, assistant_response: str) -> str:
        """Prompt for typed operational memory (PRD-159 S1).

        No transient-event exclusion: tool outcomes, task/mission learnings and
        playbook patterns are exactly the operational memories we now WANT. The
        ``type`` field classifies them instead of a blanket ban filtering them
        out.
        """
        return (
            "You are curating long-term memory for an AI assistant (\"Auto\"). "
            "From the single chat exchange below, extract durable facts worth "
            "remembering for future work — both operational knowledge (what a "
            "tool call revealed, what a mission/task taught, a reusable playbook "
            "pattern) and stable knowledge about the user, their business, their "
            "domain, and their preferences.\n\n"
            "Classify each fact with a `type` from this taxonomy:\n"
            "- tool_outcome: a notable result of a tool/integration call "
            "(a failure + its cause, an auth quirk, a rate limit, a new channel/"
            "record id, a schema surprise)\n"
            "- task_learning: what was learned from a mission or task succeeding "
            "or failing\n"
            "- playbook_pattern: a reusable pattern/approach surfaced while "
            "working\n"
            "- user_fact: a stable fact about the user\n"
            "- business_fact: a stable fact about their business or domain\n"
            "- preference: a stated preference (tone, format, tools, cadence)\n"
            "- procedure: a how-to or standing instruction for getting something "
            "done\n\n"
            "Write each `fact` as a standalone, third-person statement that makes "
            "sense without the surrounding conversation. Preserve specifics "
            "(names, standards, numbers, ids, spellings). Set `importance` in "
            "[0,1] (0.8+ = load-bearing, 0.3 = minor). Skip pure pleasantries and "
            "chit-chat with no durable content.\n\n"
            "Return ONLY a JSON array of objects "
            "{\"fact\": str, \"type\": str, \"importance\": number}. "
            "If nothing durable is worth keeping, return an empty array [].\n\n"
            f"User: {user_message}\n"
            f"Assistant: {assistant_response}\n\n"
            "Typed durable facts (JSON array):"
        )

    @staticmethod
    def _parse_distilled_facts(content: str) -> Optional[List[Dict[str, Any]]]:
        """Parse the LLM output into a list of typed ``{fact, type, importance}``.

        Tolerates prose and ```json code fences around the array. Each item is
        normalised: ``type`` is validated against ``MEMORY_FACT_TYPES`` (unknown
        → ``DEFAULT_FACT_TYPE``) and ``importance`` is coerced to a [0,1] float
        (default 0.5). Items without a non-empty ``fact`` string are dropped.
        Returns the parsed list (possibly empty), or ``None`` if no JSON array
        can be found.
        """
        if not content:
            return None
        text = content.strip()
        # Strip a ```json … ``` (or bare ```) code fence if present.
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Prefer a clean parse; otherwise grab the first […] array in the text.
        candidate = text
        if not (candidate.startswith("[") and candidate.endswith("]")):
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return None
            candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            return None
        if not isinstance(parsed, list):
            return None

        out: List[Dict[str, Any]] = []
        for item in parsed:
            # Tolerate a bare string (older shape) by typing it as the default.
            if isinstance(item, str):
                fact_text = item.strip()
                ftype, importance = DEFAULT_FACT_TYPE, 0.5
            elif isinstance(item, dict):
                fact_text = str(item.get("fact", "")).strip()
                ftype = str(item.get("type", DEFAULT_FACT_TYPE)).strip()
                if ftype not in MEMORY_FACT_TYPES:
                    ftype = DEFAULT_FACT_TYPE
                try:
                    importance = float(item.get("importance", 0.5))
                except (ValueError, TypeError):
                    importance = 0.5
                importance = max(0.0, min(1.0, importance))
            else:
                continue
            if not fact_text:
                continue
            out.append({"fact": fact_text, "type": ftype, "importance": importance})
        return out

    # ---------------------------------------------------------------
    # Daily Log Summary (US-011 / US-012)
    # ---------------------------------------------------------------

    @staticmethod
    def _extract_summary_from_exchange(
        user_message: str,
        assistant_response: str,
    ) -> str:
        """
        Rule-based extraction of key activities from a single chat exchange.

        Extracts:
        - Topics discussed (first sentence / main noun phrases)
        - Tools invoked (Composio action names or common tool keywords)
        - Decisions made (keywords: decided, approved, confirmed, chose, set up)

        Returns a 1-3 sentence summary of the exchange.
        """
        parts: List[str] = []

        # --- Topic: use first meaningful sentence of the user message ---
        first_sentence = re.split(r'[.!?\n]', user_message.strip(), maxsplit=1)[0].strip()
        if first_sentence:
            # Cap at 120 chars
            topic = first_sentence[:120]
            parts.append(f"Discussed: {topic}")

        # --- Tools: scan assistant response for action/tool names ---
        tool_patterns = [
            r'(?:executed|called|ran|used|invoked|triggered)\s+[`"]?(\w+_\w+)[`"]?',
            r'(?:SLACK|GMAIL|GITHUB|GOOGLESHEETS|GOOGLEDOCS|HUBSPOT|JIRA|NOTION|TRELLO|ASANA|COMPOSIO)\w*',
        ]
        tools_found: set = set()
        for pattern in tool_patterns:
            matches = re.findall(pattern, assistant_response, re.IGNORECASE)
            tools_found.update(m.upper() for m in matches)
        # Also check user message for explicit tool mentions
        tool_keywords = [
            "slack", "gmail", "email", "github", "google sheets",
            "jira", "notion", "trello", "hubspot", "asana",
        ]
        combined_lower = (user_message + " " + assistant_response).lower()
        for kw in tool_keywords:
            if kw in combined_lower:
                tools_found.add(kw.upper().replace(" ", "_"))
        if tools_found:
            parts.append(f"Tools: {', '.join(sorted(tools_found)[:5])}")

        # --- Decisions: look for decision language ---
        decision_patterns = [
            r'(?:decided|approved|confirmed|chose|selected|set up|configured|created|scheduled|enabled|disabled)\s+(.{10,80}?)(?:[.\n]|$)',
        ]
        decisions: List[str] = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, assistant_response, re.IGNORECASE)
            decisions.extend(m.strip() for m in matches[:2])
        if decisions:
            parts.append(f"Actions: {'; '.join(decisions[:2])}")

        if not parts:
            # Fallback: just note the exchange happened
            return f"User query: {user_message[:80]}"

        return ". ".join(parts)

    async def store_daily_summary(
        self,
        workspace_id: str,
        user_message: str,
        assistant_response: str,
        agent_id: Optional[int] = None,
    ) -> bool:
        """
        Generate and store a daily log summary entry from a chat exchange.

        Extracts key activities (topics, tools, decisions) using rule-based
        extraction (no LLM call) and stores/appends to today's daily log in
        Mem0.

        Args:
            workspace_id: Workspace ID for scoping
            user_message: The user's message
            assistant_response: The assistant's response
            agent_id: Optional agent ID (included in metadata)

        Returns:
            True if stored successfully, False otherwise
        """
        if not user_message or not assistant_response:
            return False

        try:
            if not self.unified_service:
                logger.warning("[SmartMemory] No UnifiedMemoryService for daily summary storage")
                return False

            today_str = datetime.utcnow().strftime("%Y-%m-%d")

            # Extract summary from the exchange
            summary_line = self._extract_summary_from_exchange(
                user_message, assistant_response
            )
            timestamp = datetime.utcnow().strftime("%H:%M")
            entry = f"[{timestamp}] {summary_line}"

            metadata = {
                "type": "daily_log_entry",
                "date": today_str,
                "workspace_id": workspace_id,
            }
            if agent_id is not None:
                metadata["agent_id"] = agent_id

            result = await self.unified_service.store_daily_log(
                workspace_id=workspace_id,
                content=entry,
                agent_id=agent_id,
                metadata=metadata,
            )

            # L2: Store daily log in short-term memory for temporal retrieval (fire-and-forget)
            try:
                asyncio.create_task(
                    self.unified_service.store_short_term(
                        workspace_id=workspace_id,
                        content=entry,
                        content_type="heartbeat_log",
                        agent_id=agent_id,
                        importance=0.4,
                        metadata=metadata,
                    )
                )
            except Exception:
                logger.debug(
                    "[SmartMemory] L2 store_short_term for daily log failed ws=%s",
                    workspace_id,
                    exc_info=True,
                )

            success = bool(result and not result.get("error"))
            if success:
                logger.info("[SmartMemory] Daily summary stored for %s", today_str)
            else:
                logger.warning(
                    "[SmartMemory] Daily summary storage failed: %s",
                    result.get("error") if result else "no result",
                )
            return success

        except Exception as e:
            logger.error("[SmartMemory] store_daily_summary failed: %s", e, exc_info=True)
            return False

    async def get_daily_logs(
        self,
        workspace_id: str,
        max_chars: int = 2000,
    ) -> str:
        """
        Fetch today and yesterday's daily logs for injection into the system
        prompt.

        Args:
            workspace_id: Workspace ID for scoping
            max_chars: Maximum character length (~500 tokens at ~4 chars/token)

        Returns:
            Formatted string of daily logs, or empty string if none exist
        """
        try:
            if not self.unified_service:
                return ""

            today = datetime.utcnow()
            today_str = today.strftime("%Y-%m-%d")
            yesterday_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
            target_dates = {today_str, yesterday_str}

            all_memories = await self.unified_service.get_all_daily_logs(
                workspace_id=workspace_id, limit=50,
            )

            if not all_memories:
                return ""

            # Filter to today + yesterday entries (supports both old
            # "daily_log" bulk records and new "daily_log_entry" per-exchange)
            daily_entries: Dict[str, str] = {}
            for mem in all_memories:
                meta = mem.get("metadata") or mem.get("metadata_") or {}
                mem_type = meta.get("type", "")
                if mem_type not in ("daily_log", "daily_log_entry"):
                    continue
                date_val = meta.get("date", "")
                if date_val not in target_dates:
                    continue
                content = mem.get("memory") or mem.get("content") or ""
                if content:
                    existing = daily_entries.get(date_val, "")
                    daily_entries[date_val] = (
                        f"{existing}\n{content}" if existing else content
                    )

            if not daily_entries:
                return ""

            # Build formatted output (today first, then yesterday)
            lines: List[str] = ["Recent activity log:"]
            for date_key in [today_str, yesterday_str]:
                if date_key not in daily_entries:
                    continue
                label = "Today" if date_key == today_str else "Yesterday"
                lines.append(f"\n[{label} - {date_key}]")
                lines.append(daily_entries[date_key])

            result = "\n".join(lines)

            # Trim to max_chars, cutting oldest entries first
            if len(result) > max_chars:
                # If we have both days and it's too long, drop yesterday
                if yesterday_str in daily_entries and today_str in daily_entries:
                    lines_today = [
                        "Recent activity log:",
                        f"\n[Today - {today_str}]",
                        daily_entries[today_str],
                    ]
                    result = "\n".join(lines_today)

                # If still too long, truncate from the beginning of entries
                if len(result) > max_chars:
                    result = result[:max_chars].rsplit("\n", 1)[0] + "\n..."

            return result

        except Exception as e:
            logger.error("[SmartMemory] get_daily_logs failed: %s", e)
            return ""

    async def cleanup_old_daily_logs(
        self,
        workspace_id: str,
        retention_days: int = 7,
    ) -> int:
        """
        Delete daily logs older than retention_days.

        Intended to be called by the heartbeat service cleanup job.

        Args:
            workspace_id: Workspace ID for scoping
            retention_days: Number of days to retain (default 7)

        Returns:
            Number of deleted log entries
        """
        try:
            if not self.unified_service:
                return 0

            cutoff = datetime.utcnow() - timedelta(days=retention_days)
            cutoff_str = cutoff.strftime("%Y-%m-%d")

            all_memories = await self.unified_service.get_all_daily_logs(
                workspace_id=workspace_id, limit=100,
            )

            if not all_memories:
                return 0

            deleted = 0
            for mem in all_memories:
                meta = mem.get("metadata") or mem.get("metadata_") or {}
                if meta.get("type") not in ("daily_log", "daily_log_entry"):
                    continue
                date_val = meta.get("date", "")
                mem_id = mem.get("id")
                if date_val and date_val < cutoff_str and mem_id:
                    success = await self.unified_service.delete_memory(mem_id)
                    if success:
                        deleted += 1
                        logger.info(
                            "[SmartMemory] Deleted old daily log: date=%s id=%s",
                            date_val, mem_id,
                        )

            logger.info(
                "[SmartMemory] Daily log cleanup: deleted %d entries older than %s",
                deleted, cutoff_str,
            )
            return deleted

        except Exception as e:
            logger.error("[SmartMemory] cleanup_old_daily_logs failed: %s", e, exc_info=True)
            return 0

    def _invalidate_cache(self, workspace_id: str, agent_id: Optional[int]):
        """Invalidate cached memories for a user."""
        prefix = f"{workspace_id}:{agent_id}:"
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]

    def get_user_name(self, memories_result: MemoryResult) -> Optional[str]:
        """Quick helper to get user's name from memory result."""
        return memories_result.user_context.name if memories_result else None

    def clear_cache(self):
        """Clear all cached memories."""
        self._cache.clear()


# Module-level singleton
_memory_manager = None

def get_smart_memory_manager() -> SmartMemoryManager:
    """Get the global smart memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = SmartMemoryManager()
    return _memory_manager
