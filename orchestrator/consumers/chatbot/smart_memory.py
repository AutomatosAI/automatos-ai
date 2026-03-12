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
import logging
import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


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

    def _classify_memory_tier(self, user_message: str, assistant_response: str) -> str:
        """
        Classify where a memory should be stored.

        Returns: "global", "agent", or "both"

        Classification rules:
        - Personal facts (name, location, job, general info) → global
        - Tool/workflow-related (Slack, email patterns, contacts for tools) → agent
        - Preferences → both (useful everywhere)

        IMPORTANT: Only classify based on the USER message, not the assistant
        response. The assistant mentioning "slack" or "github" in an explanation
        shouldn't misclassify a personal question as agent-specific.
        """
        # Only use USER message for classification — assistant response
        # contains tool names in explanations that cause false positives
        combined = user_message.lower()

        # Tool/workflow-specific keywords → agent-specific memory
        # These are things specific to how tools are used
        tool_keywords = [
            # Communication tools
            "slack", "channel", "#", "dm", "message to",
            "gmail", "email to", "send email", "cc", "bcc", "forward",
            # Dev tools
            "github", "repository", "repo", "branch", "pr", "pull request",
            "jira", "ticket", "issue",
            # Data tools
            "database", "table", "query", "sql", "api", "endpoint",
            # File tools
            "spreadsheet", "document", "file", "folder", "drive", "upload",
            # Workflow patterns
            "when i ask", "for this agent", "in this context"
        ]

        # Personal/global keywords → global memory (who the user IS)
        personal_keywords = [
            "my name", "i am", "i'm", "call me",
            "i work at", "i work for", "my job", "my role",
            "i live", "from ireland", "from portugal", "i'm from",
            "born", "age", "founder", "ceo", "coo", "manager",
            "my company", "my team", "my organization"
        ]

        # Preference keywords → both tiers (useful everywhere)
        preference_keywords = [
            "prefer", "favorite", "like to", "don't like",
            "usually", "my style", "i want", "i need"
        ]

        # Strong agent indicators (override others)
        strong_agent_keywords = [
            "always cc", "default channel", "send to", "post to",
            "my slack", "my email", "contact", "@"
        ]

        has_tool = any(kw in combined for kw in tool_keywords)
        has_personal = any(kw in combined for kw in personal_keywords)
        has_preference = any(kw in combined for kw in preference_keywords)
        has_strong_agent = any(kw in combined for kw in strong_agent_keywords)

        # Strong agent indicators take precedence
        if has_strong_agent:
            return "agent"
        elif has_personal and has_tool:
            return "both"  # Personal + tool context → store everywhere
        elif has_preference and has_tool:
            return "both"  # Tool preference → useful in both tiers
        elif has_preference:
            return "both"  # General preference → both
        elif has_tool:
            return "agent"  # Pure tool-specific
        elif has_personal:
            return "global"  # Pure personal fact
        else:
            return "both"  # Default to both — let Mem0 decide what's worth extracting

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

            messages = [
                {"role": "user", "content": user_message[:1500]},
                {"role": "assistant", "content": assistant_response[:1500]}
            ]

            base_metadata = {
                "chat_id": chat_id,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace_id": workspace_id,
                "agent_id": agent_id,
            }

            results = await self.unified_service.store_two_tier(
                workspace_id=workspace_id,
                messages=messages,
                agent_id=agent_id,
                tier=tier,
                metadata=base_metadata,
            )

            # Check if any storage succeeded
            success = any(r[1] and not r[1].get("error") for r in results)

            if success:
                tiers_stored = [r[0] for r in results if r[1] and not r[1].get("error")]
                logger.info("[SmartMemory] Stored in tiers: %s", tiers_stored)
                # Invalidate cache
                self._invalidate_cache(workspace_id, agent_id)
                return True
            else:
                errors = [f"{r[0]}: {r[1].get('error') if r[1] else 'None'}" for r in results]
                logger.warning("[SmartMemory] Storage failed: %s", errors)
                return False

        except Exception as e:
            logger.error("[SmartMemory] Storage failed: %s", e, exc_info=True)
            return False

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
