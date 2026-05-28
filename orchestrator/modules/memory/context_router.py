"""
Context Router
==============

Intelligent pre-LLM context assembly layer that analyses user queries and
decides which memory layers to fetch BEFORE the agent sees the prompt.

Two responsibilities:
  1. **Signal detection** (``analyze_query``) — fast regex, <10 ms, no I/O.
  2. **Context assembly** (``retrieve_context``) — fetches from L1/L2/L3
     based on signals and assembles a budget-constrained ContextBundle.

Usage:
    from modules.memory.context_router import ContextRouter

    router = ContextRouter()
    signals = router.analyze_query("What did we discuss last week?")
    # signals.is_temporal == True

    bundle = await router.retrieve_context(
        workspace_id="ws-123", agent_id=1, query="What did we discuss last week?"
    )
    # bundle.total_tokens_estimate <= 4000
"""

import asyncio
import re
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context budget weights (PRD-141 US-011)
# ---------------------------------------------------------------------------
# Each section gets a fixed proportion of the *usable* context window
# (usable = 80% of the raw window, reserving 20% for the model's response).
# Weights sum to 0.80 of the usable window — the remaining 0.20 is slack for
# estimator error and untracked overhead. ``tools`` and ``system_prompt`` are
# reserved headroom the router does not fill itself — they keep the memory
# sections from claiming space the prompt assembler needs.
_CONTEXT_BUDGET_WEIGHTS: Dict[str, float] = {
    "session": 0.10,
    "long_term": 0.15,
    "temporal": 0.10,
    "daily": 0.08,
    "awareness": 0.05,
    "tools": 0.20,
    "system_prompt": 0.12,
}
_USABLE_WINDOW_FRACTION = 0.80


# ---------------------------------------------------------------------------
# ContextSignals — output of query analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextSignals:
    """
    Signals extracted from a user query that guide context assembly.

    Each flag indicates which memory layer(s) should be consulted.
    Multiple flags can be True simultaneously (e.g. a temporal + personal
    fact query like "What was my preference last week?").
    """

    is_temporal: bool = False
    is_personal_fact: bool = False
    is_session_continuation: bool = False
    is_knowledge_query: bool = False
    is_live_data: bool = False
    temporal_window: Optional[Tuple[datetime, datetime]] = None


# ---------------------------------------------------------------------------
# ContextBundle — assembled context ready for system prompt injection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBundle:
    """
    Pre-assembled context bundle returned by ``ContextRouter.retrieve_context()``.

    Each field corresponds to a system-prompt section. The caller formats and
    injects these into the prompt template. ``total_tokens_estimate`` ensures
    the bundle stays within the configured budget.
    """

    session_summary: str = ""
    long_term_memories: Tuple[Dict[str, Any], ...] = ()
    temporal_results: Tuple[Dict[str, Any], ...] = ()
    daily_logs: str = ""
    knowledge_awareness: str = ""
    total_tokens_estimate: int = 0
    signals: Optional[ContextSignals] = None


# ---------------------------------------------------------------------------
# Compiled regex patterns — compiled once at module load
# ---------------------------------------------------------------------------

# Temporal patterns: relative time references
_TEMPORAL_PATTERNS = re.compile(
    r"""(?xi)               # verbose + case-insensitive
    \b(?:
        last\s+(?:week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)
      | yesterday
      | today
      | earlier\s+(?:today|this\s+(?:week|month))
      | this\s+(?:week|month|morning|afternoon|evening)
      | (?:a\s+)?(?:few|couple(?:\s+of)?)\s+(?:days?|weeks?|hours?|months?)\s+ago
      | (?:\d+)\s+(?:days?|weeks?|hours?|months?|minutes?)\s+ago
      | recently
      | the\s+other\s+day
      | last\s+time
      | (?:before|since|after)\s+(?:yesterday|last\s+(?:week|month))
      | previous(?:ly)?
      | in\s+the\s+past\s+(?:\d+\s+)?(?:days?|weeks?|months?)
      | back\s+(?:then|when)
    )\b
    """
)

# Personal fact patterns: references to user identity or preferences
_PERSONAL_FACT_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        my\s+(?:name|email|role|title|company|team|timezone|preference|style)
      | i\s+(?:prefer|like|want|need|use|hate|love|always|usually|tend\s+to)
      | remember\s+(?:when|that|what)\b
      | you\s+(?:told|said|mentioned|know)\s+(?:me|that|about)
      | (?:do\s+you|you)\s+remember
      | what\s+do\s+you\s+know\s+about\s+me
      | we\s+(?:agreed|decided|discussed|talked\s+about)
      | as\s+i\s+(?:said|mentioned)
    )\b
    """
)

# Session continuation patterns: references to current conversation
_SESSION_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        (?:as\s+)?(?:i|we)\s+(?:just|were)\s+(?:said|mentioned|discussed|talking\s+about)
      | going\s+back\s+to
      | (?:continuing|picking\s+up)\s+(?:where|from|on)
      | earlier\s+in\s+(?:this|our)\s+(?:conversation|chat|discussion)
      | what\s+(?:i|we)\s+(?:just|were)\s+(?:saying|discussing|talking)
      | like\s+i\s+(?:said|mentioned)\s+(?:earlier|before|above)
      | (?:that|the)\s+(?:thing|topic|point)\s+(?:we|i)\s+(?:mentioned|discussed)
      | you\s+just\s+(?:said|mentioned|told)
    )\b
    """
)

# Knowledge query patterns: document or policy lookups
_KNOWLEDGE_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        find\s+(?:the|a|that)?\s*(?:doc(?:ument)?|file|policy|guide|report|article)
      | search\s+(?:for|the|our)\b
      | look\s+(?:up|for|into)\b
      | what(?:'s|\s+is)\s+(?:our|the)\s+(?:policy|process|procedure|guideline|standard)
      | (?:is\s+there|do\s+we\s+have)\s+(?:a|any)\s+(?:doc(?:ument)?|guide|policy|wiki)
      | check\s+(?:the|our)\s+(?:docs?|documentation|wiki|knowledge\s+base|confluence)
      | according\s+to\s+(?:our|the)\s+(?:docs?|documentation|policy)
      | where\s+(?:can\s+i|do\s+i)\s+find
    )\b
    """
)

# Live data patterns: metrics, counts, real-time queries
_LIVE_DATA_PATTERNS = re.compile(
    r"""(?xi)
    \b(?:
        (?:current|latest|live|real[\s-]?time|up[\s-]?to[\s-]?date)\s+\w+
      | how\s+many\s+(?:users?|customers?|orders?|transactions?|subscriptions?|signups?)
      | (?:total|average|count\s+of|number\s+of)\s+\w+
      | mrr|arr|revenue|churn|dau|mau|wau
      | (?:users?|customers?)\s+(?:signed?\s+up|registered|churned|converted)
      | what(?:'s|\s+is)\s+(?:our|the|my)\s+(?:mrr|arr|revenue|growth|churn)
      | (?:latest|recent|last)\s+(?:deploy|deployment|release|build|commit)
      | (?:show|give|get)\s+me\s+(?:the\s+)?(?:stats?|statistics|metrics|numbers|data)
      | (?:sales|conversion|retention|signup)\s+(?:rate|count|numbers?|stats?)
    )\b
    """
)


# ---------------------------------------------------------------------------
# Temporal window calculation
# ---------------------------------------------------------------------------

def _compute_temporal_window(
    query: str,
    now: datetime,
) -> Optional[Tuple[datetime, datetime]]:
    """
    Convert a relative time reference in *query* into an absolute
    (start, end) datetime window.

    Returns None if no recognisable temporal reference is found.
    """
    q = query.lower()

    # "yesterday"
    if re.search(r"\byesterday\b", q):
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1) - timedelta(microseconds=1)
        return (start, end)

    # "today" / "earlier today" / "this morning/afternoon/evening"
    if re.search(r"\b(?:today|this\s+(?:morning|afternoon|evening))\b", q):
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return (start, now)

    # "last week"
    if re.search(r"\blast\s+week\b", q):
        start = now - timedelta(days=7)
        return (start, now)

    # "last month"
    if re.search(r"\blast\s+month\b", q):
        start = now - timedelta(days=30)
        return (start, now)

    # "last year"
    if re.search(r"\blast\s+year\b", q):
        start = now - timedelta(days=365)
        return (start, now)

    # "this week"
    if re.search(r"\bthis\s+week\b", q):
        # Monday of current week
        start = (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (start, now)

    # "this month"
    if re.search(r"\bthis\s+month\b", q):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return (start, now)

    # "N days/weeks/hours/months ago" or "a few days ago"
    m = re.search(r"\b(\d+)\s+(days?|weeks?|hours?|months?|minutes?)\s+ago\b", q)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).rstrip("s")
        delta_map = {
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "hour": timedelta(hours=amount),
            "month": timedelta(days=amount * 30),
            "minute": timedelta(minutes=amount),
        }
        delta = delta_map.get(unit)
        if delta:
            return (now - delta, now)

    # "a few days/weeks ago", "couple of days ago"
    m = re.search(r"\b(?:a\s+)?(?:few|couple(?:\s+of)?)\s+(days?|weeks?|hours?|months?)\s+ago\b", q)
    if m:
        unit = m.group(1).rstrip("s")
        # "a few" ≈ 3
        delta_map = {
            "day": timedelta(days=3),
            "week": timedelta(weeks=3),
            "hour": timedelta(hours=3),
            "month": timedelta(days=90),
        }
        delta = delta_map.get(unit)
        if delta:
            return (now - delta, now)

    # "in the past N days/weeks"
    m = re.search(r"\bin\s+the\s+past\s+(\d+)\s+(days?|weeks?|months?)\b", q)
    if m:
        amount = int(m.group(1))
        unit = m.group(2).rstrip("s")
        delta_map = {
            "day": timedelta(days=amount),
            "week": timedelta(weeks=amount),
            "month": timedelta(days=amount * 30),
        }
        delta = delta_map.get(unit)
        if delta:
            return (now - delta, now)

    # "recently" / "the other day" / "last time" — fuzzy 7-day window
    if re.search(r"\b(?:recently|the\s+other\s+day|last\s+time)\b", q):
        return (now - timedelta(days=7), now)

    # Named day of week: "last monday", "last tuesday", etc.
    m = re.search(
        r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q
    )
    if m:
        target_day = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6,
        }[m.group(1)]
        days_back = (now.weekday() - target_day) % 7
        if days_back == 0:
            days_back = 7  # "last Monday" when today is Monday → 7 days ago
        target = (now - timedelta(days=days_back)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return (target, target + timedelta(days=1) - timedelta(microseconds=1))

    return None


# ---------------------------------------------------------------------------
# ContextRouter
# ---------------------------------------------------------------------------

class ContextRouter:
    """
    Pre-LLM context assembly layer.

    Analyses user queries with fast regex patterns (<10 ms) and returns
    ContextSignals that downstream code uses to decide which memory layers
    to fetch and inject into the system prompt.
    """

    def analyze_query(self, query: str) -> ContextSignals:
        """
        Classify a user query into context signals.

        All detection is regex-based — no LLM calls, no I/O.
        Target latency: <10 ms.
        """
        if not query or not query.strip():
            return ContextSignals()

        now = datetime.now(timezone.utc)

        is_temporal = bool(_TEMPORAL_PATTERNS.search(query))
        is_personal_fact = bool(_PERSONAL_FACT_PATTERNS.search(query))
        is_session = bool(_SESSION_PATTERNS.search(query))
        is_knowledge = bool(_KNOWLEDGE_PATTERNS.search(query))
        is_live_data = bool(_LIVE_DATA_PATTERNS.search(query))

        temporal_window = _compute_temporal_window(query, now) if is_temporal else None

        return ContextSignals(
            is_temporal=is_temporal,
            is_personal_fact=is_personal_fact,
            is_session_continuation=is_session,
            is_knowledge_query=is_knowledge,
            is_live_data=is_live_data,
            temporal_window=temporal_window,
        )

    # ------------------------------------------------------------------
    # Context assembly (US-017)
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap token estimate: ~4 chars per token."""
        return len(text) // 4 if text else 0

    @staticmethod
    def _compute_budgets(context_window: Optional[int]) -> Dict[str, int]:
        """Resolve per-section token budgets for context assembly.

        When the model's ``context_window`` is known (a positive int), each
        section gets a fixed proportion of the *usable* window
        (``usable = int(context_window * 0.80)``), so budgets scale with the
        model — a 128K model gets far larger sections than an 8K model.

        When the window is unknown (``None`` / non-positive), fall back to the
        static ``CONTEXT_BUDGET_*`` config values. The config values are
        therefore a fallback only, never the primary source when a window is
        available.

        Returns a dict keyed by section name with token budgets.
        """
        from config import config

        if not context_window or context_window <= 0:
            return {
                "session": config.CONTEXT_BUDGET_SESSION,
                "long_term": config.CONTEXT_BUDGET_LONG_TERM,
                "temporal": config.CONTEXT_BUDGET_TEMPORAL,
                "daily": config.CONTEXT_BUDGET_DAILY,
                "awareness": config.CONTEXT_BUDGET_AWARENESS,
                "tools": config.CONTEXT_BUDGET_TOOLS,
                "system_prompt": config.CONTEXT_BUDGET_SYSTEM_PROMPT,
            }

        usable = int(context_window * _USABLE_WINDOW_FRACTION)
        return {
            name: int(usable * weight)
            for name, weight in _CONTEXT_BUDGET_WEIGHTS.items()
        }

    @staticmethod
    def _truncate_to_budget(text: str, token_budget: int) -> str:
        """Truncate *text* so its estimated token count fits within *token_budget*."""
        max_chars = token_budget * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    @staticmethod
    def _memories_to_text(memories: List[Dict[str, Any]], token_budget: int) -> Tuple[List[Dict[str, Any]], str]:
        """
        Convert memory dicts to a text block and trim to fit *token_budget*.

        Returns (kept_memories, text_block).
        """
        if not memories:
            return [], ""
        lines: List[str] = []
        kept: List[Dict[str, Any]] = []
        char_budget = token_budget * 4
        total = 0
        for mem in memories:
            content = mem.get("memory") or mem.get("content") or ""
            if not content:
                continue
            line = f"- {content}"
            if total + len(line) > char_budget:
                break
            lines.append(line)
            kept.append(mem)
            total += len(line) + 1  # +1 for newline
        return kept, "\n".join(lines)

    async def retrieve_context(
        self,
        workspace_id: str,
        agent_id: int,
        query: str,
        conversation_id: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> ContextBundle:
        """
        Assemble a budget-constrained context bundle by fetching from L1/L2/L3
        based on query signals.

        Fetch strategy (driven by ``analyze_query`` signals):
          - **session_continuation** or default with conversation_id → L1 session
          - **temporal** → L2 short-term with time filter
          - **personal_fact** or default → L3 long-term via Mem0 (cached)
          - **knowledge_query** / **live_data** → awareness text only (no pre-fetch)

        Default (no strong signal): L3 top-5 memories + L1 session summary.

        All layer fetches are concurrent via ``asyncio.gather``.
        Any single-layer failure is logged and skipped — never breaks the bundle.
        """
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()
        signals = self.analyze_query(query)

        budgets = self._compute_budgets(context_window)
        budget_session = budgets["session"]
        budget_long_term = budgets["long_term"]
        budget_temporal = budgets["temporal"]
        budget_daily = budgets["daily"]
        budget_awareness = budgets["awareness"]

        # ----- Determine which fetches to launch -----
        fetch_session = (
            conversation_id is not None
            and (signals.is_session_continuation or not any([
                signals.is_temporal,
                signals.is_knowledge_query,
                signals.is_live_data,
            ]))
        )
        fetch_long_term = (
            signals.is_personal_fact
            or not any([
                signals.is_temporal,
                signals.is_knowledge_query,
                signals.is_live_data,
            ])
        )
        fetch_temporal = signals.is_temporal and signals.temporal_window is not None
        # Daily logs on default path (no strong signal)
        fetch_daily = not any([
            signals.is_temporal,
            signals.is_personal_fact,
            signals.is_session_continuation,
            signals.is_knowledge_query,
            signals.is_live_data,
        ])

        # ----- Launch concurrent fetches -----
        async def _noop():
            return None

        session_task = (
            self._safe_fetch("L1 session", service.get_session(workspace_id, conversation_id))
            if fetch_session else _noop()
        )
        long_term_task = (
            self._safe_fetch("L3 long-term", service.search_long_term(workspace_id, query, agent_id=agent_id, limit=5))
            if fetch_long_term else _noop()
        )
        temporal_task = (
            self._safe_fetch("L2 temporal", service.search_short_term(workspace_id, query, days=self._window_days(signals.temporal_window)))
            if fetch_temporal else _noop()
        )
        daily_task = (
            self._safe_fetch("daily logs", service.get_all_daily_logs(workspace_id, limit=10))
            if fetch_daily else _noop()
        )

        session_result, lt_result, temporal_result, daily_result = await asyncio.gather(
            session_task, long_term_task, temporal_task, daily_task,
        )

        # ----- Assemble bundle with budget constraints -----

        # Session summary
        session_text = ""
        if session_result is not None:
            raw_summary = getattr(session_result, "summary", "") or ""
            exchange_count = getattr(session_result, "exchange_count", 0)
            if raw_summary:
                session_text = self._truncate_to_budget(
                    f"Conversation so far ({exchange_count} exchanges):\n{raw_summary}",
                    budget_session,
                )

        # Long-term memories
        lt_memories: List[Dict[str, Any]] = lt_result if isinstance(lt_result, list) else []
        kept_lt, lt_text = self._memories_to_text(lt_memories, budget_long_term)

        # Temporal results
        temporal_memories: List[Dict[str, Any]] = temporal_result if isinstance(temporal_result, list) else []
        kept_temporal, temporal_text = self._memories_to_text(temporal_memories, budget_temporal)

        # Daily logs
        daily_text = ""
        if daily_result and isinstance(daily_result, list):
            daily_lines: List[str] = []
            for mem in daily_result:
                content = mem.get("memory") or mem.get("content") or ""
                if content:
                    daily_lines.append(f"- {content}")
            daily_text = self._truncate_to_budget("\n".join(daily_lines), budget_daily)

        # Knowledge awareness (dynamic per-workspace capability map)
        awareness_text = ""
        if signals.is_knowledge_query or signals.is_live_data:
            try:
                awareness_text = await self.build_knowledge_awareness(workspace_id)
            except Exception:
                logger.warning(
                    "[ContextRouter] build_knowledge_awareness failed, using fallback",
                    exc_info=True,
                )
                awareness_text = self._build_fallback_awareness(signals)
            awareness_text = self._truncate_to_budget(awareness_text, budget_awareness)

        # Total token estimate
        total_tokens = sum(
            self._estimate_tokens(t)
            for t in (session_text, lt_text, temporal_text, daily_text, awareness_text)
        )

        return ContextBundle(
            session_summary=session_text,
            long_term_memories=tuple(kept_lt),
            temporal_results=tuple(kept_temporal),
            daily_logs=daily_text,
            knowledge_awareness=awareness_text,
            total_tokens_estimate=total_tokens,
            signals=signals,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_fetch(label: str, coro):
        """Run a coroutine and return None on failure instead of raising."""
        try:
            return await coro
        except Exception:
            logger.warning(
                "[ContextRouter] %s fetch failed — skipping",
                label,
                exc_info=True,
            )
            return None

    @staticmethod
    def _window_days(window: Optional[Tuple[datetime, datetime]]) -> int:
        """Convert a temporal window to a number-of-days value for search_short_term."""
        if window is None:
            return 7
        start, end = window
        delta = end - start
        return max(1, delta.days + 1)

    @staticmethod
    def _build_fallback_awareness(signals: ContextSignals) -> str:
        """
        Build a static fallback awareness text when dynamic query fails.

        Used only when ``build_knowledge_awareness()`` cannot reach the DB or
        Redis cache.
        """
        lines = ["## What You Can Look Up"]
        lines.append("You have access to organizational knowledge. Don't guess — look things up:")
        if signals.is_knowledge_query:
            lines.append("- **Company documents**: Use `search_knowledge` to search uploaded docs, policies, guides")
        if signals.is_live_data:
            lines.append("- **Business data**: Use `query_data` to ask questions about metrics, users, revenue, etc.")
        lines.append("- **Past conversations**: Your memories include recent interactions — check before asking the user to repeat themselves")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dynamic knowledge awareness (US-018)
    # ------------------------------------------------------------------

    async def build_knowledge_awareness(self, workspace_id: str) -> str:
        """
        Build a dynamic per-workspace capability map describing available
        knowledge sources (connected databases, documents, tools).

        Cached in Redis for ``MEMORY_AWARENESS_CACHE_TTL_SECONDS`` (default
        10 min) since workspace capabilities change infrequently.

        Returns:
            A ``## What You Can Look Up`` text block (<200 tokens).
        """
        from modules.memory.unified_memory_service import get_unified_memory_service

        service = get_unified_memory_service()

        # --- Try Redis cache first ---
        cached = await self._get_cached_awareness(service, workspace_id)
        if cached is not None:
            return cached

        # --- Cache miss: query DB for workspace capabilities ---
        try:
            capabilities = await asyncio.get_event_loop().run_in_executor(
                None, self._query_workspace_capabilities, workspace_id
            )
        except Exception:
            logger.warning(
                "[ContextRouter] build_knowledge_awareness DB query failed ws=%s",
                workspace_id,
                exc_info=True,
            )
            capabilities = {}

        text = self._format_awareness_text(capabilities)

        # Cache the result (fire-and-forget)
        asyncio.ensure_future(self._set_cached_awareness(service, workspace_id, text))

        return text

    @staticmethod
    def _query_workspace_capabilities(workspace_id: str) -> Dict[str, Any]:
        """
        Query Postgres for workspace knowledge sources (synchronous).

        Returns a dict with keys: databases, doc_count, tools.
        Runs in an executor — safe for the async context.
        """
        from contextlib import suppress
        from core.database.database import get_db_session
        from sqlalchemy import func

        result: Dict[str, Any] = {
            "databases": [],
            "doc_count": 0,
            "tools": [],
        }

        with suppress(Exception):
            with get_db_session() as db:
                # Connected external databases
                try:
                    from core.models.database_knowledge import DatabaseKnowledgeSource

                    db_sources = (
                        db.query(
                            DatabaseKnowledgeSource.name,
                            DatabaseKnowledgeSource.dialect,
                        )
                        .filter(
                            DatabaseKnowledgeSource.workspace_id == workspace_id,
                            DatabaseKnowledgeSource.is_active.is_(True),
                        )
                        .all()
                    )
                    result["databases"] = [
                        {"name": row.name, "dialect": row.dialect}
                        for row in db_sources
                    ]
                except Exception:
                    logger.debug(
                        "[ContextRouter] _query_workspace_capabilities databases failed",
                        exc_info=True,
                    )

                # Document count
                try:
                    from core.models.core import Document

                    doc_count = (
                        db.query(func.count(Document.id))
                        .filter(
                            Document.workspace_id == workspace_id,
                            Document.status == "processed",
                        )
                        .scalar()
                    ) or 0
                    result["doc_count"] = doc_count
                except Exception:
                    logger.debug(
                        "[ContextRouter] _query_workspace_capabilities documents failed",
                        exc_info=True,
                    )

                # Connected tools (Composio connections)
                try:
                    from core.models.composio import ComposioConnection, ComposioEntity

                    tool_rows = (
                        db.query(ComposioConnection.app_name)
                        .join(
                            ComposioEntity,
                            ComposioConnection.entity_id == ComposioEntity.id,
                        )
                        .filter(
                            ComposioEntity.workspace_id == workspace_id,
                            ComposioConnection.status == "active",
                        )
                        .distinct()
                        .all()
                    )
                    result["tools"] = [row.app_name.title() for row in tool_rows]
                except Exception:
                    logger.debug(
                        "[ContextRouter] _query_workspace_capabilities tools failed",
                        exc_info=True,
                    )

        return result

    @staticmethod
    def _format_awareness_text(capabilities: Dict[str, Any]) -> str:
        """
        Format a ``## What You Can Look Up`` text block from capability data.

        Output is kept under ~200 tokens.
        """
        databases = capabilities.get("databases", [])
        doc_count = capabilities.get("doc_count", 0)
        tools = capabilities.get("tools", [])

        lines = ["## What You Can Look Up"]
        lines.append(
            "You have access to organizational knowledge. Don't guess — look things up:"
        )

        if doc_count > 0:
            lines.append(
                f"- **Company documents** ({doc_count} indexed): Use `search_knowledge` to search uploaded docs, policies, guides"
            )
        else:
            lines.append(
                "- **Company documents**: Use `search_knowledge` to search uploaded docs, policies, guides"
            )

        if databases:
            db_descriptions = ", ".join(
                f"{d['name']} ({d['dialect']})" for d in databases[:5]
            )
            lines.append(
                f"- **Connected databases**: {db_descriptions}. Use `query_data` to ask questions about business metrics, users, revenue"
            )
        else:
            lines.append(
                "- **Business data**: Use `query_data` to ask questions about metrics, users, revenue, etc."
            )

        if tools:
            tool_list = ", ".join(tools[:10])
            lines.append(
                f"- **External tools**: Connected — {tool_list}. Use your connected tools for tasks"
            )

        lines.append(
            "- **Past conversations**: Your memories include recent interactions — check before asking the user to repeat themselves"
        )

        return "\n".join(lines)

    # --- Awareness cache helpers ---

    @staticmethod
    async def _get_cached_awareness(
        service: Any, workspace_id: str
    ) -> Optional[str]:
        """Read awareness text from Redis cache. Returns None on miss."""
        from modules.memory.unified_memory_service import MemoryNamespace

        redis_client = service._get_redis()
        if redis_client is None:
            return None
        ns = MemoryNamespace(workspace_id=str(workspace_id))
        key = ns.awareness()
        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()
            raw: Optional[str] = await loop.run_in_executor(None, conn.get, key)
            if raw is not None:
                logger.debug(
                    "[ContextRouter] awareness CACHE HIT key=%s", key
                )
            return raw
        except Exception:
            logger.debug(
                "[ContextRouter] _get_cached_awareness failed key=%s",
                key,
                exc_info=True,
            )
            return None

    @staticmethod
    async def _set_cached_awareness(
        service: Any, workspace_id: str, text: str
    ) -> None:
        """Write awareness text to Redis with configured TTL."""
        from config import config
        from modules.memory.unified_memory_service import MemoryNamespace

        redis_client = service._get_redis()
        if redis_client is None:
            return
        ns = MemoryNamespace(workspace_id=str(workspace_id))
        key = ns.awareness()
        ttl = config.MEMORY_AWARENESS_CACHE_TTL_SECONDS
        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()
            await loop.run_in_executor(
                None, lambda: conn.setex(key, ttl, text)
            )
            logger.debug(
                "[ContextRouter] _set_cached_awareness key=%s ttl=%ds",
                key,
                ttl,
            )
        except Exception:
            logger.debug(
                "[ContextRouter] _set_cached_awareness failed key=%s",
                key,
                exc_info=True,
            )
