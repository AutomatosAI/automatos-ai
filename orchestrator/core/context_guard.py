"""
Context Window Guard
====================

Monitors token usage before each LLM call and auto-compacts the conversation
when it approaches the model's context limit.

Strategy:
1. Count tokens in the full message payload (system + user + assistant + tool)
2. If below 80% of model context → pass through unchanged
3. If above 80% → compact: summarize older turns, keep recent context
4. Flush key facts to Mem0 before discarding messages

This prevents context_length_exceeded errors and keeps conversations going
indefinitely without manual truncation.

Usage:
    guard = ContextGuard()
    messages, was_compacted = await guard.check_and_compact(
        messages=llm_messages,
        model_name="gpt-4",
        llm_manager=agent_runtime.llm_manager,
        workspace_id="ws123",
        agent_id=1,
    )
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token counting (reuses tiktoken, already a project dependency)
# ---------------------------------------------------------------------------

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None
    logger.warning("[ContextGuard] tiktoken unavailable — falling back to word-based estimation")


def count_tokens(text: str) -> int:
    """Count tokens in a text string."""
    if not text:
        return 0
    if _encoding:
        return len(_encoding.encode(text))
    # Rough fallback: ~4 chars per token
    return max(1, len(text) // 4)


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """
    Estimate total tokens across all messages.

    Accounts for role/content overhead (~4 tokens per message for OpenAI format).
    """
    total = 0
    for msg in messages:
        total += 4  # message overhead (role, separators)
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            # Multi-part content (vision, etc.)
            for part in content:
                if isinstance(part, dict) and part.get("text"):
                    total += count_tokens(part["text"])
        # Tool calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            total += count_tokens(json.dumps(tool_calls))
    total += 2  # reply priming
    return total


def count_tool_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """Estimate tokens consumed by the tools/functions parameter."""
    if not tools:
        return 0
    return count_tokens(json.dumps(tools))


# ---------------------------------------------------------------------------
# Model context window lookup
# ---------------------------------------------------------------------------

# Default for models not in the DB registry. Modern models are 128K+;
# using a low fallback (e.g. 8K) causes the guard to aggressively strip
# tools and compact messages when it shouldn't.
_DEFAULT_CONTEXT_WINDOW = 128_000


def get_context_window(model_name: str, db_session=None) -> int:
    """
    Look up the context window for a model from the model registry DB.

    Falls back to a safe 128K default if the model isn't registered.
    """
    if not model_name:
        return _DEFAULT_CONTEXT_WINDOW

    if db_session:
        try:
            from core.models import LLMModel

            # Try exact match first (covers both "openai/gpt-5.4" and "gpt-4o")
            row = (
                db_session.query(LLMModel.context_window)
                .filter(LLMModel.model_id == model_name)
                .first()
            )
            if row and row[0]:
                return int(row[0])

            # Try without provider prefix: "openai/gpt-5.4" → "gpt-5.4"
            if "/" in model_name:
                bare = model_name.split("/", 1)[1]
                row = (
                    db_session.query(LLMModel.context_window)
                    .filter(LLMModel.model_id == bare)
                    .first()
                )
                if row and row[0]:
                    return int(row[0])
        except Exception:
            pass

    return _DEFAULT_CONTEXT_WINDOW


# ---------------------------------------------------------------------------
# Context Guard
# ---------------------------------------------------------------------------

# Thresholds
COMPACT_THRESHOLD = 0.80   # Compact when >80% of context used
KEEP_RECENT_TURNS = 6      # Always keep the last N user+assistant messages
SUMMARY_MAX_TOKENS = 500   # Max tokens for the compaction summary

# PRD-123 Pattern #7: Proactive compaction thresholds
PROACTIVE_COMPACT_AFTER_TURNS = int(
    __import__("os").getenv("PROACTIVE_COMPACT_AFTER_TURNS", "8")
)
PROACTIVE_COMPACT_KEEP_RECENT = int(
    __import__("os").getenv("PROACTIVE_COMPACT_KEEP_RECENT", "4")
)


class ContextGuard:
    """
    Monitors and auto-compacts conversation context before LLM calls.
    """

    async def check_and_compact(
        self,
        messages: List[Dict[str, Any]],
        model_name: str,
        llm_manager: Any,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        db_session=None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], bool, Optional[List[Dict[str, Any]]]]:
        """
        Check if messages + tools fit within context window; compact if needed.

        Args:
            messages: LLM-formatted messages (system + user/assistant/tool)
            model_name: Current model name for context window lookup
            llm_manager: LLMManager instance for summarization calls
            workspace_id: For memory flush
            agent_id: For memory flush
            db_session: Optional DB session for model registry lookup
            tools: Optional list of tool schemas (counted toward context budget)

        Returns:
            (messages, was_compacted, tools) — tools may be None if they don't fit
        """
        context_window = get_context_window(model_name, db_session)
        tool_tokens = count_tool_tokens(tools)
        current_tokens = count_message_tokens(messages)
        total_tokens = current_tokens + tool_tokens
        threshold = int(context_window * COMPACT_THRESHOLD)

        logger.debug(
            "[ContextGuard] tokens=%d (msgs=%d tools=%d) / %d (%.0f%% of %d window)",
            total_tokens, current_tokens, tool_tokens, threshold,
            (total_tokens / context_window) * 100, context_window,
        )

        # If tools alone exceed 60% of context, drop them entirely
        if tools and tool_tokens > int(context_window * 0.6):
            logger.warning(
                "[ContextGuard] Tools alone use %d tokens (%.0f%% of %d context) — "
                "dropping tools to fit within context window",
                tool_tokens, (tool_tokens / context_window) * 100, context_window,
            )
            tools = None
            total_tokens = current_tokens

        if total_tokens <= threshold:
            return messages, False, tools

        logger.info(
            "[ContextGuard] Context at %d/%d tokens (%.0f%%) — compacting",
            total_tokens, context_window,
            (total_tokens / context_window) * 100,
        )

        compacted = await self._compact(
            messages=messages,
            llm_manager=llm_manager,
            workspace_id=workspace_id,
            agent_id=agent_id,
        )

        new_tokens = count_message_tokens(compacted)
        logger.info(
            "[ContextGuard] Compacted: %d → %d tokens (saved %d)",
            current_tokens, new_tokens, current_tokens - new_tokens,
        )

        return compacted, True, tools

    async def _compact(
        self,
        messages: List[Dict[str, Any]],
        llm_manager: Any,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Compact messages by summarizing older turns.

        Strategy:
        1. Split messages into: system_msgs | old_turns | recent_turns
        2. Summarize old_turns into a single context message
        3. Flush key facts from old_turns to Mem0
        4. Return: system_msgs + [summary] + recent_turns
        """
        # Separate system messages (always keep) from conversation turns
        system_msgs = []
        conversation = []
        for msg in messages:
            if msg.get("role") == "system":
                system_msgs.append(msg)
            else:
                conversation.append(msg)

        # If conversation is short enough, keep everything
        if len(conversation) <= KEEP_RECENT_TURNS:
            return messages

        # Split: old turns (to summarize) | recent turns (to keep)
        old_turns = conversation[:-KEEP_RECENT_TURNS]
        recent_turns = conversation[-KEEP_RECENT_TURNS:]

        # Build text from old turns for summarization
        old_text = self._turns_to_text(old_turns)

        # Summarize old turns
        summary = await self._summarize(old_text, llm_manager)

        # Flush key facts to memory (fire-and-forget)
        if workspace_id:
            await self._flush_to_memory(old_text, workspace_id, agent_id)

        # Construct compacted message list
        summary_msg = {
            "role": "system",
            "content": (
                "## Earlier Conversation Summary\n"
                "The following is a summary of the earlier part of this conversation. "
                "Use it for context but prioritize the recent messages below.\n\n"
                f"{summary}"
            ),
        }

        return system_msgs + [summary_msg] + recent_turns

    def _turns_to_text(self, turns: List[Dict[str, Any]]) -> str:
        """Convert message turns into readable text for summarization."""
        parts = []
        for msg in turns:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    async def _summarize(self, text: str, llm_manager: Any) -> str:
        """Summarize conversation text using the LLM."""
        if not text.strip():
            return "No significant conversation history."

        # Truncate input if very long (avoid context overflow in the summary call itself)
        max_input_chars = 12000
        if len(text) > max_input_chars:
            text = text[:max_input_chars] + "\n... [truncated]"

        summary_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a conversation summarizer. Produce a concise summary of the "
                    "conversation below. Focus on: key decisions, user preferences, "
                    "action items, and important facts. Keep under 300 words."
                ),
            },
            {
                "role": "user",
                "content": f"Summarize this conversation:\n\n{text}",
            },
        ]

        try:
            response = await llm_manager.generate_response(
                messages=summary_prompt,
                tools=None,
            )
            return response.content or "Summary generation failed."
        except Exception as exc:
            logger.warning("[ContextGuard] Summarization failed: %s", exc)
            # Fallback: extract last few lines
            lines = text.strip().split("\n")
            return "Key points from earlier conversation:\n" + "\n".join(lines[-8:])

    async def _flush_to_memory(
        self,
        text: str,
        workspace_id: str,
        agent_id: Optional[int] = None,
    ):
        """Flush key facts from compacted turns to Mem0 for long-term retention."""
        try:
            from consumers.chatbot.smart_memory import get_smart_memory_manager

            memory_mgr = get_smart_memory_manager()
            # Extract a compact summary for memory storage
            key_facts = text[:2000]  # First 2000 chars contain most relevant context
            await memory_mgr.store_conversation(
                workspace_id=workspace_id,
                agent_id=agent_id,
                user_message="[Context compaction — key facts from earlier conversation]",
                assistant_response=key_facts,
            )
            logger.info("[ContextGuard] Flushed key facts to Mem0")
        except Exception as exc:
            logger.warning("[ContextGuard] Memory flush failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# PRD-123 Pattern #7: Proactive Transcript Compaction
# ---------------------------------------------------------------------------


async def maybe_compact_session(
    messages: List[Dict[str, Any]],
    turn_count: int,
    llm_manager: Any,
    workspace_id: Optional[str] = None,
    agent_id: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Proactively compact conversation every N turns to prevent context bloat.

    Unlike ContextGuard (reactive, triggers at 80% capacity), this function
    compacts based on turn count — ensuring long conversations stay lean
    even when individual messages are small.

    Args:
        messages: Full message list (system + conversation).
        turn_count: Number of user turns in this session so far.
        llm_manager: LLMManager for summarization.
        workspace_id: For memory flush.
        agent_id: For memory flush.

    Returns:
        (messages, was_compacted) — compacted messages if threshold met.
    """
    if turn_count < PROACTIVE_COMPACT_AFTER_TURNS:
        return messages, False

    # Only compact on exact multiples to avoid compacting every turn
    if turn_count % PROACTIVE_COMPACT_AFTER_TURNS != 0:
        return messages, False

    # Separate system messages from conversation
    system_msgs = []
    conversation = []
    for msg in messages:
        if msg.get("role") == "system":
            system_msgs.append(msg)
        else:
            conversation.append(msg)

    keep = PROACTIVE_COMPACT_KEEP_RECENT * 2  # user + assistant pairs
    if len(conversation) <= keep:
        return messages, False

    old_turns = conversation[:-keep]
    recent_turns = conversation[-keep:]

    # Build text and summarize
    guard = ContextGuard()
    old_text = guard._turns_to_text(old_turns)
    summary = await guard._summarize(old_text, llm_manager)

    # Flush to memory
    if workspace_id:
        await guard._flush_to_memory(old_text, workspace_id, agent_id)

    summary_msg = {
        "role": "system",
        "content": (
            "## Earlier Conversation Summary\n"
            "The following is a summary of the earlier part of this conversation. "
            "Use it for context but prioritize the recent messages below.\n\n"
            f"{summary}"
        ),
    }

    compacted = system_msgs + [summary_msg] + recent_turns
    old_tokens = count_message_tokens(messages)
    new_tokens = count_message_tokens(compacted)
    logger.info(
        "[ProactiveCompact] Turn %d: compacted %d → %d tokens (saved %d)",
        turn_count, old_tokens, new_tokens, old_tokens - new_tokens,
    )

    return compacted, True
