"""PRD-206 S2 — thread checkpoints: the LLM summarisation L1 never got.

A checkpoint distills a conversation thread into ``chats.summary``
(``{topic, decisions[], open_questions[], last_summary, next_step,
updated_at, checkpointed_at, trigger}``) and stores the NEW decisions and
open loops as typed L3 memories (via the S1 write contract, linked by
``chat_id``). This is the "Phase 2 adds LLM summarisation" promise at
``unified_memory_service.update_session`` finally delivered — on the chat
row, where resume (S3) can read it.

Two triggers, both silent (Q3):
  - ``idle_sweep`` — the memory-jobs scheduler checkpoints recently-idle
    threads (services/memory_jobs.py),
  - ``on_demand`` — the ``platform_checkpoint_thread`` tool when the user
    says "save where we are".

Idempotence: re-checkpointing UPDATES ``chats.summary`` (a new dict — never
an in-place JSONB mutation, the PRD-220 ORM lesson) and stores only items
not already present in the prior checkpoint; an unchanged ``last_summary``
writes no new ``thread_summary`` memory. The Q3 exclusion validator applies
to every stored item — this is a memory-write path like any other.

The pure pieces (prompt build, parse, compose, plan) are separated from the
I/O (`run_thread_checkpoint`) so they unit-test with plain dicts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from modules.memory.write_contract import (
    SOURCE_TYPE_DISTILLED,
    build_memory_metadata,
    violates_exclusions,
)

logger = logging.getLogger(__name__)

# Rendering caps for the checkpoint transcript — the distill model is the
# cheap tier; a runaway thread must not become a runaway prompt.
MAX_MESSAGES = 40
PER_MESSAGE_CHARS = 500
TOTAL_TRANSCRIPT_CHARS = 8000


# ---------------------------------------------------------------------------
# Pure: transcript rendering
# ---------------------------------------------------------------------------

def extract_message_text(parts: Any) -> str:
    """Join the text parts of an AI-SDK ``messages.parts`` payload."""
    if isinstance(parts, str):
        return parts
    if not isinstance(parts, list):
        return ""
    texts = [
        str(p.get("text", ""))
        for p in parts
        if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
    ]
    return "\n".join(t for t in texts if t)


def render_transcript(messages: List[Dict[str, Any]]) -> str:
    """Render (role, text) message dicts to the prompt transcript, capped."""
    lines: List[str] = []
    for msg in messages[-MAX_MESSAGES:]:
        role = msg.get("role", "user")
        text = extract_message_text(msg.get("parts", msg.get("content", "")))
        if not text:
            continue
        label = "User" if role == "user" else "Auto"
        lines.append(f"{label}: {text[:PER_MESSAGE_CHARS]}")
    return "\n".join(lines)[-TOTAL_TRANSCRIPT_CHARS:]


# ---------------------------------------------------------------------------
# Pure: prompt + parse
# ---------------------------------------------------------------------------

def build_checkpoint_prompt(transcript: str, prior_summary: Optional[Dict]) -> str:
    prior_block = ""
    if prior_summary:
        prior_block = (
            "A previous checkpoint of this thread exists:\n"
            f"{json.dumps({k: prior_summary.get(k) for k in ('topic', 'last_summary', 'next_step', 'decisions', 'open_questions')}, ensure_ascii=False)}\n"
            "Carry forward items that are still true or still open; do NOT "
            "re-list decisions or questions from it unless they changed. "
            "Resolved open questions are simply omitted.\n\n"
        )
    return (
        "You are checkpointing a conversation thread for an AI assistant "
        "(\"Auto\") so it can later answer \"where did we leave off?\".\n\n"
        f"{prior_block}"
        "From the transcript below, produce ONLY a JSON object with exactly "
        "these keys:\n"
        '- "topic": a short (<=8 words) name for what this thread is about\n'
        '- "last_summary": 2-3 sentences on where the conversation got to\n'
        '- "next_step": the single most likely next action, or null\n'
        '- "decisions": array of standalone statements of decisions made, '
        "each with its reason when stated\n"
        '- "open_questions": array of standalone statements of things left '
        "unresolved or promised for later\n\n"
        "Write every item so it makes sense without the transcript. NEVER "
        "include secrets, credentials, passwords, API keys or tokens, card "
        "or bank numbers, or one-time codes in any field.\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Checkpoint (JSON object):"
    )


def _string_list(value: Any) -> List[str]:
    """Normalise an LLM array field to a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("fact") or "").strip()
        else:
            text = ""
        if text:
            out.append(text)
    return out


def parse_checkpoint(content: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM output into the checkpoint shape, or None.

    Tolerates prose and ``` fences around the object (same tolerance as the
    fact distiller). Normalises: strings stripped, arrays coerced to string
    lists, missing keys defaulted.
    """
    if not content:
        return None
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    candidate = text
    if not (candidate.startswith("{") and candidate.endswith("}")):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    next_step = parsed.get("next_step")
    next_step = str(next_step).strip() if isinstance(next_step, str) and next_step.strip() else None
    return {
        "topic": str(parsed.get("topic") or "").strip(),
        "last_summary": str(parsed.get("last_summary") or "").strip(),
        "next_step": next_step,
        "decisions": _string_list(parsed.get("decisions")),
        "open_questions": _string_list(parsed.get("open_questions")),
    }


# ---------------------------------------------------------------------------
# Pure: compose the new summary + plan the typed memory writes
# ---------------------------------------------------------------------------

def compose_summary(parsed: Dict[str, Any], trigger: str) -> Dict[str, Any]:
    """The new ``chats.summary`` value — a NEW dict every time.

    ``checkpointed_at`` (epoch seconds) is the sweep's staleness watermark:
    comparable in SQL against ``EXTRACT(EPOCH FROM updated_at)`` without the
    ISO-vs-postgres text-format trap.
    """
    now = datetime.now(timezone.utc)
    return {
        "topic": parsed["topic"],
        "last_summary": parsed["last_summary"],
        "next_step": parsed["next_step"],
        "decisions": list(parsed["decisions"]),
        "open_questions": list(parsed["open_questions"]),
        "updated_at": now.isoformat(),
        "checkpointed_at": time.time(),
        "trigger": trigger,
    }


def plan_typed_memories(
    parsed: Dict[str, Any],
    prior: Optional[Dict[str, Any]],
    *,
    owner: Optional[str],
    chat_id: str,
    workspace_id: str,
    trigger: str,
) -> List[Dict[str, Any]]:
    """Which L3 memories this checkpoint should write — pure, deterministic.

    Only items NOT present in the prior checkpoint are planned (exact-string
    diff — re-checkpointing an unchanged thread plans nothing), and every
    item passes the Q3 exclusion validator or is dropped with a log line.
    Returns ``[{content, metadata, subject_id}]`` ready for the L3 store.
    """
    prior = prior or {}
    extra = {"workspace_id": workspace_id, "source": "thread_checkpoint", "trigger": trigger}
    planned: List[Dict[str, Any]] = []

    def _plan(content: str, fact_type: str, importance: float) -> None:
        violation = violates_exclusions(content)
        if violation:
            logger.warning(
                "[ThreadCheckpoint] Exclusion validator dropped a %s (rule=%s)",
                fact_type, violation,
            )
            return
        planned.append({
            "content": content,
            "subject_id": owner,
            "metadata": build_memory_metadata(
                fact_type=fact_type,
                importance=importance,
                source_type=SOURCE_TYPE_DISTILLED,
                owner=owner,
                chat_id=chat_id,
                extra=extra,
            ),
        })

    prior_decisions = set(prior.get("decisions") or [])
    for decision in parsed["decisions"]:
        if decision not in prior_decisions:
            _plan(decision, "decision", 0.7)

    prior_questions = set(prior.get("open_questions") or [])
    for question in parsed["open_questions"]:
        if question not in prior_questions:
            _plan(question, "open_loop", 0.6)

    if parsed["last_summary"] and parsed["last_summary"] != (prior.get("last_summary") or ""):
        topic = parsed["topic"] or "This thread"
        content = f"{topic} — where we left off: {parsed['last_summary']}"
        if parsed["next_step"]:
            content += f" Next step: {parsed['next_step']}"
        _plan(content, "thread_summary", 0.5)

    return planned


# ---------------------------------------------------------------------------
# I/O: the checkpoint run
# ---------------------------------------------------------------------------

def _load_chat(db, workspace_id: str, chat_id: str):
    from core.models.core import Chat

    return (
        db.query(Chat)
        .filter(Chat.id == chat_id, Chat.workspace_id == workspace_id)
        .first()
    )


def _load_messages(db, chat_id: str) -> List[Dict[str, Any]]:
    from core.models.core import Message

    rows = (
        db.query(Message)
        .filter(Message.chat_id == chat_id)
        .order_by(Message.created_at.desc())
        .limit(MAX_MESSAGES)
        .all()
    )
    return [{"role": r.role, "parts": r.parts} for r in reversed(rows)]


async def _distill_checkpoint(
    prompt: str, *, workspace_id: str, agent_id: Optional[int]
) -> Optional[str]:
    """One cheap-tier LLM call; None on failure (caller stores nothing)."""
    try:
        # Imported per call (not at module top) so tests can monkeypatch
        # ``core.llm.create_llm_manager`` — same idiom as the fact distiller.
        from core.llm import create_llm_manager
        from config import config

        llm = create_llm_manager(
            service_name="memory_integration",
            model=config.MEMORY_DISTILL_MODEL,
            workspace_id=workspace_id,
            agent_id=agent_id,
            request_type="thread_checkpoint",
        )
        response = await llm.generate_response(
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content if hasattr(response, "content") else str(response)
    except Exception:
        logger.warning("[ThreadCheckpoint] LLM call failed", exc_info=True)
        return None


async def run_thread_checkpoint(
    db,
    *,
    workspace_id: str,
    chat_id: str,
    trigger: str,
    min_messages: int = 4,
) -> Dict[str, Any]:
    """Checkpoint one thread: update ``chats.summary`` + store new typed memories.

    Never raises — the idle sweep and the tool handler both consume the
    ``{success, ...}`` dict. A failed LLM/parse stores NOTHING (no fallback
    summary — same honesty rule as the fact distiller).
    """
    chat = _load_chat(db, workspace_id, chat_id)
    if chat is None:
        return {"success": False, "error": "chat not found in this workspace"}

    messages = _load_messages(db, chat_id)
    if len(messages) < min_messages:
        return {"success": False, "error": "thread too short to checkpoint", "skipped": True}

    transcript = render_transcript(messages)
    if not transcript:
        return {"success": False, "error": "thread has no text content", "skipped": True}

    prior = dict(chat.summary) if isinstance(chat.summary, dict) else None
    raw = await _distill_checkpoint(
        build_checkpoint_prompt(transcript, prior),
        workspace_id=workspace_id,
        agent_id=None,
    )
    parsed = parse_checkpoint(raw) if raw is not None else None
    if parsed is None:
        return {"success": False, "error": "checkpoint distill failed"}

    owner = f"user:{chat.user_id}" if getattr(chat, "user_id", None) else None
    planned = plan_typed_memories(
        parsed, prior,
        owner=owner, chat_id=str(chat_id),
        workspace_id=str(workspace_id), trigger=trigger,
    )

    # New dict assignment — never mutate the ORM JSONB in place (PRD-220).
    chat.summary = compose_summary(parsed, trigger)
    try:
        db.commit()
    except Exception:
        db.rollback()
        logger.error("[ThreadCheckpoint] chats.summary write failed", exc_info=True)
        return {"success": False, "error": "summary write failed"}

    stored = 0
    if planned:
        try:
            from modules.memory.unified_memory_service import get_unified_memory_service

            service = get_unified_memory_service()
            for item in planned:
                results = await service.store_two_tier(
                    workspace_id=str(workspace_id),
                    messages=[{"role": "user", "content": item["content"]}],
                    agent_id=None,
                    tier="global",
                    metadata=item["metadata"],
                    subject_id=item["subject_id"],
                )
                if any(r[1] and not r[1].get("error") for r in results):
                    stored += 1
        except Exception:
            # The summary is already committed — partial success is reported
            # honestly rather than rolled into a blanket failure.
            logger.error("[ThreadCheckpoint] typed-memory store failed", exc_info=True)

    logger.info(
        "[ThreadCheckpoint] chat=%s trigger=%s planned=%d stored=%d",
        chat_id, trigger, len(planned), stored,
    )
    return {
        "success": True,
        "topic": parsed["topic"],
        "planned_memories": len(planned),
        "stored_memories": stored,
        "summary": chat.summary,
    }
