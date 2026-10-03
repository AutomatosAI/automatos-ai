"""The ATOM chat path's prompt, memory and attachments (PRD-68, PRD-127, F232).

These were the body of ChatService._prepare_atom_path, which had grown to 101
lines. F232 adds the product facts that the full path's ProductFactsSection
renders. ATOM skips the sections, and a short question ("can Hana use it from
home?") is exactly the kind of turn that takes this path.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _last_user_text(messages: List[Dict[str, Any]]) -> Any:
    return next(
        (m.get("content", "") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )


async def atom_memory_block(orchestrator: Any, messages: List[Dict[str, Any]], *, workspace_id: Any,
                            agent_id: Any, widget_mode: bool, viewer_subject_id: Any) -> str:
    """The "What you remember about this user" block, or "" when there is nothing to add."""
    manager = getattr(orchestrator, "memory_manager", None) if orchestrator else None
    user_msg = _last_user_text(messages)
    if not user_msg or not manager:
        return ""
    try:
        result = await manager.retrieve_memories(
            workspace_id=str(workspace_id),
            agent_id=agent_id,
            query=user_msg if len(user_msg) > 5 else "user context",
            widget_mode=widget_mode,
            # PRD-206 S7: Q7 private-scope guard needs the viewer.
            viewer_subject_id=viewer_subject_id,
        )
    except Exception:
        logger.exception("[PRD-68] ATOM memory retrieval failed; the turn goes on without memories")
        return ""
    if not (result and result.formatted_context):
        return ""
    logger.info("[PRD-68] ATOM memory: %d memories injected", len(result.memories))
    return f"\n\n## What you remember about this user:\n{result.formatted_context}\n"


def time_of_day(now: Optional[datetime] = None) -> str:
    """The greeting for the hour (UTC, as the ATOM prompt always used)."""
    hour = (now or datetime.utcnow()).hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"


def atom_system_prompt(metadata: Any, *, identity: str, memory_block: str, facts: str) -> str:
    """The ATOM system prompt: who the agent is, how it talks and, since F232,
    what Automatos is in this workspace (``facts``, empty on a widget turn)."""
    description = str(metadata.description).strip() if metadata.description else ""
    description_block = f"\n\n## Agent Description\n{description}\n" if description else ""
    persona_block = f"\n\n{metadata.persona}\n" if metadata.persona else ""
    facts_block = f"\n\n{facts}\n" if facts else ""
    return (
        f"You are {metadata.name}, an AI assistant on the Automatos platform.\n\n"
        f"{time_of_day()}.{identity} "
        "Read the conversation and match the user's energy. "
        "If they're frustrated, be direct — skip the niceties and lead with the answer. "
        "If they're curious, explain the why. If they're casual, be casual back. "
        "If they're formal, match it. Never be artificially cheerful when someone is having a bad time. "
        "Never be robotic when someone is being warm.\n\n"
        "You adapt. That's what makes you good at this.\n"
        f"{description_block}"
        f"{persona_block}"
        f"{facts_block}"
        f"{memory_block}"
    )


async def resolve_atom_attachments(db: Any, llm_messages: List[Dict[str, Any]], attachment_ids: List[str], *,
                                   workspace_id: Any, model_id: Optional[str]) -> None:
    """PRD-127: put the turn's attachments into its last user message. The full
    path's ContextService.build_context does this; ATOM bypasses it."""
    from uuid import UUID

    from modules.attachments.resolver import (
        AttachmentResolver,
        VisionNotSupportedError,
        inject_parts_into_last_user_message,
    )

    try:
        parts, failures = await AttachmentResolver(db_session=db).resolve(
            attachment_ids=[UUID(a) for a in attachment_ids],
            workspace_id=UUID(str(workspace_id)),
            model_id=model_id or "",
        )
        if parts:
            inject_parts_into_last_user_message(llm_messages, parts)
            logger.info("[PRD-127] ATOM path: resolved %d attachment parts from %d ids",
                        len(parts), len(attachment_ids))
        if failures:
            # PRD-223 S0.3: the unavailable-marker part is already in `parts`,
            # so the model will say what it cannot see.
            logger.warning("[PRD-223] ATOM path: %d attachment(s) unavailable", len(failures))
    except VisionNotSupportedError as err:
        logger.warning("[PRD-127] ATOM vision not supported: %s", err)
    except Exception:
        logger.exception("[PRD-127] ATOM attachment resolution failed")
