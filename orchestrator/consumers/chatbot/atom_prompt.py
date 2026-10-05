"""The ATOM chat path's prompt, memory and attachments (PRD-68, PRD-127, F232).

These were the body of ChatService._prepare_atom_path, which had grown to 101
lines. F232 adds the product facts that the full path's ProductFactsSection
renders. ATOM skips the sections, and a short question ("can Hana use it from
home?") is exactly the kind of turn that takes this path.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from modules.context.remembered_figures import labels_what_is_remembered  # F316 (night 9b)

logger = logging.getLogger(__name__)


def _last_user_text(messages: List[Dict[str, Any]]) -> Any:
    return next(
        (m.get("content", "") for m in reversed(messages) if isinstance(m, dict) and m.get("role") == "user"),
        "",
    )


@labels_what_is_remembered  # F316 (night 9b): a remembered figure is never today's
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


async def todays_line_off_loop(db: Any, workspace_id: Any) -> str:
    """Today's date in the workspace's zone (``services.todays_date``), read on a worker thread:
    the zone is a database read, and a pool wait on the event loop stops the process (F105)."""
    from services.todays_date import today_line

    return await asyncio.to_thread(today_line, db, workspace_id)


def atom_system_prompt(metadata: Any, *, identity: str, memory_block: str, facts: str,
                       today: Optional[str] = None) -> str:
    """The ATOM system prompt: who the agent is, how it talks and, since F232,
    what Automatos is in this workspace (``facts``, empty on a widget turn). F323
    (night 9b): it carried no date, and Auto searched the web for "current date"
    (chat 32bb645f); it now says today's date. F337 (night 10): in the workspace's
    zone (``today``, from :func:`todays_line_off_loop`) as the full path's
    DatetimeContextSection does, so late on a Sunday in London is Monday, not UTC's
    Sunday; UTC without one. The greeting stays on the UTC hour."""
    from services.brief_facts import AUTO_OWNER_RULES
    from services.todays_date import today_line

    description = str(metadata.description).strip() if metadata.description else ""
    description_block = f"\n\n## Agent Description\n{description}\n" if description else ""
    persona_block = f"\n\n{metadata.persona}\n" if metadata.persona else ""
    facts_block = f"\n\n{facts}\n" if facts else ""
    # F322 and F327 (night 9b): the owner's turns, not a widget visitor's, carry Auto's rules on their facts.
    owner_block = f"\n\n## What I Avoid\n{AUTO_OWNER_RULES}\n" if facts else ""
    return (
        f"You are {metadata.name}, an AI assistant on the Automatos platform.\n\n"
        f"{time_of_day()}. {today or today_line(None, None)}{identity} "
        "Read the conversation and match the user's energy. "
        "If they're frustrated, be direct — skip the niceties and lead with the answer. "
        "If they're curious, explain the why. If they're casual, be casual back. "
        "If they're formal, match it. Never be artificially cheerful when someone is having a bad time. "
        "Never be robotic when someone is being warm.\n\n"
        "You adapt. That's what makes you good at this.\n"
        f"{description_block}"
        f"{persona_block}"
        f"{facts_block}"
        f"{owner_block}"
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
