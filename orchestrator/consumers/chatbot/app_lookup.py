"""A turn's connected-apps lookup, split from its merge (voice latency D-1).

Before its first model call a chat turn asks Composio which actions to offer: the tool search
(``ComposioToolService.get_tools_for_step``) and, when that offers no tools, the action hints
(``ComposioHintService.build_hints``). Both ran inside ``StreamingChatService._inject_composio_tools``,
in one block with the code that puts their answer into the turn's tools and messages.

They are two things here:

* ``look_up_apps`` only reads. It takes the agent, the workspace, the person's message and the
  tool hints, and needs nothing from the turn's prompt. Each lookup still runs on the Composio
  lookup threads with a session of its own (``core.composio.off_loop``).
* ``apply_app_lookup`` only merges: it rebuilds the tool list and puts one system message at
  index 2, the statements ``_inject_composio_tools`` held, in the same order.

Nothing a turn is given changes. The split exists so the lookup can be started before the
prompt is built (D-2) while the merge stays where it is.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COMPOSIO_EXECUTE = "composio_execute"  # the generic tool the typed per-action tools replace
SCOPE_MESSAGE_INDEX = 2                # after the identity and the context messages
SYSTEM_ROLE = "system"


@dataclasses.dataclass(frozen=True)
class AppLookup:
    """What a turn's connected-apps lookup found."""

    found: Any = None     # ComposioToolService.get_tools_for_step's result; None: gave up or failed
    hints: Any = None     # ComposioHintService.build_hints' result; asked only when `found` offers no tools
    failed: bool = False  # a lookup raised (not a timeout)


def tool_hints_of(complexity_assessment: Any) -> Any:
    """The tool hints the turn's complexity assessment carries, or none."""
    if complexity_assessment and hasattr(complexity_assessment, "tool_hints"):
        return complexity_assessment.tool_hints
    return []


def asks_the_apps(latest_text: str, agent_id: Any, workspace_id: Any, skip_composio: bool) -> bool:
    """Whether this turn looks up its connected apps at all."""
    return bool(latest_text and agent_id and workspace_id and not skip_composio)


async def look_up_apps(agent_id: int, workspace_id: Any, latest_text: str, tool_hints: Any) -> AppLookup:
    """The tool search, then the action hints when the search offers no tools. A lookup that
    gives up in time (already warned where it gave up) or raises ends the lookup with what
    was found so far: the turn goes on without Composio tools. Never raises but for a cancel."""
    from core.composio.off_loop import ComposioLookupTimeout, composio_lookup

    found = None
    try:
        from modules.tools.services.composio_tool_service import ComposioToolService

        search_prompt = " ".join(tool_hints) if tool_hints else latest_text
        found = await composio_lookup(
            lambda db: ComposioToolService(db).get_tools_for_step(
                agent_id=agent_id,
                workspace_id=workspace_id,
                task_prompt=search_prompt,
                tool_hints=tool_hints,
            ),
            step=f"chat turn (agent {agent_id}): tool search",
        )
        if found and found.tools:
            return AppLookup(found=found)
        from modules.tools.services.composio_hint_service import ComposioHintService

        hints = await composio_lookup(
            lambda db: ComposioHintService(db).build_hints(
                agent_id=agent_id,
                prompt=latest_text,
                workspace_id=workspace_id,
            ),
            step=f"chat turn (agent {agent_id}): action hints",
        )
        return AppLookup(found=found, hints=hints)
    except ComposioLookupTimeout:
        return AppLookup(found=found)  # warned where it gave up
    except Exception as exc:  # noqa: BLE001 — a failed lookup never breaks the turn; it goes on without the apps
        logger.warning(f"Composio tool injection failed for agent {agent_id}: {exc}")
        return AppLookup(found=found, failed=True)


def apply_app_lookup(llm_messages: List[Dict[str, Any]], use_tools: Optional[List[Dict[str, Any]]],
                     lookup: AppLookup, agent_id: int) -> Optional[List[Dict[str, Any]]]:
    """Put what the lookup found into the turn: the typed per-action tools in place of the
    generic one, with the scope message; or the hint lines. Returns the tools to offer, a new
    list when they changed (the list given is never changed in place); ``llm_messages`` is the
    caller's own list and gains at most one message. A failure after the tools were rebuilt
    still returns the rebuilt list."""
    try:
        found = lookup.found
        if found and found.tools:
            if use_tools:
                use_tools = [
                    t for t in use_tools
                    if t.get("function", {}).get("name") != COMPOSIO_EXECUTE
                ] + found.tools
            else:
                use_tools = found.tools
            from api.recipe_executor import _composio_scope_message

            llm_messages.insert(SCOPE_MESSAGE_INDEX, {
                "role": SYSTEM_ROLE,
                "content": _composio_scope_message(found.app_names),
            })
            logger.info(
                f"[ComposioToolService] Agent {agent_id}: strategy={found.strategy} "
                f"actions={len(found.action_set)} search_ms={found.search_ms}"
            )
        elif lookup.hints is not None and lookup.hints.hint_lines:
            hints = lookup.hints
            llm_messages.insert(SCOPE_MESSAGE_INDEX, {"role": SYSTEM_ROLE, "content": "\n".join(hints.hint_lines)})
            logger.info(
                f"[Composio Hints fallback] Agent {agent_id}: strategy={hints.strategy_used} "
                f"apps={hints.allowed_apps} matches={len(hints.matched_actions)}"
            )
    except Exception as exc:  # noqa: BLE001 — a failed merge never breaks the turn
        logger.warning(f"Composio tool injection failed for agent {agent_id}: {exc}")
    return use_tools


__all__ = ["AppLookup", "apply_app_lookup", "asks_the_apps", "look_up_apps", "tool_hints_of"]
