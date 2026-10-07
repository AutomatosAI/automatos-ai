"""F381 (night 11, 7 Oct): a how-to question from the owner changes nothing; Auto answers it and offers.

Night 11, iteration 4 (chat a576f1de, B-I4-1): the owner asked "how do I give my agent two
photos?". Auto called platform_update_agent twice and platform_assign_tool_to_agent (DROPBOX onto
the Social Media Director, #348), and filed task 2163 to upload "the photos on your Mac" to a
temporary Dropbox folder. It changed an agent and planned a write to an outside drive without
asking, and only the owner's "no" on question #1826 stopped the upload. An hour later the same
question, "Where do I add photos?", got an answer and changed nothing: the persona's "I prefer
action over explanation" decided it, a coin toss. The owner: "it never changes an agent's tools
to answer a question."

So a turn whose owner message asks how, where or whether to do something ("how do I give my
agent two photos?", "where do I add photos?", "can I…?", "is there a way to…?") refuses, for that
turn, every call that changes an agent, its tools, skills or plugins, a connection, the board's
tasks and missions, the brand kit or a setting, with the answer to give instead: say how and where
they do it, and offer to do it for them. A request is not a question: "can you make me a post?",
"can I have…", "please…", "… for me", "go ahead" and an instruction sentence beside the question
("Where do they go? Give my agent Dropbox.") all leave the turn as it was. The words are the
owner's latest message in the chat the call is made in (``owner_turn``); outside a person's chat
nothing is checked. A fault here never stops a call: it is logged and the call goes on.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# The calls a how-to question never makes: an agent and what it carries, a connection, the board's
# work, the brand kit and the settings.
REFUSED_ON_A_QUESTION = frozenset({
    "platform_create_agent", "platform_update_agent", "platform_delete_agent", "platform_configure_agent_heartbeat",
    "platform_assign_tool_to_agent", "platform_unassign_tool_from_agent", "platform_assign_plugin_to_agent",
    "platform_assign_skill_to_agent", "platform_unassign_skill_from_agent", "platform_install_plugin",
    "platform_uninstall_plugin", "platform_install_skill", "platform_install_marketplace_agent",
    "platform_install_package", "platform_install_model", "platform_connect_channel", "platform_revoke_api_key",
    "platform_create_task", "platform_assign_task", "platform_create_mission", "platform_schedule_task",
    "platform_schedule_playbook", "platform_update_system_setting", "platform_update_workspace_settings",
    "platform_update_brand_kit", "platform_propose_brand_kit", "platform_save_approved_brand_kit",
    "platform_create_workspace_skill", "platform_update_skill", "platform_set_skill_script_execution",
})
ANSWER_IT = ("The owner asked a question, {asked}, not for a change. Nothing was changed. Answer it in words: how "
             "and where they do it themselves, in the product. Then offer to do it for them; if they say yes in "
             "their next message, make the change then.")
QUOTED_CHARS = 160

_LEAD = r"^\s*(?:(?:hi|hey|hello|ok(?:ay)?|so|right|quick question|question|auto)\b[\s,:!.\-–—]*)*"
_HOW_TO = re.compile(
    _LEAD + r"(?:how\s+(?:do|can|could|would|should|might|does|did)\s+(?:i|we|one|you|my|our|the)\b"
    r"|how\s+(?:to|is|are)\b|where\s+(?:do|can|could|should|would|is|are|does)\b|where['’]s\b"
    r"|what(?:['’]s|\s+is)\s+the\s+(?:best\s+|easiest\s+|right\s+)?way\s+to\b|is\s+(?:there|it\s+possible)\b"
    r"|(?:can|could|may)\s+(?:i|we)\b|am\s+i\s+able\b|do\s+(?:i|we)\s+(?:need|have)\s+to\b|should\s+(?:i|we)\b)",
    re.I)
_REQUEST = re.compile(r"\b(?:can|could|would|will)\s+you\b|(?<!how )\b(?:can|could|may)\s+(?:i|we)\s+(?:have|get|grab)\b"
                      r"|\bplease\b|\bgo ahead\b|\bdo it\b|\bfor (?:me|us)\b"
                      r"|\bi(?:['’]d| would) like you to\b|\bi want you to\b|\bset (?:it|that|this) up\b", re.I)
_INSTRUCTION = re.compile(r"^\s*(?:give|add|connect|assign|make|create|set|update|change|install|remove|delete|put|"
                          r"use|send|upload|attach|switch|turn|move|start|run|file|draft|write|build|let['’]?s)\b",
                          re.I)
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")


def asks_how(text: object) -> bool:
    """Whether ``text`` asks how, where or whether to do something, and asks for nothing to be done:
    no request ("can you…", "please", "for me") and no instruction sentence beside it."""
    said = str(text or "").strip()
    if not said or _REQUEST.search(said):
        return False
    sentences = [part.strip() for part in _SENTENCES.split(said) if part.strip()]
    if not any(_HOW_TO.match(sentence) for sentence in sentences):
        return False
    return not any(_INSTRUCTION.match(sentence) for sentence in sentences)


def refusal_on_a_question(db: Any, workspace_id: Any, action: str, caller_context: Any) -> Optional[str]:
    """Why ``action`` is refused in this turn (its owner asked how, not for a change), or None."""
    if action not in REFUSED_ON_A_QUESTION:
        return None
    from modules.tools.discovery.owner_turn import owner_turn

    try:
        turn = owner_turn(db, workspace_id, caller_context)
    except Exception:
        logger.exception("[F381] could not read the owner's words for %s; the call goes on", action)
        return None
    if turn is None or not asks_how(turn.latest):
        return None
    asked = json.dumps(" ".join(turn.latest.split())[:QUOTED_CHARS], ensure_ascii=False)
    return ANSWER_IT.format(asked=asked)


Execute = Callable[..., Awaitable[Dict[str, Any]]]


def answers_the_question_first(execute: Execute) -> Execute:
    """Wrap PlatformExecutor.execute: in a turn whose owner asked how, a call that changes an agent,
    a connection, the board's work, the kit or a setting is refused, with the answer to give instead."""
    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        db, workspace_id = getattr(self, "db", None), getattr(self, "workspace_id", None)
        refusal = refusal_on_a_question(db, workspace_id, action_name, caller_context)
        if refusal:
            logger.info("[F381] %s refused: the owner asked a question", action_name)
            return {"success": False, "error": refusal}
        return await execute(self, action_name, params, caller_context)
    return wrapped


__all__ = ["ANSWER_IT", "REFUSED_ON_A_QUESTION", "answers_the_question_first", "asks_how", "refusal_on_a_question"]
