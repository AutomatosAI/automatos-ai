"""PRD-268 §6.7 (US-C13b) — the reply chips eval: each gold question asked of Auto's chat model.

Each line of ``reply_chips.jsonl`` is one turn's ask, a choice question or an open one::

    {"choice": true, "ask": "Book the delivery call for Friday.",
     "history": [{"role": "user" | "assistant", "content": "..."}]}

``history`` (optional) is the conversation before the ask. The set is 20 choice questions and 10
open ones (§6.7); the bars are shares, so a set of another size is still scored.

The model is Auto's (the platform ``orchestrator_llm``, booked ``orchestrator``), one ask in the
``chat`` lane per question, offered ``platform_offer_replies`` as the platform offers it. The turn
runs through the chat's own tool loop (``ToolLoopExecutor``), so its rule holds as it does in chat:
a round whose chips the tool kept ends the turn on that round's text with no further model call,
and a round with no text goes on to the next. Each call runs the tool's own handler, which writes
nothing; a refused call is answered with its sentence, and the model may correct it.

The system prompt is one line naming Auto, not the chat's whole context (its persona, memory and
the workspace's sections need the live chat), and the turn is offered this one tool: the number
measures the tool's own description on Auto's model, which is where the rule for chips is written.

What the turn offered is scored (``score_reply_chips``): the chips its answer carries, else the
last replies it tried, as it sent them (a try the tool refused is outside the limits), else none.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from core.llm.usage_context import LANE_CHAT
from evals.proactive.gold import GoldSetUnavailable, read_gold
from evals.proactive.runtime import eval_workspace, tool_schema
from evals.proactive.scoring import ChipsTurn, Score, score_reply_chips
from modules.tools.discovery.handlers_replies import OFFER_REPLIES, REPLIES_PARAM, offer_replies
from modules.tools.execution.params_text import json_container
from modules.tools.execution.reply_chips_turn import REPLY_CHIPS
from modules.tools.execution.tool_loop import ToolLoopExecutor
from services.credits.asks import ask_scope

__all__ = ["CHAT_PURPOSE", "NAME", "Question", "ask", "chat_llm", "evaluate", "read_questions"]

NAME = "reply_chips"
# Auto's tier, the platform's orchestrator_llm (core/llm/manager.py SERVICE_CATEGORY_MAP).
AUTO_SERVICE = "orchestrator"
# Auto's own chat calls' purpose, customer work (services/credits/classification.py). Written out:
# tests/test_prd265_call_site_classes.py reads it from this module.
CHAT_PURPOSE = "orchestrator"
# The turn's rounds: a choice is a question and its chips, perhaps one correction, then the answer.
MAX_ROUNDS = 3
SYSTEM_PROMPT = "You are Auto, the assistant in this Automatos workspace. Answer the person in plain, short sentences."
ROLES = ("user", "assistant")
UNKNOWN_TOOL = "There is no tool {name}. Answer the person in words."


@dataclass(frozen=True)
class Question:
    """One gold question: a choice or an open one, and the conversation it ends."""

    choice: bool
    messages: Tuple[Mapping[str, str], ...]


def chat_llm(workspace_id: str) -> Any:
    """Auto's model: the platform ``orchestrator_llm``, its calls booked as Auto's own."""
    from core.llm.manager import LLMManager

    return LLMManager(service_name=AUTO_SERVICE, workspace_id=workspace_id, request_type=CHAT_PURPOSE)


def _history(case: Mapping[str, Any], number: int) -> Tuple[Mapping[str, str], ...]:
    history = case.get("history") or []
    fits = isinstance(history, list) and all(
        isinstance(m, Mapping) and m.get("role") in ROLES and isinstance(m.get("content"), str) for m in history)
    if not fits:
        raise GoldSetUnavailable(f"{NAME} case {number}: history is a list of {{role: user | assistant, content}}.")
    return tuple({"role": m["role"], "content": m["content"]} for m in history)


def _question(case: Mapping[str, Any], number: int) -> Question:
    choice, said = case.get("choice"), case.get("ask")
    if not isinstance(choice, bool):
        raise GoldSetUnavailable(f"{NAME} case {number}: choice is true (a choice question) or false (an open one).")
    if not isinstance(said, str) or not said.strip():
        raise GoldSetUnavailable(f"{NAME} case {number}: ask is the person's message, as text.")
    return Question(choice=choice, messages=(*_history(case, number), {"role": "user", "content": said}))


def read_questions() -> List[Question]:
    """Every gold question, read and checked before the first model call."""
    return [_question(case, number) for number, case in enumerate(read_gold(NAME), start=1)]


def _offered(raw: Any) -> Tuple[Any, ...]:
    """The replies a call tried, as it sent them (a list sent as JSON text is read)."""
    replies = json_container(raw)
    return tuple(replies) if isinstance(replies, list) else ()


async def ask(llm: Any, question: Question, tools: Sequence[Mapping[str, Any]], workspace_id: str) -> ChipsTurn:
    """Ask ``question`` as one chat turn and return what it answered and offered."""
    tried: List[Tuple[Any, ...]] = []

    async def run_tool(name: str, args: Dict[str, Any], call_id: str, ws: Any) -> Dict[str, Any]:
        if name != OFFER_REPLIES:
            return {"success": False, "error": UNKNOWN_TOOL.format(name=name)}
        params = args if isinstance(args, dict) else {}
        tried.append(_offered(params.get(REPLIES_PARAM)))
        return await offer_replies(None, workspace_id, params)

    async def call_model(messages: List[Dict[str, Any]], schemas: Optional[List[Dict[str, Any]]]) -> Any:
        return await llm.generate_response(messages, tools=schemas)

    offered = [dict(tool) for tool in tools]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *(dict(m) for m in question.messages)]
    loop = ToolLoopExecutor(llm_callback=call_model, tool_callback=run_tool, max_iterations=MAX_ROUNDS, promises=True)
    with ask_scope(LANE_CHAT, workspace_id=workspace_id, new=True):
        first = await call_model(messages, offered)
        result = await loop.run(initial_response=first, messages=messages, tools=offered, workspace_id=workspace_id)
    shown = getattr(result.response, REPLY_CHIPS, None)
    replies = tuple(shown) if shown else (tried[-1] if tried else None)
    return ChipsTurn(choice=question.choice, answer=str(getattr(result.response, "content", "") or ""),
                     replies=replies)


async def evaluate() -> Score:
    """Ask every gold question and score the chips, their limits and the answers they replaced."""
    questions = read_questions()
    workspace_id = eval_workspace()
    llm = chat_llm(workspace_id)
    tools = (tool_schema(OFFER_REPLIES),)
    turns = [await ask(llm, question, tools, workspace_id) for question in questions]
    return score_reply_chips(turns)
