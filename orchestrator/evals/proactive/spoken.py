"""PRD-268 §6.7 (US-C13c) — the spoken brevity eval: each gold spoken-mode ask, as the person hears it.

Each line of ``spoken.jsonl`` is one ask made by voice::

    {"ask": "What needs me this morning?",
     "history": [{"role": "user" | "assistant", "content": "..."}],
     "brief": {"spoken": "Good morning, Ana. Two things need you: ...", ...}}

``history`` (optional) is the conversation before the ask. ``brief`` (optional) makes the case a
brief: it is the person's ``Brief@1`` as ``platform_get_brief`` answers it (its ``spoken`` words at
least). The set is 30 asks, 5 of them briefs (§6.7); the bars are a median and a maximum, so a set
of another size is still scored.

The turn is a spoken turn: the system prompt names Auto and ends with ``SPOKEN_OUTPUT_CONTRACT``, as
the streaming service appends it on a spoken turn. It runs on Auto's model (the platform
``orchestrator_llm``, booked ``orchestrator``), one ask in the ``chat`` lane, through the chat's own
tool loop. A brief case is offered ``platform_get_brief``, and a call to it is answered with the
case's brief, its spoken words and the tool's note, as the live tool answers (the whole brief: a
section asked for is not cut out); any other case is offered no tool.

What is scored is what the person would hear: the answer's text through the voice's own stream
(``spoken_text_frames``: whole speech units, each through ``speechify``, joined as they are sent).
``score_spoken`` counts its words (an answer heard as nothing is a miss: no words would flatter the
median) and finds any markdown, URL, path or id left in it. §6.7 also asks
for every answer to be audited by hand (OpenDuet's checker missed 2 of 12), so each ask and what was
heard are carried in the score's audit lines, which the command line prints.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence, Tuple

from core.llm.usage_context import LANE_CHAT
from evals.proactive.gold import GoldSetUnavailable, conversation, read_gold
from evals.proactive.runtime import chat_llm, eval_workspace, tool_schema
from evals.proactive.scoring import Score, score_spoken
from modules.tools.discovery.handlers_brief import GET_BRIEF, ROWS_ARE_DATA
from modules.tools.execution.tool_loop import ToolLoopExecutor
from modules.voice.spoken_stream import TEXT_FRAME_PREFIX, spoken_text_frames
from modules.voice.spoken_style import SPOKEN_OUTPUT_CONTRACT
from services.credits.asks import ask_scope

__all__ = ["NAME", "SpokenAsk", "answer", "evaluate", "heard", "read_asks"]

NAME = "spoken"
# The turn's rounds: a brief is one call for the brief, then the answer.
MAX_ROUNDS = 2
SYSTEM_PROMPT = "You are Auto, the assistant in this Automatos workspace."
UNKNOWN_TOOL = "There is no tool {name}. Answer the person in words."
SPOKEN_KEY = "spoken"
AUDIT_HEADING = "every answer as it would be heard, for the hand audit (§6.7):"
AUDIT_ASK = "{number}. {kind}: {ask}"
AUDIT_HEARD = "   heard ({words} words): {said}"
BRIEF_KIND, ASK_KIND = "brief", "ask"


@dataclass(frozen=True)
class SpokenAsk:
    """One gold ask: the conversation it ends, and the brief the brief tool answers (None when the
    ask is not a brief)."""

    messages: Tuple[Mapping[str, str], ...]
    brief: Optional[Mapping[str, Any]] = None


def _brief(case: Mapping[str, Any], number: int) -> Optional[Mapping[str, Any]]:
    brief = case.get("brief")
    if brief is None:
        return None
    if not isinstance(brief, Mapping) or not str(brief.get(SPOKEN_KEY) or "").strip():
        raise GoldSetUnavailable(f"{NAME} case {number}: brief is the Brief@1 the brief tool answers, with its "
                                 "spoken words.")
    return dict(brief)


def _ask(case: Mapping[str, Any], number: int) -> SpokenAsk:
    return SpokenAsk(messages=conversation(NAME, case, number), brief=_brief(case, number))


def read_asks() -> List[SpokenAsk]:
    """Every gold ask, read and checked before the first model call."""
    return [_ask(case, number) for number, case in enumerate(read_gold(NAME), start=1)]


async def _one_frame(text: str) -> AsyncIterator[str]:
    yield TEXT_FRAME_PREFIX + json.dumps(text)


async def heard(text: str) -> str:
    """What the voice says of ``text``: the spoken stream's own speech units, each through speechify."""
    return "".join([frame async for frame in spoken_text_frames(_one_frame(text), str)])


def _brief_answer(brief: Mapping[str, Any]) -> Dict[str, Any]:
    """``platform_get_brief``'s answer for ``brief``: the brief, its spoken words and the tool's note."""
    return {"success": True, "data": {"brief": dict(brief), SPOKEN_KEY: brief[SPOKEN_KEY], "note": ROWS_ARE_DATA}}


async def answer(llm: Any, case: SpokenAsk, tools: Sequence[Mapping[str, Any]], workspace_id: str) -> str:
    """Ask ``case`` as one spoken chat turn and return its answer's text, as written."""

    async def run_tool(name: str, args: Dict[str, Any], call_id: str, ws: Any) -> Dict[str, Any]:
        if name == GET_BRIEF and case.brief is not None:
            return _brief_answer(case.brief)
        return {"success": False, "error": UNKNOWN_TOOL.format(name=name)}

    async def call_model(messages: List[Dict[str, Any]], schemas: Optional[List[Dict[str, Any]]]) -> Any:
        return await llm.generate_response(messages, tools=schemas)

    offered = [dict(tool) for tool in tools] or None
    system = {"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{SPOKEN_OUTPUT_CONTRACT}"}
    messages = [system, *(dict(m) for m in case.messages)]
    loop = ToolLoopExecutor(llm_callback=call_model, tool_callback=run_tool, max_iterations=MAX_ROUNDS, promises=True)
    with ask_scope(LANE_CHAT, workspace_id=workspace_id, new=True):
        first = await call_model(messages, offered)
        result = await loop.run(initial_response=first, messages=messages, tools=offered, workspace_id=workspace_id)
    return str(getattr(result.response, "content", "") or "")


def _audit(asks: Sequence[SpokenAsk], said: Sequence[str]) -> Tuple[str, ...]:
    lines = [AUDIT_HEADING]
    for number, (case, words) in enumerate(zip(asks, said), start=1):
        kind = BRIEF_KIND if case.brief is not None else ASK_KIND
        lines.append(AUDIT_ASK.format(number=number, kind=kind, ask=case.messages[-1]["content"]))
        lines.append(AUDIT_HEARD.format(words=len(words.split()), said=words))
    return tuple(lines)


async def evaluate() -> Score:
    """Ask every gold ask by voice and score what is heard: words, markdown, URLs, paths and ids."""
    asks = read_asks()
    workspace_id = eval_workspace()
    llm = chat_llm(workspace_id)
    brief_tools = (tool_schema(GET_BRIEF),)
    said = [await heard(await answer(llm, case, brief_tools if case.brief is not None else (), workspace_id))
            for case in asks]
    return replace(score_spoken(said), audit=_audit(asks, said))
