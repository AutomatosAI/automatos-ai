"""How the tool loop nudges a reply, and what it keeps when a nudge gets nothing back.

F295 (night 8): eight agent runs failed "Task execution failed after 2 attempts:
Empty response from LLM" (#0207 twice, #0233, #0245 twice, #0260 twice, #0273), on
three claude-sonnet-4 agents over OpenRouter. Each first reply was a real answer
(107 to 860 tokens) that the loop took for narrated actions or for a claim with no
action behind it, and nudged. The nudge went as a system message after the model's
own reply. OpenRouter folds system messages into Anthropic's system prompt, so the
conversation the model saw ended on its own reply, and it added nothing: every one
of the 16 nudged retries came back with 3 tokens (llm_usage), each after a 502
"Server tool openrouter:web_search failed: upstream returned an invalid response"
that F264 sent again without the search. The empty retry replaced the answer. All
27 such 502s of the night came before a 2- or 3-token reply; the three sonnet nudges
that got a real reply had none.

Now a nudge is the user's turn, so the conversation ends on it on every route, and
a nudge that still gets an empty reply leaves the reply it was about standing.

F297 (night 8): when the reply straight after a round of tool calls is empty, the
run is asked once for its answer (``MISSING_ANSWER_MSG``). It used to end on its
tool results, which became the card's answer ("Based on the tool results:
**workspace_write_file**: {…}" on #0234; #0408.3's step wrote its notes into the
mission field, then answered with 2 tokens).

Stdlib only at import, like the loop (the refused-write nudge reads Auto's rule from
services.brief_facts when it is sent).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Messages = List[Dict[str, Any]]
LLMCall = Callable[[Messages, Optional[List[Dict[str, Any]]]], Awaitable[Any]]

NARRATION_RECOVERY_MSG = (
    "Your previous reply described actions (\"let me create…\", \"now let me "
    "assign…\", \"both created\") but made NO tool call, so nothing was executed "
    "and nothing you reported exists. Either call the tools now, in this "
    "response, or state plainly that you did not do it and what you need. "
    "Never describe an action as done without a tool result, and never "
    "invent ids, models or statuses."
)
# F099 (night 3): a reply that names a tool as its source when no tool ran in
# this turn is repeating something from memory — an earlier conversation's
# answer, labelled as if it were a fresh search.
UNRUN_SOURCE_RECOVERY_MSG = (
    "Your previous reply gives {tool} as its source, but no tool ran in this "
    "turn: what you wrote came from memory of an earlier conversation and may "
    "be out of date. Call {tool} now, in this response, or say plainly that the "
    "answer is from an earlier conversation and was not searched again."
)
# F108 (night 3): "I've approved the mission. It's now running" — it wasn't.
CLAIMED_ACTION_RECOVERY_MSG = (
    "Your previous reply says something was {claim}, but no tool call in this turn "
    "did that, so it has not happened. Make the call now, in this response, or say "
    "plainly that it has not been done and what you need. Never report an action "
    "as done without a tool result."
)
# PRD-256 US-006 (F108, night 10b): the claim came after a write that was REFUSED — "Missing
# required params: mission_id", then "I've approved the mission". The nudge names the refusal
# and carries Auto's rule (services.brief_facts.REFUSED_WRITE_RULE, the same words as its prompt).
REFUSED_WRITE_NOTE = (
    "A write in this turn was refused: {refused}. {rule} Make the call again as the refusal says, in this "
    "response, or tell the owner plainly that it was refused and why."
)
REFUSAL_SHOWN_CHARS = 200
REFUSED_SHOWN = 3
# finish_reason "length" mid tool call: the arguments' JSON was cut (moved here from
# tool_loop.py, which is over its size limit, for F328's decorator).
LENGTH_RECOVERY_MSG = (
    "Your previous response was truncated (output token limit reached) "
    "while writing tool call arguments. The JSON was incomplete and could "
    "not be parsed. Please retry with SHORTER content — use concise text, "
    "fewer sections, or summarise instead of writing full prose in the "
    "tool arguments."
)
MISSING_ANSWER_MSG = (
    "Your tool calls have run and their results are above, but your reply had no "
    "answer in it. Write your answer now: the finished work itself (the email, the "
    "table, the text, the figures), not a description of what you did or of the tools "
    "you used. If you saved the work to a file, put the work itself in your answer too."
)

# F306 (night 9): a reply that ends announcing the step it is about to take, and takes
# none: "Let me try a more specific query:" (#1879 twice), "Now let me get the total
# kilograms per account …:" (#1881), "I'll attempt a query to list all tables in"
# (#1888, cut off). It became the card's answer.
ANNOUNCED_STEP_MSG = (
    "Your previous reply ends by announcing a next step (\"{step}\") but made no tool call, "
    "so the step never ran and there is no answer yet. Make that call now, in this response, "
    "or write your answer: the finished work itself, or plainly what is missing."
)
_STEP_CUE = re.compile(r"\b(?:let me|let's|i'?ll|i will|i'?m going to|i am going to)\b", re.IGNORECASE)
_NOT_A_STEP = re.compile(r"\blet me know\b", re.IGNORECASE)
_SENTENCE_END = (".", "!", "?", ")", "\"", "'", "`", "*", "”", "’")
STEP_SHOWN_CHARS = 160


def _refusal(result: Any) -> Optional[str]:
    """What refused a call (its result's error, on one line), or None when it did not fail. A call
    held for the owner's click (PRD-256 US-004's ask) is waiting, not refused."""
    if not isinstance(result, dict) or (result.get("success") is not False and result.get("successful") is not False):
        return None
    if result.get("requires_confirmation") or result.get("owner_only"):
        return None
    said = str(result.get("error") or result.get("message") or "it reported a failure").strip()
    return " ".join(said.split())[:REFUSAL_SHOWN_CHARS]


def refused_writes(outcomes: List[Any]) -> List[str]:
    """The turn's refused writes that no later call of the same action made good, each as
    '<action> (the tool said: "<what refused it>")', from the tracker's outcomes; reads are left out."""
    from .turn_account import is_read

    done = {action for action, _params, result in outcomes if _refusal(result) is None}
    refused: Dict[str, str] = {}
    for action, _params, result in outcomes:
        said = _refusal(result)
        if said and action not in done and not is_read(action):
            # quoted as the tool's own words: a service's error never reads as the owner's (the nudge is a user turn)
            refused.setdefault(action, f"{action} (the tool said: {json.dumps(said, ensure_ascii=False)})")
    return list(refused.values())[:REFUSED_SHOWN]


def claimed_action_nudge(claim: str, outcomes: List[Any]) -> str:
    """F108's nudge for a reply that says something was ``claim`` with no action behind it;
    PRD-256 US-006: after a refused write, the nudge names the refusal and the rule."""
    said = CLAIMED_ACTION_RECOVERY_MSG.format(claim=claim)
    refused = refused_writes(outcomes)
    if not refused:
        return said
    from services.brief_facts import REFUSED_WRITE_RULE

    return f"{said} {REFUSED_WRITE_NOTE.format(refused='; '.join(refused), rule=REFUSED_WRITE_RULE)}"


def announced_step(text: str) -> Optional[str]:
    """The step a reply's last line announces and never took, else None. The line has
    a cue ("let me", "now let me", "I'll") and either ends on a colon or stops
    mid-sentence; "let me know" is never a step."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1]
    if not _STEP_CUE.search(last) or _NOT_A_STEP.search(last):
        return None
    if last.endswith(":") or not last.endswith(_SENTENCE_END):
        return last[:STEP_SHOWN_CHARS]
    return None


# Night 9b (6586c8bf, 8578eeaf): a nudge in the user's turn was read as the owner
# speaking. Auto answered "You are absolutely right to call me out on that, Gerard. My
# apologies…" to the platform's own claim check. Every nudge now says what it is, and
# an apology a reply opens with anyway is taken off (``without_the_apology``).
PLATFORM_CHECK = ("[An automatic check by the platform, not a message from the owner. Do not answer this "
                  "note, thank anyone or apologise: do what it asks, then reply to the owner's last message "
                  "as if the check had not been needed.]")
_APOLOGY = re.compile(
    r"^(?:you(?:'re| are) (?:absolutely |completely |quite |totally )?right\b|my apologies\b|apologies\b|"
    r"i apologi[sz]e\b|(?:i'?m |i am )?(?:so |very )?sorry\b|thank you for (?:catching|pointing|calling)\b|"
    r"good catch\b|you caught\b|it seems i made a mistake\b|i made a mistake\b|i (?:clearly )?missed\b|"
    r"i will (?:ensure|make sure) (?:that )?(?:this|that|it) (?:doesn'?t|does not|won'?t)\b)",
    re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def as_a_check(text: str) -> str:
    """A nudge's words, opened by the line that says the platform wrote them."""
    return f"{PLATFORM_CHECK}\n\n{text}"


def without_the_apology(response: Any) -> Any:
    """``response`` with the apology sentences it opens with taken off (a copy; the
    response itself is never changed). Anything else is left as it is."""
    text = getattr(response, "content", None)
    if not isinstance(text, str) or not text.strip():
        return response
    sentences = _SENTENCE.split(text.lstrip())
    kept = 0
    while kept < len(sentences) and _APOLOGY.match(sentences[kept].strip()):
        kept += 1
    if kept == 0 or kept == len(sentences):
        return response
    logger.info("[tool-loop] a nudged reply opened with an apology — %d sentence(s) taken off", kept)
    return dataclasses.replace(response, content=" ".join(sentences[kept:])) \
        if dataclasses.is_dataclass(response) else response


class Nudge(dict):
    """A message the loop wrote in the user's turn. It is sent as any user message
    is; the chat reads past it when it looks for the owner's own words."""


def nudge(text: str) -> Nudge:
    """A nudge, as the user's turn: the conversation ends on it, so the model answers it."""
    return Nudge(role="user", content=as_a_check(text))


def is_nudge(message: Any) -> bool:
    """A user message the loop wrote, not the person."""
    return isinstance(message, Nudge)


def blank(response: Any) -> bool:
    """A reply with no tool call and no words."""
    return not getattr(response, "tool_calls", None) and not (getattr(response, "content", "") or "").strip()


def kept_if_blank(nudged: Any, retry: Any) -> Any:
    """The retry a nudge got, unless it came back empty: then the reply it nudged stands."""
    if blank(retry) and not blank(nudged):
        logger.warning("[tool-loop] the nudge got an empty reply — keeping the reply it was about")
        return nudged
    return without_the_apology(retry)


async def nudge_about(llm: LLMCall, reply: Any, messages: Messages, tools: Optional[List[Dict[str, Any]]],
                      text: str) -> Any:
    """Nudge ``reply`` once: it stays in the history, the nudge is the user's turn, and
    the retry is returned, or ``reply`` itself when the retry came back empty."""
    messages.append({"role": "assistant", "content": getattr(reply, "content", "") or ""})
    messages.append(nudge(text))
    return kept_if_blank(reply, await llm(messages, tools))


def after_tool_results(messages: Messages) -> bool:
    """The conversation's last turn is a round of tool results."""
    return bool(messages) and messages[-1].get("role") == "tool"


async def ask_for_the_answer(llm: LLMCall, messages: Messages, tools: Optional[List[Dict[str, Any]]]) -> Any:
    """F297: the reply straight after a round of tool calls had nothing in it. Asked
    once for the answer itself, through ``llm`` (the loop's callback). None when the
    reply did not follow tool results, or the answer is still empty."""
    if not after_tool_results(messages):
        return None
    logger.warning("[tool-loop] the reply after the tool calls was empty — asking once for the answer")
    messages.append(nudge(MISSING_ANSWER_MSG))
    retry = await llm(messages, tools)
    return None if blank(retry) else without_the_apology(retry)
