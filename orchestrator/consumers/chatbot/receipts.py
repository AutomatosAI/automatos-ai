"""PRD-256 US-001: the receipts, what Auto did this turn, written by the platform.

Eleven customer nights found Auto's worst habit: claiming work it did not do, most often
after a call that FAILED on its arguments and was then reported as done. A reply is no
longer the only account of its turn. After the tool loop the platform writes its own, from
the calls that ran: the in-process ``ToolExecutionTracker`` (``outcomes``, ``skipped``)
read with ``call_effects``, never ``tool_execution_logs`` (D8: it keeps parameter names
only, no result and no chat).

- One receipt per call: ``{action, kind, status, subject, effect, link, reason}``. The kind
  is read or write; the status done, refused or skipped. The subject is the thing by number
  or name (card #0422, an agent, a document's title); the effect is what the call did, in
  plain words ("moved to Done", "sent back to its agent"); a refused or skipped call says
  why, on one line. A link is the page of what a done call touched (its card, its agent).
- The reads the turn makes before the model's first call (retrieval first, the Needs-you
  read, the team's findings: ``prefetched``) fold into ONE read receipt.

The turn streams them as one ``receipts`` frame, with the model that answered: after the
loop, before the answer's additions, or, for a turn that ran no loop, before its finish. It
saves them as the message's ``receipts`` part (``narration.reply_parts``; ``[]`` when nothing
ran). They are built after the model is done: no prompt names them and no tool can set them.

The wiring is decorators on the turn's seams, so the turn's own long functions are untouched:
``writes_its_receipts`` (the turn, whichever agent answers it: Auto, or a specialist on a
DELEGATE turn), ``its_reads_are_receipted`` (retrieval first), ``the_loop_writes_receipts``
(the tool loop), ``notes_the_answering_model`` (the answer's additions) and
``saves_the_turns_receipts`` (the saved parts). The turn's state is a few context variables,
set once per turn and never mutated.
"""
from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from modules.tools.execution.call_effects import (
    AGENT_SET, DOCUMENT_MAKES, SENT_BACK, STEPS_CHECKED, STEPS_UNCHECKED, answers_in, call_effects, result_effects,
)
from modules.tools.execution.tool_execution_tracker import TRACKERS_MADE
from modules.tools.execution.turn_account import is_read, thing_of, what_it_did

Receipt = Dict[str, Any]
Prefetched = Sequence[Tuple[str, Dict[str, Any]]]

PART = FRAME = "receipts"
READ, WRITE = "read", "write"
DONE, REFUSED, SKIPPED = "done", "refused", "skipped"
REASON_CHARS = 200
SUBJECT_CHARS = 80
LOOKED_UP = "looked up"
FAILED = "it reported a failure"
# The automatic reads, folded: "read your documents and the board".
AUTOMATIC_READS = "automatic_reads"
DOCUMENTS, BOARD = "your documents", "the board"
FOLD_EFFECT = "read {what}"
# Why a call was not run, in the owner's words (the tracker's own are written for the model).
SKIPPED_REPEAT = "the same request already ran in this reply"
SKIPPED_CAP = "this reply reached its limit for it"
_CAP_WORDS = ("limit", "ceiling")
# Where a card moved, by the status the board gives it (call_effects reads the owner's words).
_MOVES = {"done": "moved to Done", "cancelled": "moved to Cancelled", "assigned": "sent back to its agent",
          "review": "moved to Review", "in_progress": "started", "inbox": "moved to the Inbox",
          "blocked": "marked blocked"}
_SAID = {SENT_BACK: "sent back to its agent", AGENT_SET: "agent set",
         STEPS_CHECKED: "each step waits for your OK", STEPS_UNCHECKED: "its steps run without your check"}
# Calls whose name reads badly as "<thing> <past verb>".
_OWN_WORDS = {"assign_tool_to_agent": "tool added", "unassign_tool_from_agent": "tool removed"}
DOCUMENT_MADE = "document made"   # F351: the calls that make a document the owner finds in Deliverables
_NAME_KEYS = ("title", "document_title", "agent_name", "name", "filename", "file_name")
_CARD_KEYS = ("task_id", "task_number", "number")
# The UI's pages for what a call touched (frontend/lib/ticket-links.ts, the agents page).
CARD_LINK = "/command-center?tab=board&task_id={id}"
AGENT_LINK = "/agents?agent={id}"


def _refused(result: Any) -> bool:
    """The tracker's test (F108): a result that says it failed."""
    return isinstance(result, dict) and (result.get("success") is False or result.get("successful") is False)


def _one_line(text: str, limit: int) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    first = lines[0] if lines else ""
    return first if len(first) <= limit else first[: limit - 1].rstrip() + "…"


def _number(answers: Iterable[Dict[str, Any]], params: Dict[str, Any]) -> str:
    """The card's number as the board shows it: from the answer, else as the call gave it."""
    for answer in answers:
        task = answer.get("task") if isinstance(answer.get("task"), dict) else {}
        for said in (answer.get("number"), task.get("number")):
            if isinstance(said, str) and said.startswith("#"):
                return said
    for key in _CARD_KEYS:
        said = params.get(key)
        if isinstance(said, str) and said.strip().startswith("#"):
            return said.strip()
    return ""


def _name(answers: Sequence[Dict[str, Any]], params: Dict[str, Any]) -> str:
    for source in (*answers, params):
        for key in _NAME_KEYS:
            said = source.get(key)
            if isinstance(said, str) and said.strip():
                return _one_line(said, SUBJECT_CHARS)
    return ""


def _subject(answers: Sequence[Dict[str, Any]], params: Dict[str, Any]) -> str:
    """The thing by number or name: card #0422, an agent, a document's title; '' when none is said."""
    return _number(answers, params) or _name(answers, params)


def _effect(action: str, params: Dict[str, Any], result: Any) -> str:
    """What a write did (or tried), in plain words."""
    tags = [effect.rsplit(":", 1)[-1] for effect in call_effects(action, params)] + list(result_effects(result))
    words = [_SAID.get(tag) or _MOVES.get(tag) or f"moved to {tag.replace('_', ' ')}" for tag in tags]
    if words:
        return ", ".join(dict.fromkeys(words))
    if any(stem in action for stem in DOCUMENT_MAKES):
        return DOCUMENT_MADE
    return _OWN_WORDS.get(action.removeprefix("platform_")) or what_it_did(action).lower()


def _link(answers: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The page of the card or agent a done call's own answer names (this workspace's, by its answer)."""
    for key, page in (("task_id", CARD_LINK), ("agent_id", AGENT_LINK)):
        for answer in answers:
            said = answer.get(key)
            if isinstance(said, int) and not isinstance(said, bool):
                return page.format(id=said)
    return None


def _reason(answers: Sequence[Dict[str, Any]]) -> str:
    for answer in answers:
        said = answer.get("error") or answer.get("message")
        if isinstance(said, str) and said.strip():
            return _one_line(said, REASON_CHARS)
    return FAILED


def receipt(action: str, params: Dict[str, Any], result: Any) -> Receipt:
    """The receipt of one call that ran: done, or refused with its reason."""
    params = params if isinstance(params, dict) else {}
    answers = answers_in(result)
    refused = _refused(result)
    reads = is_read(action)
    return {
        "action": action,
        "kind": READ if reads else WRITE,
        "status": REFUSED if refused else DONE,
        "subject": _subject(answers, params) or (thing_of(action) if reads else ""),
        "effect": LOOKED_UP if reads else _effect(action, params, result),
        "link": None if refused else _link(answers),
        "reason": _reason(answers) if refused else None,
    }


def skipped_receipt(action: str, params: Dict[str, Any], why: str) -> Receipt:
    """The receipt of a call the turn did not run (a repeat, a cap)."""
    params = params if isinstance(params, dict) else {}
    reads = is_read(action)
    capped = any(word in (why or "").lower() for word in _CAP_WORDS)
    return {
        "action": action,
        "kind": READ if reads else WRITE,
        "status": SKIPPED,
        "subject": _subject([], params) or (thing_of(action) if reads else ""),
        "effect": LOOKED_UP if reads else _effect(action, params, None),
        "link": None,
        "reason": SKIPPED_CAP if capped else SKIPPED_REPEAT,
    }


def folded_reads(prefetched: Prefetched) -> Optional[Receipt]:
    """The automatic reads as ONE receipt: 'read your documents and the board'; None when none ran."""
    from consumers.chatbot.knowledge_prefetch import PREFETCH_TOOL

    names = {name for name, _args in prefetched}
    if not names:
        return None
    what = [place for place, read in ((DOCUMENTS, PREFETCH_TOOL in names), (BOARD, bool(names - {PREFETCH_TOOL})))
            if read]
    return {"action": AUTOMATIC_READS, "kind": READ, "status": DONE, "subject": "",
            "effect": FOLD_EFFECT.format(what=" and ".join(what)), "link": None, "reason": None}


def build_receipts(tracker: Any, prefetched: Prefetched = ()) -> List[Receipt]:
    """The turn's receipts, in order: its automatic reads (folded), each call that ran, each
    call it skipped. ``tracker`` is the loop's ToolExecutionTracker, or None when no loop ran."""
    folded = folded_reads(prefetched or ())
    ran = [receipt(action, params, result) for action, params, result in getattr(tracker, "outcomes", ())]
    not_run = [skipped_receipt(action, params, why) for action, params, why in getattr(tracker, "skipped", ())]
    return ([folded] if folded else []) + ran + not_run


def turn_receipts(loop_receipts: Optional[List[Receipt]], prefetched: Prefetched = ()) -> List[Receipt]:
    """The receipts of a whole turn: the loop's when the turn ran it (a forced synthesis keeps
    them), else the automatic reads alone; ``[]`` for a turn that ran nothing."""
    return list(loop_receipts) if loop_receipts is not None else build_receipts(None, prefetched)


def model_of(response: Any) -> Optional[str]:
    """The model that answered, when its provider says (``response.model``)."""
    model = getattr(response, "model", None)
    return model if isinstance(model, str) and model else None


def receipts_frame(handler: Any, receipts: List[Receipt], model: Optional[str] = None) -> str:
    """The one ``receipts`` frame of a turn, with the model that answered when it is known."""
    data: Dict[str, Any] = {"receipts": list(receipts)}
    if model:
        data["model"] = model
    return handler.format_aisdk_data(FRAME, data)


# ── the turn ───────────────────────────────────────────────────────────────
# One value per variable, set by the turn's seams, never mutated: whether a chat turn is
# running, its automatic reads, its loop's receipts once the loop ended, the model that
# answered, and whether its frame went out.
_IN_TURN: ContextVar[bool] = ContextVar("receipts_in_turn", default=False)
_PREFETCHED: ContextVar[Prefetched] = ContextVar("receipts_prefetched", default=())
_LOOP: ContextVar[Optional[List[Receipt]]] = ContextVar("receipts_loop", default=None)
_MODEL: ContextVar[Optional[str]] = ContextVar("receipts_model", default=None)
_SENT: ContextVar[bool] = ContextVar("receipts_sent", default=False)

Stream = Callable[..., AsyncGenerator[Any, None]]
Additions = Callable[[Any, Any], List[str]]
Parts = Callable[..., List[Dict[str, Any]]]


def current_receipts() -> Optional[List[Receipt]]:
    """This chat turn's receipts: its loop's once the loop ended, else its automatic reads
    (``[]`` when nothing ran); None outside a chat turn."""
    if not _IN_TURN.get():
        return None
    return turn_receipts(_LOOP.get(), _PREFETCHED.get())


def _frame_once(chat: Any, receipts: List[Receipt], model: Optional[str]) -> Optional[str]:
    """The turn's frame, the first time it is asked for. A public widget visitor is sent none
    (F155: a visitor sees no internals); the receipts are still saved with the message."""
    handler = getattr(chat, "streaming_handler", None)
    if _SENT.get() or handler is None:
        return None
    _SENT.set(True)
    if getattr(chat, "widget_mode", False):
        return None
    return receipts_frame(handler, receipts, model)


def writes_its_receipts(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService.stream_response_with_agent``: every turn has receipts,
    whichever agent answers it. A turn that ran no loop sends its frame before its finish."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        for var, fresh in ((_IN_TURN, True), (_PREFETCHED, ()), (_LOOP, None), (_MODEL, None), (_SENT, False)):
            var.set(fresh)
        handler = getattr(chat, "streaming_handler", None)
        finish = handler.format_aisdk_finish() if handler is not None else None
        try:
            async for chunk in turn(chat, *args, **kwargs):
                frame = _frame_once(chat, current_receipts() or [], _MODEL.get()) if finish and chunk == finish else None
                if frame:
                    yield frame
                yield chunk
        finally:
            _IN_TURN.set(False)
    return wrapped


def its_reads_are_receipted(retrieval_first: Stream) -> Stream:
    """Wrap ``StreamingChatService._retrieval_first``: the turn keeps its automatic reads
    (``prefetched``, filled by the reads under it) for the folded read receipt."""
    @functools.wraps(retrieval_first)
    async def wrapped(chat: Any, latest_text: str, llm_messages: List[Dict[str, Any]], agent_runtime: Any,
                      chat_id: str, prefetched: List[Any]) -> AsyncGenerator[Any, None]:
        _PREFETCHED.set(prefetched)
        async for frame in retrieval_first(chat, latest_text, llm_messages, agent_runtime, chat_id, prefetched):
            yield frame
    return wrapped


def the_loop_writes_receipts(loop: Stream) -> Stream:
    """Wrap ``StreamingChatService._stream_tool_loop``: once the loop ends, its receipts are
    built from its own tracker (the first one made while it runs) and its frame goes out,
    before the answer and its additions."""
    @functools.wraps(loop)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        made: List[Any] = []
        TRACKERS_MADE.set(made)
        try:
            async for chunk in loop(chat, *args, **kwargs):
                if isinstance(chunk, dict) and chunk.get("_final_response") is not None:
                    receipts = build_receipts(made[0] if made else None, kwargs.get("prefetched") or ())
                    _LOOP.set(receipts)
                    frame = _frame_once(chat, receipts, model_of(chunk["_final_response"]))
                    if frame:
                        yield frame
                yield chunk
        finally:
            TRACKERS_MADE.set(None)
    return wrapped


def notes_the_answering_model(answer_additions: Additions) -> Additions:
    """Wrap ``StreamingChatService._answer_additions``: the model of the round that answered
    (the loop's, a forced synthesis', or the first reply's) is the frame's."""
    @functools.wraps(answer_additions)
    def wrapped(f187_verdict: Any, final_round: Any) -> List[str]:
        _MODEL.set(model_of(final_round))
        return answer_additions(f187_verdict, final_round)
    return wrapped


def saves_the_turns_receipts(reply_parts: Parts) -> Parts:
    """Wrap ``narration.reply_parts``: the message a chat turn saves carries its receipts."""
    @functools.wraps(reply_parts)
    def wrapped(reasoning: str, narration: str, answer: str,
                receipts: Optional[List[Receipt]] = None) -> List[Dict[str, Any]]:
        return reply_parts(reasoning, narration, answer, receipts if receipts is not None else current_receipts())
    return wrapped


__all__ = ["AUTOMATIC_READS", "DONE", "FRAME", "PART", "READ", "REFUSED", "SKIPPED", "WRITE", "build_receipts",
           "current_receipts", "folded_reads", "its_reads_are_receipted", "model_of", "notes_the_answering_model",
           "receipt", "receipts_frame", "saves_the_turns_receipts", "skipped_receipt", "the_loop_writes_receipts",
           "turn_receipts", "writes_its_receipts"]
