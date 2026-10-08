"""PRD-256 US-001: the receipts, what Auto did this turn, written by the platform.

Eleven customer nights found Auto's worst habit: claiming work it did not do, most often
after a call that FAILED on its arguments and was then reported as done. A reply is no
longer the only account of its turn. After the tool loop the platform writes its own, from
the calls that ran: the in-process ``ToolExecutionTracker`` (``outcomes``, ``skipped``)
read with ``call_effects``, never the persisted tool-execution log (D8: it keeps parameter
names only, no result and no chat).

- One receipt per call: ``{action, kind, status, subject, effect, link, reason}``. The kind
  is read or write; the status done, refused, skipped or waiting (FX-004: an ask for the
  owner's click, ``card raised: <verb> <subject>``, is never a refusal). The subject is the thing by number
  or name (card #0422, an agent, a document's title); the effect is what the call did, in
  plain words ("moved to Done", "sent back to its agent"); a refused or skipped call says
  why, on one line. A link is the page of what a done call touched (its card, its agent).
- The reads the turn makes before the model's first call (retrieval first, the Needs-you
  read, the team's findings: ``prefetched``) fold into ONE read receipt.

The turn streams them as one ``receipts`` frame, with the model that answered: after the
loop, before the answer's additions, or, for a turn that ran no loop, before its finish (the
web chat reads the same data from the ``tool-data`` frame just before it). It
saves them as the message's ``receipts`` part (``narration.reply_parts``; ``[]`` when nothing
ran). They are built after the model is done: no prompt names them and no tool can set them.

The wiring is decorators on the turn's seams, so the turn's own long functions are untouched:
``writes_its_receipts`` (the turn, whichever agent answers it: Auto, or a specialist on a
DELEGATE turn), ``its_reads_are_receipted`` (retrieval first), ``the_loop_writes_receipts``
(the tool loop), ``notes_the_answering_model`` (the answer's additions) and
``saves_the_turns_receipts`` (the saved parts). The turn's state is a few context variables,
set once per turn and never mutated.

US-002, one honesty rule: the line about what was not done comes from the receipts alone
(``honesty_lines``). FX-006: it is decided per claim (``claims_backed``): a report of work done
(``COMPLETED_ACTION``, one generic pattern) with no done write of its kind behind it gets the
line, whatever else went through; a refused write gets its own line.
``the_answer_takes_the_receipts`` (the answer's additions) settles them when the loop has not; they go above the text, in the frame (``above``) and at the top of
the saved answer, never under it. FX-005: they are the only not-done line; the turn's
``no_tool_call`` notice never says it again (``says_nothing_was_done``).
"""
from __future__ import annotations

import functools
from contextvars import ContextVar
from typing import Any, AsyncGenerator, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from consumers.chatbot.claim_check import NOTHING_DONE
from consumers.chatbot.claims_backed import (
    COMPLETED_ACTION, claims_work_done, is_not_done_line, not_done_line, unbacked_claims,
)
from modules.tools.execution.call_effects import (
    AGENT_SET, DOCUMENT_MAKES, REVIEWED_BY_YOU, SENT_BACK, STATUS_IGNORED_SAID, STEPS_CHECKED, STEPS_UNCHECKED,
    answers_in, done_effects,
)
from modules.tools.execution.card_raised import is_waiting, receipt_effect
from modules.tools.execution.tool_execution_tracker import TRACKERS_MADE
from modules.tools.execution.turn_account import is_read, thing_of, what_it_did

Receipt = Dict[str, Any]
Prefetched = Sequence[Tuple[str, Dict[str, Any]]]

PART = FRAME = LIVE_KEY = "receipts"
READ, WRITE = "read", "write"
DONE, REFUSED, SKIPPED, WAITING = "done", "refused", "skipped", "waiting"
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
         STEPS_CHECKED: "each step waits for your OK", STEPS_UNCHECKED: "its steps run without your check",
         REVIEWED_BY_YOU: "card created, reviewed by you before it closes",  # FX-010 (D7)
         STATUS_IGNORED_SAID: "status ignored: a re-brief sends the card back by itself"}  # FX-013
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
    tags = [effect.rsplit(":", 1)[-1] for effect in done_effects(action, params, result)]
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
    """The receipt of one call that ran: done, refused with its reason, or waiting for the
    owner's click (an ask: its card is raised, nothing is refused)."""
    params = params if isinstance(params, dict) else {}
    answers = answers_in(result)
    if is_waiting(result):
        return _waiting_receipt(action, _subject(answers, params), result)
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


def _waiting_receipt(action: str, subject: str, result: Any) -> Receipt:
    """An ask: a write that waits for the owner's click, with no reason (it was not refused)."""
    return {"action": action, "kind": WRITE, "status": WAITING, "subject": subject,
            "effect": receipt_effect(result, action, subject), "link": None, "reason": None}


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


def _frame_data(receipts: List[Receipt], model: Optional[str], above: Sequence[str]) -> Dict[str, Any]:
    data: Dict[str, Any] = {"receipts": list(receipts)}
    if model:
        data["model"] = model
    if above:
        data[ABOVE] = list(above)
    return data


def receipts_frame(handler: Any, receipts: List[Receipt], model: Optional[str] = None,
                   above: Sequence[str] = ()) -> str:
    """The one ``receipts`` frame of a turn, with the model that answered when it is known and
    the lines the reply carries above its text (US-002) when there are any."""
    return handler.format_aisdk_data(FRAME, _frame_data(receipts, model, above))


def receipts_frames(handler: Any, receipts: List[Receipt], model: Optional[str] = None,
                    above: Sequence[str] = ()) -> Tuple[str, ...]:
    """The same data on a ``tool-data`` frame (under ``LIVE_KEY``), then the frame. The web
    chat's stream reader hands a ``tool-data`` frame to its data callback, where the live
    message takes its receipts (frontend lib/chat/use-chat-with-receipts.ts); it passes over a
    frame type it does not know."""
    live = handler.format_aisdk_tool_data({LIVE_KEY: _frame_data(receipts, model, above)})
    return live, receipts_frame(handler, receipts, model, above)


# ── US-002: one honesty rule, from the receipts alone ──────────────────────
# The line about what was not done is decided by the receipts, never by a vocabulary of
# claims (the regex families are gone, FX-007). It fires for a
# claim of the answer (one generic pattern, ``claims_backed.COMPLETED_ACTION``) that no done
# write of its kind backs, whatever else went through (FX-006); a write of its kind is never
# denied. A refused write gets its own line. Both sit above the text: in the frame
# (``above``) live, at the top of the saved answer on reload.
ABOVE = "above"
NOTHING_DONE_LINE = NOTHING_DONE   # "Just to be clear: I haven't done that yet, and nothing has changed. …"
TRIED_LINE = "I tried to {what} and it didn't go through: {reason}."
# How a refused write is named in "I tried to <what>": the board's own words where a call's
# name reads badly, else "<verb> the <thing>".
_TRIED = {"update_task_status": "move the card", "update_task": "change the card", "send_back": "send the card back",
          "assign_tool_to_agent": "give the agent that tool",
          "unassign_tool_from_agent": "take that tool off the agent",
          "store_memory": "save that to memory", "execute_playbook": "run the playbook"}
_VERBS = ("create", "update", "delete", "cancel", "assign", "approve", "reject", "schedule", "run", "send", "submit",
          "install", "pause", "resume", "upload", "add", "remove", "publish", "set", "generate", "make", "move", "post")


def _tried(r: Receipt) -> str:
    """A refused write in plain words, for "I tried to <what>": "move the card #0422"."""
    name = r["action"].lower().removeprefix("platform_")
    verb = name.split("_")[0]
    what = _TRIED.get(name) or (f"{verb} the {thing_of(name)}" if verb in _VERBS else "do that")
    subject = r.get("subject") or ""
    if not subject:
        return what
    return f"{what} {subject}" if subject.startswith("#") else f'{what} "{subject}"'


def honesty_lines(receipts: Sequence[Receipt], answer: str) -> List[str]:
    """The lines above the answer: one per write that was refused (and not then done), and the
    not-done line when a claim of the answer has no done write of its kind behind it (FX-006:
    per claim, whatever else went through; it names the claim when another write did). A write
    that waits for the owner's click (FX-004) is neither done nor refused: it writes no "I tried
    to" line, and the answer that calls it done still gets the not-done line."""
    writes = [r for r in receipts if r.get("kind") == WRITE]
    done_writes = [r for r in writes if r.get("status") == DONE]
    done = {r["action"] for r in done_writes}
    refused: Dict[str, Receipt] = {}
    for r in writes:
        if r.get("status") == REFUSED and r["action"] not in done:
            refused.setdefault(r["action"], r)          # one line per action: its first reason
    lines = [TRIED_LINE.format(what=_tried(r), reason=(r.get("reason") or FAILED).rstrip(". "))
             for r in refused.values()]
    not_done = not_done_line(answer, done_writes)
    return [*lines, not_done] if not_done else lines


def unbacked_claim(answer: str, outcomes: Iterable[Tuple[str, Dict[str, Any], Any]]) -> Optional[str]:
    """FX-007, F108's nudge from the receipts: the verb of the answer's first claim that no done
    write of the calls so far backs (``outcomes``: the loop tracker's), "done" for a claim whose
    verb is in no family ("Done.", "it's now on your board"), else None."""
    receipts = [receipt(action, params, result) for action, params, result in outcomes]
    done_writes = [r for r in receipts if r["kind"] == WRITE and r["status"] == DONE]
    unbacked = unbacked_claims(answer, done_writes)
    if not unbacked:
        return None
    verb, known = unbacked[0]
    return verb if known else "done"


def with_lines_above(answer: str, above: Sequence[str]) -> str:
    """The saved answer: the honesty lines first, then the text."""
    return "\n\n".join([*above, answer]) if above else answer


# ── the turn ───────────────────────────────────────────────────────────────
# One value per variable, set by the turn's seams, never mutated: whether a chat turn is
# running, its automatic reads, its loop's receipts once the loop ended, the model that
# answered, whether its frame went out, and the lines above its answer (US-002) once settled.
_IN_TURN: ContextVar[bool] = ContextVar("receipts_in_turn", default=False)
_PREFETCHED: ContextVar[Prefetched] = ContextVar("receipts_prefetched", default=())
_LOOP: ContextVar[Optional[List[Receipt]]] = ContextVar("receipts_loop", default=None)
_MODEL: ContextVar[Optional[str]] = ContextVar("receipts_model", default=None)
_SENT: ContextVar[bool] = ContextVar("receipts_sent", default=False)
_ABOVE: ContextVar[Optional[List[str]]] = ContextVar("receipts_above", default=None)
_VISITOR: ContextVar[bool] = ContextVar("receipts_visitor", default=False)

Stream = Callable[..., AsyncGenerator[Any, None]]
Additions = Callable[[Any, Any], List[str]]
Parts = Callable[..., List[Dict[str, Any]]]


def current_receipts() -> Optional[List[Receipt]]:
    """This chat turn's receipts: its loop's once the loop ended, else its automatic reads
    (``[]`` when nothing ran); None outside a chat turn."""
    if not _IN_TURN.get():
        return None
    return turn_receipts(_LOOP.get(), _PREFETCHED.get())


def _settle_above(receipts: Sequence[Receipt], answer: str) -> List[str]:
    """The turn's lines above its answer, decided once from its receipts and its answer. A public
    widget visitor's turn has none (F155: a refused call's reason is the platform's, not theirs)."""
    above = [] if _VISITOR.get() else honesty_lines(receipts, answer)
    _ABOVE.set(above)
    return above


def says_nothing_was_done(answer: str) -> bool:
    """Whether the turn's receipts put the not-done line above ``answer`` (FX-005: the one
    producer of that line). Another notice that would say the same defers to it."""
    receipts = current_receipts() or []
    return not _VISITOR.get() and any(is_not_done_line(line) for line in honesty_lines(receipts, answer))


def _frames_once(chat: Any, receipts: List[Receipt], model: Optional[str],
                 above: Sequence[str] = ()) -> Tuple[str, ...]:
    """The turn's frame, the first time it is asked for. A public widget visitor is sent none
    (F155: a visitor sees no internals); the receipts are still saved with the message."""
    handler = getattr(chat, "streaming_handler", None)
    if _SENT.get() or handler is None:
        return ()
    _SENT.set(True)
    if getattr(chat, "widget_mode", False):
        return ()
    return receipts_frames(handler, receipts, model, above)


def writes_its_receipts(turn: Stream) -> Stream:
    """Wrap ``StreamingChatService.stream_response_with_agent``: every turn has receipts,
    whichever agent answers it. A turn that ran no loop sends its frame before its finish."""
    @functools.wraps(turn)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        for var, fresh in ((_IN_TURN, True), (_PREFETCHED, ()), (_LOOP, None), (_MODEL, None), (_SENT, False),
                           (_ABOVE, None), (_VISITOR, bool(getattr(chat, "widget_mode", False)))):
            var.set(fresh)
        handler = getattr(chat, "streaming_handler", None)
        finish = handler.format_aisdk_finish() if handler is not None else None
        try:
            async for chunk in turn(chat, *args, **kwargs):
                frames = (_frames_once(chat, current_receipts() or [], _MODEL.get(), _ABOVE.get() or ())
                          if finish and chunk == finish else ())
                for frame in frames:
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
    before the answer's additions, with the lines above the answer (US-002). An answer that
    came back empty is forced once more by the turn: its frame then goes out before the finish."""
    @functools.wraps(loop)
    async def wrapped(chat: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        made: List[Any] = []
        TRACKERS_MADE.set(made)
        try:
            async for chunk in loop(chat, *args, **kwargs):
                final = chunk.get("_final_response") if isinstance(chunk, dict) else None
                if final is not None:
                    receipts = build_receipts(made[0] if made else None, kwargs.get("prefetched") or ())
                    _LOOP.set(receipts)
                    answer = getattr(final, "content", None)
                    frames = _frames_once(chat, receipts, model_of(final), _settle_above(receipts, answer)) \
                        if answer else ()
                    for frame in frames:
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


def the_answer_takes_the_receipts(answer_additions: Additions) -> Additions:
    """Wrap ``StreamingChatService._answer_additions`` (US-002): the lines above the answer are
    settled from the turn's receipts and the answer that is kept (the first reply, a forced
    synthesis), unless the loop already settled them on its own answer. Nothing is added
    under the text: the lines go above it, in the frame and in the saved answer."""
    @functools.wraps(answer_additions)
    def wrapped(f187_verdict: Any, final_round: Any) -> List[str]:
        if _IN_TURN.get() and _ABOVE.get() is None:
            _settle_above(current_receipts() or [], getattr(final_round, "content", None) or "")
        return answer_additions(f187_verdict, final_round)
    return wrapped


def saves_the_turns_receipts(reply_parts: Parts) -> Parts:
    """Wrap ``narration.reply_parts``: the message a chat turn saves carries its receipts, and
    its answer starts with the lines the receipts put above it (US-002)."""
    @functools.wraps(reply_parts)
    def wrapped(reasoning: str, narration: str, answer: str,
                receipts: Optional[List[Receipt]] = None) -> List[Dict[str, Any]]:
        if receipts is not None:
            return reply_parts(reasoning, narration, answer, receipts)
        return reply_parts(reasoning, narration, with_lines_above(answer, _ABOVE.get() or ()), current_receipts())
    return wrapped


__all__ = ["ABOVE", "AUTOMATIC_READS", "COMPLETED_ACTION", "DONE", "FRAME", "LIVE_KEY", "NOTHING_DONE_LINE", "PART",
           "READ", "REFUSED", "SKIPPED", "TRIED_LINE", "WAITING", "WRITE", "build_receipts", "claims_work_done",
           "current_receipts", "folded_reads", "honesty_lines", "its_reads_are_receipted", "model_of", "notes_the_answering_model",
           "receipt", "receipts_frame", "receipts_frames", "saves_the_turns_receipts", "says_nothing_was_done", "skipped_receipt",
           "the_answer_takes_the_receipts", "the_loop_writes_receipts", "turn_receipts", "unbacked_claim", "with_lines_above",
           "writes_its_receipts"]
