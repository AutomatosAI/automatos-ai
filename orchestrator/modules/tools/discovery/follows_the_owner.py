"""Auto acts only on the card the owner names, the way they asked (night 8: F241, F280, F281, F289).

The persona's first fix after night 8: "Auto acts only on the card I name, the way I
asked, and says done only when a call did it." Each platform call Auto makes in a
chat the owner drives is checked against the owner's own words (``owner_turn``)
before it runs, and refused, saying the call that does what they asked, when:

- a card's number goes to something that is not a card: platform_submit_social_post
  {post_id: "0201"}, platform_cancel_scheduled_task {task_id: 209}, a tool called
  "#0382" given to the Content Creator, blog post "0403";
- the owner named a card and the call changes something else: agent #323's own
  description for "Update #0296 …", a social post draft for "Send #0425 back";
- it makes a copy instead of acting: a new ticket #0287 for "Give #0285 to …", #0432 for
  "Update #0431 …", a second mission #0268 for "change #0267";
- it decided for the owner: an approval of #0329 when the owner only named it, an
  approval of #0422 when they said cancel. PRD-256 US-004: that is no longer read from
  their words here. An approval, a cancel and the rest of Decision D1 wait for the
  owner's click on the approval card (``owner_only``);
- it writes words as the owner's that they never wrote: notes signed "you", and briefs
  ("The description now reflects exactly what you wrote" over Auto's own checklist);
- it moves a card to In progress for a send-back, which re-runs the old brief without
  the owner's words (#0425).

Night 9b (F318): "Send card #0081 back to the Business Analyst with this: I only wanted
this September …" (chat 6586c8bf) became platform_update_task {description: …}: the
board's Re-brief. #1938 re-ran with Auto's rewording as its brief, and the card showed
times_sent_back 0 and no note: the owner's words were nowhere on it. A new brief for a
card the owner said to send back, when they asked for no new brief, is refused with the
send-back that carries their words.

Night 9b (F319): "My Business Analyst just worked out we're about 97 kg short … (card
1970). Can you get the Operations Manager to draft a reorder email …" (chat 8578eeaf):
the new card was refused as a copy of #1970, a card the owner named only as where the
figure came from, and Auto said it had assigned the work. A new card is a copy only when
the owner asked for something done to the card they named (give, send back, update,
run again).

Night 9 (F309):
- "Card 1869 needs to go back. Correction: …" became a question to the owner
  (platform_ask_human, ask #1460) and parked #1869 in Blocked; a question or a stop on a
  card the owner sent back is refused, with the send-back that carries their words;
- "approve card 1879" went to platform_approve_mission {mission_id: 1879}: a mission's
  call on a card that is not a mission's is refused, with the card's own call;
- a paraphrased note was refused with the owner's LATEST message quoted, which held no
  correction ("Send it back to the writer with that correction"), so the next call
  couldn't be right; the refusal now quotes the words the owner gave for the card, from
  whichever of their recent messages holds them, for the call to carry word for word;
- platform_update_task with a status now moves the card (``ticket_edit_moves``), so its
  move to Done is an approval the guard judges like platform_update_task_status's.

Outside a person's chat (a ticket, a playbook step, a heartbeat) nothing is checked.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from sqlalchemy.orm import Session

from modules.tools.discovery.brand_turns import refusal_on_a_brand_turn
from modules.tools.discovery.card_words_said import owners_card_words
from modules.tools.discovery.owner_turn import (
    AGAIN, GIVE, MISSION_CARD, NEW_CARD, NEW_MISSION, OTHER_VERB, RUN_CARD, SEND_BACK, SENDS_IT_BACK, STEP_CARD,
    UPDATE, NamedCard, OwnerTurn, autos_proposal, card_kind, card_words, names_card, owner_turn,
    owners_recent_words, says, values_in,
)

logger = logging.getLogger(__name__)

Execute = Callable[..., Awaitable[Dict[str, Any]]]

# The calls that act on a card, or on the mission or playbook run a card belongs to.
CARD_ACTIONS = frozenset({
    "platform_create_task", "platform_list_tasks", "platform_board_snapshot", "platform_board_summary",
    "platform_get_task", "platform_wait_for_task", "platform_assign_task", "platform_update_task",
    "platform_update_task_status", "platform_link_report_to_task", "platform_list_missions",
    "platform_get_mission", "platform_approve_mission", "platform_reject_mission", "platform_pause_mission",
    "platform_resume_mission", "platform_cancel_mission", "platform_replan_mission",
    "platform_update_mission_plan", "platform_get_playbook_execution",
})
# What a call works on, by the words in its name, as the owner would say it.
KINDS = (
    ("social_post", "a social post"), ("blog", "a blog post"), ("skill", "a skill"), ("plugin", "a tool"),
    ("tool", "a tool"), ("scheduled_task", "a scheduled task"), ("schedule", "a timer"),
    ("playbook", "a playbook"), ("agent", "an agent"), ("document", "a document"), ("report", "a report"),
    ("shopify", "a Shopify product"), ("chat_history", "the chat history"), ("ask_human", "a question"),
)
# The owner's words that name such a thing outright, so a call on it is theirs to ask for.
KIND_SAID = {
    "a social post": r"\bsocial\b|\binstagram\b|\bfacebook\b|\blinkedin\b", "a blog post": r"\bblog\b",
    "a skill": r"\bskills?\b", "a tool": r"\btools?\b|\bintegrations?\b", "a scheduled task": r"\bremind\w*\b",
    "a timer": r"\btimers?\b|\bschedul\w*\b", "a playbook": r"\bplaybooks?\b",
    "an agent": r"\bagent'?s (?:own )?(?:description|settings|name|instructions)\b",
    "a document": r"\bdocuments?\b|\bpdf\b", "a Shopify product": r"\bshopify\b",
}
PLAYBOOK = "a playbook"
NOTE_KEYS = ("note", "notes", "comment", "remark")
NOTE_SHARE, BRIEF_SHARE = 0.8, 0.6
COMMON_WORDS = frozenset({"the", "and", "for", "with", "this", "that", "your", "you", "are", "was", "but",
                          "not", "its", "it's", "from", "have", "has", "will", "into", "they", "them",
                          "then", "than", "just", "please", "card", "ticket", "task"})
NOTHING_DONE = " Nothing was done."
# The calls that move a card: platform_update_task moves it too when it carries a status (F309).
STATUS_CALLS = ("platform_update_task_status", "platform_update_task")
MISSION_CALLS = frozenset(action for action in CARD_ACTIONS if "_mission" in action)
# Where a mission's call names its mission (never a page size or a limit).
MISSION_REF_KEYS = ("mission_id", "run_id", "id")
# A card the owner can send back: one its agent has answered.
SENDABLE_BACK = ("review", "done")
# F318 (night 9b): the owner asking for a new brief, which a send-back keeps ("Put that brief on #0204").
NEW_BRIEF = re.compile(r"\bbrief\b|\bdescription\b", re.I)
# How much of the owner's words a refusal or a card's call quotes for the call to carry.
WORDS_QUOTED = 1000
OTHER_ASK = (" If the owner also asked for {kind}, ask them to confirm that in their next message.")
# A card's calls when the owner's words name no give, send-back or new brief: the model reads
# their words, and an approval or a cancel waits for their click (PRD-256 US-004).
CARD_CALLS_SAID = ("To approve it: platform_update_task_status {{task_id: \"{ref}\", status: \"done\", note: {note}}}; "
                   "to cancel it: platform_update_task_status {{task_id: \"{ref}\", status: \"cancelled\"}}. Either "
                   "waits for the owner's click on the approval card. To read it: platform_get_task "
                   "{{task_id: \"{ref}\"}}.")
MISSION_CALLS_SAID = ("To approve its plan: platform_approve_mission {{mission_id: \"{ref}\"}}; to cancel it: "
                      "platform_cancel_mission {{mission_id: \"{ref}\"}}. Either waits for the owner's click on the "
                      "approval card. To read it: platform_get_mission {{mission_id: \"{ref}\"}}.")


def follows_the_owner(execute: Execute) -> Execute:
    """Wrap PlatformExecutor.execute: a call that doesn't follow the owner's words is
    refused, before any gate or handler runs, with the call that does."""
    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        # Gerard, 7 Oct: a turn routed to the Brand designer changes no setting and starts no mission.
        refusal = (refusal_on_a_brand_turn(action_name)
                   or refusal_for(self.db, self.workspace_id, action_name, params, caller_context))
        if refusal:
            logger.info("[follows_the_owner] %s refused: %s", action_name, refusal[:160])
            return {"success": False, "error": refusal}
        return await execute(self, action_name, params, caller_context)
    return wrapped


def refusal_for(db: Session, workspace_id: Any, action: str, params: Any, caller_context: Any) -> Optional[str]:
    """Why this call doesn't follow the owner's words, or None. A fault here never
    stops a call: it is logged and the call goes on as before."""
    try:
        turn = owner_turn(db, workspace_id, caller_context)
        if turn is None:
            return None
        params = _as_dict(params)
        for rule in RULES:
            refusal = rule(db, workspace_id, turn, action, params)
            if refusal:
                return refusal
        return None
    except Exception:
        logger.exception("[follows_the_owner] could not check %s against the owner's words", action)
        return None


def _as_dict(params: Any) -> Dict[str, Any]:
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except (json.JSONDecodeError, TypeError):
            return {}
    return params if isinstance(params, dict) else {}


def _kind_of(action: str) -> str:
    return next((kind for stem, kind in KINDS if stem in action), "something that is not a card")


def _wrong_kind(db: Session, workspace_id: Any, turn: OwnerTurn, action: str, params: Dict[str, Any]) -> Optional[str]:
    """A card's number sent to a call that doesn't act on cards."""
    if action in CARD_ACTIONS or (action == "platform_ask_human" and params.get("subject_type") == "board_task"):
        return None
    for card in turn.found():
        if any(names_card(value, card) for value in values_in(params)):
            return (f"{card_words(card)} is a card on the owner's board, not {_kind_of(action)}.{NOTHING_DONE} "
                    f"{right_call(turn, card)}")
    return None


def _not_a_mission(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                   params: Dict[str, Any]) -> Optional[str]:
    """A mission's call on a card the owner named that is not a mission's (#1879, night 9)."""
    said = [params.get(key) for key in MISSION_REF_KEYS if params.get(key) is not None]
    if action not in MISSION_CALLS or not said:
        return None
    for card in turn.found():
        if card_kind(card) in (MISSION_CARD, STEP_CARD):
            continue
        if any(names_card(value, card) for value in said):
            return (f"{card_words(card)} is a card on the owner's board, not a mission.{NOTHING_DONE} "
                    f"{right_call(turn, card)}")
    return None


def _a_question_for_a_send_back(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                                params: Dict[str, Any]) -> Optional[str]:
    """A question to the owner, or a stop, for a card they sent back with their words:
    "Card 1869 needs to go back. Correction: …" became ask #1460 and Blocked (night 9)."""
    asks = action == "platform_ask_human" or (action in STATUS_CALLS and _status_of(params) == "blocked")
    if not asks or not says(SENDS_IT_BACK, (turn.latest,)):
        return None
    card = next((c for c in turn.found() if getattr(c.task, "status", None) in SENDABLE_BACK
                 and any(names_card(value, c) for value in values_in(params))), None)
    if card is None:
        return None
    return (f"The owner sent {card_words(card)} back with their correction: it goes back to its agent, who redoes "
            f"it, not to the owner as a question.{NOTHING_DONE} {_send_back_call(card.ref, _words_for_card(turn))}")


def _status_of(params: Dict[str, Any]) -> str:
    from modules.tools.execution.call_effects import STATUS_WORDS   # read the board's way, as the move is

    status = str(params.get("status") or "").strip().lower()
    return STATUS_WORDS.get(status, status)


def _not_the_card(db: Session, workspace_id: Any, turn: OwnerTurn, action: str, params: Dict[str, Any]) -> Optional[str]:
    """A change to something else when the owner named a card and not that thing."""
    found = turn.found()
    kind = _kind_of(action)
    if not found or action in CARD_ACTIONS or kind == "something that is not a card" or not _writes(action):
        return None
    if re.search(KIND_SAID.get(kind, r"(?!)"), turn.latest, re.I):
        return None
    if kind == PLAYBOOK and any(card_kind(card) == RUN_CARD for card in found):
        return None   # a playbook's own run card: running its playbook again is about that card
    card = found[0]
    return (f"The owner named {card_words(card)}, a card on their board, and {kind} isn't what they asked "
            f"about.{NOTHING_DONE} {right_call(turn, card)}{OTHER_ASK.format(kind=kind)}")


def _writes(action: str) -> bool:
    from modules.tools.discovery.action_registry import get_action_registry

    definition = get_action_registry().get(action)
    level = getattr(definition, "permission_level", None) if definition is not None else None
    return level in ("write", "destructive") or (definition is None and not action.startswith(("platform_get",
                                                                                               "platform_list")))


def _a_copy(db: Session, workspace_id: Any, turn: OwnerTurn, action: str, params: Dict[str, Any]) -> Optional[str]:
    """A new card or mission made for a card or mission the owner named."""
    found = turn.found()
    if action == "platform_create_task" and found and not says(NEW_CARD, turn.said) and _acts_on_it(turn):
        card = found[0]
        return (f"{card_words(card)} is already on the owner's board: act on that card, don't make a new one."
                f"{NOTHING_DONE} {right_call(turn, card)}")
    missions = [card for card in found if card_kind(card) == MISSION_CARD]
    if action != "platform_create_mission" or not missions or says(NEW_MISSION, turn.said):
        return None
    card, goal = missions[0], _mission_goal(db, missions[0])
    if says(AGAIN, turn.said):
        if goal and share_of_words(str(params.get("goal") or ""), [goal]) >= BRIEF_SHARE:
            return None
        return f"Start {card.ref} again with its own goal, word for word: '{goal}'.{NOTHING_DONE}"
    return (f"{card_words(card)} is the owner's mission: change it with platform_update_mission_plan "
            f"{{mission_id: \"{card.ref}\", ...}} instead of making a second one.{NOTHING_DONE}")


def _acts_on_it(turn: OwnerTurn) -> bool:
    """The owner's latest message asks for something done to the card it names: a new card
    for it would be a copy. A card named as a source ("(card 1970)") is not acted on (F319)."""
    return says(OTHER_VERB, (turn.latest,)) or says(AGAIN, (turn.latest,))


def _mission_goal(db: Session, card: NamedCard) -> str:
    from core.models.orchestration import OrchestrationRun

    run_id = getattr(card.task, "orchestration_run_id", None)
    run = db.get(OrchestrationRun, run_id) if run_id else None
    return str(getattr(run, "goal", "") or "").strip()


def _not_their_words(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                     params: Dict[str, Any]) -> Optional[str]:
    """A note signed as the owner's, or a new brief, in words the owner never wrote."""
    if action not in ("platform_update_task_status", "platform_update_task"):
        return None
    note = _signed_note(action, params)
    if note and not _theirs(db, workspace_id, turn, note, NOTE_SHARE):
        words = _their_words_for(db, workspace_id, turn, note)
        return ("A note on the card is signed as the owner's, so it is their own words, exactly as they wrote them "
                f"in this chat.{NOTHING_DONE} Make the same call again with these words as the note, word for word, "
                f"not shortened or reworded: {_quoted(words)}. Or make it without a note.")
    brief = str(params.get("description") or "")
    if action != "platform_update_task" or not brief or params.get("send_back"):
        return None
    extra = (*_the_cards_words(db, workspace_id, params), autos_proposal(db, workspace_id, turn))
    if _theirs(db, workspace_id, turn, brief, BRIEF_SHARE, extra):
        return None
    return ("A new brief is the owner's words, not yours: put the brief they wrote in this chat as the description, "
            f"word for word, or ask them for the brief.{NOTHING_DONE}")


def _signed_note(action: str, params: Dict[str, Any]) -> str:
    """The words the call would put on the card as the owner's: its note, or, for a
    send-back, the description it sends back as their correction (F279)."""
    note = next((str(params[key]) for key in NOTE_KEYS if params.get(key)), "")
    if not note and action == "platform_update_task" and params.get("send_back"):
        return str(params.get("description") or "")
    return note


def _theirs(db: Session, workspace_id: Any, turn: OwnerTurn, text: str, share: float, extra: tuple = ()) -> bool:
    """Whether ``text`` is the owner's words: from this turn, or from one of their last
    few messages ("put the brief I gave you above on #0296"), or from ``extra``."""
    if share_of_words(text, (*turn.said, *extra)) >= share:
        return True
    return share_of_words(text, (*owners_recent_words(db, workspace_id, turn), *extra)) >= share


def _the_cards_words(db: Session, workspace_id: Any, params: Dict[str, Any]) -> tuple:
    """The card's own brief and title: a new brief may keep them."""
    from core.models.core import BoardTask
    from services.ticket_refs import ticket_id_named

    task_id, _ = ticket_id_named(db, workspace_id, params.get("task_id"))
    task = (db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
            if task_id else None)
    return (str(task.description or ""), str(task.title or "")) if task is not None else ()


def _a_rerun_for_a_send_back(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                             params: Dict[str, Any]) -> Optional[str]:
    """In Progress for a card the owner sent back: it would re-run the old brief without their words.
    With their words as its note it is the board's Reject (ticket_moves, F318), so it goes on."""
    if action not in STATUS_CALLS or _status_of(params) != "in_progress" or not says(SEND_BACK, (turn.latest,)):
        return None
    if str(params.get("note") or "").strip():
        return None
    card = next((c for c in turn.found() if getattr(c.task, "status", None) in SENDABLE_BACK), None)
    if card is None:
        return None
    return (f"Moving {card.ref} to In progress would run its old brief again without the owner's words."
            f"{NOTHING_DONE} {_send_back_call(card.ref, _words_for_card(turn))}")


def _a_new_brief_for_a_send_back(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                                 params: Dict[str, Any]) -> Optional[str]:
    """A new brief for a card the owner sent back with their words (F318, chat 6586c8bf):
    the board's Re-brief, which puts no note on the card. Theirs when they asked for one."""
    if action != "platform_update_task" or not str(params.get("description") or "").strip():
        return None
    if params.get("send_back") or params.get("status") or not says(SENDS_IT_BACK, (turn.latest,)):
        return None
    if says(UPDATE, (turn.latest,)) or NEW_BRIEF.search(turn.latest):
        return None
    card = next((c for c in turn.found() if getattr(c.task, "status", None) in SENDABLE_BACK), None)
    if card is None:
        return None
    return (f"A new description would give {card.ref} a new brief, and the owner asked to send it back: their words "
            f"go on the card as the correction its redo fixes.{NOTHING_DONE} "
            f"{_send_back_call(card.ref, _words_for_card(turn))}")


def _words_for_card(turn: OwnerTurn) -> str:
    """The words the owner's latest message gives for the card ("Correction: …"), or ""."""
    return owners_card_words(turn.latest)


def _their_words_for(db: Session, workspace_id: Any, turn: OwnerTurn, note: str) -> str:
    """The owner's own words that ``note`` rewords: from whichever of their recent
    messages it shares most words with, the part after its label ("Correction: …"),
    else that whole message. Night 9: "Send it back to the writer with that correction"
    held no correction; the message before it did."""
    messages = [text for text in (*turn.said, *owners_recent_words(db, workspace_id, turn)) if text]
    if not messages:
        return turn.latest
    best = max(messages, key=lambda text: share_of_words(note, [text]))
    return owners_card_words(best) or best


def right_call(turn: OwnerTurn, card: NamedCard) -> str:
    """The call that does what the owner asked with this card: by their verb for a give, a
    send-back or a new brief; otherwise the card's calls, approving and cancelling waiting
    for the owner's click (PRD-256 US-004). The owner's words for it, when they labelled
    them ("Correction: …"), go in the call."""
    if card_kind(card) == MISSION_CARD:
        return MISSION_CALLS_SAID.format(ref=card.ref)
    note = _owners_note(_words_for_card(turn))
    for pattern, call in _CARD_CALLS:
        if pattern.search(turn.latest):
            return call.format(ref=card.ref, note=note)
    return CARD_CALLS_SAID.format(ref=card.ref, note=note)


def _owners_note(words: str) -> str:
    """A call's note: the owner's own words, quoted when they gave them for the card."""
    return f"the owner's own words, word for word: {_quoted(words)}" if words else "the owner's own words"


def _send_back_call(ref: str, words: str = "") -> str:
    return (f"To send it back: platform_update_task_status {{task_id: \"{ref}\", status: \"assigned\", "
            f"note: {_owners_note(words)}}}: its brief stays, and the agent redoes it on the same card.")


# F309 (night 9): giving comes before sending back, so "Give card 1859 to the Analyst
# instead" is a reassign, not a send-back ("instead").
_CARD_CALLS = (
    (GIVE, "To give it to an agent: platform_assign_task {{task_id: \"{ref}\", agent_name: …}}: a card its agent "
           "already answered goes to the new agent and runs again."),
    (SEND_BACK, "To send it back: platform_update_task_status {{task_id: \"{ref}\", status: \"assigned\", "
                "note: {note}}}: its brief stays, and the agent redoes it on the same card."),
    (UPDATE, "To give it a new brief: platform_update_task {{task_id: \"{ref}\", description: the owner's "
             "brief, word for word}}."),
)


def share_of_words(text: str, sources: Any) -> float:
    """How much of ``text`` (its words of three letters or more) is in ``sources``."""
    said = _words(text)
    if not said:
        return 1.0
    pool = set().union(*(_words(source) for source in sources)) if sources else set()
    return len(said & pool) / len(said)


def _words(text: str) -> set:
    return {word for word in re.findall(r"[a-z0-9£$%']{3,}", (text or "").lower()) if word not in COMMON_WORDS}


def _quoted(text: str) -> str:
    """The owner's words, as one JSON string, up to the length a note keeps."""
    return json.dumps(" ".join((text or "").split())[:WORDS_QUOTED], ensure_ascii=False)


RULES = (_wrong_kind, _not_a_mission, _a_question_for_a_send_back, _not_the_card, _a_copy, _not_their_words,
         _a_rerun_for_a_send_back, _a_new_brief_for_a_send_back)

__all__ = ["CARD_ACTIONS", "follows_the_owner", "refusal_for", "right_call", "share_of_words"]
