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
- it decides for the owner: an approval of #0329 when the owner only named it, an
  approval of #0422 when they said cancel. An approval needs the owner's approving
  words now, or in the message before when this one only says which card; a bare
  "yes" or "ok" counts only as the answer to Auto's own question about approving;
- it writes words as the owner's that they never wrote: notes signed "you", and briefs
  ("The description now reflects exactly what you wrote" over Auto's own checklist);
- it moves a card to In progress for a send-back, which re-runs the old brief without
  the owner's words (#0425).

Outside a person's chat (a ticket, a playbook step, a heartbeat) nothing is checked.
"""
from __future__ import annotations

import functools
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, Optional

from sqlalchemy.orm import Session

from modules.tools.discovery.owner_turn import (
    AGAIN, APPROVE, CANCEL, CARD_REF, GIVE, GO_AHEAD, MISSION_CARD, NEW_CARD, NEW_MISSION, NOT_YET, OTHER_VERB,
    RUN_CARD, SEND_BACK, UPDATE, NamedCard, OwnerTurn, autos_last_reply, autos_proposal, card_kind, card_words,
    names_card, owner_turn, owners_recent_words, says, values_in,
)
from modules.tools.discovery.ticket_moves import STATUS_WORDS

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
QUOTE_CHARS = 240
COMMON_WORDS = frozenset({"the", "and", "for", "with", "this", "that", "your", "you", "are", "was", "but",
                          "not", "its", "it's", "from", "have", "has", "will", "into", "they", "them",
                          "then", "than", "just", "please", "card", "ticket", "task"})
NOTHING_DONE = " Nothing was done."
OTHER_ASK = (" If the owner also asked for {kind}, ask them to confirm that in their next message.")


def follows_the_owner(execute: Execute) -> Execute:
    """Wrap PlatformExecutor.execute: a call that doesn't follow the owner's words is
    refused, before any gate or handler runs, with the call that does."""
    @functools.wraps(execute)
    async def wrapped(self: Any, action_name: str, params: Any, caller_context: Any = None) -> Dict[str, Any]:
        refusal = refusal_for(self.db, self.workspace_id, action_name, params, caller_context)
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
    if action == "platform_create_task" and found and not says(NEW_CARD, turn.said):
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


def _mission_goal(db: Session, card: NamedCard) -> str:
    from core.models.orchestration import OrchestrationRun

    run_id = getattr(card.task, "orchestration_run_id", None)
    run = db.get(OrchestrationRun, run_id) if run_id else None
    return str(getattr(run, "goal", "") or "").strip()


def _not_what_they_said(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                        params: Dict[str, Any]) -> Optional[str]:
    """An approval or a cancel the owner didn't ask for."""
    decided = _decision(action, params)
    if decided == "approve" and says(CANCEL, (turn.latest,)):
        return f"The owner said cancel, not approve.{NOTHING_DONE} {_cancel_call(action, turn)}"
    if decided == "approve" and says(NOT_YET, (turn.latest,)):
        return (f"The owner hasn't approved {_which(turn)} yet: answer what they asked first.{NOTHING_DONE}")
    if decided == "approve" and not _owner_approved(db, workspace_id, turn):
        return (f"The owner hasn't said to approve {_which(turn)}.{NOTHING_DONE} Ask them what they want done "
                "with it, in their words.")
    if decided == "cancel" and not says(CANCEL, turn.said):
        return f"The owner hasn't said to cancel {_which(turn)}.{NOTHING_DONE} Ask them first."
    return None


def _owner_approved(db: Session, workspace_id: Any, turn: OwnerTurn) -> bool:
    """The owner's go-ahead to approve: their approving words now; or in the message
    before, when this one only says which card it meant; or a yes to Auto's own question
    about approving. An "ok" in an earlier message about something else is none of these:
    #0329 was approved when the owner had only said which card they meant."""
    if says(APPROVE, (turn.latest,)):
        return True
    if says(APPROVE, (turn.earlier,)) and not says(OTHER_VERB, (turn.latest,)) and _same_cards(turn):
        return True
    return says(GO_AHEAD, (turn.latest,)) and bool(APPROVE.search(autos_last_reply(db, workspace_id, turn)))


def _same_cards(turn: OwnerTurn) -> bool:
    """The cards the earlier message named, if any, are the ones the latest names."""
    earlier = _numbers(turn.earlier)
    return not earlier or earlier <= _numbers(turn.latest)


def _numbers(text: str) -> set:
    """The card numbers in ``text``, as (number, step): "#329" and "#0329" are one card."""
    return {(int(match.group(1)), match.group(2)) for match in CARD_REF.finditer(text or "")}


def _decision(action: str, params: Dict[str, Any]) -> Optional[str]:
    if action in ("platform_approve_mission", "platform_cancel_mission"):
        return "approve" if action == "platform_approve_mission" else "cancel"
    if action != "platform_update_task_status":
        return None
    status = str(params.get("status") or "").strip().lower()
    status = STATUS_WORDS.get(status, status)
    return {"done": "approve", "cancelled": "cancel"}.get(status)


def _which(turn: OwnerTurn) -> str:
    found = turn.found()
    return found[0].ref if found else "it"


def _cancel_call(action: str, turn: OwnerTurn) -> str:
    ref = _which(turn)
    if action == "platform_approve_mission":
        return f"To cancel it: platform_cancel_mission {{mission_id: \"{ref}\"}}."
    return f"To cancel it: platform_update_task_status {{task_id: \"{ref}\", status: \"cancelled\"}}."


def _not_their_words(db: Session, workspace_id: Any, turn: OwnerTurn, action: str,
                     params: Dict[str, Any]) -> Optional[str]:
    """A note signed as the owner's, or a new brief, in words the owner never wrote."""
    if action not in ("platform_update_task_status", "platform_update_task"):
        return None
    note = _signed_note(action, params)
    if note and not _theirs(db, workspace_id, turn, note, NOTE_SHARE):
        return ("A note on the card is signed as the owner's, so it is their own words, as they wrote them in this "
                f"chat (their latest message: '{_quoted(turn.latest)}').{NOTHING_DONE} Resend it with their words, "
                "or without a note.")
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
    """In Progress for a card the owner sent back: it would re-run the old brief without their words."""
    status = str(params.get("status") or "").strip().lower()
    if action != "platform_update_task_status" or status != "in_progress" or not says(SEND_BACK, (turn.latest,)):
        return None
    card = next((c for c in turn.found() if getattr(c.task, "status", None) in ("review", "done")), None)
    if card is None:
        return None
    return (f"Moving {card.ref} to In progress would run its old brief again without the owner's words."
            f"{NOTHING_DONE} {_send_back_call(card.ref)}")


def right_call(turn: OwnerTurn, card: NamedCard) -> str:
    """The call that does what the owner asked with this card, by their verb."""
    ref = card.ref
    if card_kind(card) == MISSION_CARD:
        if says(CANCEL, (turn.latest,)):
            return f"To cancel it: platform_cancel_mission {{mission_id: \"{ref}\"}}."
        if says(APPROVE, (turn.latest,)):
            return f"To approve its plan: platform_approve_mission {{mission_id: \"{ref}\"}}."
        return f"To read it: platform_get_mission {{mission_id: \"{ref}\"}}."
    for pattern, call in _CARD_CALLS:
        if pattern.search(turn.latest):
            return call.format(ref=ref)
    return f"To read it: platform_get_task {{task_id: \"{ref}\"}}."


def _send_back_call(ref: str) -> str:
    return (f"To send it back: platform_update_task_status {{task_id: \"{ref}\", status: \"assigned\", "
            "note: the owner's own words}: its brief stays, and the agent redoes it on the same card.")


_CARD_CALLS = (
    (CANCEL, "To cancel it: platform_update_task_status {{task_id: \"{ref}\", status: \"cancelled\"}}."),
    (APPROVE, "To approve it: platform_update_task_status {{task_id: \"{ref}\", status: \"done\", "
              "note: the owner's own words}}."),
    (SEND_BACK, "To send it back: platform_update_task_status {{task_id: \"{ref}\", status: \"assigned\", "
                "note: the owner's own words}}: its brief stays, and the agent redoes it on the same card."),
    (GIVE, "To give it to an agent: platform_assign_task {{task_id: \"{ref}\", agent_name: …}}."),
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
    text = " ".join((text or "").split())
    return text if len(text) <= QUOTE_CHARS else text[:QUOTE_CHARS].rstrip() + "…"


RULES = (_wrong_kind, _not_the_card, _a_copy, _not_what_they_said, _not_their_words, _a_rerun_for_a_send_back)

__all__ = ["CARD_ACTIONS", "follows_the_owner", "refusal_for", "right_call", "share_of_words"]
