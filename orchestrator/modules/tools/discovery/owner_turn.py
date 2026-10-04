"""What the owner said in this chat turn, for the calls Auto makes in it (F241, F280, F281).

Night 8 (build 12), Auto in the owner's chat:
- acted on the wrong kind of thing for a card's number: "Approve #0201" became
  platform_submit_social_post {post_id: "0201"}, "Cancel #0209" a scheduled task,
  "Give #0211 to the Analyst" skill 211, "Give #0382 …" a tool called "#0382" on the
  Content Creator, "Update #0296 …" agent #323's own description, "Update #0403 …"
  blog post "0403". 17 of 95 first tries landed on the card;
- decided for the owner: told only which card was meant, it approved #0329 with the
  note "Reply to Raj Patel about skipping November has been reviewed and approved.",
  signed "you"; asked to cancel #0422, it approved it with "User chalked it up
  themselves.", signed "you".

This reads the turn once: the owner's latest words (and the message before, for a
"yes" that answers a question), the cards they name by number, and the verb they
used. ``follows_the_owner`` checks each call against it.

Night 9 (F309): the owner named cards in words, "card 1879", "Card 1869", "card 27.2",
and neither the note nor the guard knew them: "card 1879" was looked up as a mission,
"27.2" as a social post. A card named with a card word counts now
(``card_words_said``), by its id or its number; and "needs to go back. Correction: …"
is a send-back.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# The board's own form of a card's number: #0201, or #0428.2 for a mission's step.
CARD_REF = re.compile(r"(?<![\w&#])#(\d{3,6})(?:\.(\d{1,3}))?\b")
MAX_CARDS = 5
MISSION_CARD, STEP_CARD, RUN_CARD = "orchestration", "orchestration_task", "recipe"

# The owner's verbs, as night 8 said them.
CANCEL = re.compile(r"\bcancel\w*\b|\bscrap (?:it|this|that)\b|\bnot needed\b"
                    r"|\b(?:don'?t|do not|no longer) need (?:it|this|that|the card|the ticket)\b", re.I)
APPROVE = re.compile(r"\bapprov\w*\b|\baccept\w*\b|\bsign(?:ed)? (?:it )?off\b|\bmark (?:it |this |that )?(?:as )?done\b"
                     r"|\bthat'?s (?:the one|right|it|fine|good)\b|\blooks? (?:good|right|fine)\b|\bgood to go\b", re.I)
# A yes that answers Auto's question ("Shall I approve it?") counts as the owner's go-ahead.
GO_AHEAD = re.compile(r"\b(?:yes|yep|yeah|go ahead|go on|do it|please do|ok(?:ay)?)\b", re.I)
# Night 9 (F309): "Card 1869 needs to go back. Correction: …" is a send-back too.
SEND_BACK = re.compile(r"\bsend\b(?:\W+\w+){0,3}?\W+back\b|\bsent back\b|\breject\w*\b|\bredo\b|\bre-do\b"
                       r"|\btake (?:out|off)\b|\bfix\b|\bwrong\b|\bisn'?t right\b|\bnot right\b|\bshould (?:be|say|start)\b"
                       r"|\binstead\b|\bgo(?:es|ing)? back\b|\bcorrections?\s*:|\b(?:give|hand)\s+(?:it|this|that|them)\s+back\b",
                       re.I)
# A send-back said outright, never by "instead" alone ("Give card 1859 to the Analyst instead").
SENDS_IT_BACK = re.compile(r"\bsend\b(?:\W+\w+){0,3}?\W+back\b|\bsent back\b|\breject\w*\b|\bgo(?:es|ing)? back\b"
                           r"|\bcorrections?\s*:|\b(?:give|hand)\s+(?:it|this|that|them)\s+back\b", re.I)
# "Give it back" and "hand it back" send a card back; "give card 1859 to …" gives it (F309).
GIVE = re.compile(r"\b(?:give|assign|hand)\b(?!\s+(?:it|this|that|them)\s+back\b)"
                  r"|\bput\b(?:\W+\w+){0,4}?\W+on (?:the |my )?[A-Z]", re.I)
UPDATE = re.compile(r"\bupdat\w*\b|\bre-?brief\w*\b|\bnew brief\b|\bchange (?:its|the|that) brief\b|\brewrite\b", re.I)
NEW_CARD = re.compile(r"\b(?:new|another|separate|second|extra|fresh)\s+(?:card|ticket|task)\b"
                      r"|\b(?:create|make|add|open|start)\s+(?:a|an)\s+(?:\w+\s+){0,2}(?:card|ticket|task)\b", re.I)
NEW_MISSION = re.compile(r"\b(?:new|another|second|separate)\s+mission\b|\bstart (?:a|another) mission\b", re.I)
AGAIN = re.compile(r"\b(?:start|run|do)\s+(?:it|this|that|them|#\S+)\s+again\b|\bstart again\b|\brestart\b", re.I)
# "Before I approve #0410: is it set to stop after each step?" is not an approval, nor is
# "No, …" or "Not yet, ok?" to Auto's "Shall I approve it?".
NOT_YET = re.compile(r"\bbefore (?:i|we) approve\b|\b(?:don'?t|do not|not) approve\b|\bnot yet\b|^\W*no\b"
                     r"|\bhold (?:on|off)\b|\bhang on\b|\bwait,|\bwait a (?:sec|second|minute|moment)\b", re.I)
# The owner's other verbs: a message with one of them is not just saying which card.
OTHER_VERB = re.compile("|".join(f"(?:{pattern.pattern})" for pattern in (CANCEL, SEND_BACK, GIVE, UPDATE)), re.I)
# The owner taking Auto's own proposal ("Yes, that's it… put that brief on #0204"), unless
# they said the words are theirs ("I didn't ask you to write it… put what I wrote", #0451).
AGREES = re.compile(r"\b(?:yes|yep|yeah|that'?s it|that'?s right|exactly|agreed|perfect|sounds good"
                    r"|go with (?:that|it)|use (?:that|it|this|your)|put (?:that|it|this|your))\b", re.I)
THEIRS = re.compile(r"\bdidn'?t ask you to write\b|\bwhat i wrote\b|\bmy (?:own )?words\b|\bmy brief\b", re.I)
AUTO_ROLE = "assistant"
# How many of the owner's messages a note or a brief signed as theirs may come from: on
# night 8 the owner asked for "the brief I gave you above" two messages after giving it (#0296).
OWNER_HISTORY = 6


@dataclass(frozen=True)
class NamedCard:
    """A card the owner named: its number as they wrote it, and the card (None when
    no card on the board has that number). ``by_id``: they named it by its id, in
    words ("card 1879", F309), and ``ref`` is the board's number for it."""
    ref: str
    seq: int
    step: Optional[int]
    task: Any
    by_id: bool = False


@dataclass(frozen=True)
class OwnerTurn:
    """The owner's words this turn (``latest``, and ``earlier``, the message before)
    and the cards they named by number."""
    latest: str
    earlier: str
    cards: Tuple[NamedCard, ...]
    chat_id: Optional[UUID] = None

    @property
    def said(self) -> Tuple[str, ...]:
        return tuple(text for text in (self.latest, self.earlier) if text)

    def found(self) -> Tuple[NamedCard, ...]:
        return tuple(card for card in self.cards if card.task is not None)


def owner_turn(db: Session, workspace_id: Any, caller_context: Any) -> Optional[OwnerTurn]:
    """The turn the call is made in, or None outside a chat a person drives."""
    from core.security.surface import widget_turn
    from modules.tools.discovery.handlers_board_task_review import owner_words

    chat_id = _chat_id(caller_context)
    if chat_id is None or widget_turn():
        return None
    words = owner_words(db, _as_uuid(workspace_id), chat_id)
    if not words or not words[0]:
        return None
    latest, earlier = words[0], (words[1] if len(words) > 1 else "")
    return OwnerTurn(latest=latest, earlier=earlier or "", cards=cards_named(db, workspace_id, latest),
                     chat_id=chat_id)


def autos_proposal(db: Session, workspace_id: Any, turn: OwnerTurn) -> str:
    """Auto's last reply in the chat when the owner's latest words take it as it is
    ("Yes, that's it"), else ""."""
    if not AGREES.search(turn.latest) or THEIRS.search(turn.latest):
        return ""
    return autos_last_reply(db, workspace_id, turn)


def autos_last_reply(db: Session, workspace_id: Any, turn: OwnerTurn) -> str:
    """Auto's last reply in the chat, or "". Read in a savepoint, like the owner's words."""
    texts = _chat_words(db, workspace_id, turn, AUTO_ROLE, 1)
    return texts[0] if texts else ""


def owners_recent_words(db: Session, workspace_id: Any, turn: OwnerTurn) -> Tuple[str, ...]:
    """The owner's last ``OWNER_HISTORY`` messages in the chat, newest first: a brief
    they gave a few messages back is still their words."""
    from modules.tools.discovery.handlers_board_task_review import OWNER_ROLE

    return _chat_words(db, workspace_id, turn, OWNER_ROLE, OWNER_HISTORY)


def _chat_words(db: Session, workspace_id: Any, turn: OwnerTurn, role: str, count: int) -> Tuple[str, ...]:
    """The text of the last ``count`` messages by ``role`` in the turn's chat, newest
    first. Read in a savepoint: a refused read must not abort the caller's transaction."""
    from core.models.core import Message
    from modules.memory.thread_checkpoint import extract_message_text

    if turn.chat_id is None:
        return ()
    try:
        with db.begin_nested():
            rows = (db.query(Message.parts)
                    .filter(Message.chat_id == turn.chat_id, Message.workspace_id == _as_uuid(workspace_id),
                            Message.role == role)
                    .order_by(Message.created_at.desc()).limit(count).all())
    except Exception:
        logger.exception("[owner_turn] could not read the %s messages in chat %s", role, turn.chat_id)
        return ()
    return tuple(extract_message_text(row.parts) for row in rows)


def _chat_id(caller_context: Any) -> Optional[UUID]:
    said = (caller_context or {}).get("conversation_id") if isinstance(caller_context, dict) else None
    try:
        return UUID(str(said)) if said else None
    except ValueError:
        return None


def _as_uuid(value: Any) -> Any:
    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except ValueError:
        return value


def cards_named(db: Session, workspace_id: Any, text: str) -> Tuple[NamedCard, ...]:
    """Each card the words name, found on this workspace's board or not: by its
    number ("#0201"), or in words ("card 1879", "step 27.2", F309), each once."""
    from services.ticket_numbers import format_number, resolve_ticket_ref

    named: List[NamedCard] = []
    for match in list(CARD_REF.finditer(text or ""))[:MAX_CARDS]:
        seq, step = int(match.group(1)), (int(match.group(2)) if match.group(2) else None)
        ref = format_number(seq, step)
        named.append(NamedCard(ref=ref, seq=seq, step=step, task=_card(db, workspace_id,
                                                                       resolve_ticket_ref(db, workspace_id, ref))))
    if len(named) < MAX_CARDS:
        named += [card for card in _cards_in_words(db, workspace_id, text) if card.ref not in {n.ref for n in named}]
    return tuple(named[:MAX_CARDS])


def _cards_in_words(db: Session, workspace_id: Any, text: str) -> List[NamedCard]:
    """The cards named in words ("card 1879", "step 27.2"), as the board numbers them."""
    from modules.tools.discovery.card_words_said import card_said, word_refs
    from services.ticket_numbers import ticket_number

    cards: List[NamedCard] = []
    for said in word_refs(text)[:MAX_CARDS]:
        task_id, by_id = card_said(db, workspace_id, said.digits)
        task = _card(db, workspace_id, task_id)
        ref = (ticket_number(db, task) if task is not None else None) or _as_number(said.digits)
        seq, step = _seq_and_step(ref)
        cards.append(NamedCard(ref=ref, seq=seq, step=step, task=task, by_id=by_id))
    return cards


def _seq_and_step(ref: str) -> Tuple[int, Optional[int]]:
    """(42, None) for "#0042", (51, 3) for "#0051.3"."""
    parts = re.fullmatch(r"#(\d+)(?:\.(\d+))?", ref)
    if parts is None:
        return -1, None
    return int(parts.group(1)), (int(parts.group(2)) if parts.group(2) else None)


def _card(db: Session, workspace_id: Any, task_id: Optional[int]) -> Any:
    from core.models.core import BoardTask

    if not task_id:
        return None
    return db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()


def _as_number(digits: str) -> str:
    """The board's form of digits said in words: #0027.2 for "27.2", #1879 for "1879"."""
    from services.ticket_numbers import format_number

    seq, _, step = digits.partition(".")
    return format_number(int(seq), int(step) if step else None)


def says(pattern: re.Pattern, texts: Sequence[str]) -> bool:
    return any(pattern.search(text or "") for text in texts)


def names_card(value: Any, card: NamedCard) -> bool:
    """Whether a call's value is this card's number in any form Auto sends it:
    "#0201", "0201", "201", 201, or 428.2 for step #0428.2."""
    if isinstance(value, bool) or value is None:
        return False
    if card.by_id and card.task is not None and str(value).strip() == str(card.task.id):
        return True   # F309: "card 1879" by its id, and Auto sent 1879
    if isinstance(value, int):
        return card.step is None and value == card.seq
    if isinstance(value, float):
        return f"{value:g}" == (f"{card.seq}.{card.step}" if card.step else str(card.seq))
    match = re.fullmatch(r"#?0*(\d+)(?:[.-]0*(\d+))?", str(value).strip())
    if not match:
        return False
    step = int(match.group(2)) if match.group(2) else None
    return int(match.group(1)) == card.seq and step == card.step


def values_in(params: Any, depth: int = 0) -> List[Any]:
    """Every plain value in a call's parameters, nested ones included (to depth 3)."""
    if depth > 3:
        return []
    if isinstance(params, dict):
        return [v for value in params.values() for v in values_in(value, depth + 1)]
    if isinstance(params, list):
        return [v for value in params for v in values_in(value, depth + 1)]
    return [params]


def card_kind(card: NamedCard) -> str:
    return str(getattr(card.task, "source_type", "") or "")


def card_words(card: NamedCard) -> str:
    """'#0201 ('Reply to Hannah', in Review)' for a refusal."""
    task = card.task
    if task is None:
        return f"{card.ref} (no card on the board has that number)"
    title = str(getattr(task, "title", "") or "").strip()[:80]
    status = str(getattr(task, "status", "") or "").replace("_", " ")
    named_by = f" (id {task.id}, as the owner named it)" if card.by_id else ""
    return f"{card.ref}{named_by} ('{title}', {status})"


__all__ = ["CARD_REF", "SENDS_IT_BACK", "NamedCard", "OwnerTurn", "autos_last_reply", "autos_proposal", "card_kind", "card_words",
           "cards_named", "names_card", "owner_turn", "owners_recent_words", "says", "values_in"]
