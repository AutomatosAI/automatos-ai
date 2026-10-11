"""PRD-256 US-005 (F380): a card moves to Done only when the thing it was for exists.

Night 11's cards closed Done with nothing made: a social card whose agent answered with a
caption, a document card whose answer was instructions for the owner. F380 sent a run that
ended that way to review (``services.social_ticket_post``), but a person's Approve and
Auto's ``platform_update_task_status`` to done still closed the card. The owner: "done on
my board means something was made".

A card's kind is read from its own brief, with the detectors the platform already has:

* a **social post** card: the brief asks for a Socials post
  (``social_post_asks.asks_for_a_social_post``). It needs a post its agent made or changed
  since the run started (``social_ticket_post.runs_posts``, the actor its Socials tools
  write) that has rendered, needs no render, or a person has already approved. With
  Socials off for the workspace no post can be made, so nothing is checked, as in F380;
* a **document** card: the brief asks for customer paperwork
  (``handoffs.asks_for_paperwork``, the paperwork row's words), or names ``generate_document``, which is
  how a card Auto files from chat for a Deliverable says what to make. It needs a
  Deliverable registered on the card (``source_type='task'``, the card's id; a document
  generated while working it, a file its agent wrote, a session's files);
* any other card (a question answered in text) moves as before, and so does a Playbook's or
  a mission's card: their engines file the work under the run (``exec_workspace``'s source
  is the mission or the Playbook, not the card), and judge their own steps.

``missing_artifact`` is the ONE rule: Auto's status tool (``ticket_moves._refusal``), the
board's Approve, its drag and its PATCH all ask it. A refusal names what is missing in
plain words; the card is left where it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from sqlalchemy import text

from modules.tools.discovery.social_post_asks import DOCUMENT_ACTION, asks_for_a_social_post
from services import social_ticket_post as stp
from services.deliverable_tags import CARD_SOURCE_TYPE
from services.ticket_numbers import ticket_label

DONE = "done"
POST, DELIVERABLE = "post", "deliverable"
A_DOCUMENT = "a document"
VOWELS = "aeiou"
TITLE_CHARS = 120
# Where a post stands that makes the card done (``social_ticket_post.render_state``): rendered,
# a text post that needs no render, or one a person already approved, scheduled or published.
MADE_STATES = (stp.RENDERED, stp.NO_RENDER_NEEDED, stp.LEFT_AS_IT_IS)

NO_POST = ("{label} can't be Done: it asks for a Socials post, and its agent has saved no post for it in "
           "Socials. {stays} Send it back with what the post needs, or make the post in the Socials tab.")
NOT_RENDERED = ('{label} can\'t be Done: its Socials post "{title}" {why}. {stays} Render it in the Socials '
                "tab, then approve the card.")
STILL_RENDERING = "is still rendering"
DID_NOT_RENDER = "has not rendered"
NO_DELIVERABLE = ("{label} can't be Done: it asks for {what}, and no Deliverable is linked to it. {stays} "
                  "Send it back so its agent makes {what}, then approve the card.")
STAYS_IN_REVIEW = "It stays in Review."
STAYS_AS_IT_IS = "It stays where it is."
# The cards an engine runs (a Playbook's, a mission's own and its steps'): not read for a kind.
ENGINE_CARD_SOURCES = frozenset({"recipe", "mission", "orchestration", "orchestration_task"})

_CARD_HAS_A_DELIVERABLE = text(
    "SELECT 1 FROM deliverables WHERE workspace_id = CAST(:workspace_id AS uuid) AND source_type = :source_type "
    "AND source_id = :source_id AND deleted_at IS NULL LIMIT 1")


@dataclass(frozen=True)
class ArtifactNeed:
    """What a card is for: a Socials post, or a Deliverable (``what`` says which, in words)."""

    kind: str
    what: str


def artifact_need(task: Any) -> Optional[ArtifactNeed]:
    """The artifact ``task``'s brief asks for, or None for a card that asks for none. Pure."""
    # Read when called: the chatbot package's __init__ loads the chat service, which reaches the board.
    from consumers.chatbot.handoffs import asks_for_paperwork

    if getattr(task, "source_type", None) in ENGINE_CARD_SOURCES:
        return None
    title, description = getattr(task, "title", None) or "", getattr(task, "description", None) or ""
    if asks_for_a_social_post(title, description):
        return ArtifactNeed(POST, "a Socials post")
    brief = f"{title}\n{description}"
    paper = asks_for_paperwork(brief)
    if paper:
        return ArtifactNeed(DELIVERABLE, f"{'an' if paper[:1] in VOWELS else 'a'} {paper}")
    if DOCUMENT_ACTION in brief:
        return ArtifactNeed(DELIVERABLE, A_DOCUMENT)
    return None


def has_a_deliverable(db: Any, task: Any) -> bool:
    """A live Deliverable of this workspace is registered on the card."""
    row = db.execute(_CARD_HAS_A_DELIVERABLE, {
        "workspace_id": str(task.workspace_id), "source_type": CARD_SOURCE_TYPE, "source_id": str(task.id),
    }).first()
    return row is not None


def post_shortfall(db: Any, task: Any, said: Dict[str, str]) -> Optional[str]:
    """Why the card's Socials post is not there yet, in ``said``'s words (the card's label and
    where it stays), or None when one has rendered, needs no render or was approved, or
    Socials is off for the workspace."""
    if stp.socials_on(db, task.workspace_id) is None:
        return None
    agent = getattr(task, "assigned_agent_id", None)
    posts = stp.runs_posts(db, task.workspace_id, stp.AGENT_ACTOR.format(agent_id=agent),
                           stp.run_start(task)) if agent else []
    states = [(stp.render_state(post)[0], post) for post in posts]
    if any(state in MADE_STATES for state, _ in states):
        return None
    if not states:
        return NO_POST.format(**said)
    state, post = states[0]
    why = STILL_RENDERING if state == stp.UNDER_WAY else DID_NOT_RENDER
    return NOT_RENDERED.format(title=str(getattr(post, "title", "") or "")[:TITLE_CHARS], why=why, **said)


def missing_artifact(db: Any, task: Any) -> Optional[str]:
    """What ``task`` lacks to be Done, in plain words, or None when it has it or asks for none."""
    need = artifact_need(task)
    if need is None:
        return None
    said = {"label": ticket_label(task, capital=True),
            "stays": STAYS_IN_REVIEW if getattr(task, "status", None) == "review" else STAYS_AS_IT_IS}
    if need.kind == POST:
        return post_shortfall(db, task, said)
    return None if has_a_deliverable(db, task) else NO_DELIVERABLE.format(what=need.what, **said)


def done_refusal(db: Any, task: Any, new_status: Any) -> Optional[str]:
    """``missing_artifact`` for a move to Done; None for any other move."""
    return missing_artifact(db, task) if new_status == DONE else None


__all__ = ["ArtifactNeed", "artifact_need", "done_refusal", "has_a_deliverable", "missing_artifact"]
