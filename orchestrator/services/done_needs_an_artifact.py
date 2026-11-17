"""PRD-256 US-005 (F380): an agent's card moves to Done only when the thing it was for exists.

Night 11's cards closed Done with nothing made: a social card whose agent answered with a
caption, a document card whose answer was instructions for the owner. F380 sent a run that
ended that way to review (``services.social_ticket_post``), but a person's Approve and
Auto's ``platform_update_task_status`` to done still closed the card. The owner: "done on
my board means something was made".

A card's kind is read from its own brief, with the detectors the platform already has:

* a **social post** card: the brief asks for a Socials post
  (``social_post_asks.asks_for_a_social_post``). It needs a post its agent made or changed
  since the run started (``social_ticket_post.runs_posts``, the actor its Socials tools
  write), or one the brief names by its id that something happened to since then
  (``social_ticket_post.named_posts``, P256-FIX-SOC-4), that has rendered, needs no render, or
  a person has already approved.
  "Make the post in the Socials tab" is said only of a card that names no post. With
  Socials off for the workspace no post can be made, so nothing is checked, as in F380;
* a **document** card: the brief asks for customer paperwork
  (``handoffs.asks_for_paperwork``, the paperwork row's words), or names ``generate_document``, which is
  how a card Auto files from chat for a Deliverable says what to make. It needs a
  Deliverable registered on the card (``source_type='task'``, the card's id; a document
  generated while working it, a file its agent wrote, a session's files);
* any other card (a question answered in text) moves as before, and so does a Playbook's or
  a mission's card: their engines file the work under the run (``exec_workspace``'s source
  is the mission or the Playbook, not the card), and judge their own steps;
* a card filed for the owner's decision (P256-FIX-N14-7, ``approval_cards``) asks for nothing:
  the owner's approval is what it is for, and no agent works it.

Who moves the card decides what the rule does (#0065). The owner decides what is done on his
board; the rule is there to stop agents claiming work they did not do. #0065 asked for a demo
Socials post: Auto made it in the owner's chat and the owner published it himself, so the
card's own agent had saved none, and the board refused the owner's own Approve with no way
to close his card.

* **Not a person** (Auto's own call to its status tool, ``ticket_moves.board_refusal``; an
  agent's run; any automatic path; an in-process call that carries no person's lane; a request
  with no credential): ``done_refusal`` refuses the move with ``missing_artifact``, which names
  what is missing in plain words, and the card is left where it is.
* **A person**: the board's Approve, its drag and its PATCH on a Clerk sign-in or a paired
  phone or browser (``a_persons_request``), and the owner's click on the chat's approval card
  (``ticket_moves``, by ``owner_only.signed_by``, which only a grant he gave sets). The card
  always closes. When the rule would have refused it, the card carries a note saying what was
  missing (``closed_without``, in the refusal's own words for it). Nothing is refused, and the
  person sees no error.

A request with no credential is not a person here, though the local edition treats it as its
operator elsewhere: an agent's ``http_request`` to localhost, or a session's own code, arrives
exactly that way. On a local install, pairing the browser (Settings → Phones & devices) is what
lets the owner close a card without its artifact; unpaired, the board refuses it as before.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from sqlalchemy import text

from modules.tools.discovery.social_post_asks import DOCUMENT_ACTION, asks_for_a_social_post
from services import social_ticket_post as stp
from services.deliverable_tags import CARD_SOURCE_TYPE
from services.themes.membership import PERSON_AUTH_TYPES
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
# P256-FIX-SOC-4: a post the brief names that nothing has touched since the card's run started.
UNCHANGED = "unchanged"
NOT_CHANGED = "has not changed since this card's run started"
NO_DELIVERABLE = ("{label} can't be Done: it asks for {what}, and no Deliverable is linked to it. {stays} "
                  "Send it back so its agent makes {what}, then approve the card.")
STAYS_IN_REVIEW = "It stays in Review."
STAYS_AS_IT_IS = "It stays where it is."
# #0065: the note a person's close leaves on a card the rule would have held for an agent.
CLOSED_WITHOUT_A_POST = "Closed by you without {what} saved for this card."
CLOSED_BEFORE_ITS_POST = 'Closed by you while its Socials post "{title}" {why}.'
CLOSED_WITHOUT_A_DELIVERABLE = "Closed by you without {what}: no Deliverable is linked to this card."
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


@dataclass(frozen=True)
class Shortfall:
    """What keeps a card from Done: what it asks for and, when it has a Socials post that is
    not made yet, that post's title and why, in words (``post_title`` None: it has no post)."""

    need: ArtifactNeed
    post_title: Optional[str] = None
    why: str = ""


def artifact_need(task: Any) -> Optional[ArtifactNeed]:
    """The artifact ``task``'s brief asks for, or None for a card that asks for none. Pure."""
    # Read when called: the chatbot package's __init__ loads the chat service, which reaches the board.
    from consumers.chatbot.handoffs import asks_for_paperwork
    from modules.tools.discovery.approval_cards import is_an_approval_card

    if getattr(task, "source_type", None) in ENGINE_CARD_SOURCES:
        return None
    if is_an_approval_card(getattr(task, "planning_data", None)):  # P256-FIX-N14-7: the card is the decision
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


def shortfall(db: Any, task: Any) -> Optional[Shortfall]:
    """What ``task`` lacks to be Done, or None when it has it or asks for none. The refusal
    (``missing_artifact``) and a person's close (``closed_without``) both say it."""
    need = artifact_need(task)
    if need is None:
        return None
    if need.kind == POST:
        return post_shortfall(db, task, need)
    return None if has_a_deliverable(db, task) else Shortfall(need)


def post_shortfall(db: Any, task: Any, need: ArtifactNeed) -> Optional[Shortfall]:
    """Why the card's Socials post is not there yet (no post, or the first one not made yet,
    with why), or None when one has rendered, needs no render or was approved, or Socials is
    off for the workspace. A post the brief names by its id counts as the card's
    (P256-FIX-SOC-4): its agent was told to work on it, not to make another. It makes the card
    only once something happened to it since the run started (a render, an edit, an approval):
    a post named as an example, made long before, is not this card's work."""
    if stp.socials_on(db, task.workspace_id) is None:
        return None
    agent, since = getattr(task, "assigned_agent_id", None), stp.run_start(task)
    posts = stp.runs_posts(db, task.workspace_id, stp.AGENT_ACTOR.format(agent_id=agent), since) if agent else []
    run_ids = {post.id for post in posts}
    named = [post for post in stp.named_posts(db, task.workspace_id, task) if post.id not in run_ids]
    states = [(stp.render_state(post)[0], post) for post in posts] + [(_named_state(post, since), post) for post in named]
    if any(state in MADE_STATES for state, _ in states):
        return None
    if not states:
        return Shortfall(need)
    state, post = states[0]
    why = {stp.UNDER_WAY: STILL_RENDERING, UNCHANGED: NOT_CHANGED}.get(state, DID_NOT_RENDER)
    return Shortfall(need, str(getattr(post, "title", "") or "")[:TITLE_CHARS], why)


def _named_state(post: Any, since: Any) -> str:
    """Where a post the brief names stands for this card: its render state, or ``UNCHANGED`` when
    it was made already and nothing has happened to it since the run started."""
    state = stp.render_state(post)[0]
    return state if state not in MADE_STATES or stp.changed_since(post, since) else UNCHANGED


def missing_artifact(db: Any, task: Any) -> Optional[str]:
    """What ``task`` lacks to be Done, in plain words, or None when it has it or asks for none:
    the refusal a move to Done gets when it is not a person's (``done_refusal``)."""
    short = shortfall(db, task)
    if short is None:
        return None
    said: Dict[str, str] = {
        "label": ticket_label(task, capital=True),
        "stays": STAYS_IN_REVIEW if getattr(task, "status", None) == "review" else STAYS_AS_IT_IS}
    if short.post_title is not None:
        return NOT_RENDERED.format(title=short.post_title, why=short.why, **said)
    if short.need.kind == POST:
        return NO_POST.format(**said)
    return NO_DELIVERABLE.format(what=short.need.what, **said)


def closed_without(db: Any, task: Any) -> Optional[str]:
    """The note a person's close of ``task`` to Done leaves on the card when the rule would
    have refused an agent's: what was missing, in the refusal's own words for it. None when
    the card has its artifact or asks for none."""
    short = shortfall(db, task)
    if short is None:
        return None
    if short.post_title is not None:
        return CLOSED_BEFORE_ITS_POST.format(title=short.post_title, why=short.why)
    if short.need.kind == POST:
        return CLOSED_WITHOUT_A_POST.format(what=short.need.what)
    return CLOSED_WITHOUT_A_DELIVERABLE.format(what=short.need.what)


def a_persons_request(ctx: Any) -> bool:
    """``ctx`` is a person's own request, by the lane it came in on: a Clerk sign-in or a
    paired phone or browser (``services.themes.membership.PERSON_AUTH_TYPES``). Not a request
    with no credential, on either edition: on a local install an agent's ``http_request`` to
    localhost, or a session's own code, arrives as the anonymous operator. Nor an env or SDK
    key, or a call with no lane (None, or an in-process context such as
    ``mission_step_verdicts``' Approve for Auto)."""
    return getattr(ctx, "auth_type", None) in PERSON_AUTH_TYPES


def done_refusal(db: Any, task: Any, new_status: Any, *, ctx: Any = None) -> Optional[str]:
    """``missing_artifact`` for a move to Done that is not a person's (``ctx`` None: Auto's
    status tool, an agent's run); None for any other move, and for a person's, which closes
    the card and leaves ``closed_without``'s note on it."""
    if new_status != DONE or a_persons_request(ctx):
        return None
    return missing_artifact(db, task)


__all__ = [
    "ArtifactNeed", "CLOSED_BEFORE_ITS_POST", "CLOSED_WITHOUT_A_DELIVERABLE", "CLOSED_WITHOUT_A_POST", "Shortfall",
    "a_persons_request", "artifact_need", "closed_without", "done_refusal", "has_a_deliverable", "missing_artifact",
    "shortfall",
]
